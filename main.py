# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from comet_ml import Experiment
import argparse
import datetime
import json
import random
import time
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
from datasets.use_shards import get_loader
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from models.person_encoder import PersonEncoder, SetInfoNce


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='/mnt/NAS-TVS872XT/dataset/COCO', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--shards_path', default='/mnt/HDD12TB-1/omi/detr/datasets/shards/UCF101-24', type=str)
    return parser


def main(args):
    ex = Experiment(
        project_name="stal",
        workspace="kazukiomi",
    )

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    model.eval()
    criterion.eval()

    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    psn_encoder = PersonEncoder().to(device)
    psn_criterion = SetInfoNce().to(device)

    optimizer = torch.optim.AdamW(psn_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    data_loader_train = get_loader(shard_path=args.shards_path + "/train", batch_size=4, clip_frames=4, sampling_rate=1)
    data_loader_val = get_loader(shard_path=args.shards_path + "/val", batch_size=4, clip_frames=4, sampling_rate=1)

    output_dir = Path(args.output_dir)

    checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
    model_without_ddp.load_state_dict(checkpoint['model'])

    train_log = {"psn_loss": utils.AverageMeter()}
    val_log = {"psn_loss": utils.AverageMeter()}

    if args.eval:
        evaluate(
            model, criterion, postprocessors, data_loader_val, device, args.output_dir, psn_encoder, psn_criterion, val_log, ex
        )
        ex.log_metric("epoch_val_psn_loss", val_log["psn_loss"].avg, step=0)
        val_log["psn_loss"].reset()
        #     return

    print("Start training")
    start_time = time.time()
    pbar_epoch = tqdm(range(1, args.epochs))
    for epoch in pbar_epoch:
        pbar_epoch.set_description(f"[Epoch {epoch}]")
        train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, psn_encoder, psn_criterion, train_log, ex)

        ex.log_metric("epoch_train_psn_loss", train_log["psn_loss"].avg, step=epoch)

        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        evaluate(
            model, criterion, postprocessors, data_loader_val, device, args.output_dir, psn_encoder, psn_criterion, val_log, ex
        )
        ex.log_metric("epoch_val_psn_loss", val_log["psn_loss"].avg, step=epoch)

        train_log["psn_loss"].reset()
        val_log["psn_loss"].reset()

        # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #              **{f'test_{k}': v for k, v in test_stats.items()},
        #              'epoch': epoch,
        #              'n_parameters': n_parameters}

        # if args.output_dir and utils.is_main_process():
        #     with (output_dir / "log.txt").open("a") as f:
        #         f.write(json.dumps(log_stats) + "\n")

        # for evaluation logs
        # if coco_evaluator is not None:
        #     (output_dir / 'eval').mkdir(exist_ok=True)
        #     if "bbox" in coco_evaluator.coco_eval:
        #         filenames = ['latest.pth']
        #         if epoch % 50 == 0:
        #             filenames.append(f'{epoch:03}.pth')
        #         for name in filenames:
        #             torch.save(coco_evaluator.coco_eval["bbox"].eval,
        #                        output_dir / "eval" / name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
