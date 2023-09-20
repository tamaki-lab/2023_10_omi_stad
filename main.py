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
    # loader
    parser.add_argument('--shards_path', default='/mnt/HDD12TB-1/omi/detr/datasets/shards/UCF101-24', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--n_frames', default=8, type=int)

    # setting
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--ex_name', default='test_ex', type=str)

    # * person encoder
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # * Backbone
    parser.add_argument('--backbone', default='resnet101', type=str, choices=('resnet50', 'resnet101'),
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', default=True,
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")

    # others
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--check_dir', default="checkpoint", type=str)
    parser.add_argument('--num_workers', default=8, type=int)

    return parser


def main(args):
    device = torch.device(f"cuda:{args.device}")

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    criterion.to(device)
    model.to(device)
    model.eval()
    criterion.eval()

    psn_encoder = PersonEncoder().to(device)
    psn_criterion = SetInfoNce().to(device)

    optimizer = torch.optim.AdamW(psn_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    data_loader_train = get_loader(shard_path=args.shards_path + "/train", batch_size=args.batch_size, clip_frames=args.n_frames, sampling_rate=1, num_workers=args.num_workers)
    data_loader_val = get_loader(shard_path=args.shards_path + "/val", batch_size=args.batch_size, clip_frames=args.n_frames, sampling_rate=1, num_workers=args.num_workers)

    pretrain_path = "checkpoint/detr/" + utils.get_pretrain_path(args.backbone, args.dilation)
    model.load_state_dict(torch.load(pretrain_path)["model"])

    train_log = {"psn_loss": utils.AverageMeter(),
                 "diff_psn_score": utils.AverageMeter(),
                 "same_psn_score": utils.AverageMeter(),
                 "total_psn_score": utils.AverageMeter()}
    val_log = {"psn_loss": utils.AverageMeter(),
               "diff_psn_score": utils.AverageMeter(),
               "same_psn_score": utils.AverageMeter(),
               "total_psn_score": utils.AverageMeter()}
    # exit()

    ex = Experiment(
        project_name="stal",
        workspace="kazukiomi",
    )
    hyper_params = {
        "ex_name": args.ex_name,
        "optimizer": str(type(optimizer)).split("class")[-1][2:-2],
        "learning late": args.lr,
        "batch_size": args.batch_size,
        "n_frames": args.n_frames,
    }
    ex.log_parameters(hyper_params)

    # log loss before training
    evaluate(
        model, criterion, postprocessors, data_loader_train, device, psn_encoder, psn_criterion, train_log
    )
    ex.log_metric("epoch_train_psn_loss", train_log["psn_loss"].avg, step=0)
    ex.log_metric("epoch_train_diff_psn_score", train_log["diff_psn_score"].avg, step=0)
    ex.log_metric("epoch_train_same_psn_score", train_log["same_psn_score"].avg, step=0)
    ex.log_metric("epoch_train_total_psn_score", train_log["total_psn_score"].avg, step=0)
    [train_log[key].reset() for key in train_log.keys()]

    evaluate(
        model, criterion, postprocessors, data_loader_val, device, psn_encoder, psn_criterion, val_log
    )
    ex.log_metric("epoch_val_psn_loss", val_log["psn_loss"].avg, step=0)
    ex.log_metric("epoch_val_diff_psn_score", val_log["diff_psn_score"].avg, step=0)
    ex.log_metric("epoch_val_same_psn_score", val_log["same_psn_score"].avg, step=0)
    ex.log_metric("epoch_val_total_psn_score", val_log["total_psn_score"].avg, step=0)
    [val_log[key].reset() for key in val_log.keys()]

    print("Start training")

    pbar_epoch = tqdm(range(1, args.epochs + 1))
    for epoch in pbar_epoch:
        pbar_epoch.set_description(f"[Epoch {epoch}]")
        train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            psn_encoder, psn_criterion, train_log, ex)

        ex.log_metric("epoch_train_psn_loss", train_log["psn_loss"].avg, step=epoch)
        ex.log_metric("epoch_train_diff_psn_score", train_log["diff_psn_score"].avg, step=epoch)
        ex.log_metric("epoch_train_same_psn_score", train_log["same_psn_score"].avg, step=epoch)
        ex.log_metric("epoch_train_total_psn_score", train_log["total_psn_score"].avg, step=epoch)

        lr_scheduler.step()

        evaluate(
            model, criterion, postprocessors, data_loader_val, device, psn_encoder, psn_criterion, val_log
        )
        ex.log_metric("epoch_val_psn_loss", val_log["psn_loss"].avg, step=epoch)
        ex.log_metric("epoch_val_diff_psn_score", val_log["diff_psn_score"].avg, step=epoch)
        ex.log_metric("epoch_val_same_psn_score", val_log["same_psn_score"].avg, step=epoch)
        ex.log_metric("epoch_val_total_psn_score", val_log["total_psn_score"].avg, step=epoch)

        [train_log[key].reset() for key in train_log.keys()]
        [val_log[key].reset() for key in val_log.keys()]

        utils.save_checkpoint(model, args.check_dir, args.ex_name, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
