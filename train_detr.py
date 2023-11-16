

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Iterable
import argparse
import random
from comet_ml import Experiment
from tqdm import tqdm
import numpy as np
import torch
import os.path as osp

from datasets.use_shards import get_loader
import util.misc as utils
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    # setting #
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--ex_name', default='test', type=str)
    # loader
    parser.add_argument('--shards_path', default='/mnt/HDD12TB-1/omi/detr/datasets/shards', type=str)
    parser.add_argument('--dataset', default='ucf101-24', type=str, choices=['ucf101-24', 'jhmdb21'])
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--n_frames', default=1, type=int)
    # detr
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lr_drop', default=15, type=int)

    # Fixed settings #
    # Backbone
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

    detr, criterion, postprocessors = build_model(args)
    criterion.to(device)
    detr.to(device)

    # for name, param in detr.named_parameters():
    #     if ("class" in name) or ("bbox") in name:
    #         continue
    #     else:
    #         param.requires_grad = False

    # optimizer = torch.optim.AdamW(list(detr.class_embed.parameters()) + list(detr.bbox_embed.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(detr.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    shards_path = osp.join(args.shards_path, args.dataset)
    train_loader = get_loader(shard_path=shards_path + "/train", batch_size=args.batch_size, clip_frames=args.n_frames, sampling_rate=1, num_workers=args.num_workers)
    val_loader = get_loader(shard_path=shards_path + "/val", batch_size=args.batch_size, clip_frames=args.n_frames, sampling_rate=1, num_workers=args.num_workers)

    pretrain_path = "checkpoint/detr/" + utils.get_pretrain_path(args.backbone, args.dilation)
    detr.load_state_dict(torch.load(pretrain_path)["model"])

    train_log = {"total_loss": utils.AverageMeter(),
                 "class_loss": utils.AverageMeter(),
                 "bbox_loss": utils.AverageMeter(),
                 "giou_loss": utils.AverageMeter()}
    val_log = {"total_loss": utils.AverageMeter(),
               "class_loss": utils.AverageMeter(),
               "bbox_loss": utils.AverageMeter(),
               "giou_loss": utils.AverageMeter()}

    ex = Experiment(
        project_name="stal",
        workspace="kazukiomi",
    )
    hyper_params = {
        "dataset": args.dataset,
        "ex_name": args.ex_name,
        "batch_size": args.batch_size,
        "n_frames": args.n_frames,
    }
    ex.log_parameters(hyper_params)

    # log loss before training #
    evaluate(detr, criterion, train_loader, device, train_log)
    leave_ex(ex, "train", train_log, 0)
    evaluate(detr, criterion, val_loader, device, val_log)
    leave_ex(ex, "val", val_log, 0)

    print("Start training")

    pbar_epoch = tqdm(range(1, args.epochs + 1))
    for epoch in pbar_epoch:
        pbar_epoch.set_description(f"[Epoch {epoch}]")
        train(detr, criterion, train_loader, optimizer, device, epoch, train_log, ex)
        leave_ex(ex, "train", train_log, epoch)

        lr_scheduler.step()

        evaluate(detr, criterion, val_loader, device, val_log)
        leave_ex(ex, "val", val_log, epoch)

        utils.save_checkpoint(detr, osp.join(args.check_dir, args.dataset), args.ex_name + "/detr", epoch)


def train(detr: torch.nn.Module, criterion: torch.nn.Module,
          loader: Iterable, optimizer: torch.optim.Optimizer,
          device: torch.device, epoch: int, log: dict, ex: Experiment):

    step = len(loader) * (epoch - 1)
    detr.train()
    criterion.train()

    pbar_batch = tqdm(enumerate(loader), total=len(loader), leave=False)
    for i, (samples, targets) in pbar_batch:
        samples = samples.to(device)
        targets = [[{k: v.to(device) for k, v in t.items()} for t in vtgt] for vtgt in targets]
        targets = [t for vtgt in targets for t in vtgt]
        b, c, t, h, w = samples.size()
        samples = samples.permute(0, 2, 1, 3, 4)
        samples = samples.reshape(b * t, c, h, w)

        outputs = detr(samples)
        loss_dict, _ = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        loss_list = [loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict]
        total_loss = sum(loss_list)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        update_log(log, total_loss, loss_list, b)

        pbar_batch.set_postfix_str(f'loss_avg={round(log["total_loss"].avg,3)}, loss_val={round(log["total_loss"].val,3)}')

        ex.log_metric("batch_total_loss", log["total_loss"].avg, step=step + i)
        ex.log_metric("batch_class_loss", log["class_loss"].avg, step=step + i)
        ex.log_metric("batch_bbox_loss", log["bbox_loss"].avg, step=step + i)
        ex.log_metric("batch_giou_loss", log["giou_loss"].avg, step=step + i)


@torch.no_grad()
def evaluate(detr, criterion, loader, device, log):
    detr.eval()
    criterion.eval()

    pbar_batch = tqdm(loader, total=len(loader), leave=False)
    for samples, targets in pbar_batch:
        samples = samples.to(device)
        targets = [[{k: v.to(device) for k, v in t.items()} for t in vtgt] for vtgt in targets]
        targets = [t for vtgt in targets for t in vtgt]
        b, c, t, h, w = samples.size()
        samples = samples.permute(0, 2, 1, 3, 4)
        samples = samples.reshape(b * t, c, h, w)

        outputs = detr(samples)
        loss_dict, _ = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        loss_list = [loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict]
        total_loss = sum(loss_list)

        update_log(log, total_loss, loss_list, b)


def update_log(log, total_loss, loss_list, b):
    log["total_loss"].update(total_loss.item(), b)
    log["class_loss"].update(loss_list[0].item(), b)
    log["bbox_loss"].update(loss_list[1].item(), b)
    log["giou_loss"].update(loss_list[2].item(), b)


def leave_ex(ex, subset, log, epoch):
    ex.log_metric("epoch_" + subset + "_total_loss", log["total_loss"].avg, step=epoch)
    ex.log_metric("epoch_" + subset + "_class_loss", log["class_loss"].avg, step=epoch)
    ex.log_metric("epoch_" + subset + "_bbox_loss", log["bbox_loss"].avg, step=epoch)
    ex.log_metric("epoch_" + subset + "_giou_loss", log["giou_loss"].avg, step=epoch)

    [log[key].reset() for key in log.keys()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
