# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Iterable
import argparse
import random
from comet_ml import Experiment
from tqdm import tqdm
import numpy as np
import torch
import os.path as osp
import yaml

from util.box_ops import generalized_box_iou, box_cxcywh_to_xyxy
from datasets.use_shards import get_loader
import util.misc as utils
from models import build_model
from models.person_encoder import PersonEncoder, NPairLoss, make_same_person_list
from util.plot_utils import plot_pred_clip_boxes, plot_diff_results


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    # setting #
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--qmm_name', default='test', type=str)
    # loader
    parser.add_argument('--shards_path', default='/mnt/HDD12TB-1/omi/detr/datasets/shards', type=str)
    parser.add_argument('--dataset', default='jhmdb21', type=str, choices=['ucf101-24', 'jhmdb21'])
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--n_frames', default=8, type=int)
    parser.add_argument('--sampling_rate', default=1, type=int)

    # person encoder
    parser.add_argument('--lr_en', default=1e-4, type=float)
    parser.add_argument('--weight_decay_en', default=1e-4, type=float)
    parser.add_argument('--lr_drop_en', default=10, type=int)
    parser.add_argument('--psn_score_th', default=0.75, type=float)
    parser.add_argument('--iou_th', default=0.2, type=float)
    parser.add_argument('--is_skip', action="store_true")

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
    if args.dataset == "jhmdb21":
        params = yaml.safe_load(open(f"datasets/projects/{args.dataset}.yml"))
        args.iou_th = params["iou_th"]
        args.psn_score_th = params["psn_score_th"]

    device = torch.device(f"cuda:{args.device}")

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    detr, criterion, postprocessors = build_model(args)
    criterion.to(device)
    detr.to(device)
    detr.eval()
    criterion.eval()

    psn_encoder = PersonEncoder(skip=args.is_skip).to(device)
    psn_criterion = NPairLoss().to(device)

    optimizer_en = torch.optim.AdamW(psn_encoder.parameters(), lr=args.lr_en, weight_decay=args.weight_decay_en)
    lr_scheduler_en = torch.optim.lr_scheduler.StepLR(optimizer_en, args.lr_drop_en)

    shards_path = osp.join(args.shards_path, args.dataset)
    data_loader_train = get_loader(shard_path=shards_path + "/train", batch_size=args.batch_size, clip_frames=args.n_frames, sampling_rate=args.sampling_rate, num_workers=args.num_workers)
    data_loader_val = get_loader(shard_path=shards_path + "/val", batch_size=args.batch_size, clip_frames=args.n_frames, sampling_rate=1, num_workers=args.num_workers)

    if args.dataset == "ucf101-24":
        pretrain_path = "checkpoint/ucf101-24/w:252/detr/epoch_10.pth"
        # pretrain_path = "checkpoint/ucf101-24/w:252/detr/epoch_20.pth"
        detr.load_state_dict(torch.load(pretrain_path))
    else:
        pretrain_path = "checkpoint/detr/" + utils.get_pretrain_path(args.backbone, args.dilation)
        detr.load_state_dict(torch.load(pretrain_path)["model"])

    train_log = {"psn_loss": utils.AverageMeter(),
                 "diff_psn_score": utils.AverageMeter(),
                 "same_psn_score": utils.AverageMeter(),
                 "total_psn_score": utils.AverageMeter()}
    val_log = {"psn_loss": utils.AverageMeter(),
               "diff_psn_score": utils.AverageMeter(),
               "same_psn_score": utils.AverageMeter(),
               "total_psn_score": utils.AverageMeter()}

    ex = Experiment(
        project_name="stal",
        workspace="kazukiomi",
    )
    ex.add_tag("train qmm")
    hyper_params = {
        "dataset": args.dataset,
        "ex_name": args.qmm_name,
        "batch_size": args.batch_size,
        "n_frames": args.n_frames,
        "learning late": args.lr_en,
        "lr_drop_epoch": args.lr_drop_en,
        "psn_score_th": args.psn_score_th,
        "iou_th": args.iou_th
    }
    ex.log_parameters(hyper_params)

    ## log loss before training ##
    # evaluate(detr, criterion, postprocessors, data_loader_train, device, psn_encoder, psn_criterion, train_log)
    # leave_ex(ex, "train", train_log, 0)

    # evaluate(detr, criterion, postprocessors, data_loader_val, device, psn_encoder, psn_criterion, val_log)
    # leave_ex(ex, "val", val_log, 0)

    print("Start training")

    pbar_epoch = tqdm(range(1, args.epochs + 1))
    for epoch in pbar_epoch:
        pbar_epoch.set_description(f"[Epoch {epoch}]")
        train_one_epoch(
            detr, criterion, postprocessors, data_loader_train, optimizer_en, device, epoch,
            psn_encoder, psn_criterion, train_log, ex)

        leave_ex(ex, "train", train_log, epoch)

        lr_scheduler_en.step()

        evaluate(
            detr, criterion, postprocessors, data_loader_val, device, psn_encoder, psn_criterion, val_log
        )
        leave_ex(ex, "val", val_log, epoch)

        utils.save_checkpoint(psn_encoder, osp.join(args.check_dir, args.dataset), args.qmm_name + "/encoder", epoch)


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    postprocessors: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    psn_encoder: torch.nn.Module, psn_criterion: torch.nn.Module,
                    log: dict, ex: Experiment):

    psn_encoder.train()
    psn_criterion.train()

    step = len(data_loader) * (epoch - 1)

    pbar_batch = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)
    for i, (samples, targets) in pbar_batch:
        with torch.inference_mode():
            samples = samples.to(device)
            targets = [[{k: v.to(device) for k, v in t.items()} for t in vtgt] for vtgt in targets]
            targets = [t for vtgt in targets for t in vtgt]
            b, c, t, h, w = samples.size()
            samples = samples.permute(0, 2, 1, 3, 4)
            samples = samples.reshape(b * t, c, h, w)

            outputs = model(samples)
            _, indices_ex = criterion(outputs, targets)

            # ハンガリアンマッチングで選ばれたindices_exの中でも正しく人物を捉えているidのみを保持するように変更
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessors['bbox'](outputs, orig_target_sizes)

            score_filter_indices = [(result["scores"] > args.psn_score_th).nonzero().flatten() for result in results]
            score_filter_labels = [result["labels"]
                                   [(result["scores"] > args.psn_score_th).nonzero().flatten()] for result in results]
            psn_indices = [idx[lab == 1] for idx, lab in zip(score_filter_indices, score_filter_labels)]
            psn_boxes = [result["boxes"][p_idx].cpu() for result, p_idx in zip(results, psn_indices)]
            box_filter_indices = box_filter(psn_boxes, targets, orig_target_sizes, psn_indices, args.iou_th)
            indices = [a[0][torch.isin(a[0], b.cpu())] for a, b in zip(indices_ex, box_filter_indices)]
            decoded_queries = [outputs["queries"][0, t][idx] for t, idx in enumerate(indices)]

        labels = psn_criterion.label_rearrange(indices, b, t).to(device)
        if labels.shape[0] == 0:
            continue
        psn_embedding = psn_encoder(torch.cat(decoded_queries, 0))
        loss = psn_criterion(psn_embedding, labels)

        n_gt_bbox_list = [idx.size(0) for idx in indices]  # [frame_id] = n gt bbox
        matching_scores, _ = make_same_person_list(psn_embedding.detach(), labels, n_gt_bbox_list, b, t)

        if not loss.requires_grad:
            continue
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        update_log(log, loss, matching_scores, b)

        pbar_batch.set_postfix_str(
            f'loss={round(log["psn_loss"].avg, 3)}, match score={round(log["total_psn_score"].avg, 3)}')

        ex.log_metric("batch_psn_loss", log["psn_loss"].val, step=step + i)
        ex.log_metric("batch_diff_psn_score", log["diff_psn_score"].val, step=step + i)
        ex.log_metric("batch_same_psn_score", log["same_psn_score"].val, step=step + i)
        ex.log_metric("batch_total_psn_score", log["total_psn_score"].val, step=step + i)


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, device, psn_encoder, psn_criterion, log):
    psn_encoder.eval()
    psn_criterion.eval()

    pbar_batch = tqdm(data_loader, total=len(data_loader), leave=False)
    for samples, targets in pbar_batch:
        samples = samples.to(device)
        targets = [[{k: v.to(device) for k, v in t.items()} for t in vtgt] for vtgt in targets]
        targets = [t for vtgt in targets for t in vtgt]
        b, c, t, h, w = samples.size()
        samples = samples.permute(0, 2, 1, 3, 4)
        samples = samples.reshape(b * t, c, h, w)

        outputs = model(samples)
        _, indices_ex = criterion(outputs, targets)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        score_filter_indices = [(result["scores"] > args.psn_score_th).nonzero().flatten() for result in results]
        score_filter_labels = [result["labels"][(result["scores"] > args.psn_score_th).nonzero().flatten()]
                               for result in results]
        psn_indices = [idx[lab == 1] for idx, lab in zip(score_filter_indices, score_filter_labels)]
        psn_boxes = [result["boxes"][p_idx].cpu() for result, p_idx in zip(results, psn_indices)]
        box_filter_indices = box_filter(psn_boxes, targets, orig_target_sizes, psn_indices, args.iou_th)
        indices = [a[0][torch.isin(a[0], b.cpu())] for a, b in zip(indices_ex, box_filter_indices)]
        decoded_queries = [outputs["queries"][0, t][idx] for t, idx in enumerate(indices)]

        labels = psn_criterion.label_rearrange(indices, b, t).to(device)
        if labels.shape[0] == 0:
            continue
        psn_embedding = psn_encoder(torch.cat(decoded_queries, 0))
        loss = psn_criterion(psn_embedding, labels)

        n_gt_bbox_list = [idx.size(0) for idx in indices]  # [frame_id] = n gt bbox
        matching_scores, _ = make_same_person_list(psn_embedding, labels, n_gt_bbox_list, b, t)

        update_log(log, loss, matching_scores, b)

        pbar_batch.set_postfix_str(
            f'loss={round(log["psn_loss"].avg, 3)}, match score={round(log["total_psn_score"].avg, 3)}')

        # plot #
        continue
        target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, target_sizes)
        plot_pred_clip_boxes(samples[0:t], results[0:t], targets[0:t], plot_label=True)


def box_filter(psn_boxes, targets, org_sizes, indices_list, th=0.4):
    device = indices_list[0].device
    new_indices = []
    for pred_boxes, gt_boxes, org_size, indices in zip(psn_boxes, targets, org_sizes, indices_list):
        gt_boxes = gt_boxes["boxes"]
        if gt_boxes.size(0) == 0:
            new_indices.append(torch.Tensor().to(torch.int64).to(device))
            continue
        gt_boxes = box_cxcywh_to_xyxy(gt_boxes)
        gt_boxes[:, 0::2] = gt_boxes[:, 0::2] * org_size[1]
        gt_boxes[:, 1::2] = gt_boxes[:, 1::2] * org_size[0]
        iou = generalized_box_iou(pred_boxes, gt_boxes.cpu())
        max_v, max_idx = torch.max(iou, dim=1)
        new_indices.append(indices[max_v > th])

    return new_indices


def update_log(log, loss, matching_scores, b):
    log["psn_loss"].update(loss.item(), b)
    log["diff_psn_score"].update(matching_scores["diff_psn_score"], b)
    log["same_psn_score"].update(matching_scores["same_psn_score"], b)
    log["total_psn_score"].update(matching_scores["total_psn_score"], b)


def leave_ex(ex, subset, log, epoch):
    ex.log_metric("epoch_" + subset + "_psn_loss", log["psn_loss"].avg, step=epoch)
    ex.log_metric("epoch_" + subset + "_diff_psn_score", log["diff_psn_score"].avg, step=epoch)
    ex.log_metric("epoch_" + subset + "_same_psn_score", log["same_psn_score"].avg, step=epoch)
    ex.log_metric("epoch_" + subset + "_total_psn_score", log["total_psn_score"].avg, step=epoch)

    [log[key].reset() for key in log.keys()]


@torch.no_grad()
def diff_detr_head(args):
    device = torch.device(f"cuda:{args.device}")

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    org_detr, criterion, postprocessors = build_model(args)
    detr, _, _ = build_model(args)
    criterion.to(device)
    detr.to(device)
    detr.eval()
    org_detr.to(device)
    org_detr.eval()
    criterion.eval()

    shards_path = osp.join(args.shards_path, args.dataset)
    # data_loader = get_loader(shard_path=shards_path + "/train", batch_size=args.batch_size, clip_frames=args.n_frames, sampling_rate=1, num_workers=args.num_workers)
    data_loader = get_loader(shard_path=shards_path + "/val", batch_size=args.batch_size, clip_frames=args.n_frames, sampling_rate=1, num_workers=args.num_workers)

    org_pretrain_path = "checkpoint/detr/" + utils.get_pretrain_path(args.backbone, args.dilation)
    org_detr.load_state_dict(torch.load(org_pretrain_path)["model"])
    # pretrain_path = "checkpoint/ucf101-24/test/detr/epoch_50.pth"
    pretrain_path = "checkpoint/ucf101-24/w:252/detr/epoch_20.pth"
    detr.load_state_dict(torch.load(pretrain_path))

    pbar_batch = tqdm(data_loader, total=len(data_loader), leave=False)
    for samples, targets in pbar_batch:
        samples = samples.to(device)
        targets = [[{k: v.to(device) for k, v in t.items()} for t in vtgt] for vtgt in targets]
        targets = [t for vtgt in targets for t in vtgt]
        b, c, t, h, w = samples.size()
        samples = samples.permute(0, 2, 1, 3, 4)
        samples = samples.reshape(b * t, c, h, w)

        org_outputs = org_detr(samples)
        outputs = detr(samples)

        target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        org_results = postprocessors['bbox'](org_outputs, target_sizes)
        results = postprocessors['bbox'](outputs, target_sizes)
        plot_diff_results(samples[0:t], org_results[0:t], results[0:t], targets[0:t], plot_label=True)
        continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
    # diff_detr_head(args)
