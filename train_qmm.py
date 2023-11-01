# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Iterable
import argparse
import random
from comet_ml import Experiment
from tqdm import tqdm
import numpy as np
import torch

from datasets.use_shards import get_loader
import util.misc as utils
from models import build_model
from models.person_encoder import PersonEncoder, SetInfoNce
from util.plot_utils import plot_label_clip_boxes, plot_pred_clip_boxes, plot_pred_person_link
from models.person_encoder import make_same_person_list


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    # loader
    parser.add_argument('--shards_path', default='/mnt/HDD12TB-1/omi/detr/datasets/shards/UCF101-24', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--n_frames', default=8, type=int)

    # setting
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--ex_name', default='test_ex', type=str)

    # person encoder
    parser.add_argument('--lr_en', default=1e-4, type=float)
    parser.add_argument('--weight_decay_en', default=1e-4, type=float)
    parser.add_argument('--lr_drop_en', default=50, type=int)

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
    detr.eval()
    criterion.eval()

    psn_encoder = PersonEncoder().to(device)
    psn_criterion = SetInfoNce().to(device)

    optimizer_en = torch.optim.AdamW(psn_encoder.parameters(), lr=args.lr_en, weight_decay=args.weight_decay_en)
    lr_scheduler_en = torch.optim.lr_scheduler.StepLR(optimizer_en, args.lr_drop_en)

    data_loader_train = get_loader(shard_path=args.shards_path + "/train", batch_size=args.batch_size, clip_frames=args.n_frames, sampling_rate=1, num_workers=args.num_workers)
    data_loader_val = get_loader(shard_path=args.shards_path + "/val", batch_size=args.batch_size, clip_frames=args.n_frames, sampling_rate=1, num_workers=args.num_workers)

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
    hyper_params = {
        "ex_name": args.ex_name,
        "optimizer_en": str(type(optimizer_en)).split("class")[-1][2:-2],
        "learning late": args.lr_en,
        "batch_size": args.batch_size,
        "n_frames": args.n_frames,
    }
    ex.log_parameters(hyper_params)

    # log loss before training
    evaluate(
        detr, criterion, postprocessors, data_loader_train, device, psn_encoder, psn_criterion, train_log
    )
    ex.log_metric("epoch_train_psn_loss", train_log["psn_loss"].avg, step=0)
    ex.log_metric("epoch_train_diff_psn_score", train_log["diff_psn_score"].avg, step=0)
    ex.log_metric("epoch_train_same_psn_score", train_log["same_psn_score"].avg, step=0)
    ex.log_metric("epoch_train_total_psn_score", train_log["total_psn_score"].avg, step=0)
    [train_log[key].reset() for key in train_log.keys()]

    evaluate(
        detr, criterion, postprocessors, data_loader_val, device, psn_encoder, psn_criterion, val_log
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
            detr, criterion, data_loader_train, optimizer_en, device, epoch,
            psn_encoder, psn_criterion, train_log, ex)

        ex.log_metric("epoch_train_psn_loss", train_log["psn_loss"].avg, step=epoch)
        ex.log_metric("epoch_train_diff_psn_score", train_log["diff_psn_score"].avg, step=epoch)
        ex.log_metric("epoch_train_same_psn_score", train_log["same_psn_score"].avg, step=epoch)
        ex.log_metric("epoch_train_total_psn_score", train_log["total_psn_score"].avg, step=epoch)

        lr_scheduler_en.step()

        evaluate(
            detr, criterion, postprocessors, data_loader_val, device, psn_encoder, psn_criterion, val_log
        )
        ex.log_metric("epoch_val_psn_loss", val_log["psn_loss"].avg, step=epoch)
        ex.log_metric("epoch_val_diff_psn_score", val_log["diff_psn_score"].avg, step=epoch)
        ex.log_metric("epoch_val_same_psn_score", val_log["same_psn_score"].avg, step=epoch)
        ex.log_metric("epoch_val_total_psn_score", val_log["total_psn_score"].avg, step=epoch)

        [train_log[key].reset() for key in train_log.keys()]
        [val_log[key].reset() for key in val_log.keys()]

        utils.save_checkpoint(psn_encoder, args.check_dir, args.ex_name, epoch)


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    psn_encoder: torch.nn.Module, psn_criterion: torch.nn.Module,
                    log: dict, ex: Experiment):

    psn_encoder.train()
    psn_criterion.train()

    step = len(data_loader) * (epoch - 1)

    pbar_batch = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)
    for i, (samples, targets) in pbar_batch:
        samples = samples.to(device)
        targets = [[{k: v.to(device) for k, v in t.items()} for t in vtgt] for vtgt in targets]
        targets = [t for vtgt in targets for t in vtgt]
        b, c, t, h, w = samples.size()
        samples = samples.permute(0, 2, 1, 3, 4)
        samples = samples.reshape(b * t, c, h, w)

        outputs = model(samples)
        _, indices_ex = criterion(outputs, targets)
        p_queries = [outputs["queries"][0, t][idx[0]] for t, idx in enumerate(indices_ex)]  # if idx[0] == None: 空のテンソルが格納
        n_gt_bbox_list = [idx[0].size(0) for idx in indices_ex]  # [frame_id] = n gt bbox
        p_queries = torch.cat(p_queries, 0)
        p_feature_queries = psn_encoder(p_queries)
        p_loss, same_person_label = psn_criterion(p_feature_queries, indices_ex, b, t)
        matching_scores, same_person_lists_clip = make_same_person_list(p_feature_queries.detach(), same_person_label, n_gt_bbox_list, b, t)

        optimizer.zero_grad()
        p_loss.backward()
        optimizer.step()

        log["psn_loss"].update(p_loss.item(), b)
        log["diff_psn_score"].update(matching_scores["diff_psn_score"], b)
        log["same_psn_score"].update(matching_scores["same_psn_score"], b)
        log["total_psn_score"].update(matching_scores["total_psn_score"], b)

        pbar_batch.set_postfix_str(f'loss={log["psn_loss"].val}')
        pbar_batch.set_postfix_str(f'match score={log["total_psn_score"].val}')

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
        loss_dict, indices_ex = criterion(outputs, targets)
        p_queries = [outputs["queries"][0, t][idx[0]] for t, idx in enumerate(indices_ex)]  # if idx[0] == None: 空のテンソルが格納
        p_query_idx2org_query_idx = [(t, idx.item()) for t, idxes in enumerate(indices_ex) for idx in idxes[0]]  # [idx of p_queries] = (frame idx, idx of origin query)
        n_gt_bbox_list = [idx[0].size(0) for idx in indices_ex]
        p_queries = torch.cat(p_queries, 0)
        p_feature_queries = psn_encoder(p_queries)
        p_loss, same_person_label = psn_criterion(p_feature_queries, indices_ex, b, t)
        matching_scores, same_person_lists_clip = make_same_person_list(p_feature_queries.detach(), same_person_label, n_gt_bbox_list, b, t)

        same_person_p_queries = []  # [clip_idx][person_list_idx] = p_query (tensor size is (x, D) x is len(person_list))
        same_person_idx_lists = []  # [clip_idx][person_list_idx] = {frame_idx: origin_query_idx} len in len(person_list)
        for clip_idx, same_person_lists in enumerate(same_person_lists_clip):
            same_person_p_queries.append([])
            same_person_idx_lists.append([])
            for person_list_idx, same_person_list in enumerate(same_person_lists):
                idx_of_p_queries = torch.Tensor(same_person_list["idx_of_p_queries"]).to(torch.int64)
                same_person_p_queries[clip_idx].append(p_queries[idx_of_p_queries])
                same_person_idx_lists[clip_idx].append({p_query_idx2org_query_idx[p_query_idx][0]: p_query_idx2org_query_idx[p_query_idx][1] for p_query_idx in idx_of_p_queries})

        log["psn_loss"].update(p_loss.item(), b)
        log["diff_psn_score"].update(matching_scores["diff_psn_score"], b)
        log["same_psn_score"].update(matching_scores["same_psn_score"], b)
        log["total_psn_score"].update(matching_scores["total_psn_score"], b)

        pbar_batch.set_postfix_str(f'loss={log["psn_loss"].val}')
        pbar_batch.set_postfix_str(f'match score={log["total_psn_score"].val}')

        continue
        orig_target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        # plot
        plot_label_clip_boxes(samples[0:t], targets[0:t])
        plot_pred_clip_boxes(samples[0:t], results[0:t])
        plot_pred_person_link(samples[0:t], results[0:t], same_person_idx_lists[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
