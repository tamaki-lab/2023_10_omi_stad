# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import random
from comet_ml import Experiment
from tqdm import tqdm
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import yaml
# from PIL import Image

from datasets.dataset import VideoDataset
import util.misc as utils
from models import build_model
from models.action_head import ActionHead, ActionHead2, X3D_XS


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    # setting #
    parser.add_argument('--n_epochs', default=20, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--head_name', default='head_test', type=str)
    parser.add_argument('--link_cues', default='feature', type=str)

    # loader
    parser.add_argument('--dataset', default='jhmdb21', type=str, choices=['ucf101-24', 'jhmdb21'])

    # person encoder
    parser.add_argument('--qmm_name', default='noskip_sr:4', type=str)
    parser.add_argument('--load_epoch', default=20, type=int)
    parser.add_argument('--psn_score_th', default=0.9, type=float)
    parser.add_argument('--sim_th', default=0.5, type=float)
    parser.add_argument('--filter_length', default=8, type=int)

    # Backbone
    parser.add_argument('--backbone', default='resnet101', type=str, choices=('resnet50', 'resnet101'),
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', default=True,
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")

    # action head
    parser.add_argument('--head_type', default='vanilla', type=str, choices=["vanilla", "time_ecd:add", "time_ecd:cat", "res", "x3d"])
    parser.add_argument('--lr_head', default=1e-3, type=float)
    parser.add_argument('--weight_decay_head', default=1e-4, type=float)
    parser.add_argument('--lr_drop_head', default=15, type=int)
    parser.add_argument('--iter_update', default=64, type=int)

    # others
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--check_dir', default="checkpoint", type=str)
    parser.add_argument('--num_workers', default=8, type=int)

    return parser


def main(args, params):
    device = torch.device(f"cuda:{args.device}")

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    detr, _, _ = build_model(args)
    detr.to(device)
    detr.eval()

    if args.head_type == "vanilla":
        action_head = ActionHead(n_classes=args.n_classes, pos_ecd=(False, "", None)).to(device)
    elif args.head_type == "time_ecd:add":
        action_head = ActionHead(n_classes=args.n_classes, pos_ecd=(True, "add", None)).to(device)
    elif args.head_type == "time_ecd:cat":
        action_head = ActionHead(n_classes=args.n_classes, pos_ecd=(True, "cat", 32)).to(device)
    else:
        action_head = ActionHead2(n_classes=args.n_classes, pos_ecd=(True, "cat", 32)).to(device)

    if args.dataset == "ucf101-24":
        head_criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0] * params["num_classes"] + [0.1])).to(device)
    else:
        head_criterion = nn.CrossEntropyLoss()
    optimizer_head = torch.optim.AdamW(action_head.parameters(), lr=args.lr_head, weight_decay=args.weight_decay_head)
    lr_scheduler_head = torch.optim.lr_scheduler.StepLR(optimizer_head, args.lr_drop_head)

    if args.link_cues == "feature":
        dir = osp.join(args.check_dir, args.dataset, args.qmm_name, "qmm_tubes")
        filename = f"tube-epoch:{args.load_epoch}_pth:{args.psn_score_th}_simth:{args.sim_th}_fl:{args.filter_length}"
    elif args.link_cues == "iou":
        dir = osp.join(args.check_dir, args.dataset, "iou_link", "qmm_tubes")
        filename = f"tube-pth:{args.psn_score_th}_iouth:{args.sim_th}_fl:{args.filter_length}"

    train_loader = utils.TarIterator(dir + "/train", filename)
    val_loader = utils.TarIterator(dir + "/val", filename)

    train_dataset = VideoDataset(args.dataset, "train")
    val_dataset = VideoDataset(args.dataset, "val")
    x3d_xs = X3D_XS().to(device)
    x3d_xs.eval()

    train_log = {"loss": utils.AverageMeter(),
                 "acc1": utils.AverageMeter(),
                 "acc5": utils.AverageMeter(),
                 "acc1_wo_noaction": utils.AverageMeter(),
                 "acc5_wo_noaction": utils.AverageMeter()}
    val_log = {"loss": utils.AverageMeter(),
               "acc1": utils.AverageMeter(),
               "acc5": utils.AverageMeter(),
               "acc1_wo_noaction": utils.AverageMeter(),
               "acc5_wo_noaction": utils.AverageMeter()}

    ex = Experiment(
        project_name="stal",
        workspace="kazukiomi",
    )
    ex.add_tag("train action head")
    hyper_params = {
        "ex_name": args.qmm_name + "--" + args.head_name,
    }
    ex.log_parameters(hyper_params)

    pbar_epoch = tqdm(range(1, args.n_epochs + 1), leave=False)
    for epoch in pbar_epoch:
        pbar_epoch.set_description(f"[Epoch {epoch}]")

        ## Training ##
        action_head.train()
        step = len(train_loader) * (epoch - 1)

        pbar_tubes = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        pbar_tubes.set_description("[Training]")
        for tube_idx, tube in pbar_tubes:
            frame_indices = [x[0] for x in tube.query_indicies]
            frame_indices = [x - frame_indices[0] for x in frame_indices]

            action_label = torch.Tensor(tube.action_label).to(torch.int64).to(device)
            decoded_queries = torch.stack(tube.decoded_queries).to(device)

            if tube_idx % args.iter_update == 0:
                optimizer_head.zero_grad()

            if args.head_type == "vanilla" or args.head_type == "time_ecd:add":
                outputs = action_head(decoded_queries)
            elif args.head_type == "time_ecd:cat":
                outputs = action_head(decoded_queries, frame_indices)
            else:
                if args.head_type == "res":
                    frame_features = utils.get_frame_features(detr.backbone, tube.video_name, frame_indices, train_dataset, device, True)
                elif args.head_type == "x3d":
                    frame_features = utils.get_frame_features(x3d_xs, tube.video_name, frame_indices, train_dataset, device, True)
                outputs = action_head(frame_features, decoded_queries, frame_indices)

            tube.log_pred(outputs)
            loss = head_criterion(outputs, action_label) / args.iter_update
            loss.backward()

            acc_dict = utils.calc_acc(tube.action_pred, action_label, params["num_classes"])
            if tube_idx % args.iter_update == args.iter_update - 1:
                optimizer_head.step()

            train_log["loss"].update(loss.item() * args.iter_update)
            train_log["acc1"].update(acc_dict["acc1"].item())
            train_log["acc5"].update(acc_dict["acc5"].item())
            ex.log_metric("iter_action_loss", train_log["loss"].avg, step=step + tube_idx)
            ex.log_metric("iter_action_acc1", train_log["acc1"].avg, step=step + tube_idx)
            ex.log_metric("iter_action_acc5", train_log["acc5"].avg, step=step + tube_idx)
            if acc_dict["acc1_wo"] == -1:
                continue
            train_log["acc1_wo_noaction"].update(acc_dict["acc1_wo"].item())
            train_log["acc5_wo_noaction"].update(acc_dict["acc5_wo"].item())
            ex.log_metric("iter_action_acc1_wo_noaction", train_log["acc1_wo_noaction"].avg, step=step + tube_idx)
            ex.log_metric("iter_action_acc5_wo_noaction", train_log["acc5_wo_noaction"].avg, step=step + tube_idx)

            pbar_tubes.set_postfix_str(f'loss={round(train_log["loss"].avg, 3)}, acc1: {round(train_log["acc1"].avg, 3)}, acc1_w/o: {round(train_log["acc1_wo_noaction"].avg, 3)}')

        leave_ex(ex, "train", train_log, epoch)

        # validation
        with torch.inference_mode():
            action_head.eval()

            pbar_tubes = tqdm(enumerate(val_loader), total=len(val_loader), leave=False)
            pbar_tubes.set_description("[Validation]")
            for tube_idx, tube in pbar_tubes:
                action_label = torch.Tensor(tube.action_label).to(torch.int64).to(device)
                decoded_queries = torch.stack(tube.decoded_queries).to(device)

                frame_indices = [x[0] for x in tube.query_indicies]
                frame_indices = [x - frame_indices[0] for x in frame_indices]

                if args.head_type == "vanilla" or args.head_type == "time_ecd:add":
                    outputs = action_head(decoded_queries)
                elif args.head_type == "time_ecd:cat":
                    outputs = action_head(decoded_queries, frame_indices)
                else:
                    if args.head_type == "res":
                        frame_features = utils.get_frame_features(detr.backbone, tube.video_name, frame_indices, val_dataset, device, True)
                    elif args.head_type == "x3d":
                        frame_features = utils.get_frame_features(x3d_xs, tube.video_name, frame_indices, val_dataset, device, True)
                    outputs = action_head(frame_features, decoded_queries, frame_indices)

                tube.log_pred(outputs)
                loss = head_criterion(outputs, action_label)

                acc_dict = utils.calc_acc(tube.action_pred, action_label, params["num_classes"])

                val_log["loss"].update(loss.item())
                val_log["acc1"].update(acc_dict["acc1"].item())
                val_log["acc5"].update(acc_dict["acc5"].item())
                if acc_dict["acc1_wo"] == -1:
                    continue
                val_log["acc1_wo_noaction"].update(acc_dict["acc1_wo"].item())
                val_log["acc5_wo_noaction"].update(acc_dict["acc5_wo"].item())

                pbar_tubes.set_postfix_str(f'loss={round(val_log["loss"].avg, 3)}, acc1: {round(val_log["acc1"].avg, 3)}, acc1_w/o: {round(val_log["acc1_wo_noaction"].avg, 3)}')

        leave_ex(ex, "val", val_log, epoch)

        lr_scheduler_head.step()

        if args.link_cues == "feature":
            utils.save_checkpoint(action_head, osp.join(args.check_dir, args.dataset), args.qmm_name + "/head/" + args.head_name, epoch)
        else:
            utils.save_checkpoint(action_head, osp.join(args.check_dir, args.dataset), "iou_link/head/" + args.head_name, epoch)


def leave_ex(ex, subset, log, epoch):
    ex.log_metric(subset + "_action_loss", log["loss"].avg, step=epoch)
    ex.log_metric(subset + "_action_acc1", log["acc1"].avg, step=epoch)
    ex.log_metric(subset + "_action_acc5", log["acc5"].avg, step=epoch)
    ex.log_metric(subset + "_action_acc1_wo_noaction", log["acc1_wo_noaction"].avg, step=epoch)
    ex.log_metric(subset + "_action_acc5_wo_noaction", log["acc5_wo_noaction"].avg, step=epoch)
    [log[key].reset() for key in log.keys()]


def random_idx_list(length):
    lst = [x for x in range(length)]
    percentage = random.uniform(0.7, 1.0)
    idx_list = sorted(random.sample(lst, int(length * percentage)))
    return idx_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Tube evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    params = yaml.safe_load(open(f"datasets/projects/{args.dataset}.yml"))
    params["label_list"].append("no action")
    args.n_classes = len(params["label_list"])

    main(args, params)
