# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import random
from comet_ml import Experiment
from tqdm import tqdm
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import yaml
import random


from datasets.dataset import get_video_loader, get_sequential_loader
import util.misc as utils
from util.plot_utils import make_video_with_tube
from models import build_model
from models.person_encoder import PersonEncoder
from models.action_head import ActionHead
from models.tube import ActionTubes


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    # setting #
    parser.add_argument('--n_epochs', default=20, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--write_ex_name', default='head_test', type=str)
    # loader
    parser.add_argument('--dataset', default='jhmdb21', type=str, choices=['ucf101-24', 'jhmdb21'])
    parser.add_argument('--n_frames', default=128, type=int)

    # person encoder
    parser.add_argument('--load_ex_name', default='jhmdb_wd:e4', type=str)
    parser.add_argument('--load_epoch', default=15, type=int)
    parser.add_argument('--psn_score_th', default=0.8, type=float)
    parser.add_argument('--sim_th', default=0.5, type=float)
    parser.add_argument('--iou_th', default=0.3, type=float)

    # Backbone
    parser.add_argument('--backbone', default='resnet101', type=str, choices=('resnet50', 'resnet101'),
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', default=True,
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")

    # action head
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

    detr, criterion, postprocessors = build_model(args)
    detr.to(device)
    detr.eval()
    pretrain_path = "checkpoint/detr/" + utils.get_pretrain_path(args.backbone, args.dilation)
    detr.load_state_dict(torch.load(pretrain_path)["model"])
    criterion.to(device)
    criterion.eval()

    psn_encoder = PersonEncoder().to(device)
    psn_encoder.eval()
    trained_psn_encoder_path = osp.join(args.check_dir, args.dataset, args.load_ex_name, "encoder", f"epoch_{args.load_epoch}.pth")
    psn_encoder.load_state_dict(torch.load(trained_psn_encoder_path))

    action_head = ActionHead(n_classes=args.n_classes).to(device)
    # head_criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0] * args.n_classes + [0.05])).to(device)
    head_criterion = nn.CrossEntropyLoss()
    optimizer_head = torch.optim.AdamW(action_head.parameters(), lr=args.lr_head, weight_decay=args.weight_decay_head)
    lr_scheduler_head = torch.optim.lr_scheduler.StepLR(optimizer_head, args.lr_drop_head)

    # train_loader = get_video_loader(args.dataset, "train")
    dir = osp.join(args.check_dir, args.dataset, args.load_ex_name, "qmm_tubes")
    filename = f"tube-epoch:{args.load_epoch}_pth:{args.psn_score_th}_simth:{args.sim_th}"
    train_loader = utils.TarIterator(dir + "/train", filename)
    val_loader = utils.TarIterator(dir+"/val", filename)

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
    hyper_params = {
        "ex_name": args.load_ex_name + "--" + args.write_ex_name,
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
            # idx_list = random_idx_list(len(tube.action_label))
            # tube.action_label = [tube.action_label[idx] for idx in idx_list]
            # tube.decoded_queries = [tube.decoded_queries[idx] for idx in idx_list]
            # frame_indices = [tube.query_indicies[idx][0] for idx in idx_list]
            frame_indices = [x[0] for x in tube.query_indicies]
            frame_indices = [x - frame_indices[0] for x in frame_indices]

            action_label = torch.Tensor(tube.action_label).to(torch.int64).to(device)
            decoded_queries = torch.stack(tube.decoded_queries).to(device)

            if tube_idx % args.iter_update == 0:
                optimizer_head.zero_grad()

            # outputs = action_head(decoded_queries)
            outputs = action_head(decoded_queries, frame_indices)
            tube.log_pred(outputs)
            loss = head_criterion(outputs, action_label) / args.iter_update
            loss.backward()

            acc_dict = utils.calc_acc(tube.action_pred, action_label, args.n_classes)
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

            # make new video with tube
            continue
            video_path = osp.join(params["dataset_path_video"], tube.video_name + ".avi")
            make_video_with_tube(video_path, params["label_list"], tube.tubes, video_ano)
            os.remove("test.avi")
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

                outputs = action_head(decoded_queries, frame_indices)
                # outputs = action_head(decoded_queries)
                tube.log_pred(outputs)
                loss = head_criterion(outputs, action_label)

                acc_dict = utils.calc_acc(tube.action_pred, action_label, args.n_classes)

                val_log["loss"].update(loss.item())
                val_log["acc1"].update(acc_dict["acc1"].item())
                val_log["acc5"].update(acc_dict["acc5"].item())
                if acc_dict["acc1_wo"] == -1:
                    continue
                val_log["acc1_wo_noaction"].update(acc_dict["acc1_wo"].item())
                val_log["acc5_wo_noaction"].update(acc_dict["acc5_wo"].item())

                pbar_tubes.set_postfix_str(f'loss={round(val_log["loss"].avg, 3)}, acc1: {round(val_log["acc1"].avg, 3)}, acc1_w/o: {round(val_log["acc1_wo_noaction"].avg, 3)}')

                ## make new video with tube ##
                continue
                video_path = osp.join(params["dataset_path_video"], tube.video_name + ".avi")
                make_video_with_tube(video_path, params["label_list"], tube.tubes, video_ano)
                os.remove("test.avi")

        leave_ex(ex, "val", val_log, epoch)

        lr_scheduler_head.step()

        utils.save_checkpoint(action_head, osp.join(args.check_dir, args.dataset), args.load_ex_name + "/head/" + args.write_ex_name, epoch)


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
