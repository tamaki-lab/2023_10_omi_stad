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

from datasets.dataset import get_video_loader, get_sequential_loader
import util.misc as utils
from util.plot_utils import make_video_with_tube
from models import build_model
from models.person_encoder import PersonEncoder
from models.action_head import ActionHead
from models.tube import ActionTube


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
    parser.add_argument('--psn_score_th', default=0.7, type=float)
    parser.add_argument('--sim_th', default=0.7, type=float)
    parser.add_argument('--iou_th', default=0.3, type=float)

    # Backbone
    parser.add_argument('--backbone', default='resnet101', type=str, choices=('resnet50', 'resnet101'),
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', default=True,
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")

    # action head
    parser.add_argument('--lr_head', default=1e-4, type=float)
    parser.add_argument('--weight_decay_head', default=1e-4, type=float)
    parser.add_argument('--lr_drop_head', default=15, type=int)
    parser.add_argument('--iter_update', default=8, type=int)

    # others
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--check_dir', default="checkpoint", type=str)
    parser.add_argument('--num_workers', default=8, type=int)

    return parser


# @torch.no_grad()
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

    train_loader = get_video_loader(args.dataset, "train")
    val_loader = get_video_loader(args.dataset, "val", shuffle=False)

    train_log = {"action_loss": utils.AverageMeter(),
                 "action_acc1": utils.AverageMeter(),
                 "action_acc5": utils.AverageMeter(),
                 "action_acc1_wo_noaction": utils.AverageMeter(),
                 "action_acc5_wo_noaction": utils.AverageMeter()}
    val_log = {"action_loss": utils.AverageMeter(),
               "action_acc1": utils.AverageMeter(),
               "action_acc5": utils.AverageMeter(),
               "action_acc1_wo_noaction": utils.AverageMeter(),
               "action_acc5_wo_noaction": utils.AverageMeter()}

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
        pbar_videos = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        pbar_videos.set_description("[Training]")

        for video_idx, (img_paths, video_ano) in pbar_videos:
            sequential_loader = get_sequential_loader(img_paths, video_ano, args.n_frames)
            tube = ActionTube()

            # make person lists
            with torch.inference_mode():
                pbar_video = tqdm(enumerate(sequential_loader), total=len(sequential_loader), leave=False)
                pbar_video.set_description("[Frames Iteration]")
                for clip_idx, (samples, targets) in pbar_video:
                    samples = samples.to(device)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    outputs = detr(samples)

                    orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
                    results = postprocessors['bbox'](outputs, orig_target_sizes)

                    score_filter_indices = [(result["scores"] > args.psn_score_th).nonzero().flatten() for result in results]
                    score_filter_labels = [result["labels"]
                                           [(result["scores"] > args.psn_score_th).nonzero().flatten()] for result in results]
                    psn_indices = [idx[lab == 1] for idx, lab in zip(score_filter_indices, score_filter_labels)]
                    decoded_queries = [outputs["queries"][0, t][p_idx] for t, p_idx in enumerate(psn_indices)]
                    psn_boxes = [result["boxes"][p_idx].cpu() for result, p_idx in zip(results, psn_indices)]

                    psn_embedding = psn_encoder(torch.cat(decoded_queries, 0))
                    psn_embedding = utils.arrange_list(psn_indices, psn_embedding)

                    for t, (d_queries, p_embed) in enumerate(zip(decoded_queries, psn_embedding)):
                        frame_idx = args.n_frames * clip_idx + t
                        tube.update(d_queries, p_embed, frame_idx, psn_indices[t], psn_boxes[t])

            tube.filter()

            ## give a label for each query in person list ##
            utils.give_label(video_ano, tube.tubes, params["num_classes"], args.iou_th)

            # train head
            if len(tube.tubes) == 0:
                continue
            action_label = [person_list["action_label"] for person_list in tube.tubes]
            queries_list = [person_list["d_query"] for person_list in tube.tubes]
            if video_idx % args.iter_update == 0:
                optimizer_head.zero_grad()
            total_loss = torch.zeros(1).to(device)
            for list_idx, (input_queries, label) in enumerate(zip(queries_list, action_label)):
                input_queries = torch.stack(input_queries, 0).to(device)
                label = torch.Tensor(label).to(torch.int64).to(device)
                outputs = action_head(input_queries)
                loss = head_criterion(outputs, label)
                utils.give_pred(tube.tubes[list_idx], outputs)
                total_loss += loss
            total_loss = total_loss / (args.iter_update * len(queries_list))
            total_loss.backward()
            acc_dict = utils.calc_acc(tube.tubes, args.n_classes)
            if video_idx % args.iter_update == args.iter_update - 1:
                optimizer_head.step()

            train_log["action_loss"].update(total_loss.item())
            train_log["action_acc1"].update(acc_dict["acc1"].item())
            train_log["action_acc5"].update(acc_dict["acc5"].item())
            ex.log_metric("iter_action_loss", train_log["action_loss"].avg, step=step + video_idx)
            ex.log_metric("iter_action_acc1", train_log["action_acc1"].avg, step=step + video_idx)
            ex.log_metric("iter_action_acc5", train_log["action_acc5"].avg, step=step + video_idx)
            if acc_dict["acc1_wo"] == -1:
                continue
            train_log["action_acc1_wo_noaction"].update(acc_dict["acc1_wo"].item())
            train_log["action_acc5_wo_noaction"].update(acc_dict["acc5_wo"].item())
            ex.log_metric("iter_action_acc1_wo_noaction", train_log["action_acc1_wo_noaction"].avg, step=step + video_idx)
            ex.log_metric("iter_action_acc5_wo_noaction", train_log["action_acc5_wo_noaction"].avg, step=step + video_idx)

            # make new video with tube
            continue
            video_name = "/".join(img_paths[0].parts[-3:-1])
            video_path = osp.join(params["dataset_path_video"], video_name + ".avi")
            make_video_with_tube(video_path, params["label_list"], tube.tubes, video_ano)
            os.remove("test.avi")
        leave_ex(ex, "train", train_log, epoch)

        # validation
        with torch.inference_mode():
            action_head.eval()
            pbar_videos = tqdm(enumerate(val_loader), total=len(val_loader), leave=False)
            pbar_videos.set_description("[Validation]")
            for video_idx, (img_paths, video_ano) in pbar_videos:
                sequential_loader = get_sequential_loader(img_paths, video_ano, args.n_frames)
                tube = ActionTube()

                pbar_video = tqdm(enumerate(sequential_loader), total=len(sequential_loader), leave=False)
                pbar_video.set_description("[Frames Iteration]")
                for clip_idx, (samples, targets) in pbar_video:
                    samples = samples.to(device)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    outputs = detr(samples)

                    orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
                    results = postprocessors['bbox'](outputs, orig_target_sizes)

                    score_filter_indices = [(result["scores"] > args.psn_score_th).nonzero().flatten() for result in results]
                    score_filter_labels = [result["labels"]
                                           [(result["scores"] > args.psn_score_th).nonzero().flatten()] for result in results]
                    psn_indices = [idx[lab == 1] for idx, lab in zip(score_filter_indices, score_filter_labels)]
                    decoded_queries = [outputs["queries"][0, t][p_idx] for t, p_idx in enumerate(psn_indices)]
                    psn_boxes = [result["boxes"][p_idx].cpu() for result, p_idx in zip(results, psn_indices)]

                    psn_embedding = psn_encoder(torch.cat(decoded_queries, 0))
                    psn_embedding = utils.arrange_list(psn_indices, psn_embedding)

                    for t, (d_queries, p_embed) in enumerate(zip(decoded_queries, psn_embedding)):
                        frame_idx = args.n_frames * clip_idx + t
                        tube.update(d_queries, p_embed, frame_idx, psn_indices[t], psn_boxes[t])

                tube.filter()

                utils.give_label(video_ano, tube.tubes, args.n_classes, args.iou_th)

                if len(tube.tubes) == 0:
                    continue
                action_label = [person_list["action_label"] for person_list in tube.tubes]
                queries_list = [person_list["d_query"] for person_list in tube.tubes]

                total_loss = torch.zeros(1).to(device)
                for list_idx, (input_queries, label) in enumerate(zip(queries_list, action_label)):
                    input_queries = torch.stack(input_queries, 0).to(device)
                    label = torch.Tensor(label).to(torch.int64).to(device)
                    outputs = action_head(input_queries)
                    loss = head_criterion(outputs, label)
                    utils.give_pred(tube.tubes[list_idx], outputs)
                    total_loss += loss
                total_loss = total_loss / (args.iter_update * len(queries_list))
                acc_dict = utils.calc_acc(tube.tubes, args.n_classes)

                val_log["action_loss"].update(total_loss.item())
                val_log["action_acc1"].update(acc_dict["acc1"].item())
                val_log["action_acc5"].update(acc_dict["acc5"].item())
                if acc_dict["acc1_wo"] == -1:
                    continue
                val_log["action_acc1_wo_noaction"].update(acc_dict["acc1_wo"].item())
                val_log["action_acc5_wo_noaction"].update(acc_dict["acc5_wo"].item())

                ## make new video with tube ##
                continue
                video_name = "/".join(img_paths[0].parts[-3:-1])
                video_path = osp.join(params["dataset_path_video"], video_name + ".avi")
                make_video_with_tube(video_path, params["label_list"], tube.tubes, video_ano)
                os.remove("test.avi")

        leave_ex(ex, "val", val_log, epoch)

        utils.save_checkpoint(action_head, osp.join(args.check_dir, args.dataset), args.load_ex_name + "/head/" + args.write_ex_name, epoch)



def leave_ex(ex, subset, log, epoch):
    ex.log_metric(subset + "_action_loss", log["action_loss"].avg, step=epoch)
    ex.log_metric(subset + "_action_acc1", log["action_acc1"].avg, step=epoch)
    ex.log_metric(subset + "_action_acc5", log["action_acc5"].avg, step=epoch)
    ex.log_metric(subset + "_action_acc1_wo_noaction", log["action_acc1_wo_noaction"].avg, step=epoch)
    ex.log_metric(subset + "_action_acc5_wo_noaction", log["action_acc5_wo_noaction"].avg, step=epoch)
    [log[key].reset() for key in log.keys()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Tube evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    params = yaml.safe_load(open(f"datasets/projects/{args.dataset}.yml"))
    params["label_list"].append("no action")
    args.n_classes = len(params["label_list"])
    if args.dataset == "jhmdb21":
        args.psn_score_th = params["psn_score_th"]
        args.iou_th = params["iou_th"]

    main(args, params)
