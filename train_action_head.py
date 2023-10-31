# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import random
from comet_ml import Experiment
from tqdm import tqdm
import os
import os.path as osp
import av
import cv2
import numpy as np
import torch
import torch.nn as nn
import copy

from datasets.dataset import get_video_loader, get_sequential_loader
import util.misc as utils
from util.plot_utils import make_video_with_tube_ano
from util.box_ops import generalized_box_iou
from models import build_model
from models.person_encoder import PersonEncoder, SetInfoNce
from models.action_head import ActionHead
from models.tube import ActionTube


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    # loader
    parser.add_argument('--n_frames', default=8, type=int)

    # setting
    parser.add_argument('--load_ex_name', default='noskip_th0.7', type=str)
    parser.add_argument('--write_ex_name', default='test', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--load_epoch', default=100, type=int)
    parser.add_argument('--psn_score_th', default=0.7, type=float)
    parser.add_argument('--sim_th', default=0.7, type=float)
    parser.add_argument('--iou_th', default=0.4, type=float)
    parser.add_argument('--n_classes', default=24, type=int)
    parser.add_argument('--n_epochs', default=20, type=int)

    # Backbone
    parser.add_argument('--backbone', default='resnet101', type=str, choices=('resnet50', 'resnet101'),
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', default=True,
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")

    # action head
    parser.add_argument('--lr_head', default=1e-4, type=float)
    parser.add_argument('--weight_decay_head', default=1e-4, type=float)
    parser.add_argument('--lr_drop_head', default=50, type=int)
    parser.add_argument('--iter_update', default=8, type=int)

    # others
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--check_dir', default="checkpoint", type=str)
    parser.add_argument('--num_workers', default=8, type=int)

    return parser


# @torch.no_grad()
def main(args):
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
    trained_psn_encoder_path = osp.join(args.check_dir, args.load_ex_name, "encoder", f"epoch_{args.load_epoch}.pth")
    psn_encoder.load_state_dict(torch.load(trained_psn_encoder_path))
    psn_criterion = SetInfoNce().to(device)
    psn_criterion.eval()

    action_head = ActionHead(n_classes=args.n_classes).to(device)
    # head_criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0] * args.n_classes + [0.05])).to(device)
    head_criterion = nn.CrossEntropyLoss()
    optimizer_head = torch.optim.AdamW(action_head.parameters(), lr=args.lr_head, weight_decay=args.weight_decay_head)
    lr_scheduler_head = torch.optim.lr_scheduler.StepLR(optimizer_head, args.lr_drop_head)

    train_loader = get_video_loader("ucf101-24", "train")
    val_loader = get_video_loader("ucf101-24", "val", shuffle=False)

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

                    orig_target_sizes = torch.stack([t["size"] for t in targets], dim=0)
                    results = postprocessors['bbox'](outputs, orig_target_sizes)

                    score_filter_indices = [(result["scores"] > args.psn_score_th).nonzero().flatten() for result in results]
                    score_filter_labels = [result["labels"]
                                           [(result["scores"] > args.psn_score_th).nonzero().flatten()] for result in results]
                    psn_indices = [idx[lab == 1] for idx, lab in zip(score_filter_indices, score_filter_labels)]
                    decoded_queries = [outputs["queries"][0, t][p_idx] for t, p_idx in enumerate(psn_indices)]
                    psn_boxes = [result["boxes"][p_idx].cpu() for result, p_idx in zip(results, psn_indices)]

                    psn_embedding = psn_encoder(torch.cat(decoded_queries, 0))
                    psn_embedding = arrange_list(psn_indices, psn_embedding)

                    for t, (d_queries, p_embed) in enumerate(zip(decoded_queries, psn_embedding)):
                        frame_idx = args.n_frames * clip_idx + t
                        tube.update(d_queries, p_embed, frame_idx, psn_indices[t], psn_boxes[t])

            tube.filter()

            # give a label for each query in person list
            video_ano_fixed = fix_ano_scale(video_ano, resize_scale=512 / 320)  # TODO change
            give_label(video_ano_fixed, tube.tubes, args.n_classes, args.iou_th)

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
                give_pred(tube.tubes[list_idx], outputs)
                total_loss += loss
            total_loss = total_loss / (args.iter_update * len(queries_list))
            total_loss.backward()
            acc_dict = calc_acc(tube.tubes, args.n_classes)
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
            video_path = "/".join(img_paths[0].parts[-3:-1])
            video_path = "/mnt/NAS-TVS872XT/dataset/UCF101/video/" + video_path + ".avi"
            make_video_with_tube(video_path, tube.tubes, video_ano)
            os.remove("test.avi")
        ex.log_metric("train_action_loss", train_log["action_loss"].avg, step=epoch)
        ex.log_metric("train_action_acc1", train_log["action_acc1"].avg, step=epoch)
        ex.log_metric("train_action_acc5", train_log["action_acc5"].avg, step=epoch)
        ex.log_metric("train_action_acc1_wo_noaction", train_log["action_acc1_wo_noaction"].avg, step=epoch)
        ex.log_metric("train_action_acc5_wo_noaction", train_log["action_acc5_wo_noaction"].avg, step=epoch)

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

                    orig_target_sizes = torch.stack([t["size"] for t in targets], dim=0)
                    results = postprocessors['bbox'](outputs, orig_target_sizes)

                    score_filter_indices = [(result["scores"] > args.psn_score_th).nonzero().flatten() for result in results]
                    score_filter_labels = [result["labels"]
                                           [(result["scores"] > args.psn_score_th).nonzero().flatten()] for result in results]
                    psn_indices = [idx[lab == 1] for idx, lab in zip(score_filter_indices, score_filter_labels)]
                    decoded_queries = [outputs["queries"][0, t][p_idx] for t, p_idx in enumerate(psn_indices)]
                    psn_boxes = [result["boxes"][p_idx].cpu() for result, p_idx in zip(results, psn_indices)]

                    psn_embedding = psn_encoder(torch.cat(decoded_queries, 0))
                    psn_embedding = arrange_list(psn_indices, psn_embedding)

                    for t, (d_queries, p_embed) in enumerate(zip(decoded_queries, psn_embedding)):
                        frame_idx = args.n_frames * clip_idx + t
                        tube.update(d_queries, p_embed, frame_idx, psn_indices[t], psn_boxes[t])

                tube.filter()

                video_ano_fixed = fix_ano_scale(video_ano, resize_scale=512 / 320)  # TODO change
                give_label(video_ano_fixed, tube.tubes, args.n_classes, args.iou_th)

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
                    give_pred(tube.tubes[list_idx], outputs)
                    total_loss += loss
                total_loss = total_loss / (args.iter_update * len(queries_list))
                acc_dict = calc_acc(tube.tubes, args.n_classes)

                val_log["action_loss"].update(total_loss.item())
                val_log["action_acc1"].update(acc_dict["acc1"].item())
                val_log["action_acc5"].update(acc_dict["acc5"].item())
                if acc_dict["acc1_wo"] == -1:
                    continue
                val_log["action_acc1_wo_noaction"].update(acc_dict["acc1_wo"].item())
                val_log["action_acc5_wo_noaction"].update(acc_dict["acc5_wo"].item())

                ## make new video with tube ##
                continue
                video_path = "/".join(img_paths[0].parts[-3:-1])
                video_path = "/mnt/NAS-TVS872XT/dataset/UCF101/video/" + video_path + ".avi"
                make_video_with_tube_ano(video_path, tube.tubes, video_ano)
                exit()
                os.remove("test.avi")

        ex.log_metric("val_action_loss", val_log["action_loss"].avg, step=epoch)
        ex.log_metric("val_action_acc1", val_log["action_acc1"].avg, step=epoch)
        ex.log_metric("val_action_acc5", val_log["action_acc5"].avg, step=epoch)
        ex.log_metric("val_action_acc1_wo_noaction", val_log["action_acc1_wo_noaction"].avg, step=epoch)
        ex.log_metric("val_action_acc5_wo_noaction", val_log["action_acc5_wo_noaction"].avg, step=epoch)

        [log.reset() for _, log in train_log.items()]
        [log.reset() for _, log in val_log.items()]
        dir = args.check_dir + "/" + args.load_ex_name + "/head"
        utils.save_checkpoint(action_head, dir, args.write_ex_name, epoch)


def arrange_list(x, y):
    """
    yの要素は変えずにxと同じリストの形にする
    xは2重リスト,yはリストでその長さはxのサブリストの長さの和
    """
    result = []
    for sublist in x:
        size = len(sublist)
        result.append(y[:size])
        y = y[size:]
    return result


def fix_ano_scale(video_ano, resize_scale=512 / 320):
    video_ano_fixed = copy.deepcopy(video_ano)
    for frame_idx, frame_ano in video_ano.items():
        for tube_idx, box_ano in frame_ano.items():
            video_ano_fixed[frame_idx][tube_idx][:4] = [x * resize_scale for x in box_ano[:4]]
    return video_ano_fixed


def give_label(video_ano, person_lists, no_action_id=-1, iou_th=0.4):
    for person_list in person_lists:
        person_list["action_label"] = []
        for i, (frame_idx, _) in enumerate(person_list["idx_of_p_queries"]):
            if frame_idx in video_ano:
                gt_ano = [ano for tube_idx, ano in video_ano[frame_idx].items()]
                gt_boxes = torch.tensor(gt_ano)[:, :4]
                iou = generalized_box_iou(person_list["bbox"][i].reshape(-1, 4), gt_boxes)
                max_v, max_idx = torch.max(iou, dim=1)
                if max_v.item() > iou_th:
                    person_list["action_label"].append(gt_ano[max_idx][4])
                else:
                    person_list["action_label"].append(no_action_id)
                continue
            else:
                person_list["action_label"].append(no_action_id)


def give_pred(person_list, outputs):
    person_list["action_pred"] = outputs.cpu().detach()


def calc_acc(person_lists, n_classes):
    acc_dict = {"acc1": torch.zeros(1),
                "acc5": torch.zeros(1),
                "acc1_wo": torch.zeros(1),
                "acc5_wo": torch.zeros(1)}
    n_all_wo = 0
    for person_list in person_lists:
        output = person_list["action_pred"]
        label = person_list["action_label"]
        label = torch.Tensor(label).to(torch.int64)
        acc1, acc5 = utils.accuracy(output, label, topk=(1, 5))
        acc_dict["acc1"] += acc1
        acc_dict["acc5"] += acc5
        if (label != n_classes).sum() == 0:
            n_all_wo += 1
            continue
        else:
            acc1_wo, acc5_wo = utils.accuracy(output[label != n_classes], label[label != n_classes], topk=(1, 5))
            acc_dict["acc1_wo"] += acc1_wo
            acc_dict["acc5_wo"] += acc5_wo
    acc_dict["acc1"] = acc_dict["acc1"] / len(person_lists)
    acc_dict["acc5"] = acc_dict["acc5"] / len(person_lists)
    if len(person_lists) != n_all_wo:
        acc_dict["acc1_wo"] = acc_dict["acc1_wo"] / (len(person_lists) - n_all_wo)
        acc_dict["acc5_wo"] = acc_dict["acc5_wo"] / (len(person_lists) - n_all_wo)
    else:
        acc_dict["acc1_wo"] = -1
        acc_dict["acc5_wo"] = -1
    return acc_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Tube evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
