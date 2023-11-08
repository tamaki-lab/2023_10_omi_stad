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
import yaml

from datasets.dataset import get_video_loader, get_sequential_loader
import util.misc as utils
from util.plot_utils import make_video_with_tube
from util.gt_tubes import make_gt_tubes_ucf
from util.box_ops import tube_iou
from models import build_model
from models.person_encoder import PersonEncoder, NPairLoss
from models.tube import ActionTube


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    # setting #
    parser.add_argument('--dataset', default='ucf101-24', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--ex_name', default='lr:e4_pth:0.5_iouth:0.4', type=str)
    # loader
    parser.add_argument('--n_frames', default=128, type=int)
    parser.add_argument('--n_classes', default=24, type=int)
    # person encoder
    parser.add_argument('--load_epoch', default=10, type=int)
    parser.add_argument('--psn_score_th', default=0.5, type=float)
    parser.add_argument('--sim_th', default=0.7, type=float)
    parser.add_argument('--iou_th', default=0.5, type=float)
    parser.add_argument('--tiou_th', default=0.4, type=float)

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


@torch.no_grad()
def main(args):
    params = yaml.safe_load(open(f"datasets/projects/{args.dataset}.yml"))
    params["label_list"].append("no action")

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
    trained_psn_encoder_path = osp.join(args.check_dir, args.ex_name, "encoder", f"epoch_{args.load_epoch}.pth")
    psn_encoder.load_state_dict(torch.load(trained_psn_encoder_path))
    psn_criterion = NPairLoss().to(device)
    psn_criterion.eval()

    val_loader = get_video_loader("ucf101-24", "val")

    pred_tubes = []
    video_names = []
    total_tubes = 0

    pbar_videos = tqdm(enumerate(val_loader), total=len(val_loader), leave=False)
    for video_idx, (img_paths, video_ano) in pbar_videos:
        video_name = "/".join(img_paths[0].parts[-3: -1])
        video_names.append(video_name)
        sequential_loader = get_sequential_loader(img_paths, video_ano, args.n_frames)
        tube = ActionTube(video_name, args.sim_th)

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

        # make new video with tube
        pred_tubes.append(tube)
        total_tubes += len(tube.tubes)
        continue
        video_path = osp.join(params["dataset_path_video"], video_name + ".avi")
        utils.give_label(video_ano, tube.tubes, args.n_classes, args.iou_th)
        make_video_with_tube(video_path, params["label_list"], tube.tubes, video_ano=video_ano, plot_label=True)
        os.remove("test.avi")

    gt_tubes = make_gt_tubes_ucf("val", params)
    calc_precision_recall(pred_tubes, gt_tubes, video_names, args.tiou_th)


def calc_precision_recall(pred_tubes, gt_tubes, video_names, tiou_th):
    pred_tubes = [(video_tubes.video_name, video_tubes.extract(tube)) for video_tubes in pred_tubes for tube in video_tubes.tubes]

    for video_name, tubes_ano in gt_tubes.copy().items():
        gt_tubes[video_name] = [tube_ano["boxes"] for tube_ano in tubes_ano]
    gt_tubes = {name: tubes for name, tubes in gt_tubes.items() if name in video_names}

    correct_map = {name: [False] * len(tube) for name, tube in gt_tubes.items() if name in video_names}

    n_gt = sum([len(tubes) for _, tubes in gt_tubes.items()])
    tp = 0
    n_pred = 0

    for _, (video_name, pred_tube) in enumerate(pred_tubes):
        video_gt_tubes = gt_tubes[video_name]
        tiou_list = []
        for _, gt_tube in enumerate(video_gt_tubes):
            tiou_list.append(tube_iou(pred_tube, gt_tube, label_centric=True))
        max_tiou = max(tiou_list)   # TODO gt bboxが1つも存在しない場合の考慮
        max_index = tiou_list.index(max_tiou)
        if max_tiou > tiou_th:
            if correct_map[video_name][max_index]:
                continue
            else:
                correct_map[video_name][max_index] = True
                tp += 1
                n_pred += 1
        else:
            n_pred += 1

    print("Settings")
    print(f"psn_socore_th: {args.psn_score_th}")
    print(f"sim_th: {args.sim_th}")
    print(f"iou_th: {args.iou_th}")
    print(f"tiou_th: {args.tiou_th}")
    print("-------")
    print(f"n_gt_tubes: {n_gt}")
    print(f"n_pred_tubes: {len(pred_tubes)}, n_pred_tubes (filter): {n_pred}")
    print(f"True Positive: {tp}")
    print("-------")
    print(f"Precision: {tp/len(pred_tubes)}, Precision (filter): {tp/n_pred}")
    print(f"Recall: {tp/n_gt}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Tube evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
