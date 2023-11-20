
import argparse
import random
from comet_ml import Experiment
from tqdm import tqdm
import os
import os.path as osp
import torch
import torch.nn as nn
import numpy as np
import torch
import yaml

from datasets.dataset import get_video_loader, get_sequential_loader
import util.misc as utils
from util.plot_utils import make_video_with_actiontube
from models import build_model
from models.person_encoder import PersonEncoder, NPairLoss
from models.action_head import ActionHead
from models.tube import ActionTube
from util.gt_tubes import make_gt_tubes
from util.video_map import calc_video_map


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    # loader
    parser.add_argument('--dataset', default='jhmdb21', type=str, choices=['ucf101-24', 'jhmdb21'])
    parser.add_argument('--n_frames', default=128, type=int)
    parser.add_argument('--subset', default="val", type=str, choices=["train", "val"])

    # setting
    parser.add_argument('--load_ex_name', default='jhmdb_wd:e4', type=str)
    parser.add_argument('--write_ex_name', default='head_test', type=str)
    parser.add_argument('--device', default=1, type=int)
    parser.add_argument('--load_epoch_encoder', default=15, type=int)
    parser.add_argument('--load_epoch_head', default=14, type=int)
    parser.add_argument('--psn_score_th', default=0.7, type=float)
    parser.add_argument('--sim_th', default=0.5, type=float)
    parser.add_argument('--tiou_th', default=0.2, type=float)
    parser.add_argument('--iou_th', default=0.4, type=float)

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
def main(args, params):
    device = torch.device(f"cuda:{args.device}")
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    detr, _, postprocessors = build_model(args)
    detr.to(device)
    detr.eval()
    pretrain_path_detr = "checkpoint/detr/" + utils.get_pretrain_path(args.backbone, args.dilation)
    detr.load_state_dict(torch.load(pretrain_path_detr)["model"])

    psn_encoder = PersonEncoder().to(device)
    psn_encoder.eval()
    pretrain_path_encoder = osp.join(args.check_dir, args.dataset, args.load_ex_name, "encoder", f"epoch_{args.load_epoch_encoder}.pth")
    psn_encoder.load_state_dict(torch.load(pretrain_path_encoder))

    action_head = ActionHead(n_classes=args.n_classes).to(device)
    action_head.eval()
    pretrain_path_head = osp.join(args.check_dir, args.dataset, args.load_ex_name, "head", args.write_ex_name, f"epoch_{args.load_epoch_head}.pth")
    action_head.load_state_dict(torch.load(pretrain_path_head))

    loader = get_video_loader(args.dataset, args.subset, shuffle=False)

    pred_tubes = []
    video_names = []
    total_tubes = 0

    pbar_videos = tqdm(enumerate(loader), total=len(loader), leave=False)
    pbar_videos.set_description("[Validation]")
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

        utils.give_label(video_ano, tube.tubes, params["num_classes"], args.iou_th)

        if len(tube.tubes) == 0:
            continue
        queries_list = [person_list["d_query"] for person_list in tube.tubes]

        for list_idx, input_queries in enumerate(queries_list):
            input_queries = torch.stack(input_queries, 0).to(device)
            outputs = action_head(input_queries)
            utils.give_pred(tube.tubes[list_idx], outputs)

        ## make new video with action tube ##
        tube.split()
        pred_tubes.append(tube)
        total_tubes += len(tube.tubes)
        pbar_videos.set_postfix_str(f'total_tubes: {total_tubes}, n_tubes: {len(tube.tubes)}')
        continue
        video_path = osp.join(params["dataset_path_video"], video_name + ".avi")
        make_video_with_actiontube(video_path, params["label_list"], tube.tubes, video_ano, plot_label=True)
        os.remove("test.avi")

    pred_tubes = [tube for video_tubes in pred_tubes for tube in video_tubes.tubes]
    print(f"num of pred tubes: {len(pred_tubes)}")
    pred_tubes = [tube for tube in pred_tubes if tube[1]["class"] != params["num_classes"]]
    print(f"num of pred tubes w/o no action: {len(pred_tubes)}")

    gt_tubes = make_gt_tubes(args.dataset, args.subset, params)
    gt_tubes = {name: tube for name, tube in gt_tubes.items() if name in video_names}   # for debug with less data from loader

    video_ap = calc_video_map(pred_tubes, gt_tubes, params["num_classes"], args.tiou_th)
    for class_name, ap in zip(params["label_list"][:-1], video_ap):
        print(f"{class_name}: {round(ap,4)}")
    print(f"v-mAP: {round(sum(video_ap) / len(video_ap),4)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Tube evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    params = yaml.safe_load(open(f"datasets/projects/{args.dataset}.yml"))
    params["label_list"].append("no action")
    args.n_classes = len(params["label_list"])
    if args.dataset == "jhmdb21":
        args.psn_score_th = params["psn_score_th"]
        args.iou_th = params["iou_th"]

    main(args, params)
