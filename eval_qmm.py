# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import random
from tqdm import tqdm
import os
import os.path as osp
import numpy as np
import torch
import yaml


from datasets.dataset import get_video_loader, get_sequential_loader
import util.misc as utils
from util.plot_utils import make_video_with_tube
from util.gt_tubes import make_gt_tubes
from util.box_ops import tube_iou, get_motion_ctg
from models import build_model
from models.person_encoder import PersonEncoder, NPairLoss
from models.tube import ActionTubes


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    # setting #
    parser.add_argument('--dataset', default='jhmdb21', type=str, choices=['ucf101-24', 'jhmdb21'])
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--qmm_name', default='noskip_sr:4', type=str)
    parser.add_argument('--link_cues', default='feature', type=str)
    # loader
    parser.add_argument('--n_frames', default=128, type=int)
    parser.add_argument('--subset', default="val", type=str, choices=["train", "val"])
    # person encoder
    parser.add_argument('--load_epoch', default=20, type=int)
    parser.add_argument('--psn_score_th', default=0.9, type=float)  # jhmdb:0.9, ucf:0.5
    parser.add_argument('--sim_th', default=0.5, type=float)  # jhmdb:0.5, ucf:0.25
    parser.add_argument('--tiou_th', default=0.2, type=float)
    parser.add_argument('--iou_th', default=0.2, type=float)    # for visualization
    parser.add_argument('--is_skip', action="store_true")
    parser.add_argument('--filter_length', default=8, type=int)  # jhmdb:8, ucf:16

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
def main(args, params):

    device = torch.device(f"cuda:{args.device}")

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    detr, _, postprocessors = build_model(args)
    detr.to(device)
    detr.eval()
    if args.dataset == "ucf101-24":
        pretrain_path = "checkpoint/ucf101-24/w:252/detr/epoch_20.pth"
        detr.load_state_dict(torch.load(pretrain_path))
    else:
        pretrain_path = "checkpoint/detr/" + utils.get_pretrain_path(args.backbone, args.dilation)
        detr.load_state_dict(torch.load(pretrain_path)["model"])

    psn_encoder = PersonEncoder(skip=args.is_skip).to(device)
    psn_encoder.eval()
    dir_encoder = osp.join(args.check_dir, args.dataset, args.qmm_name, "encoder")
    trained_psn_encoder_path = osp.join(dir_encoder, f"epoch_{args.load_epoch}.pth")
    psn_encoder.load_state_dict(torch.load(trained_psn_encoder_path))
    psn_criterion = NPairLoss().to(device)
    psn_criterion.eval()

    loader = get_video_loader(args.dataset, args.subset)

    pred_tubes = []
    video_names = []

    pbar_videos = tqdm(enumerate(loader), total=len(loader), leave=False)
    for video_idx, (img_paths, video_ano) in pbar_videos:
        video_name = "/".join(img_paths[0].parts[-3: -1])
        video_names.append(video_name)
        sequential_loader = get_sequential_loader(img_paths, video_ano, args.n_frames)
        tube = ActionTubes(video_name, args.sim_th, ano=video_ano)

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
                tube.update(d_queries, p_embed, frame_idx, psn_indices[t], psn_boxes[t], args.link_cues)

        tube.filter(filter_length=args.filter_length)
        # tube.filter()
        pred_tubes.append(tube)
        tube.give_action_label(params["num_classes"], args.iou_th)

        # make new video with tube
        continue
        video_path = osp.join(params["dataset_path_video"], video_name + ".avi")
        make_video_with_tube(video_path, params["label_list"], tube.tubes, video_ano=video_ano, plot_label=True)
        os.remove("test.avi")

    if args.link_cues == "feature":
        dir = osp.join(args.check_dir, args.dataset, args.qmm_name, "qmm_tubes", args.subset)
        filename = f"epoch:{args.load_epoch}_pth:{args.psn_score_th}_simth:{args.sim_th}_fl:{args.filter_length}"
    elif args.link_cues == "iou":
        dir = osp.join(args.check_dir, args.dataset, "iou_link", "qmm_tubes", args.subset)
        filename = f"pth:{args.psn_score_th}_iouth:{args.sim_th}_fl:{args.filter_length}"
    utils.write_tar(pred_tubes, dir, "videotubes-" + filename)
    video_names = [tube.video_name for tube in pred_tubes]
    pred_tubes = [tube for video_tubes in pred_tubes for tube in video_tubes.tubes]
    utils.write_tar(pred_tubes, dir, "tube-" + filename)

    gt_tubes = make_gt_tubes(args.dataset, args.subset, params)
    calc_precision_recall(pred_tubes, gt_tubes, video_names, args.tiou_th)


def calc_precision_recall(pred_tubes, gt_tubes, video_names=None, tiou_th=0.5):
    pred_tubes = [(tube.video_name, tube.make_region_pred()) for tube in pred_tubes]

    for video_name, tubes_ano in gt_tubes.copy().items():
        gt_tubes[video_name] = [tube_ano["boxes"] for tube_ano in tubes_ano]

    if video_names is None:
        correct_map = {name: [False] * len(tube) for name, tube in gt_tubes.items()}
    else:
        gt_tubes = {name: tubes for name, tubes in gt_tubes.items() if name in video_names}
        correct_map = {name: [False] * len(tube) for name, tube in gt_tubes.items() if name in video_names}

    n_gt = sum([len(tubes) for _, tubes in gt_tubes.items()])
    tp = 0
    n_pred = 0

    pbar_preds = tqdm(enumerate(pred_tubes), total=len(pred_tubes), leave=False)
    pbar_preds.set_description("[Calc TP]")
    for _, (video_name, pred_tube) in pbar_preds:
        video_gt_tubes = gt_tubes[video_name]
        tiou_list = []
        for _, gt_tube in enumerate(video_gt_tubes):
            # tiou_list.append(tube_iou(pred_tube, gt_tube, label_centric=True, frame_iou_set=(True, 0.35)))
            tiou_list.append(tube_iou(pred_tube, gt_tube, label_centric=True, frame_iou_set=(False, 0.35)))
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

        pbar_preds.set_postfix_str(f'TP={tp}, Pre: {round(tp/n_pred, 3)}, Rec: {round(tp/n_gt, 3)}')

    print("Settings")
    print(f"psn_socore_th: {args.psn_score_th}")
    print(f"sim_th: {args.sim_th}")
    print(f"tiou_th: {args.tiou_th}")
    print("-------")
    print(f"n_gt_tubes: {n_gt}")
    print(f"n_pred_tubes: {len(pred_tubes)}, n_pred_tubes (filter): {n_pred}")
    print(f"True Positive: {tp}")
    print("-------")
    print(f"Precision: {round(tp/len(pred_tubes),3)}, Precision (filter): {round(tp/n_pred,3)}")
    print(f"Recall: {round(tp/n_gt,3)}")

def calc_precision_recall_per_class(pred_tubes, gt_tubes, label_list, video_names=None, tiou_th=0.5):
    print("Settings")
    print(f"psn_socore_th: {args.psn_score_th}")
    print(f"sim_th: {args.sim_th}")
    print(f"tiou_th: {args.tiou_th}")
    print("-----------------------")

    pred_tubes = [(tube.video_name, tube.make_region_pred()) for tube in pred_tubes]
    pred_tubes_per_class = {label: [] for label in label_list}
    for pred_tube in pred_tubes:
        cls = pred_tube[0].split("/")[0]
        pred_tubes_per_class[cls].append(pred_tube)

    for video_name, tubes_ano in gt_tubes.copy().items():
        gt_tubes[video_name] = [tube_ano["boxes"] for tube_ano in tubes_ano]

    if video_names is None:
        correct_map = {name: [False] * len(tube) for name, tube in gt_tubes.items()}
    else:
        gt_tubes = {name: tubes for name, tubes in gt_tubes.items() if name in video_names}
        correct_map = {name: [False] * len(tube) for name, tube in gt_tubes.items() if name in video_names}

    gt_tubes_per_class = {label: {} for label in label_list}
    for name, tube in gt_tubes.items():
        cls = name.split("/")[0]
        gt_tubes_per_class[cls][name] = tube

    for cls in label_list:
        n_gt = sum([len(tubes) for _, tubes in gt_tubes_per_class[cls].items()])
        tp = 0
        n_pred = 0

        pbar_preds = tqdm(enumerate(pred_tubes_per_class[cls]), total=len(pred_tubes_per_class[cls]), leave=False)
        pbar_preds.set_description(f"[Calc TP@{cls}]")
        for _, (video_name, pred_tube) in pbar_preds:
            video_gt_tubes = gt_tubes_per_class[cls][video_name]
            tiou_list = []
            for _, gt_tube in enumerate(video_gt_tubes):
                # tiou_list.append(tube_iou(pred_tube, gt_tube, label_centric=True, frame_iou_set=(True, 0.35)))
                tiou_list.append(tube_iou(pred_tube, gt_tube, label_centric=True, frame_iou_set=(False, 0.35)))
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

        print("-------")
        print(cls)
        print(f"n_gt_tubes: {n_gt}")
        print(f"n_pred_tubes: {len(pred_tubes_per_class[cls])}, n_pred_tubes (filter): {n_pred}")
        print(f"True Positive: {tp}")
        print(f"Precision: {round(tp/len(pred_tubes_per_class[cls]),3)}, Precision (filter): {round(tp/n_pred,3)}")
        print(f"Recall: {round(tp/n_gt,3)}")
        print("-------")


def calc_precision_recall_per_motion(pred_tubes, gt_tubes, video_names=None, tiou_th=0.5):
    print("Settings")
    print(f"psn_socore_th: {args.psn_score_th}")
    print(f"sim_th: {args.sim_th}")
    print(f"tiou_th: {args.tiou_th}")
    print("-----------------------")

    motion_ctgs = ["small", "medium", "large"]
    pred_tubes = [(tube.video_name, tube.make_region_pred()) for tube in pred_tubes]
    pred_tubes_per_motion = {ctg: [] for ctg in motion_ctgs}
    for pred_tube in pred_tubes:
        ctg = get_motion_ctg(pred_tube[1])
        pred_tubes_per_motion[ctg].append(pred_tube)

    for video_name, tubes_ano in gt_tubes.copy().items():
        gt_tubes[video_name] = [tube_ano["boxes"] for tube_ano in tubes_ano]

    if video_names is None:
        correct_map = {name: [False] * len(tube) for name, tube in gt_tubes.items()}
    else:
        gt_tubes = {name: tubes for name, tubes in gt_tubes.items() if name in video_names}
        correct_map = {name: [False] * len(tube) for name, tube in gt_tubes.items() if name in video_names}

    gt_tubes_per_motion = {ctg: {} for ctg in motion_ctgs}
    for name, tubes in gt_tubes.items():
        for tube in tubes:
            ctg = get_motion_ctg(tube)
            if name not in gt_tubes_per_motion[ctg]:
                gt_tubes_per_motion[ctg][name] = [tube]
            else:
                gt_tubes_per_motion[ctg][name].append(tube)

    for ctg in motion_ctgs:
        n_gt = sum([len(tubes) for _, tubes in gt_tubes_per_motion[ctg].items()])
        tp = 0
        n_pred = 0

        pbar_preds = tqdm(enumerate(pred_tubes_per_motion[ctg]), total=len(pred_tubes_per_motion[ctg]), leave=False)
        pbar_preds.set_description(f"[Calc TP@{ctg}]")
        for _, (video_name, pred_tube) in pbar_preds:
            if video_name not in gt_tubes_per_motion[ctg]:
                n_pred += 1
                continue
            video_gt_tubes = gt_tubes_per_motion[ctg][video_name]
            tiou_list = []
            for _, gt_tube in enumerate(video_gt_tubes):
                # tiou_list.append(tube_iou(pred_tube, gt_tube, label_centric=True, frame_iou_set=(True, 0.35)))
                tiou_list.append(tube_iou(pred_tube, gt_tube, label_centric=True, frame_iou_set=(False, 0.35)))
            max_tiou = max(tiou_list)
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

        print("-------")
        print(ctg)
        print(f"n_gt_tubes: {n_gt}")
        print(f"n_pred_tubes: {len(pred_tubes_per_motion[ctg])}, n_pred_tubes (filter): {n_pred}")
        print(f"True Positive: {tp}")
        print(f"Precision: {round(tp/len(pred_tubes_per_motion[ctg]),3)}, Precision (filter): {round(tp/n_pred,3)}")
        print(f"Recall: {round(tp/n_gt,3)}")
        print("-------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Tube evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    params = yaml.safe_load(open(f"datasets/projects/{args.dataset}.yml"))
    params["label_list"].append("no action")

    # main(args, params)
    # exit()

    # load qmm outputs #
    id = 0  # 0-> calc precision and recall, 1-> visualization
    name = ["tube-", "videotubes-"][id]
    if args.link_cues == "feature":
        dir = osp.join(args.check_dir, args.dataset, args.qmm_name, "qmm_tubes", args.subset)
        filename = f"epoch:{args.load_epoch}_pth:{args.psn_score_th}_simth:{args.sim_th}_fl:{args.filter_length}"
    elif args.link_cues == "iou":
        dir = osp.join(args.check_dir, args.dataset, "iou_link", "qmm_tubes", args.subset)
        filename = f"pth:{args.psn_score_th}_iouth:{args.sim_th}_fl:{args.filter_length}"
    pred_tubes = [obj for obj in tqdm(utils.TarIterator(dir, name + filename))]

    gt_tubes = make_gt_tubes(args.dataset, args.subset, params)

    if id == 0:  # calc precision and recall
        # calc_precision_recall(pred_tubes, gt_tubes, None, args.tiou_th)
        # calc_precision_recall_per_class(pred_tubes, gt_tubes, params["label_list"][:-1], None, args.tiou_th)
        calc_precision_recall_per_motion(pred_tubes, gt_tubes, None, args.tiou_th)
    elif id == 1:  # visualization
        for tube in pred_tubes:
            video_name = tube.video_name
            video_ano = tube.ano
            video_path = osp.join(params["dataset_path_video"], video_name + ".avi")
            make_video_with_tube(video_path, params["label_list"], tube.tubes, video_ano=video_ano, plot_label=True)
            os.remove("test.avi")
