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
from util.box_ops import tube_iou
from models import build_model
from models.person_encoder import PersonEncoder, NPairLoss
from models.tube import ActionTubes


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    # setting #
    parser.add_argument('--dataset', default='jhmdb21', type=str, choices=['ucf101-24', 'jhmdb21'])
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--ex_name', default='jhmdb_wd:e4', type=str)
    # loader
    parser.add_argument('--n_frames', default=128, type=int)
    parser.add_argument('--subset', default="val", type=str, choices=["train", "val"])
    # person encoder
    parser.add_argument('--load_epoch', default=15, type=int)
    parser.add_argument('--psn_score_th', default=0.9, type=float)
    parser.add_argument('--sim_th', default=0.5, type=float)
    parser.add_argument('--tiou_th', default=0.2, type=float)
    parser.add_argument('--iou_th', default=0.2, type=float)    # for visualization
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


@torch.no_grad()
def main(args, params):

    device = torch.device(f"cuda:{args.device}")

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    detr, criterion, postprocessors = build_model(args)
    detr.to(device)
    detr.eval()
    if args.dataset == "ucf101-24":
        pretrain_path = "checkpoint/ucf101-24/detr:headtune/detr/epoch_30.pth"
        detr.load_state_dict(torch.load(pretrain_path))
    else:
        pretrain_path = "checkpoint/detr/" + utils.get_pretrain_path(args.backbone, args.dilation)
        detr.load_state_dict(torch.load(pretrain_path)["model"])
    criterion.to(device)
    criterion.eval()

    psn_encoder = PersonEncoder(skip=args.is_skip).to(device)
    psn_encoder.eval()
    dir_encoder = osp.join(args.check_dir, args.dataset, args.ex_name, "encoder")
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
                tube.update(d_queries, p_embed, frame_idx, psn_indices[t], psn_boxes[t])

        tube.filter()
        pred_tubes.append(tube)

        # make new video with tube
        continue
        video_path = osp.join(params["dataset_path_video"], video_name + ".avi")
        utils.give_label(video_ano, tube.tubes, params["num_classes"], args.iou_th)
        make_video_with_tube(video_path, params["label_list"], tube.tubes, video_ano=video_ano, plot_label=True)
        os.remove("test.avi")

    dir = osp.join(args.check_dir, args.dataset, args.ex_name, "qmm_tubes", args.subset)
    filename = f"epoch:{args.load_epoch}_pth:{args.psn_score_th}_simth:{args.sim_th}"
    utils.write_tar(pred_tubes, dir, filename)

    gt_tubes = make_gt_tubes(args.dataset, args.subset, params)
    video_names = [tubes.video_name for tubes in pred_tubes]
    calc_precision_recall(pred_tubes, gt_tubes, video_names, args.tiou_th)


def calc_precision_recall(pred_tubes, gt_tubes, video_names=None, tiou_th=0.5):
    pred_tubes = [(video_tubes.video_name, video_tubes.extract(tube)) for video_tubes in pred_tubes for tube in video_tubes.tubes]

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Tube evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    params = yaml.safe_load(open(f"datasets/projects/{args.dataset}.yml"))
    params["label_list"].append("no action")

    if args.dataset == "jhmdb21":
        args.psn_score_th = params["psn_score_th"]
        args.iou_th = params["iou_th"]

    main(args, params)

    # load qmm outputs and calc precision and recall #
    # dir = osp.join(args.check_dir, args.dataset, args.ex_name, "qmm_tubes", args.subset)
    # filename = f"epoch:{args.load_epoch}_pth:{args.psn_score_th}_simth:{args.sim_th}"
    # pred_tubes = [obj for obj in utils.TarIterator(dir, filename)]
    # gt_tubes = make_gt_tubes(args.dataset, args.subset, params)
    # video_names = [tubes.video_name for tubes in pred_tubes]
    # calc_precision_recall(pred_tubes, gt_tubes, video_names, args.tiou_th)

    # visualization #
    # for tube in pred_tubes:
    #     video_name = tube.video_name
    #     video_ano = tube.ano
    #     video_path = osp.join(params["dataset_path_video"], video_name + ".avi")
    #     utils.give_label(video_ano, tube.tubes, params["num_classes"], args.iou_th)
    #     make_video_with_tube(video_path, params["label_list"], tube.tubes, video_ano=video_ano, plot_label=True)
    #     os.remove("test.avi")
