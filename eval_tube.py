# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import random
from comet_ml import Experiment
from tqdm import tqdm
import os.path as osp

import numpy as np
import torch
from torch.utils.data import DataLoader

# from datasets.use_shards import get_loader
from datasets.dataset import get_video_loader, get_sequential_loader
import util.misc as utils
from engine import evaluate, train_one_epoch
from models import build_model
from models.person_encoder import PersonEncoder, SetInfoNce


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    # loader
    parser.add_argument('--n_frames', default=8, type=int)

    # setting
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--ex_name', default='noskip_th0.7', type=str)
    parser.add_argument('--load_epoch', default=100, type=int)
    parser.add_argument('--psn_score_th', default=0.7, type=float)

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
    pretrain_path = "checkpoint/detr/" + utils.get_pretrain_path(args.backbone, args.dilation)
    detr.load_state_dict(torch.load(pretrain_path)["model"])

    psn_encoder = PersonEncoder().to(device)
    trained_psn_encoder_path = osp.join(args.check_dir, args.ex_name, f"epoch_{args.load_epoch}.pth")
    psn_encoder.load_state_dict(torch.load(trained_psn_encoder_path))
    psn_criterion = SetInfoNce().to(device)

    train_loader = get_video_loader("ucf101-24", "train")
    val_loader = get_video_loader("ucf101-24", "val")

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
    }
    ex.log_parameters(hyper_params)

    pbar_videos = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for viceo_idx, (img_paths, video_ano) in pbar_videos:
        sequential_loader = get_sequential_loader(img_paths, video_ano, args.n_frames)
        person_lists = []
        end_list_idx_set = set()

        pbar_video = tqdm(enumerate(sequential_loader), total=len(sequential_loader), leave=False)
        for clip_idx, (samples, targets) in pbar_video:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = detr(samples)

            orig_target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['bbox'](outputs, orig_target_sizes)

            score_filter_indices = [(result["scores"] > args.psn_score_th).nonzero().flatten() for result in results]
            score_filter_labels = [result["labels"][(result["scores"] > args.psn_score_th).nonzero().flatten()] for result in results]
            psn_indices = [idx[lab == 1] for idx, lab in zip(score_filter_indices, score_filter_labels)]
            psn_queries = [outputs["queries"][0, t][p_idx] for t, p_idx in enumerate(psn_indices)]
            psn_boxes = [result["boxes"][p_idx] for result, p_idx in zip(results, psn_indices)]

            psn_queries = torch.cat(psn_queries, 0)

            psn_features = psn_encoder(psn_queries)
            psn_features = arrange_list(psn_indices, psn_features)

            for t, psn_features_frame in enumerate(psn_features):
                frame_idx = args.n_frames * clip_idx + t
                diff_list = [frame_idx - person_list["idx_of_p_queries"][-1][0] for person_list in person_lists]
                for list_idx, d in enumerate(diff_list):
                    if d >= 8:
                        end_list_idx_set.add(list_idx)
                print("")
                print(frame_idx)
                # print(diff_list)
                # print(end_list_idx_set)
                print("")

                if len(person_lists) == 0:
                    for idx, psn_feature in enumerate(psn_features_frame):
                        query_idx = psn_indices[t][idx].item()
                        person_lists.append({"query": [psn_feature], "idx_of_p_queries": [(frame_idx, query_idx)], "bbox": [psn_boxes[t][idx]]})
                    continue

                final_queries_in_spl = torch.stack([person_list["query"][-1] for person_list in person_lists])
                dot_product = torch.mm(psn_features_frame, final_queries_in_spl.t())
                norm_frame_p_f_queries = torch.norm(psn_features_frame, dim=1).unsqueeze(1)
                norm_final_queries_in_spl = torch.norm(final_queries_in_spl, dim=1).unsqueeze(0)
                sim_scores = dot_product / (norm_frame_p_f_queries * norm_final_queries_in_spl)
                sim_scores = check_end(sim_scores, end_list_idx_set)
                indices_generator = find_max_indices(sim_scores.cpu().detach())
                used_j_list = []
                for i, j in indices_generator:
                    query_idx = psn_indices[t][i].item()
                    if (sim_scores[i, j] > args.psn_score_th) and (j not in used_j_list):
                        person_lists[j]["query"].append(psn_features_frame[i])
                        person_lists[j]["idx_of_p_queries"].append((frame_idx, query_idx))
                        person_lists[j]["bbox"].append(psn_boxes[t][i])
                        used_j_list.append(j)
                    else:
                        person_lists.append({"query": [psn_features_frame[i]], "idx_of_p_queries": [(frame_idx, query_idx)], "bbox": [psn_boxes[t][i]]})

            continue
        print("")

        # make new video with tube


def find_max_indices(tensor):
    for _ in range(tensor.size(0)):
        i, j = np.unravel_index(np.argmax(tensor), tensor.shape)
        yield i, j
        tensor[i, :] = 0
        tensor[:, j] = 0


def arrange_list(x, y):
    """
    yの要素は変えずにxと同じリストの形にする
    xは2重リストでyはリストでその長さはxのサブリストの長さの和となっている
    """
    result = []
    for sublist in x:
        size = len(sublist)
        result.append(y[:size])
        y = y[size:]
    return result


def check_end(tensor, end_idx_set):
    for idx in end_idx_set:
        tensor[:, idx] = 0

    return tensor


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Tube evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
