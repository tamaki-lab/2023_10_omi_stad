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
from util.box_ops import generalized_box_iou
from models import build_model
from models.person_encoder import PersonEncoder, SetInfoNce
from models.action_head import ActionHead


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    # loader
    parser.add_argument('--n_frames', default=8, type=int)

    # setting
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--ex_name', default='noskip_th0.7', type=str)
    parser.add_argument('--load_epoch', default=100, type=int)
    parser.add_argument('--psn_score_th', default=0.7, type=float)
    parser.add_argument('--sim_th', default=0.7, type=float)
    parser.add_argument('--iou_th', default=0.4, type=float)
    parser.add_argument('--n_classes', default=24, type=int)

    # Backbone
    parser.add_argument('--backbone', default='resnet101', type=str, choices=('resnet50', 'resnet101'),
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', default=True,
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")

    # action head
    parser.add_argument('--lr_head', default=1e-4, type=float)
    parser.add_argument('--weight_decay_head', default=1e-4, type=float)
    parser.add_argument('--lr_drop_head', default=50, type=int)

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
    trained_psn_encoder_path = osp.join(args.check_dir, args.ex_name, f"epoch_{args.load_epoch}.pth")
    psn_encoder.load_state_dict(torch.load(trained_psn_encoder_path))
    psn_criterion = SetInfoNce().to(device)
    psn_criterion.eval()

    action_head = ActionHead(n_classes=args.n_classes).to(device)
    # head_criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0] * args.n_classes + [0.05]))
    head_criterion = nn.CrossEntropyLoss()
    optimizer_head = torch.optim.AdamW(action_head.parameters(), lr=args.lr_head, weight_decay=args.weight_decay_head)
    lr_scheduler_head = torch.optim.lr_scheduler.StepLR(optimizer_head, args.lr_drop_head)

    train_loader = get_video_loader("ucf101-24", "train")
    # val_loader = get_video_loader("ucf101-24", "val")

    train_log = {"action_loss": utils.AverageMeter(),
                 "action_acc1": utils.AverageMeter(),
                 "action_acc5": utils.AverageMeter(),
                 "action_acc1_wo_noaction": utils.AverageMeter(),
                 "action_acc5_wo_noaction": utils.AverageMeter()}
    # val_log = {"psn_loss": utils.AverageMeter(),
    #            "diff_psn_score": utils.AverageMeter(),
    #            "same_psn_score": utils.AverageMeter(),
    #            "total_psn_score": utils.AverageMeter()}

    ex = Experiment(
        project_name="stal",
        workspace="kazukiomi",
    )
    hyper_params = {
        "ex_name": args.ex_name + "--train_head",
    }
    ex.log_parameters(hyper_params)

    pbar_videos = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for video_idx, (img_paths, video_ano) in pbar_videos:
        sequential_loader = get_sequential_loader(img_paths, video_ano, args.n_frames)
        person_lists = []
        end_list_idx_set = set()

        # flag = False
        # for frame_idx, frame_ano in video_ano.items():
        #     if len(frame_ano) > 2:
        #         flag = True
        # if not flag:
        #     continue

        pbar_video = tqdm(enumerate(sequential_loader), total=len(sequential_loader), leave=False)
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

            psn_queries = psn_encoder(torch.cat(decoded_queries, 0))
            psn_queries = arrange_list(psn_indices, psn_queries)

            # process per frame
            for t, (d_queries, p_queries) in enumerate(zip(decoded_queries, psn_queries)):
                frame_idx = args.n_frames * clip_idx + t
                add_query_to_list(person_lists, d_queries, p_queries, frame_idx,
                                  psn_indices[t], psn_boxes[t], end_list_idx_set, sim_th=args.sim_th)

        # filter person list
        # print(f"num_lists(before filterling):{len(person_lists)}")
        person_lists = [person_list for person_list in person_lists if len(person_list["idx_of_p_queries"]) > 8]
        # print(f"num_lists(after filterling):{len(person_lists)}")

        # give a label for each query in person list
        org_video_ano = copy.deepcopy(video_ano)
        fix_ano_scale(video_ano)
        give_label(video_ano, person_lists, args.n_classes, args.iou_th)

        # train head
        action_label = [person_list["action_label"] for person_list in person_lists]
        queries_list = [person_list["d_query"] for person_list in person_lists]
        if video_idx % 8 == 0:
            optimizer_head.zero_grad()
        total_value = {"loss": torch.zeros(1).to(device),
                       "acc1": torch.zeros(1).to(device),
                       "acc5": torch.zeros(1).to(device),
                       "acc1_wo_noaction": torch.zeros(1).to(device),
                       "acc5_wo_noaction": torch.zeros(1).to(device)}
        for list_idx, (input_queries, label) in enumerate(zip(queries_list, action_label)):
            input_queries = torch.stack(input_queries, 0).to(device)
            label = torch.Tensor(label).to(torch.int64).to(device)
            outputs = action_head(input_queries)
            loss = head_criterion(outputs, label)
            loss = loss / (8 * len(queries_list))
            loss.backward()
            give_pred(person_lists[list_idx], outputs)
            calc_total_value(total_value, outputs, label, len(queries_list), args.n_classes)  # TODO calc from person_lists outside of "for"
            total_value["loss"] += loss
        # calc_total_value(total_value, outputs, label, len(queries_list), args.n_classes)
            optimizer_head.step()

        train_log["action_loss"].update(total_value["loss"].item())
        train_log["action_acc1"].update(total_value["acc1"].item())
        train_log["action_acc5"].update(total_value["acc5"].item())
        train_log["action_acc1_wo_noaction"].update(total_value["acc1_wo_noaction"].item())
        train_log["action_acc5_wo_noaction"].update(total_value["acc5_wo_noaction"].item())
        ex.log_metric("action_loss", train_log["action_loss"].val, step=video_idx)
        ex.log_metric("action_acc1", train_log["action_acc1"].val, step=video_idx)
        ex.log_metric("action_acc5", train_log["action_acc5"].val, step=video_idx)
        ex.log_metric("action_acc1_wo_noaction", train_log["action_acc1_wo_noaction"].val, step=video_idx)
        ex.log_metric("action_acc5_wo_noaction", train_log["action_acc5_wo_noaction"].val, step=video_idx)

        # make new video with tube
        continue
        video_path = "/".join(img_paths[0].parts[-3:-1])
        video_path = "/mnt/NAS-TVS872XT/dataset/UCF101/video/" + video_path + ".avi"
        make_video_with_tube(video_path, person_lists, org_video_ano)
        os.remove("test.avi")


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


def add_query_to_list(person_lists, d_queries, p_queries, frame_idx, psn_indices, psn_boxes, end_list_idx_set, sim_th=0.7):
    diff_list = [frame_idx - person_list["idx_of_p_queries"][-1][0] for person_list in person_lists]
    for list_idx, d in enumerate(diff_list):
        if d >= 8:
            end_list_idx_set.add(list_idx)

    if len(person_lists) == 0:
        for idx, (d_query, p_query) in enumerate(zip(d_queries, p_queries)):
            query_idx = psn_indices[idx].item()
            person_lists.append({"d_query": [d_query.detach()],
                                 "p_query": [p_query.detach()],
                                 "idx_of_p_queries": [(frame_idx, query_idx)],
                                 "bbox": [psn_boxes[idx]]})
    else:
        sim_scores = get_sim_scores(person_lists, p_queries)
        indices_generator = find_max_indices(sim_scores.cpu().detach(), end_list_idx_set)
        for i, j in indices_generator:
            query_idx = psn_indices[i].item()
            if (sim_scores[i, j] > sim_th) and (j != -1):
                person_lists[j]["d_query"].append(d_queries[i].detach())
                person_lists[j]["p_query"].append(p_queries[i].detach())
                person_lists[j]["idx_of_p_queries"].append((frame_idx, query_idx))
                person_lists[j]["bbox"].append(psn_boxes[i])
            else:
                person_lists.append({"d_query": [d_queries[i].detach()],
                                     "p_query": [p_queries[i].detach()],
                                     "idx_of_p_queries": [(frame_idx, query_idx)],
                                     "bbox": [psn_boxes[i]]})


def get_sim_scores(person_lists, psn_features_frame):
    final_queries_in_spl = torch.stack([person_list["p_query"][-1] for person_list in person_lists])
    dot_product = torch.mm(psn_features_frame, final_queries_in_spl.t())
    norm_frame_p_f_queries = torch.norm(psn_features_frame, dim=1).unsqueeze(1)
    norm_final_queries_in_spl = torch.norm(final_queries_in_spl, dim=1).unsqueeze(0)
    sim_scores = dot_product / (norm_frame_p_f_queries * norm_final_queries_in_spl)
    return sim_scores


def find_max_indices(tensor, end_list_idx_set):
    used_i_list = []
    for j in end_list_idx_set:
        tensor[:, j] = -1
    for _ in range(tensor.size(0)):
        i, j = np.unravel_index(np.argmax(tensor), tensor.shape)
        if tensor[i, j] != -1:
            used_i_list.append(i)
            tensor[i, :] = -1
            tensor[:, j] = -1
            yield i, j
        else:
            break  # そのフレームにおいて全てのリストにクエリが割り当てられた場合は強制的に新しい人物とする
    not_used_i_list = [x for x in range(tensor.size(0)) if x not in used_i_list]
    for i in not_used_i_list:
        yield i, -1


def fix_ano_scale(video_ano, resize_scale=512 / 320):
    for frame_idx, frame_ano in video_ano.items():
        for tube_idx, box_ano in frame_ano.items():
            video_ano[frame_idx][tube_idx][:4] = [x * resize_scale for x in box_ano[:4]]


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


def calc_total_value(total_value, outputs, label, n_lists, n_classes):
    acc1, acc5 = utils.accuracy(outputs, label, topk=(1, 5))  # to assume list length is the batch size
    acc1_wo, acc5_wo = utils.accuracy(outputs[label != n_classes], label[label != n_classes], topk=(1, 5))
    total_value["acc1"] += acc1 / n_lists
    total_value["acc5"] += acc5 / n_lists
    total_value["acc1_wo_noaction"] += acc1_wo / n_lists
    total_value["acc5_wo_noaction"] += acc5_wo / n_lists


def make_video_with_tube(video_path, person_lists, video_ano, plot_label=True):
    # color_map = random_colors(100)
    color_map = get_color_list()

    # make video
    container = av.open(str(video_path))
    stream = container.streams.video[0]

    width = stream.width
    height = stream.height
    codec = stream.codec_context.name
    base_rate = stream.base_rate
    pix_fmt = stream.pix_fmt

    new_container = av.open("test.avi", mode="w")
    new_stream = new_container.add_stream(codec, rate=base_rate)
    new_stream.width = width
    new_stream.height = height
    new_stream.pix_fmt = pix_fmt

    for frame_idx, frame in enumerate(container.decode(video=0)):
        frame = frame.to_ndarray(format="rgb24")

        for list_idx, person_list in enumerate(person_lists):
            for idx, idx_of_query in enumerate(person_list["idx_of_p_queries"]):
                if frame_idx > idx_of_query[0]:
                    continue
                elif frame_idx < idx_of_query[0]:
                    break
                else:
                    box = person_list["bbox"][idx]
                    x1, y1, x2, y2 = box * 320 / 512
                    x1 = int(max(min(x1, 320), 0))
                    x2 = int(max(min(x2, 320), 0))
                    y1 = int(max(min(y1, 240), 0))
                    y2 = int(max(min(y2, 240), 0))
                    action_id = person_list["action_label"][idx]
                    # print(f"frame:{frame_idx}, list_idx:{list_idx}, query_idx:{idx}")

                    cv2.rectangle(
                        frame, pt1=(x1, y1), pt2=(x2, y2),
                        color=color_map[list_idx % 10], thickness=2, lineType=cv2.LINE_4, shift=0,
                    )
                    # cv2.rectangle(
                    #     frame, pt1=(x1, y1), pt2=(x2, y2), color=color_map[list_idx], thickness=2, lineType=cv2.LINE_4, shift=0,
                    # )
                    cv2.putText(
                        frame, text=f"{list_idx}, {action_id}", org=(x1, y1),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3,
                        color=color_map[list_idx % 10], thickness=1, lineType=cv2.LINE_4
                    )

        if (plot_label) and (frame_idx in video_ano):
            for tube_idx, frame_ano in video_ano[frame_idx].items():
                x1, y1, x2, y2 = frame_ano[:4]
                action_id = frame_ano[4]
                cv2.rectangle(
                    frame, pt1=(x1, y1), pt2=(x2, y2),
                    color=(0, 0, 0), thickness=2, lineType=cv2.LINE_4, shift=0,
                )
                cv2.putText(
                    frame, text=f"{tube_idx}, {action_id}", org=(x1, y1),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3,
                    color=(0, 0, 0), thickness=1, lineType=cv2.LINE_4
                )

        frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in new_stream.encode(frame):
            new_container.mux(packet)

    for packet in new_stream.encode():
        new_container.mux(packet)
    new_container.close()


def random_colors(n_hue, brightness=1.0, saturation=1.0, seed=0.1):
    """Random colormap generation
    inspred by
    https://github.com/matterport/Mask_RCNN/blob/3deaec5d902d16e1daf56b62d5971d428dc920bc/mrcnn/visualize.py#L59
    """
    from colorsys import hsv_to_rgb
    from random import shuffle
    color_map = [hsv_to_rgb(hue / n_hue, saturation, brightness)
                 for hue in range(n_hue)]
    shuffle(color_map, random=lambda: seed)
    color_map = (np.array(color_map) * 255).astype(np.uint8)
    color_map = [(int(x[0]), int(x[1]), int(x[2])) for x in color_map]
    return color_map


def get_color_list():
    color_map = [
        (255, 0, 0),  # red
        (0, 255, 0),  # green
        (0, 0, 255),  # blue
        (255, 255, 0),  # yellow
        (255, 255, 255),  # white
        (128, 0, 128),  # purple
        (128, 128, 0),  # olive
        (0, 255, 255),  # mizuiro
        (128, 128, 128),  # gray
        (255, 0, 255),  # mazenda
    ]
    return color_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Tube evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
