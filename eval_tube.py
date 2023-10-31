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

from datasets.dataset import get_video_loader, get_sequential_loader
import util.misc as utils
from models import build_model
from models.person_encoder import PersonEncoder, SetInfoNce
from models.tube import ActionTube


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
    psn_criterion = SetInfoNce().to(device)
    psn_criterion.eval()

    val_loader = get_video_loader("ucf101-24", "val")

    pbar_videos = tqdm(enumerate(val_loader), total=len(val_loader), leave=False)
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

        # make new video with tube
        video_path = "/".join(img_paths[0].parts[-3:-1])
        video_path = "/mnt/NAS-TVS872XT/dataset/UCF101/video/" + video_path + ".avi"
        make_video_with_tube(video_path, tube.tubes)
        if video_idx == 0:
            exit()
        os.remove("test.avi")
        continue


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


def make_video_with_tube(video_path, person_lists):
    # color_map = random_colors(100)
    color_map = get_color_list()

    # filter person list
    print(f"num_lists(before filterling):{len(person_lists)}")
    person_lists = [person_list for person_list in person_lists if len(person_list["idx_of_p_queries"]) > 8]
    print(f"num_lists(after filterling):{len(person_lists)}")

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
                    # print(f"frame:{frame_idx}, list_idx:{list_idx}, query_idx:{idx}")

                    cv2.rectangle(
                        frame, pt1=(x1, y1), pt2=(x2, y2), color=color_map[list_idx % 10], thickness=2, lineType=cv2.LINE_4, shift=0,
                    )
                    # cv2.rectangle(
                    #     frame, pt1=(x1, y1), pt2=(x2, y2), color=color_map[list_idx], thickness=2, lineType=cv2.LINE_4, shift=0,
                    # )

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
