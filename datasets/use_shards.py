from typing import Tuple, List
from functools import partial
import pickle
import json
import os
from pathlib import Path
import random
import io
from io import BytesIO
import webdataset as wds
import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import av

# from utils import info_from_json  # for debug

from datasets.utils import info_from_json


def video_decorder_for_detr(
    video_pickle,
    num_sample_frame,
    sampling_rate,
):
    nf = num_sample_frame
    sr = sampling_rate

    jpg_byte_list, video_stats = video_pickle
    video_stats = json.loads(video_stats)

    n_frames = video_stats["n_frames"]
    bbox_ano = video_stats["label_bbox"]

    clip, frame_indices_list = get_clip(jpg_byte_list, n_frames, nf, sr)
    new_clip, scale = resize(clip)

    label = [{} for _ in range(nf)]
    bbox_anno = get_clip_label(bbox_ano, frame_indices_list, scale)
    for i in range(nf):
        label[i]["boxes"] = torch.Tensor(bbox_anno[i][:, :4])
        label[i]["labels"] = torch.Tensor(bbox_anno[i][:, 4]).to(torch.int64)
        label[i]["orig_size"] = torch.as_tensor([int(512), int(512)])
        label[i]["size"] = torch.as_tensor([int(512), int(512)])

    is_org_label = [True for _ in range(nf)]  # ucf and jhmdb are laebed all frames

    label = (label, is_org_label)

    return new_clip, label


# https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/15403b5371a64defb2a7c74e162c6e880a7f462c/efficientdet/dataset.py#L110
def resize(clip: list, img_size=512) -> Tuple[torch.Tensor, float]:
    height, width, _ = np.array(clip[0]).shape
    if height > width:
        scale = img_size / height
        resized_height = img_size
        resized_width = int(width * scale)
    else:
        scale = img_size / width
        resized_height = int(height * scale)
        resized_width = img_size

    new_clip = []
    for frame in clip:
        img = np.array(frame).astype(np.float32) / 255.0
        img = cv2.resize(
            img, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
        )

        new_img = np.zeros((img_size, img_size, 3))
        new_img[0:resized_height, 0:resized_width] = img
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        new_img = (new_img - mean) / std
        new_clip.append(new_img)

    new_clip = [torch.from_numpy(img).to(torch.float32) for img in new_clip]
    new_clip = torch.stack(new_clip, 0)
    new_clip = torch.permute(new_clip, (3, 0, 1, 2))

    return new_clip, scale


def get_clip(jpg_byte_list: list, n_frames: int, nf: int, sr: int) -> Tuple[list, list]:
    """
    Args:
        jpg_byte_list (list): original clip
        n_frames (int): num of frames in original clip
        nf (int): num of frames in the clip to got
        sr (int): sampling rate

    Returns:
        Tuple[List, List]: clip, list of frame index in original clip
    """
    tmp_start_frame = max(0, n_frames - sr * (nf - 1))
    start_frame = random.randint(0, tmp_start_frame)
    frame_indices = range(start_frame, start_frame + nf * sr, sr)
    frame_indices_list = list(frame_indices)

    clip = []
    for i, f_idx in enumerate(frame_indices):
        if f_idx < n_frames - 1:
            clip.append(jpg_byte_list[f_idx])
        else:
            clip.append(jpg_byte_list[-1])
            frame_indices_list[i] = frame_indices_list[i - 1]

    clip = [Image.open(BytesIO(img)) for img in clip]

    return clip, frame_indices_list


def get_clip_label(
    bbox_ano: dict, frame_indices_list: list, scale: float
) -> List[np.ndarray]:
    """
    Args:
        bbox_ano (dict): annotation
        frame_indices_list (list): list of frame index in original clip
        scale (float): resize scale

    Returns:
        List[np.ndarray]: clip annotation
    """
    label = []
    for f_idx in frame_indices_list:
        annotations = np.zeros((0, 5))
        if str(f_idx) in bbox_ano:
            for _, one_bbox_ano in bbox_ano[str(f_idx)].items():
                annotation = np.zeros((1, 5))
                annotation[0, :] = one_bbox_ano
                annotation[0, :4] *= scale
                annotations = np.append(annotations, annotation, axis=0)
        label.append(annotations)
    return label


def video_decorder(
    video_pickle,
    num_sample_frame,
    sampling_rate,
):
    nf = num_sample_frame
    sr = sampling_rate

    jpg_byte_list, video_stats = video_pickle
    video_stats = json.loads(video_stats)

    n_frames = video_stats["n_frames"]
    bbox_ano = video_stats["label_bbox"]

    clip, frame_indices_list = get_clip(jpg_byte_list, n_frames, nf, sr)

    new_clip, scale = resize(clip)

    label = get_clip_label(bbox_ano, frame_indices_list, scale)

    is_org_label = [True for _ in range(nf)]  # ucf and jhmdb are laebed all frames

    label = (label, is_org_label)

    return new_clip, label


def get_frame_indices(
    n_sample_frames,
    sampling_rate,
    total_frames,
    key_frame_idx=0,
    is_random_sampling=False,
    clip_duration=10,
):
    nf = n_sample_frames
    sr = sampling_rate

    if is_random_sampling:
        tmp_start_frame = max(0, total_frames - sr * (nf - 1))
        start_frame = random.randint(0, tmp_start_frame - 1)
        frame_indices = range(start_frame, start_frame + nf * sr, sr)
        return frame_indices

    key_frame_indices = np.linspace(0, total_frames, clip_duration + 1)
    key_frame_indices = [int(k) for k in key_frame_indices]
    key_idx = random.choice(key_frame_indices[1:-1])
    frame_indices = range(
        key_idx - key_frame_idx * sr, key_idx + (nf - key_frame_idx) * sr, sr
    )

    return frame_indices


def get_clip_untrimmed(video_bytes, n_frames, nf, frame_indices):
    container = av.open(io.BytesIO(video_bytes))

    clip = [None for _ in range(nf)]
    frame_indices_list = list(frame_indices)
    n_front_shortage = len([x for x in frame_indices_list if x < 0])
    n_rear_shortage = len([x for x in frame_indices_list if x > n_frames - 1])
    current_frame = n_front_shortage

    for i, frame in enumerate(container.decode(video=0)):
        if i < frame_indices_list[current_frame]:
            continue
        img = frame.to_ndarray(format="rgb24")
        clip[current_frame] = img
        current_frame += 1
        if current_frame == nf - n_rear_shortage:
            break
    for i in range(n_front_shortage):
        clip[i] = clip[n_front_shortage]
        frame_indices_list[i] = frame_indices_list[n_front_shortage]
    for i in range(n_rear_shortage):
        clip[current_frame + i] = clip[current_frame - 1]
        frame_indices_list[i] = frame_indices_list[n_rear_shortage]

    clip = np.stack(clip, 0)

    return clip, frame_indices_list


def get_clip_untrimmed_label(
    bbox_ano, frame_indices, n_frames, scale, clip_duration=10, ano_comp=False
):
    key_frames = np.linspace(0, n_frames, clip_duration + 1)
    key_frames = [int(k) for k in key_frames]
    org_ano = {}
    for i, key_frame in enumerate(key_frames):
        if str(i) in bbox_ano:
            org_ano[key_frame] = bbox_ano[str(i)]
        else:
            org_ano[key_frame] = {"None": None}

    label = []
    is_org_label = []
    for fi in frame_indices:
        annotations = np.zeros((0, 5))
        if fi in org_ano.keys():
            is_org_label.append(True)
            for idx, one_bbox_ano in org_ano[fi].items():
                if idx == "None":
                    break
                ano = np.zeros((1, 5))
                ano[0, :] = one_bbox_ano
                ano[0, :4] *= scale
                annotations = np.append(annotations, ano, axis=0)
        else:  # complemant annotation
            is_org_label.append(False)
            if not ano_comp:
                label.append(annotations)
                continue
            near_key_frame_index = np.abs(np.asarray(key_frames) - fi).argmin()
            near_key_frame = key_frames[near_key_frame_index]
            if fi - near_key_frame > 0:
                pre_ano_fi = near_key_frame
                post_ano_fi = key_frames[near_key_frame_index + 1]
            else:
                pre_ano_fi = key_frames[near_key_frame_index - 1]
                post_ano_fi = near_key_frame

            if (pre_ano_fi in org_ano) and (post_ano_fi in org_ano):
                pre_ano = org_ano[pre_ano_fi]
                post_ano = org_ano[post_ano_fi]

                for key in pre_ano.keys():
                    if key in post_ano:
                        ano = np.zeros((1, 5))
                        ano1 = np.zeros((1, 5))
                        ano2 = np.zeros((1, 5))

                        d1, d2 = fi - pre_ano_fi, post_ano_fi - fi
                        r1 = d1 / (d1 + d2)
                        r2 = d2 / (d1 + d2)
                        ano1[0, :] = org_ano[pre_ano_fi][key]
                        ano2[0, :] = org_ano[post_ano_fi][key]
                        ano1[0, :4] *= r1
                        ano2[0, :4] *= r2
                        ano[0, :4] = ano1[0, :4] + ano2[0, :4]
                        ano[0, :4] *= scale
                        ano[0, 4] = ano1[0, 4]
                        annotations = np.append(annotations, ano, axis=0)
                    else:
                        continue
        label.append(annotations)
    return label, is_org_label


def untrimmed_video_decorder(
    video_pickle,
    num_sample_frame,
    sampling_rate,
):
    video_bytes, video_stats = video_pickle
    video_stats = json.loads(video_stats)

    n_frames = video_stats["n_frames"]
    bbox_ano = video_stats["label_bbox"]

    nf = num_sample_frame
    sr = sampling_rate

    frame_indices = get_frame_indices(nf, sr, n_frames)

    clip, frame_indices_list = get_clip_untrimmed(
        video_bytes, n_frames, nf, frame_indices
    )

    new_clip, scale = resize(clip)

    label, is_org_label = get_clip_untrimmed_label(
        bbox_ano, frame_indices_list, n_frames, scale
    )
    label = (label, is_org_label)

    return new_clip, label


def make_dataset(
    shards_url,
    dataset_size,
    shuffle_buffer_size,
    clip_frames,
    sampling_rate,
):
    dataset_name = shards_url[0].split("/")[-3]
    if dataset_name in ["UCF101-24", "JHMDB"]:
        decoder = video_decorder_for_detr
        # decoder = video_decorder
    elif dataset_name == "AVA":
        decoder = untrimmed_video_decorder
    else:
        raise NameError(f"invalide dataset name: {dataset_name}")

    decode_video = partial(
        decoder,
        num_sample_frame=clip_frames,
        sampling_rate=sampling_rate,
    )

    dataset = wds.WebDataset(shards_url)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.decode(
        wds.handle_extension("video.pickle", lambda x: (pickle.loads(x))),
    )
    dataset = dataset.to_tuple(
        "video.pickle",
    )
    dataset = dataset.map_tuple(
        decode_video,
    )
    dataset = dataset.with_length(dataset_size)

    return dataset


def collater(data):
    batch = {}
    is_org_label = [d[0][1][1] for d in data]
    annots = [d[0][1][0] for d in data]
    num_frame = data[0][0][0].shape[1]
    max_num_annots = 0
    for video_ano in annots:
        num_ano = max(f_ano.shape[0] for f_ano in video_ano)
        if num_ano > max_num_annots:
            max_num_annots = num_ano

    if max_num_annots > 0:
        annots_padded = torch.ones((len(annots), num_frame, max_num_annots, 5)) * -1
        for idx, video_annot in enumerate(annots):
            for jdx, frame_annot in enumerate(video_annot):
                if frame_annot.shape[0] > 0:
                    frame_annot = torch.from_numpy(frame_annot)
                    annots_padded[idx, jdx, : frame_annot.shape[0], :] = frame_annot
    else:
        annots_padded = torch.ones((len(annots), num_frame, 1, 5)) * -1

    videos = [d[0][0] for d in data]
    videos = torch.stack(videos, dim=0)

    batch["clip"] = videos
    batch["ano"] = annots_padded
    batch["is_org_label"] = is_org_label

    return batch


def collater_for_detr(data):
    videos = [d[0][0] for d in data]
    videos = torch.stack(videos, dim=0)
    targets = [d[0][1][0] for d in data]

    return videos, targets


def get_loader(
    shard_path,
    batch_size,
    clip_frames=4,
    sampling_rate=8,
    num_workers=16,
    shuffle_buffer_size=100,
):
    shards_path = [
        str(path) for path in Path(shard_path).glob("*.tar") if not path.is_dir()
    ]
    dataset_size, n_classes = info_from_json(shard_path)
    dataset = make_dataset(
        shards_url=shards_path,
        dataset_size=dataset_size,
        shuffle_buffer_size=shuffle_buffer_size,
        clip_frames=clip_frames,
        sampling_rate=sampling_rate,
    )
    loader = DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=collater_for_detr,
        # collate_fn=collater,
        drop_last=True,
    )

    return loader


def save_fig(clip, labels, file_name):
    frame_list = []
    for i in range(16):
        img = clip[:, i, :, :].numpy().transpose(1, 2, 0)
        label = labels[i]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (512, 512))
        img = np.clip(img * 255, a_min=0, a_max=255).astype(np.uint8)
        num_max_bbox = label.shape[0]
        for idx in range(num_max_bbox):
            if label[idx, 4].item() == -1:
                break
            x1 = int(label[idx, 0].item())
            y1 = int(label[idx, 1].item())
            x2 = int(label[idx, 2].item())
            y2 = int(label[idx, 3].item())
            cv2.rectangle(
                img,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=(0, 255, 0),
                thickness=3,
                lineType=cv2.LINE_4,
                shift=0,
            )
        frame_list.append(img)

    rows = 4
    cols = 4
    frame_id = 0

    fig, axes = plt.subplots(rows, cols, figsize=(16, 16), tight_layout=True)
    for i in range(rows):
        for j in range(cols):
            img = frame_list[frame_id]
            subplot_title = "frame:" + str(frame_id)
            axes[i, j].set_title(subplot_title)
            axes[i, j].imshow(img)
            frame_id = frame_id + 1
    os.makedirs("/".join(file_name.split("/")[:-1]), exist_ok=True)
    plt.savefig(file_name)
