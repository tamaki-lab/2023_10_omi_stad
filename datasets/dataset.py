import pathlib
from pathlib import Path
import os
import os.path as osp
import json
import torch
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
import numpy as np
import cv2
import yaml
import av

from datasets.utils import xyxy2cxcywh, box_normalize


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, subset):
        params = yaml.safe_load(open(f"datasets/projects/{dataset_name}.yml", encoding="UTF-8"))
        video_name_list_txt = params[subset + "_videos_file_path"]
        with open(video_name_list_txt, encoding="UTF-8") as f:
            video_name_list = f.readlines()

        self.dataset_path = params["dataset_path"]
        self.video_name_list = [video.replace("\n", "") for video in video_name_list]
        self.ano = read_ano(dataset_name, subset, params)

    def __len__(self):
        return len(self.video_name_list)

    def __getitem__(self, idx):
        video_name = self.video_name_list[idx]
        img_paths = [
            path
            for path in Path(osp.join(self.dataset_path, video_name)).glob("**/*.*")
            if path.suffix in [".png", ".jpg"]
        ]
        img_paths = sorted(img_paths)
        video_ano = self.ano[video_name]
        return img_paths, video_ano


def read_ano(dataset_name, subset, params) -> dict:
    """
    Returns:
        dict[video_name][frame_id][object_id]=[x1, y1, x2, y2, cls_id]
    """
    if dataset_name == "ucf101-24":
        return read_ucf_ano(subset, params)
    elif dataset_name == "jhmdb21":
        return read_jhmdb_ano(subset, params)
    else:
        raise NameError("invalide dataset name")


def read_ucf_ano(subset, params):
    video_list_txt = params[subset + "_videos_file_path"]
    with open(video_list_txt, encoding="UTF-8") as f:
        video_list = f.readlines()
    video_list = [video.replace("\n", "") for video in video_list]

    file_path = params["annotation_file_path"]
    df = pd.read_json(file_path)

    ano_dict = {}
    for video_name in video_list:
        ano_dict[video_name] = {}
        for obj_id in range(len(df[video_name]["annotations"])):
            sf = df[video_name]["annotations"][obj_id]["sf"]
            bbox_list = df[video_name]["annotations"][obj_id]["boxes"]
            cls_id = df[video_name]["annotations"][obj_id]["label"]
            for j, box in enumerate(bbox_list):
                box[2] = box[0] + box[2]
                box[3] = box[1] + box[3]
                if j + sf not in ano_dict[video_name]:
                    ano_dict[video_name][j + sf] = {}
                ano_dict[video_name][j + sf][obj_id] = box
                ano_dict[video_name][j + sf][obj_id].append(cls_id)

    return ano_dict


def read_jhmdb_ano(subset, params):
    video_list_txt = params[subset + "_videos_file_path"]
    with open(video_list_txt, encoding="UTF-8") as f:
        video_list = f.readlines()
    video_list = [video.replace("\n", "") for video in video_list]

    json_load = json.load(open(params["annotation_file_path"], encoding="UTF-8"))
    class_list = sorted(os.listdir(params["dataset_path"]))[1:]
    cls_to_idx = {cls: i for i, cls in enumerate(class_list)}

    ano_dict = {}
    for video_name in video_list:
        cls = video_name.split("/")[0]
        cls_id = cls_to_idx[cls]
        ano_dict[video_name] = {}
        for i, box in enumerate(json_load["gttubes"][video_name][str(cls_id)][0]):
            ano_dict[video_name][i] = {}
            ano_dict[video_name][i][0] = box[1:]
            ano_dict[video_name][i][0].append(cls_id)

    return ano_dict


def simple_collater(data):
    img_paths = data[0][0]  # Assuming batch size 1
    video_ano = data[0][1]

    return img_paths, video_ano


def get_video_loader(dataset_name, subset):
    return DataLoader(VideoDataset(dataset_name, subset), batch_size=1, collate_fn=simple_collater)


class VideoData(torch.utils.data.Dataset):
    """
    1つのvideoから1フレームを取り出すクラス
    loaderにする際にshuffle=Falseにし,取り出すフレーム数はbatch sizeで指定
    """

    def __init__(self, img_paths: list[pathlib.PosixPath], video_ano: dict):

        self.img_paths = img_paths
        self.ano = video_ano

        self.org_size = (Image.open(img_paths[0]).height, Image.open(img_paths[0]).width)
        self.resize_size = (512, 512)
        self.resize_scale, self.resized_hw = self.get_resize_scale(self.org_size[0], self.org_size[1], self.resize_size[0])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = self.transform(Image.open(img_path))
        label = self.get_label(idx)
        return img, label

    def get_resize_scale(self, width, height, resize_size=512):
        if height > width:
            scale = resize_size / height
            resized_h = resize_size
            resized_w = int(width * scale)
        else:
            scale = resize_size / width
            resized_h = int(height * scale)
            resized_w = resize_size
        return scale, (resized_h, resized_w)

    def transform(self, img):
        img = np.array(img).astype(np.float32) / 255.0

        # resize
        img = cv2.resize(
            img, (self.resized_hw[0], self.resized_hw[1]), interpolation=cv2.INTER_LINEAR
        )
        new_img = np.zeros((self.resize_size[0], self.resize_size[1], 3))
        new_img[0:self.resized_hw[1], 0:self.resized_hw[0]] = img

        # normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        new_img = (new_img - mean) / std

        new_img = torch.from_numpy(new_img).to(torch.float32)
        new_img = torch.permute(new_img, (2, 0, 1))

        return new_img

    def get_label(self, frame_idx):
        annotations = np.zeros((0, 6))
        if frame_idx in self.ano:
            frame_ano = self.ano[frame_idx]
            for obj_id, one_bbox_ano in frame_ano.items():
                annotation = np.zeros((1, 6))  # [x,y,x,y,class_id, person_id]
                annotation[0, :5] = one_bbox_ano
                annotation[0, :4] *= self.resize_scale
                annotation[0, :4] = xyxy2cxcywh(annotation[0, :4])
                annotation[0, :4] = box_normalize(annotation[0, :4], self.resize_size)
                annotation[0, 5] = int(obj_id)
                annotations = np.append(annotations, annotation, axis=0)

        label = {}
        label["boxes"] = torch.Tensor(annotations[:, 4])
        label["action_labels"] = torch.Tensor(annotations[:, 4]).to(torch.int64)
        label["labels"] = torch.ones(annotations.shape[0], dtype=torch.int64)   # object class (id of person class in detr is 1)
        label["person_id"] = torch.Tensor(annotations[:, 5]).to(torch.int64)
        label["orig_size"] = torch.as_tensor(self.org_size)
        label["size"] = torch.as_tensor(self.resize_size)

        return label


def get_sequential_loader(img_paths, video_ano, n_frames):
    dataset = VideoData(img_paths, video_ano)
    loader = DataLoader(dataset, batch_size=n_frames, collate_fn=collater)
    return loader


def collater(data):
    imgs = [d[0] for d in data]
    clip = torch.stack(imgs, dim=0)
    targets = [d[1] for d in data]

    return clip, targets
