from typing import Tuple, Dict
import torch
from pathlib import Path
import os.path as osp
import pandas as pd
from tqdm import tqdm
import av
import json
import os


def make_gt_tubes_ucf(subset: str, params: Dict) -> Dict[str, list[Dict]]:
    """raad annotation file and make tube annotation"""

    video_list_txt = params[subset + "_videos_file_path"]
    with open(video_list_txt) as f:
        video_list = f.readlines()
    video_list = [video.replace("\n", "") for video in video_list]

    file_path = params["annotation_file_path"]
    df = pd.read_json(file_path)

    gt_tubes = {}
    for video_name in video_list:
        gt_tubes[video_name] = {}
        for psn_id in range(len(df[video_name]["annotations"])):
            sf = df[video_name]["annotations"][psn_id]["sf"]
            bbox_list = df[video_name]["annotations"][psn_id]["boxes"]
            cls_id = df[video_name]["annotations"][psn_id]["label"]
            gt_tubes[video_name][psn_id] = {"class": cls_id, "boxes": {}}
            for j, box in enumerate(bbox_list):
                box[2] = box[0] + box[2]
                box[3] = box[1] + box[3]
                gt_tubes[video_name][psn_id]["boxes"][sf + j] = torch.Tensor(box)

    for video_name, tubes_in_video in gt_tubes.copy().items():
        gt_tubes[video_name] = [tube_ano for tube_key, tube_ano in tubes_in_video.items()]

    return gt_tubes


def make_gt_tubes_jhmdb(subset: str, params: Dict) -> Dict[str, list[Dict]]:
    """raad annotation file and make tube annotation"""

    video_list_txt = params[subset + "_videos_file_path"]
    with open(video_list_txt) as f:
        video_list = f.readlines()
    video_list = [video.replace("\n", "") for video in video_list]

    json_load = json.load(open(params["annotation_file_path"]))
    class_list = sorted(os.listdir(params["dataset_path"]))[1:]
    cls_to_idx = {cls: i for i, cls in enumerate(class_list)}

    gt_tubes = {}
    for video_name in video_list:
        cls = video_name.split("/")[0]
        cls_id = cls_to_idx[cls]
        gt_tubes[video_name] = [{"class": cls_id, "boxes": {}}]
        for i, box in enumerate(json_load["gttubes"][video_name][str(cls_id)][0]):
            gt_tubes[video_name][0]["boxes"][i] = torch.Tensor(box[1:])

    return gt_tubes


def make_gt_tubes_ava(subset: str, params: Dict) -> Dict[str, list[Dict]]:
    """raad annotation file (given in bbox) and make tube annotation"""

    org_label_list_file = params["org_label_list_file_path"]
    label_list_file = params["label_list_file_path"]
    action_id2id = {}  # csv annofile id to classification id(only used class)
    org_label_list = []
    with open(org_label_list_file) as f:
        for line in f:
            if "name" in line:
                label_name = line.split('name: "')[1].replace('"\n', "")
                org_label_list.append(label_name)
    tgt_label_list = []
    with open(label_list_file) as f:
        for line in f:
            if "name" in line:
                label_name = line.split('name: "')[1].replace('"\n', "")
                tgt_label_list.append(label_name)
    for i, label in enumerate(org_label_list):
        if label not in tgt_label_list:
            continue
        tgt_id = tgt_label_list.index(label)
        action_id2id[i + 1] = tgt_id

    video_scale = get_ava_scale(osp.join(params["dataset_path"], subset))
    df = pd.read_csv(
        params[subset + "_annotation_file_path"],
        header=None,
        names=["video_name", "sec", "x1", "y1", "x2", "y2", "cls_id", "psn_id"],
    )
    gt_tubes = {}
    video_name_list = list(set(df["video_name"].to_list()))
    for video_name in video_name_list:
        gt_tubes[video_name] = {}

    for index, row in tqdm(df.iterrows(), total=len(df)):
        video_name = row["video_name"]
        sec = row["sec"]
        psn_id = row["psn_id"]
        x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
        cls_id = row["cls_id"]
        if cls_id not in action_id2id:
            continue
        cls_id = action_id2id[cls_id]
        width = video_scale[video_name][0]
        height = video_scale[video_name][1]
        resize_scale = max([width, height]) / 512

        x1, y1, x2, y2 = int(x1 * resize_scale), int(y1 * resize_scale), int(x2 * resize_scale), int(y2 * resize_scale) # predのスケールにアノテーションを合わせる
        x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)  # predのスケールをアノテーションに合わせる

        # 同一人物かつ同一行動を確認するためのキー
        key_name = str(psn_id) + "-" + str(cls_id)
        if key_name not in gt_tubes[video_name]:
            gt_tubes[video_name][key_name] = {"class": cls_id, "boxes": {}}
        gt_tubes[video_name][key_name]["boxes"][sec] = [x1, y1, x2, y2]

    for video_name, tubes_in_video in gt_tubes.copy().items():
        gt_tubes[video_name] = [tube_ano for tube_key, tube_ano in tubes_in_video.items()]

    return gt_tubes


def get_ava_scale(videos_dir: str) -> Dict[str, Tuple[int, int]]:
    """
    Args:
        videos_dir (str): video directory

    Returns:
        Dict[str, Tuple[int, int]]: key is video_name, value is (width, height)
    """

    scale_dict = {}

    video_paths = Path(videos_dir).glob("**/*.*")
    for video_path in video_paths:
        video_name = video_path.stem
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        width = stream.codec_context.width
        height = stream.codec_context.height
        scale_dict[video_name] = (width, height)

    return scale_dict


def make_gt_tubes(dataset: str, subset: str, params: Dict) -> Dict[str, list[Dict]]:
    """raad annotation file (given in bbox) and make tube annotation
    Args:
        subset (str): "jhmdb21", "ucf101-24", "ava"
        subset (str): "train" or "val"
        params (Dict):
    Returns:
        Dict[str, list[Dict]]:
            - key is video_name, value is list(tubes annotation)
                - list length is num of tubes in one video
                - The elements of the list are dict(keys are "class_id" and "boxes")
                    - "class_id": class_id
                    - "boxes": {"frame_idx": [x1,y1,x2,y2]}
                        - dict length is len(tube) (=num of bbox)
            {video_name:[{"class_id":class_id, "boxes":[{"frame_idx":bbox},...]},...]}
    """

    if dataset == "ucf101-24":
        return make_gt_tubes_ucf(subset, params)
    elif dataset == "jhmdb21":
        return make_gt_tubes_jhmdb(subset, params)
    elif dataset == "ava":
        return make_gt_tubes_ava(subset, params)
    else:
        raise NameError(f"Invalide dataset name: {dataset}")
