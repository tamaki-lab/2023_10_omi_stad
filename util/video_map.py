from typing import Tuple, Dict
import torch
import numpy as np
import torch

from util.box_ops import tube_iou, get_motion_ctg


def voc_ap(pr, use_07_metric=False, num_complement=11):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    step = 1 / (num_complement - 1)
    rec, prec = pr[:, 1], pr[:, 0]
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1 + step, step):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / num_complement
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def calc_video_ap(
        pred_tubes: list[Tuple[str, Tuple[float, Dict[str, torch.Tensor]]]],
        gt_tubes: Dict[str, list[Dict[str, torch.Tensor]]],
        tiou_thresh: float,
        num_complement: int) -> float:
    """Calculate video AP

    Args:
        pred_tubes (list[Tuple[str, Tuple[float, Dict[str, torch.Tensor]]]]):
            - list length is num of predicted tubes
            - list element is Tuple
                - the first element is "video name"
                - the secand element is Tuple ("score", {frame_idx: bbox})
            [(video_name, (score, {frame_idx:bbox}))]
        gt_tubes (Dict[str, list[Dict[str, torch.Tensor]]]):
            - key is video_name, value is list
                - list length is num of tubes in one video
                - list element is dict ({frame_idx: bbox})
                    - dict length is len(tube) (=num of bbox)
            video_name:[{frame_idx:[x1,y1,x2,y2]}]
        tiou_thresh (float): Threshold of tIoU (extend IoU temporal)
        num_complement (int): Num of points in precision-recall graph used in the calculation

    Returns:
        float: video_ap
    """

    if len(pred_tubes) == 0:
        return 0

    pred_tubes.sort(key=lambda x: x[1][0], reverse=True)

    pr = np.empty((len(pred_tubes) + 1, 2), dtype=np.float32)
    pr[0, 0] = 1.0
    # pr[0, 0] = 0.0
    pr[0, 1] = 0.0
    tp = 0
    fn = sum([len(tubes) for _, tubes in gt_tubes.items()])
    fp = 0

    for i, (video_name, pred_tube) in enumerate(pred_tubes):
        video_gt_tubes = gt_tubes[video_name]
        tiou_list = []
        for tube_idx, gt_tube in enumerate(video_gt_tubes):
            # tiou_list.append(tube_iou(pred_tube[1], gt_tube, label_centric=True))
            tiou_list.append(tube_iou(pred_tube[1], gt_tube))
        if len(tiou_list) == 0:
            fp += 1
        elif max(tiou_list) > tiou_thresh:
            tp += 1  # TODO 既に正解したものに対して2回目以降に正解した場合の考慮
            fn -= 1
        else:
            fp += 1
        pr[i + 1, 0] = float(tp) / float(tp + fp)
        pr[i + 1, 1] = float(tp) / float(tp + fn + 0.00001)

    ap = voc_ap(pr, num_complement=num_complement)

    return ap


def calc_video_map(
        pred_tubes: list[Tuple[str, Dict]],
        gt_tubes: Dict[str, list[Dict]],
        num_class: int,
        tiou_thresh: float = 0.2,
        num_complement: int = 11) -> list[int]:
    """Calculate video mAP

    Args:
        pred_tubes (list[Tuple[str, Dict]]):
            [video_name,{class:class_idx,score:score,boxes:{frameidx:tensor(shape=4)}]
            - list length is the num of predicted tubes
            - list element is Tuple
                - the first element is "video name"
                - the secand element is Dict (keys are "class", "score" and "boxes")
        gt_tubes (Dict[str, list[Dict]]):
            {video_name:[{class:class_idx,boxes:{frameidx:tensor(shape=4)}}]}
            - key is video_name, value is list
                - list length is num of tubes in one video
                - list element is dict(keys are "class" and "boxes")
        num_class (int): Num of classes
        tiou_thresh (float): Threshold of tIoU (extend IoU temporal)
        num_complement (int): Num of points in precision-recall graph used in the calculation

    Returns:
        list[int]: video ap list (v-mAP = sum(video_ap_list)/num_class)
    """

    pred_tubes_class = [[] for _ in range(num_class)]
    gt_tubes_class = [{} for _ in range(num_class)]

    for video_name, pred_tube in pred_tubes:
        pred_tubes_class[pred_tube["class"]].append((video_name, (pred_tube["score"], pred_tube["boxes"])))

    for video_name in gt_tubes.keys():
        for class_idx in range(num_class):
            gt_tubes_class[class_idx][video_name] = []
    for video_name, video_tubes in gt_tubes.items():
        for video_tube in video_tubes:
            gt_tubes_class[video_tube["class"]][video_name].append(video_tube["boxes"])

    video_ap = []
    for class_idx in range(num_class):
        video_ap.append(calc_video_ap(pred_tubes_class[class_idx], gt_tubes_class[class_idx], tiou_thresh, num_complement))

    return video_ap


def calc_motion_ap(
        pred_tubes: list[Tuple[str, Dict]],
        gt_tubes: Dict[str, list[Dict]],
        tiou_thresh: float = 0.2,
        num_complement: int = 11) -> list[int]:
    """Calculate video mAP

    Args:
        pred_tubes (list[Tuple[str, Dict]]):
            [video_name,{class:class_idx,score:score,boxes:{frameidx:tensor(shape=4)}]
            - list length is the num of predicted tubes
            - list element is Tuple
                - the first element is "video name"
                - the secand element is Dict (keys are "class", "score" and "boxes")
        gt_tubes (Dict[str, list[Dict]]):
            {video_name:[{class:class_idx,boxes:{frameidx:tensor(shape=4)}}]}
            - key is video_name, value is list
                - list length is num of tubes in one video
                - list element is dict(keys are "class" and "boxes")
        num_class (int): Num of classes
        tiou_thresh (float): Threshold of tIoU (extend IoU temporal)
        num_complement (int): Num of points in precision-recall graph used in the calculation

    Returns:
        list[int]: motion ap list (@small, medium, large)
    """

    motion_ctgs = ["small", "medium", "large"]

    pred_tubes_motion = [[] for _ in motion_ctgs]
    gt_tubes_motion = [{} for _ in motion_ctgs]

    for video_name, pred_tube in pred_tubes:
        # ctg = "medium"
        ctg = get_motion_ctg(pred_tube["boxes"])
        ctg_index = motion_ctgs.index(ctg)
        pred_tubes_motion[ctg_index].append((video_name, (pred_tube["class"], pred_tube["score"], pred_tube["boxes"])))

    for video_name in gt_tubes.keys():
        for motion_ctg in motion_ctgs:
            ctg_index = motion_ctgs.index(motion_ctg)
            gt_tubes_motion[ctg_index][video_name] = []
    for video_name, video_tubes in gt_tubes.items():
        for video_tube in video_tubes:
            # ctg = "medium"
            ctg = get_motion_ctg(video_tube["boxes"])
            ctg_index = motion_ctgs.index(ctg)
            gt_tubes_motion[ctg_index][video_name].append((video_tube["class"], video_tube["boxes"]))

    # ap = calc_motion_one_ap(pred_tubes_motion[1], gt_tubes_motion[1], tiou_thresh, num_complement)
    # print(ap)

    for i, motion_ctg in enumerate(motion_ctgs):
        ap = calc_motion_one_ap(pred_tubes_motion[i], gt_tubes_motion[i], tiou_thresh, num_complement)
        print(f"{motion_ctg}: {round(ap, 4)}")


def calc_motion_one_ap(
        pred_tubes: list[Tuple[str, Tuple[float, Dict[str, torch.Tensor]]]],
        gt_tubes: Dict[str, list[Dict[str, torch.Tensor]]],
        tiou_thresh: float,
        num_complement: int) -> float:
    """Calculate video AP

    Args:
        pred_tubes (list[Tuple[str, Tuple[float, Dict[str, torch.Tensor]]]]):
            - list length is num of predicted tubes
            - list element is Tuple
                - the first element is "video name"
                - the secand element is Tuple ("class", "score", {frame_idx: bbox})
            [(video_name, (score, {frame_idx:bbox}))]
        gt_tubes (Dict[str, list[Dict[str, torch.Tensor]]]):
            - key is video_name, value is list
                - list length is num of tubes in one video
                - list element is dict ({frame_idx: bbox})
                    - dict length is len(tube) (=num of bbox)
            video_name:[{frame_idx:[x1,y1,x2,y2]}]
        tiou_thresh (float): Threshold of tIoU (extend IoU temporal)
        num_complement (int): Num of points in precision-recall graph used in the calculation

    Returns:
        float: video_ap
    """

    if len(pred_tubes) == 0:
        return 0

    pred_tubes.sort(key=lambda x: x[1][1], reverse=True)

    pr = np.empty((len(pred_tubes) + 1, 2), dtype=np.float32)
    pr[0, 0] = 1.0
    # pr[0, 0] = 0.0
    pr[0, 1] = 0.0
    tp = 0
    fn = sum([len(tubes) for _, tubes in gt_tubes.items()])
    fp = 0

    for i, (video_name, pred_tube) in enumerate(pred_tubes):
        video_gt_tubes = gt_tubes[video_name]
        tiou_list = []
        for tube_idx, gt_tube in enumerate(video_gt_tubes):
            # tiou_list.append(tube_iou(pred_tube[1], gt_tube, label_centric=True))
            tiou_list.append(tube_iou(pred_tube[2], gt_tube[1]))
        if len(tiou_list) == 0:
            fp += 1
        else:
            max_v = max(tiou_list)
            max_idx = tiou_list.index(max_v)
            if (max_v > tiou_thresh) and (video_gt_tubes[max_idx][0] == pred_tube[0]):
                tp += 1  # TODO 既に正解したものに対して2回目以降に正解した場合の考慮
                fn -= 1
            else:
                fp += 1
        pr[i + 1, 0] = float(tp) / float(tp + fp)
        pr[i + 1, 1] = float(tp) / float(tp + fn + 0.00001)

    ap = voc_ap(pr, num_complement=num_complement)
    print(f"tp:{tp}")

    return ap
