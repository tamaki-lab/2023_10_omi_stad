# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
from typing import Tuple, Dict
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_unnormalize(boxes: torch.Tensor, size: torch.Tensor):
    img_h, img_w = size.unbind(-1)
    boxes = boxes * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return boxes


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    return iou
    # lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    # rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    # wh = (rb - lt).clamp(min=0)  # [N,M,2]
    # area = wh[:, :, 0] * wh[:, :, 1]

    # return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


def tube_iou(tube1: Dict[str, torch.Tensor], tube2: Dict[str, torch.Tensor], label_centric: bool = False) -> int:
    """Calculate tIoU (iou3d)

    Args:
        tube1 (Dict[str, torch.Tensor]): The key is the frame index of the tube, the value is bbox ([x1,y1,x2,y2])
        tube2 (Dict[str, torch.Tensor]):
        label_centric (bool): If True, calculate iou on labeled frames (tube2) only

    Returns:
        int: tIoU
    """
    tube_iou = 0
    tube1_frame_idx = list(tube1.keys())
    tube2_frame_idx = list(tube2.keys())
    if label_centric:
        frame_idx = tube2_frame_idx
    else:
        frame_idx = tube1_frame_idx + tube2_frame_idx
        frame_idx = set(frame_idx)

    for i in frame_idx:
        if i in tube1 and i in tube2:
            frame_iou, _ = box_iou(tube1[i].reshape(-1, 4), tube2[i].reshape(-1, 4))
        else:
            frame_iou = 0
        tube_iou += frame_iou
    tube_iou /= len(frame_idx)

    return tube_iou
