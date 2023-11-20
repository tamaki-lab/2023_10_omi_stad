"""
Plotting utilities to visualize training logs.
"""
import torch
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import av
from pathlib import Path

from .box_ops import box_unnormalize, box_cxcywh_to_xyxy


def plot_pred_clip_boxes(clip_sample, results, labels, plot_label=False, th=0.75):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    filter_labels = [result["labels"][(result["scores"] > th).nonzero().flatten()] for result in results]
    filter_boxes = [result["boxes"][(result["scores"] > th).nonzero().flatten()] for result in results]
    filter_scores = [result["scores"][(result["scores"] > th).nonzero().flatten()] for result in results]

    frame_list = []
    for t in range(len(results)):
        img = clip_sample[t].permute(1, 2, 0).cpu()
        img = img * std + mean
        img = img.numpy()

        img = np.clip(img * 255, a_min=0, a_max=255).astype(np.uint8).copy()
        for i, (box, label, score) in enumerate(zip(filter_boxes[t], filter_labels[t], filter_scores[t])):
            if label != 1:
                continue
            x1, y1, x2, y2 = box.unbind()
            cv2.rectangle(
                img,
                pt1=(int(x1.item()), int(y1.item())),
                pt2=(int(x2.item()), int(y2.item())),
                color=(0, 255, 0),
                thickness=2,
                lineType=cv2.LINE_4,
                shift=0,
            )
            cv2.putText(img,
                        text=f"{round(score.item(),2)}",
                        org=(int(x1.item()), int(y1.item())),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(255, 0, 0),
                        thickness=1,
                        lineType=cv2.LINE_4)

        if plot_label:
            labels[t]["boxes"] = box_unnormalize(labels[t]["boxes"].cpu(), labels[t]["size"])
            labels[t]["boxes"] = box_cxcywh_to_xyxy(labels[t]["boxes"])
            for i, box in enumerate(labels[t]["boxes"]):
                x1, y1, x2, y2 = box.unbind()
                cv2.rectangle(
                    img,
                    pt1=(int(x1.item()), int(y1.item())),
                    pt2=(int(x2.item()), int(y2.item())),
                    color=(0, 0, 0),
                    thickness=2,
                    lineType=cv2.LINE_4,
                    shift=0,
                )

        frame_list.append(img)

    rows = 1
    cols = 8
    frame_id = 0
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False, tight_layout=True)
    for i in range(rows):
        for j in range(cols):
            img = frame_list[frame_id]
            subplot_title = "frame:" + str(frame_id)
            axes[i, j].set_title(subplot_title)
            axes[i, j].imshow(img)
            frame_id = frame_id + 1

    plt.imshow(img)
    os.makedirs("test_img", exist_ok=True)
    plt.savefig("test_img/test.png")
    plt.close()


def plot_diff_results(clip_sample, results1, results2, labels, plot_label=False, th=0.75):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    filter_labels1 = [result["labels"][(result["scores"] > th).nonzero().flatten()] for result in results1]
    filter_boxes1 = [result["boxes"][(result["scores"] > th).nonzero().flatten()] for result in results1]
    filter_scores1 = [result["scores"][(result["scores"] > th).nonzero().flatten()] for result in results1]

    filter_labels2 = [result["labels"][(result["scores"] > th).nonzero().flatten()] for result in results2]
    filter_boxes2 = [result["boxes"][(result["scores"] > th).nonzero().flatten()] for result in results2]
    filter_scores2 = [result["scores"][(result["scores"] > th).nonzero().flatten()] for result in results2]

    frame_list = []
    for t in range(len(results1)):
        img = clip_sample[t].permute(1, 2, 0).cpu()
        img = img * std + mean
        img = img.numpy()

        img = np.clip(img * 255, a_min=0, a_max=255).astype(np.uint8).copy()
        for i, (box, label, score) in enumerate(zip(filter_boxes1[t], filter_labels1[t], filter_scores1[t])):
            if label != 1:
                continue
            x1, y1, x2, y2 = box.unbind()
            cv2.rectangle(
                img,
                pt1=(int(x1.item()), int(y1.item())),
                pt2=(int(x2.item()), int(y2.item())),
                color=(255, 0, 0),
                thickness=2,
                lineType=cv2.LINE_4,
                shift=0,
            )
            cv2.putText(img,
                        text=f"{round(score.item(),2)}",
                        org=(int(x1.item()), int(y1.item())),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(255, 0, 0),
                        thickness=1,
                        lineType=cv2.LINE_4)

        for i, (box, label, score) in enumerate(zip(filter_boxes2[t], filter_labels2[t], filter_scores2[t])):
            if label != 1:
                continue
            x1, y1, x2, y2 = box.unbind()
            cv2.rectangle(
                img,
                pt1=(int(x1.item()), int(y1.item())),
                pt2=(int(x2.item()), int(y2.item())),
                color=(0, 255, 0),
                thickness=2,
                lineType=cv2.LINE_4,
                shift=0,
            )
            cv2.putText(img,
                        text=f"{round(score.item(),2)}",
                        org=(int(x1.item()), int(y1.item())),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 255, 0),
                        thickness=1,
                        lineType=cv2.LINE_4)

        if plot_label:
            labels[t]["boxes"] = box_unnormalize(labels[t]["boxes"].cpu(), labels[t]["size"])
            labels[t]["boxes"] = box_cxcywh_to_xyxy(labels[t]["boxes"])
            for i, box in enumerate(labels[t]["boxes"]):
                x1, y1, x2, y2 = box.unbind()
                cv2.rectangle(
                    img,
                    pt1=(int(x1.item()), int(y1.item())),
                    pt2=(int(x2.item()), int(y2.item())),
                    color=(0, 0, 0),
                    thickness=2,
                    lineType=cv2.LINE_4,
                    shift=0,
                )

        frame_list.append(img)

    rows = 1
    cols = 8
    frame_id = 0
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False, tight_layout=True)
    for i in range(rows):
        for j in range(cols):
            img = frame_list[frame_id]
            subplot_title = "frame:" + str(frame_id)
            axes[i, j].set_title(subplot_title)
            axes[i, j].imshow(img)
            frame_id = frame_id + 1

    plt.imshow(img)
    os.makedirs("test_img", exist_ok=True)
    plt.savefig("test_img/test.png")
    plt.close()


def make_video_with_tube(video_path, label_list, tubes, video_ano, plot_label=True):
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
    resize_scale = max([width, height]) / 512

    new_container = av.open("test.avi", mode="w")
    new_stream = new_container.add_stream(codec, rate=base_rate)
    new_stream.width = width
    new_stream.height = height
    new_stream.pix_fmt = pix_fmt

    for frame_idx, frame in enumerate(container.decode(video=0)):
        frame = frame.to_ndarray(format="rgb24")

        for list_idx, tube in enumerate(tubes):
            for idx, idx_of_query in enumerate(tube["idx_of_p_queries"]):
                if frame_idx > idx_of_query[0]:
                    continue
                elif frame_idx < idx_of_query[0]:
                    break
                else:
                    box = tube["bbox"][idx]
                    x1, y1, x2, y2 = box
                    x1 = int(max(min(x1, width), 0))
                    x2 = int(max(min(x2, width), 0))
                    y1 = int(max(min(y1, height), 0))
                    y2 = int(max(min(y2, height), 0))
                    action_id = tube["action_label"][idx]

                    cv2.rectangle(
                        frame, pt1=(x1, y1), pt2=(x2, y2),
                        color=color_map[list_idx % 10], thickness=2, lineType=cv2.LINE_4, shift=0,
                    )
                    cv2.putText(
                        frame, text=f"label: {label_list[action_id]}, idx: {list_idx}", org=(x1, y1),
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
                    frame, text=f"{label_list[action_id]}, {tube_idx}", org=(x1, y1),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3,
                    color=(0, 0, 0), thickness=1, lineType=cv2.LINE_4
                )

        frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in new_stream.encode(frame):
            new_container.mux(packet)

    for packet in new_stream.encode():
        new_container.mux(packet)
    new_container.close()


def make_video_with_actiontube(video_path, label_list, tubes, video_ano, plot_label=True):
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
    resize_scale = max([width, height]) / 512

    new_container = av.open("test.avi", mode="w")
    new_stream = new_container.add_stream(codec, rate=base_rate)
    new_stream.width = width
    new_stream.height = height
    new_stream.pix_fmt = pix_fmt

    for frame_idx, frame in enumerate(container.decode(video=0)):
        frame = frame.to_ndarray(format="rgb24")

        for list_idx, (name, tube) in enumerate(tubes):
            if frame_idx in tube["boxes"]:
                box = tube["boxes"][frame_idx]
                x1, y1, x2, y2 = box
                # x1, y1, x2, y2 = box * resize_scale
                x1 = int(max(min(x1, width), 0))
                x2 = int(max(min(x2, width), 0))
                y1 = int(max(min(y1, height), 0))
                y2 = int(max(min(y2, height), 0))
                action_id = tube["class"]
                score = tube["score"]

                cv2.rectangle(
                    frame, pt1=(x1, y1), pt2=(x2, y2),
                    color=color_map[list_idx % 10], thickness=2, lineType=cv2.LINE_4, shift=0,
                )
                cv2.putText(
                    frame, text=f"{label_list[action_id]}, {round(score,2)}, idx:{list_idx}",
                    # frame, text=f"{label_list[action_id]}, score: {round(score,2)}, tube_idx:{list_idx}",
                    org=(x1, y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3,
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
                    frame, text=f"{label_list[action_id]}", org=(x1+5, y1+10),
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
