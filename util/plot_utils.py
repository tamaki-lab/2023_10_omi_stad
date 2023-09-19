"""
Plotting utilities to visualize training logs.
"""
import torch
import pandas as pd
import cv2
import numpy as np
import os
# import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path, PurePath

from .box_ops import box_unnormalize, box_cxcywh_to_xyxy


def plot_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0, log_name='log.txt'):
    '''
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
              - fields = which results to plot from each log file - plots both training and test for each field.
              - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
              - log_name = optional, name of log file if different than default 'log.txt'.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

    '''
    func_name = "plot_utils.py::plot_logs"

    # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # convert single Path to list to avoid 'not iterable' error

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(f"{func_name} info: logs param expects a list argument, converted to list[Path].")
        else:
            raise ValueError(f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}")

    # Quality checks - verify valid dir(s), that every item in list is Path object, and that log_name exists in each dir
    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}")
        if not dir.exists():
            raise ValueError(f"{func_name} - invalid directory in logs argument:\n{dir}")
        # verify log_name exists
        fn = Path(dir / log_name)
        if not fn.exists():
            print(f"-> missing {log_name}.  Have you gotten to Epoch 1 in training?")
            print(f"--> full path of missing log file: {fn}")
            return

    # load log file(s) and plot
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]

    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            if field == 'mAP':
                coco_eval = pd.DataFrame(
                    np.stack(df.test_coco_eval_bbox.dropna().values)[:, 1]
                ).ewm(com=ewm_col).mean()
                axs[j].plot(coco_eval, c=color)
            else:
                df.interpolate().ewm(com=ewm_col).mean().plot(
                    y=[f'train_{field}', f'test_{field}'],
                    ax=axs[j],
                    color=[color] * 2,
                    style=['-', '--']
                )
    for ax, field in zip(axs, fields):
        ax.legend([Path(p).name for p in logs])
        ax.set_title(field)


def plot_precision_recall(files, naming_scheme='iter'):
    if naming_scheme == 'exp_id':
        # name becomes exp_id
        names = [f.parts[-3] for f in files]
    elif naming_scheme == 'iter':
        names = [f.stem for f in files]
    else:
        raise ValueError(f'not supported {naming_scheme}')
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    for f, color, name in zip(files, sns.color_palette("Blues", n_colors=len(files)), names):
        data = torch.load(f)
        # precision is n_iou, n_points, n_cat, n_area, max_det
        precision = data['precision']
        recall = data['params'].recThrs
        scores = data['scores']
        # take precision for all classes, all areas and 100 detections
        precision = precision[0, :, :, 0, -1].mean(1)
        scores = scores[0, :, :, 0, -1].mean(1)
        prec = precision.mean()
        rec = data['recall'][0, :, 0, -1].mean()
        print(f'{naming_scheme} {name}: mAP@50={prec * 100: 05.1f}, ' +
              f'score={scores.mean():0.3f}, ' +
              f'f1={2 * prec * rec / (prec + rec + 1e-8):0.3f}'
              )
        axs[0].plot(recall, precision, c=color)
        axs[1].plot(recall, scores, c=color)

    axs[0].set_title('Precision / Recall')
    axs[0].legend(names)
    axs[1].set_title('Scores / Recall')
    axs[1].legend(names)
    return fig, axs


def plot_frame_boxes(sample, result):
    score_filter_indices = (result["scores"] > 0.95).nonzero().flatten()
    filter_boxes = result["boxes"][score_filter_indices]
    img = sample.permute(1, 2, 0).cpu()
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    img = img * std + mean
    img = img.numpy()

    img = np.clip(img * 255, a_min=0, a_max=255).astype(np.uint8).copy()
    for i, box in enumerate(filter_boxes):
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
                    text=str(score_filter_indices[i].item()),
                    org=(int(x1.item()), int(y1.item())),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 0, 0),
                    thickness=1,
                    lineType=cv2.LINE_4)
    plt.imshow(img)
    plt.savefig("test.png")


def plot_label_clip_boxes(clip_sample, labels):
    frame_list = []
    for t in range(len(labels)):
        img = clip_sample[t].permute(1, 2, 0).cpu()
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        img = img * std + mean
        img = img.numpy()

        img = np.clip(img * 255, a_min=0, a_max=255).astype(np.uint8).copy()
        labels[t]["boxes"] = box_unnormalize(labels[t]["boxes"].cpu(), labels[t]["size"])
        labels[t]["boxes"] = box_cxcywh_to_xyxy(labels[t]["boxes"])
        for i, box in enumerate(labels[t]["boxes"]):
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
    plt.savefig("test_img/test_0.png")
    plt.close()


def plot_pred_clip_boxes(clip_sample, results, th=0.85):
    score_filter_indices = [(result["scores"] > th).nonzero().flatten() for result in results]
    filter_labels = [result["labels"][(result["scores"] > th).nonzero().flatten()] for result in results]
    filter_boxes = [result["boxes"][indices] for result, indices in zip(results, score_filter_indices)]
    # filter_boxes_label = [(result["boxes"][indices], result["labels"][indices]) for result, indices in zip(results, score_filter_indices)]
    frame_list = []
    for t in range(len(results)):
        img = clip_sample[t].permute(1, 2, 0).cpu()
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        img = img * std + mean
        img = img.numpy()

        img = np.clip(img * 255, a_min=0, a_max=255).astype(np.uint8).copy()
        for i, (box, label) in enumerate(zip(filter_boxes[t], filter_labels[t])):
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
                        text=str(score_filter_indices[t][i].item()),
                        org=(int(x1.item()), int(y1.item())),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(255, 0, 0),
                        thickness=1,
                        lineType=cv2.LINE_4)
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
    plt.savefig("test_img/test_1.png")
    plt.close()


def plot_pred_person_link(clip_sample, results, same_psn_idx_lists):
    id2coloer = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 255, 255), 3: (255, 255, 255)}

    filter_boxes_lists = []
    for i, psn_list in enumerate(same_psn_idx_lists):
        filter_boxes_lists.append({})
        for frame_idx, origin_query_idx in psn_list.items():
            filter_boxes_lists[i][frame_idx] = results[frame_idx]["boxes"][origin_query_idx]

    frame_list = []
    for t in range(len(results)):
        img = clip_sample[t].permute(1, 2, 0).cpu()
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        img = img * std + mean
        img = img.numpy()

        img = np.clip(img * 255, a_min=0, a_max=255).astype(np.uint8).copy()
        for list_idx, filter_boxes in enumerate(filter_boxes_lists):
            if t not in filter_boxes:
                continue
            x1, y1, x2, y2 = filter_boxes[t].unbind()
            cv2.rectangle(
                img,
                pt1=(int(x1.item()), int(y1.item())),
                pt2=(int(x2.item()), int(y2.item())),
                color=id2coloer[list_idx],
                thickness=2,
                lineType=cv2.LINE_4,
                shift=0,
            )
            cv2.putText(img,
                        text=str(list_idx) + ",  " + str(same_psn_idx_lists[list_idx][t]),
                        org=(int(x1.item()), int(y1.item())),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(255, 0, 0),
                        thickness=1,
                        lineType=cv2.LINE_4)
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
    plt.savefig("test_img/test_2.png")
    plt.close()
