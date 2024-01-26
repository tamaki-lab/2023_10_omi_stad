
import argparse
import random
from tqdm import tqdm
import os.path as osp
import torch
import numpy as np
import yaml

from datasets.dataset import get_video_loader
import util.misc as utils
from models import build_model
from models.person_encoder import PersonEncoder
from models.action_head import ActionHead, ActionHead2, X3D_XS
from util.gt_tubes import make_gt_tubes
from util.video_map import calc_video_map, calc_motion_ap
from util.plot_utils import make_video_with_actiontube, make_video_with_action_pred
from datasets.dataset import VideoDataset


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    # metric
    parser.add_argument('--metric', default='v-mAP', type=str, choices=['v-mAP', 'motion-AP'])

    # loader
    parser.add_argument('--dataset', default='jhmdb21', type=str, choices=['ucf101-24', 'jhmdb21'])
    parser.add_argument('--n_frames', default=128, type=int)
    parser.add_argument('--subset', default="val", type=str, choices=["train", "val"])
    parser.add_argument('--link_cues', default='feature', type=str)

    # setting
    parser.add_argument('--qmm_name', default='noskip_sr:4', type=str)
    parser.add_argument('--head_type', default='vanilla', type=str, choices=["vanilla", "time_ecd:add", "time_ecd:cat", "res", "x3d"])
    parser.add_argument('--head_name', default='vanilla', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--load_epoch_qmm', default=20, type=int)
    parser.add_argument('--load_epoch_head', default=20, type=int)
    parser.add_argument('--psn_score_th', default=0.9, type=float)
    parser.add_argument('--sim_th', default=0.5, type=float)
    parser.add_argument('--tiou_th', default=0.2, type=float)
    parser.add_argument('--filter_length', default=8, type=int)
    # parser.add_argument('--iou_th', default=0.3, type=float)
    parser.add_argument('--topk', default=1, type=int)

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
def main(args, params):
    device = torch.device(f"cuda:{args.device}")
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    detr, _, _ = build_model(args)
    detr.to(device)
    detr.eval()
    if args.dataset == "ucf101-24":
        pretrain_path = "checkpoint/ucf101-24/w:252/detr/epoch_20.pth"
        detr.load_state_dict(torch.load(pretrain_path))
    else:
        pretrain_path = "checkpoint/detr/" + utils.get_pretrain_path(args.backbone, args.dilation)
        detr.load_state_dict(torch.load(pretrain_path)["model"])

    psn_encoder = PersonEncoder().to(device)
    psn_encoder.eval()
    pretrain_path_encoder = osp.join(args.check_dir, args.dataset, args.qmm_name, "encoder", f"epoch_{args.load_epoch_qmm}.pth")
    psn_encoder.load_state_dict(torch.load(pretrain_path_encoder))

    if args.head_type == "vanilla":
        action_head = ActionHead(n_classes=args.n_classes, pos_ecd=(False, "", None)).to(device)
    elif args.head_type == "time_ecd:add":
        action_head = ActionHead(n_classes=args.n_classes, pos_ecd=(True, "add", None)).to(device)
    elif args.head_type == "time_ecd:cat":
        action_head = ActionHead(n_classes=args.n_classes, pos_ecd=(True, "cat", 32)).to(device)
    else:
        action_head = ActionHead2(n_classes=args.n_classes, pos_ecd=(True, "cat", 32)).to(device)
    action_head.eval()

    if args.link_cues == "feature":
        pretrain_path_head = osp.join(args.check_dir, args.dataset, args.qmm_name, "head", args.head_name, f"epoch_{args.load_epoch_head}.pth")
    else:
        pretrain_path_head = osp.join(args.check_dir, args.dataset, "iou_link", "head", args.head_name, f"epoch_{args.load_epoch_head}.pth")
    action_head.load_state_dict(torch.load(pretrain_path_head))

    loader = get_video_loader(args.dataset, args.subset, shuffle=False)
    dir = osp.join(args.check_dir, args.dataset, args.qmm_name, "qmm_tubes")
    filename = f"videotubes-epoch:{args.load_epoch_qmm}_pth:{args.psn_score_th}_simth:{args.sim_th}_fl:{args.filter_length}"
    loader = utils.TarIterator(dir + "/" + args.subset, filename)

    dataset = VideoDataset(args.dataset, args.subset)
    x3d_xs = X3D_XS().to(device)
    x3d_xs.eval()

    pred_tubes = []
    video_names = set()

    pbar_tubes = tqdm(enumerate(loader), total=len(loader), leave=False)
    pbar_vtubes = tqdm(enumerate(loader), total=len(loader), leave=False)
    pbar_tubes.set_description("[Validation]")
    for video_idx, tubes in pbar_vtubes:
        pred_v_tubes = []

        v_list = ["v_Basketball_g01_c02",
                  "v_Basketball_g01_c04",
                  "v_Basketball_g01_c05",
                  "v_Basketball_g01_c06",
                  "v_Basketball_g01_c07",
                  "v_Basketball_g02_c06",
                  "v_Basketball_g03_c01",
                  "v_Basketball_g03_c05",
                  "v_Basketball_g04_c01",
                  "v_Basketball_g04_c02",
                  "v_Basketball_g04_c03",
                  "v_Basketball_g05_c03",
                  "v_Basketball_g07_c01"]
        v_list = ["v_Basketball_g07_c01"]
        if tubes.video_name.split("/")[1] not in v_list:
            continue

        for tube in tubes.tubes:
            video_names.add(tube.video_name)
            decoded_queries = torch.stack(tube.decoded_queries).to(device)
            frame_indices = [x[0] for x in tube.query_indicies]
            frame_indices = [x - frame_indices[0] for x in frame_indices]

            if args.head_type == "vanilla" or args.head_type == "time_ecd:add":
                outputs = action_head(decoded_queries)
            elif args.head_type == "time_ecd:cat":
                outputs = action_head(decoded_queries, frame_indices)
            else:
                if args.head_type == "res":
                    frame_features = utils.get_frame_features(detr.backbone, tube.video_name, frame_indices, dataset, device, True)
                elif args.head_type == "x3d":
                    frame_features = utils.get_frame_features(x3d_xs, tube.video_name, frame_indices, dataset, device, True)
                outputs = action_head(frame_features, decoded_queries, frame_indices)

            tube.log_pred(outputs, args.topk)

            action_tubes = tube.split_by_action()
            pred_v_tubes.extend(action_tubes)
        pred_tubes.extend(pred_v_tubes)

        video_path = osp.join(params["dataset_path_video"], tubes.video_name + ".avi")
        make_video_with_action_pred(video_path, tubes, params["label_list"], tubes.ano, False)
        make_video_with_actiontube(video_path, params["label_list"], pred_v_tubes, tubes.ano, plot_label=True)
        continue

    print(f"num of pred tubes: {len(pred_tubes)}")
    pred_tubes = [tube for tube in pred_tubes if tube[1]["class"] != params["num_classes"]]
    print(f"num of pred tubes w/o no action: {len(pred_tubes)}")

    pred_tubes = [tube for tube in pred_tubes if len(tube[1]["boxes"]) > 8]
    print(f"num of pred tubes (after filtering): {len(pred_tubes)}")

    gt_tubes = make_gt_tubes(args.dataset, args.subset, params)
    video_names = list(video_names)
    gt_tubes = {name: tube for name, tube in gt_tubes.items()}
    # gt_tubes = {name: tube for name, tube in gt_tubes.items() if name in video_names}   # for debug with less data from loader

    if args.metric == "v-mAP":
        video_ap = calc_video_map(pred_tubes, gt_tubes, params["num_classes"], args.tiou_th)
        for class_name, ap in zip(params["label_list"][:-1], video_ap):
            print(f"{class_name}: {round(ap,4)}")
        print(f"v-mAP: {round(sum(video_ap) / len(video_ap),4)}")
    elif args.metric == "motion-AP":
        calc_motion_ap(pred_tubes, gt_tubes, args.tiou_th)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Tube evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    params = yaml.safe_load(open(f"datasets/projects/{args.dataset}.yml"))
    params["label_list"].append("no action")
    args.n_classes = len(params["label_list"])

    main(args, params)
