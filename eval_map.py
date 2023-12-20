
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
from util.video_map import calc_video_map
from util.plot_utils import make_video_with_actiontube, make_video_with_action_pred
from datasets.dataset import VideoDataset


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    # loader
    parser.add_argument('--dataset', default='jhmdb21', type=str, choices=['ucf101-24', 'jhmdb21'])
    parser.add_argument('--n_frames', default=128, type=int)
    parser.add_argument('--subset', default="val", type=str, choices=["train", "val"])

    # setting
    parser.add_argument('--qmm_name', default='jhmdb_wd:e4', type=str)
    parser.add_argument('--head_name', default='head_test', type=str)
    parser.add_argument('--device', default=1, type=int)
    parser.add_argument('--load_epoch_encoder', default=15, type=int)
    parser.add_argument('--load_epoch_head', default=20, type=int)
    parser.add_argument('--psn_score_th', default=0.8, type=float)
    parser.add_argument('--sim_th', default=0.5, type=float)
    parser.add_argument('--tiou_th', default=0.2, type=float)
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
    pretrain_path_encoder = osp.join(args.check_dir, args.dataset, args.qmm_name, "encoder", f"epoch_{args.load_epoch_encoder}.pth")
    psn_encoder.load_state_dict(torch.load(pretrain_path_encoder))

    # action_head = ActionHead(n_classes=args.n_classes).to(device)
    action_head = ActionHead2(n_classes=args.n_classes).to(device)
    action_head.eval()
    pretrain_path_head = osp.join(args.check_dir, args.dataset, args.qmm_name, "head", args.head_name, f"epoch_{args.load_epoch_head}.pth")
    action_head.load_state_dict(torch.load(pretrain_path_head))

    loader = get_video_loader(args.dataset, args.subset, shuffle=False)
    dir = osp.join(args.check_dir, args.dataset, args.qmm_name, "qmm_tubes")
    filename = f"videotubes-epoch:{args.load_epoch_encoder}_pth:{args.psn_score_th}_simth:{args.sim_th}"
    # filename = f"tube-epoch:{args.load_epoch_encoder}_pth:{args.psn_score_th}_simth:{args.sim_th}"
    loader = utils.TarIterator(dir + "/" + args.subset, filename)

    val_dataset = VideoDataset(args.dataset, args.subset)
    x3d_xs = X3D_XS().to(device)
    x3d_xs.eval()

    pred_tubes = []
    video_names = set()

    pbar_tubes = tqdm(enumerate(loader), total=len(loader), leave=False)
    pbar_vtubes = tqdm(enumerate(loader), total=len(loader), leave=False)
    pbar_tubes.set_description("[Validation]")
    # for tube_idx, tube in pbar_tubes:
    for video_idx, tubes in pbar_vtubes:
        pred_v_tubes = []
        video_path = osp.join(params["dataset_path_video"], tubes.video_name + ".avi")

        for tube in tubes.tubes:
            # action_label = torch.Tensor(tube.action_label).to(torch.int64).to(device)
            video_names.add(tube.video_name)
            decoded_queries = torch.stack(tube.decoded_queries).to(device)
            frame_indices = [x[0] for x in tube.query_indicies]
            frame_indices = [x - frame_indices[0] for x in frame_indices]

            frame_features = utils.get_frame_features(x3d_xs, tube.video_name, frame_indices, val_dataset, device)
            # frame_features = utils.get_frame_features(detr.backbone, tube.video_name, frame_indices, val_dataset, device)

            # outputs = action_head(decoded_queries)
            # outputs = action_head(decoded_queries, frame_indices)
            outputs = action_head(frame_features, decoded_queries, frame_indices)

            tube.log_pred(outputs, args.topk)

            action_tubes = tube.split_by_action()
            pred_v_tubes.extend(action_tubes)
        pred_tubes.extend(pred_v_tubes)

        # make_video_with_action_pred(video_path, tubes, params["label_list"], tubes.ano)
        # make_video_with_actiontube(video_path, params["label_list"], pred_v_tubes, tubes.ano, plot_label=True)

    print(f"num of pred tubes: {len(pred_tubes)}")
    pred_tubes = [tube for tube in pred_tubes if tube[1]["class"] != params["num_classes"]]
    print(f"num of pred tubes w/o no action: {len(pred_tubes)}")
    # pred_tubes = [tube for tube in pred_tubes if len(tube[1]["boxes"]) > 16]
    # print(f"num of pred tubes (after filtering): {len(pred_tubes)}")


    gt_tubes = make_gt_tubes(args.dataset, args.subset, params)
    video_names = list(video_names)
    gt_tubes = {name: tube for name, tube in gt_tubes.items() if name in video_names}   # for debug with less data from loader

    video_ap = calc_video_map(pred_tubes, gt_tubes, params["num_classes"], args.tiou_th)
    for class_name, ap in zip(params["label_list"][:-1], video_ap):
        print(f"{class_name}: {round(ap,4)}")
    print(f"v-mAP: {round(sum(video_ap) / len(video_ap),4)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Tube evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    params = yaml.safe_load(open(f"datasets/projects/{args.dataset}.yml"))
    params["label_list"].append("no action")
    args.n_classes = len(params["label_list"])

    main(args, params)
