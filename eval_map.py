
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
from models.action_head import ActionHead
from util.gt_tubes import make_gt_tubes
from util.video_map import calc_video_map


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    # loader
    parser.add_argument('--dataset', default='jhmdb21', type=str, choices=['ucf101-24', 'jhmdb21'])
    parser.add_argument('--n_frames', default=128, type=int)
    parser.add_argument('--subset', default="val", type=str, choices=["train", "val"])

    # setting
    parser.add_argument('--load_ex_name', default='jhmdb_wd:e4', type=str)
    parser.add_argument('--write_ex_name', default='head_test', type=str)
    parser.add_argument('--device', default=1, type=int)
    parser.add_argument('--load_epoch_encoder', default=15, type=int)
    parser.add_argument('--load_epoch_head', default=20, type=int)
    parser.add_argument('--psn_score_th', default=0.8, type=float)
    parser.add_argument('--sim_th', default=0.5, type=float)
    parser.add_argument('--tiou_th', default=0.2, type=float)
    parser.add_argument('--iou_th', default=0.3, type=float)

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
    pretrain_path_detr = "checkpoint/detr/" + utils.get_pretrain_path(args.backbone, args.dilation)
    detr.load_state_dict(torch.load(pretrain_path_detr)["model"])

    psn_encoder = PersonEncoder().to(device)
    psn_encoder.eval()
    pretrain_path_encoder = osp.join(args.check_dir, args.dataset, args.load_ex_name, "encoder", f"epoch_{args.load_epoch_encoder}.pth")
    psn_encoder.load_state_dict(torch.load(pretrain_path_encoder))

    action_head = ActionHead(n_classes=args.n_classes).to(device)
    action_head.eval()
    pretrain_path_head = osp.join(args.check_dir, args.dataset, args.load_ex_name, "head", args.write_ex_name, f"epoch_{args.load_epoch_head}.pth")
    action_head.load_state_dict(torch.load(pretrain_path_head))

    loader = get_video_loader(args.dataset, args.subset, shuffle=False)
    dir = osp.join(args.check_dir, args.dataset, args.load_ex_name, "qmm_tubes")
    filename = f"tube-epoch:{args.load_epoch_encoder}_pth:{args.psn_score_th}_simth:{args.sim_th}"
    loader = utils.TarIterator(dir + "/" + args.subset, filename)

    pred_tubes = []
    video_names = set()
    total_tubes = 0

    pbar_tubes = tqdm(enumerate(loader), total=len(loader), leave=False)
    pbar_tubes.set_description("[Validation]")
    for tube_idx, tube in pbar_tubes:
        # action_label = torch.Tensor(tube.action_label).to(torch.int64).to(device)
        video_names.add(tube.video_name)
        decoded_queries = torch.stack(tube.decoded_queries).to(device)
        frame_indices = [x[0] for x in tube.query_indicies]
        frame_indices = [x - frame_indices[0] for x in frame_indices]

        # outputs = action_head(decoded_queries)
        outputs = action_head(decoded_queries, frame_indices)
        tube.log_pred(outputs)

        action_tubes = tube.split_by_action()
        pred_tubes.extend(action_tubes)
        total_tubes += len(action_tubes)
        pbar_tubes.set_postfix_str(f'total_tubes: {total_tubes}, n_tubes: {len(action_tubes)}')

        # video_path = osp.join(params["dataset_path_video"], tube.video_name + ".avi")
        # make_video_with_actiontube(video_path, )

    pred_tubes = [tube for tube in pred_tubes]
    # pred_tubes = [tube for video_tubes in pred_tubes for tube in video_tubes.tubes]
    print(f"num of pred tubes: {len(pred_tubes)}")
    pred_tubes = [tube for tube in pred_tubes if tube[1]["class"] != params["num_classes"]]
    print(f"num of pred tubes w/o no action: {len(pred_tubes)}")

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
