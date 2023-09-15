import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import (
    miners,
    losses,
    reducers,
    distances,
)
import statistics
import numpy as np

from .detr import MLP


class PersonEncoder(nn.Module):
    def __init__(self, in_d=256, out_d=256):
        super().__init__()
        # self.psn_query_embed = nn.Linear(in_d, out_d)
        # self.psn_query_embed = MLP(in_d, out_d, out_d, 5)
        self.psn_query_embed = MLPDrop(in_d, out_d, out_d, 3)

    def forward(self, p_queries):
        p_queries = self.psn_query_embed(p_queries)
        return p_queries


class MLPDrop(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.dropouts = nn.ModuleList(nn.Dropout(dropout) for _ in range(num_layers - 1))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.dropouts[i](F.relu(layer(x))) if i < self.num_layers - 1 else layer(x)
            # x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SetInfoNce(nn.Module):
    def __init__(self):
        super().__init__()
        distance = distances.CosineSimilarity()
        reducer = reducers.MeanReducer()
        self.loss_func = losses.NTXentLoss(temperature=0.07, distance=distance, reducer=reducer)
        self.mining_func = miners.BatchEasyHardMiner(
            pos_strategy=miners.BatchEasyHardMiner.ALL,
            neg_strategy=miners.BatchEasyHardMiner.ALL,
            distance=distance
        )

    def make_posneg_label(self, indices_ex, bs, n_frames) -> torch.Tensor:
        """ Make positibe/negatibe label for metric learning
        同一人物が同じインデックスとなるラベルを返す

        Args:
            indices_ex:
                A list of size batch_size (bs*n_frames), containing tuples of (index_i, index_j, index_k) where:
                    - index_i is the indices of the selected predictions (in order)
                    - index_j is the indices of the corresponding selected targets (in order)
                    - index_k is the indices of person_id (ただし1動画内で割り当てられたものであり,異なる動画における同じidは意味をなさない)
            bs (int): batch size
            n_frames (int): n_frames in clip

        Returns:
            (torch.Tensor):A tensor size is
        """
        n_label_in_clip = []    # n_label_in_clip[clip_idx] = n_label_in_clip
        n_people_in_clip = []    # n_label_in_clip[clip_idx] = n_people_in_clip
        same_psn_label = torch.Tensor([])

        for i in range(bs):
            clip_indices_ex = indices_ex[i * n_frames:(i + 1) * n_frames]
            same_psn_label_in_clip = torch.cat([psn_id for _, _, psn_id in clip_indices_ex])
            n_label_in_clip.append(same_psn_label_in_clip.size(0))
            n_people_in_clip.append(torch.unique(same_psn_label_in_clip).size(0))

            sorted_list = sorted(torch.unique(same_psn_label_in_clip))
            new_same_psn_label_in_clip = [sorted_list.index(x) + sum(n_people_in_clip[:-1]) for x in same_psn_label_in_clip]
            same_psn_label = torch.cat((same_psn_label, torch.Tensor(new_same_psn_label_in_clip)))
        return same_psn_label.to(torch.int64)

    def forward(self, p_f_queries, indices_ex, bs, n_frames):
        label = self.make_posneg_label(indices_ex, bs, n_frames)
        miner_out = self.mining_func(p_f_queries, label)
        loss = self.loss_func(p_f_queries, indices_tuple=miner_out)
        return loss, label


def make_same_person_list(p_f_queries, same_person_label, n_gt_bbox_list, bs, n_frames):
    split_p_f_queries = [[0] * n_frames for _ in range(bs)]  # [clip_idx][frame_idx] = torch.Tensor
    split_same_person_label = [[None] * n_frames for _ in range(bs)]
    split_idx_list = [[None] * n_frames for _ in range(bs)]
    total = 0
    for i, n_gt_boxes in enumerate(n_gt_bbox_list):
        split_idx_list[i // bs][i % n_frames] = total
        split_p_f_queries[i // bs][i % n_frames] = p_f_queries[total:total + n_gt_boxes]
        split_same_person_label[i // bs][i % n_frames] = same_person_label[total:total + n_gt_boxes]
        total += n_gt_boxes
    tauple_list = [make_same_person_list_in_clip(clip_p_f_queries, clip_same_person_label, clip_split_idx) for clip_p_f_queries, clip_same_person_label, clip_split_idx in zip(split_p_f_queries, split_same_person_label, split_idx_list)]
    scores_list = [tauple[0] for tauple in tauple_list]
    same_person_lists_clip = [tauple[1] for tauple in tauple_list]

    scores_avg = {}
    scores_avg["diff_psn_score"] = statistics.mean([scores[0] for scores in scores_list if scores is not None])
    scores_avg["same_psn_score"] = statistics.mean([scores[1] for scores in scores_list if scores is not None])
    scores_avg["total_psn_score"] = statistics.mean([scores[2] for scores in scores_list if scores is not None])

    return scores_avg, same_person_lists_clip


def make_same_person_list_in_clip(clip_p_f_queries, clip_same_person_label, clip_split_idx):
    same_person_lists = []

    for i, (frame_p_f_queries, frame_same_person_label, frame_split_idx) in enumerate(zip(clip_p_f_queries, clip_same_person_label, clip_split_idx)):
        sim_score = calc_sim(same_person_lists, frame_p_f_queries, frame_same_person_label, frame_split_idx)

    scores = val_same_person_lists(same_person_lists)

    return scores, same_person_lists


def calc_sim(same_person_lists, frame_p_f_queries, frame_same_person_label, frame_split_idx, th=0.30):
    if len(same_person_lists) == 0:
        for i, (p_f_query, target_psn_id) in enumerate(zip(frame_p_f_queries, frame_same_person_label)):
            same_person_lists.append({"query": [p_f_query], "target_id": [target_psn_id.item()], "idx_of_p_queries": [frame_split_idx + i]})
        return -1

    final_queries_in_spl = torch.stack([same_person_list["query"][-1] for same_person_list in same_person_lists])
    dot_product = torch.mm(frame_p_f_queries, final_queries_in_spl.t())
    norm_frame_p_f_queries = torch.norm(frame_p_f_queries, dim=1).unsqueeze(1)
    norm_final_queries_in_spl = torch.norm(final_queries_in_spl, dim=1).unsqueeze(0)
    sim_scores = dot_product / (norm_frame_p_f_queries * norm_final_queries_in_spl)

    indices_generator = find_max_indices(sim_scores.cpu())
    for idx, (i, j) in enumerate(indices_generator):
        if sim_scores[i, j] > th:
            same_person_lists[j]["query"].append(frame_p_f_queries[i])
            same_person_lists[j]["target_id"].append(frame_same_person_label[i].item())
            same_person_lists[j]["idx_of_p_queries"].append(frame_split_idx + i)
        else:
            same_person_lists.append({"query": [frame_p_f_queries[i]], "target_id": [frame_same_person_label[i].item()], "idx_of_p_queries": [frame_split_idx + i]})
    return sim_scores


def find_max_indices(tensor):
    for _ in range(tensor.size(0)):
        i, j = np.unravel_index(np.argmax(tensor), tensor.shape)
        yield i, j
        tensor[i, :] = 0
        tensor[:, j] = 0


def val_same_person_lists(same_person_lists):
    if len(same_person_lists) == 0:
        return

    log = {}

    person_ids = list(set([id for person_list in same_person_lists for id in person_list["target_id"]]))

    log["person_ids"] = person_ids
    log["n_lists"] = len(same_person_lists)
    log["n_diff_id"] = [len(set(person_list["target_id"])) for person_list in same_person_lists]

    log["n_included_list"] = {}  # key is person id, value is number of listings in which the person id appears
    for p_id in person_ids:
        log["n_included_list"][p_id] = len([True for person_list in same_person_lists if p_id in person_list["target_id"]])

    diff_person_score = statistics.mean(log["n_diff_id"])  # Preferably close to 1
    same_person_score = statistics.mean([v for k, v in log["n_included_list"].items()])  # Preferably close to 1
    total_score = 1 - (((1 - diff_person_score) + (1 - same_person_score)) / 2)  # Preferably close to 1

    # [print(person_list["target_id"]) for person_list in same_person_lists]
    # print(log["n_lists"])
    # print(log["n_diff_id"])
    # print(log["n_included_list"])
    # print(diff_person_score)
    # print(same_person_score)
    # print(total_score)
    # print("")

    return diff_person_score, same_person_score, total_score
