import torch
import torch.nn as nn
import torch.nn.functional as F
import statistics
import numpy as np


class PersonEncoder(nn.Module):
    def __init__(self, in_d=256, out_d=256, skip=False):
        super().__init__()
        self.is_skip = skip
        self.psn_query_embed = MLPDrop(in_d, out_d, out_d, 3, skip=skip)

    def forward(self, p_queries):
        x = p_queries
        p_queries = self.psn_query_embed(p_queries)
        if self.is_skip:
            return p_queries + x
        else:
            return p_queries


class MLPDrop(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.1, skip=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.dropouts = nn.ModuleList(nn.Dropout(dropout) for _ in range(num_layers - 1))

        if skip:
            for layer in self.layers:
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.dropouts[i](F.relu(layer(x))) if i < self.num_layers - 1 else layer(x)
        return x


class NPairLoss(nn.Module):
    def __init__(self, tau=1):
        super().__init__()
        self.tau = tau

    def forward(self, embs, labels):
        device = labels.device
        embs = embs / embs.norm(p=2, dim=-1, keepdim=True)
        cos_sims = torch.matmul(embs, embs.t())
        sims = torch.exp(cos_sims / self.tau)

        loss = torch.zeros([1]).to(device)
        n_no_pos = 0  # 正例のないアンカー数

        for i, sim in enumerate(sims):
            label_indices_without_i = torch.cat(
                (torch.arange(i), torch.arange(i + 1, len(labels)))).to(device)  # i以外のインデックス列を生成
            labels_wo_i = torch.index_select(labels, 0, label_indices_without_i)  # 自分自身以外のラベルを取得
            sim_wo_i = torch.index_select(sim, 0, label_indices_without_i)  # 自分自身以外の類似度を取得

            pos_indices = torch.nonzero(labels_wo_i == labels[i])  # 正例のインデックスを取得

            if pos_indices.size(0) == 0:
                n_no_pos += 1
                continue

            if pos_indices.dim() >= 2:
                pos_indices = pos_indices.squeeze()

            total_pos_sim = torch.index_select(sim_wo_i, 0, pos_indices).sum()  # 正例の類似度の和

            total_sim = sim_wo_i.sum()  # 類似度の和

            loss_i = -torch.log(total_pos_sim / total_sim)
            loss += loss_i

        if labels.shape[0] == n_no_pos:
            loss = torch.zeros([1]).to(device)
        else:
            loss /= (labels.shape[0] - n_no_pos)

        return loss

    def label_rearrange(self, indices, bs, n_frames) -> torch.Tensor:
        """ Make positibe/negatibe label for metric learning
            同一人物が同じ値となるラベルを返す

            Args:
                indices_ex:
                    A list of size batch_size (bs*n_frames), containing index_k where:
                        - index_k is the indices of person_id (ただし1動画内で割り当てられたものであり,異なる動画における同じidは意味をなさない)
                bs (int): batch size (!=bs*n_frames)
                n_frames (int): n_frames in clip

            Returns:
                (torch.Tensor):A tensor size is
            """
        # n_label_in_clip = []    # n_label_in_clip[clip_idx] = n_label_in_clip
        n_people_in_clip = []    # n_label_in_clip[clip_idx] = n_people_in_clip
        psn_label = torch.Tensor([])

        for i in range(bs):
            clip_indices = indices[i * n_frames:(i + 1) * n_frames]
            psn_label_in_clip = torch.cat(clip_indices)
            # n_label_in_clip.append(psn_label_in_clip.size(0))
            n_people_in_clip.append(torch.unique(psn_label_in_clip).size(0))

            sorted_list = sorted(torch.unique(psn_label_in_clip))
            new_psn_label_in_clip = [sorted_list.index(x) + sum(n_people_in_clip[:-1]) for x in psn_label_in_clip]
            psn_label = torch.cat((psn_label, torch.Tensor(new_psn_label_in_clip)))
        return psn_label.to(torch.int64)


def make_same_person_list(p_f_queries, same_person_label, n_gt_bbox_list, bs, n_frames):
    split_p_f_queries = [[0] * n_frames for _ in range(bs)]  # [clip_idx][frame_idx] = torch.Tensor
    split_same_person_label = [[None] * n_frames for _ in range(bs)]
    split_idx_list = [[None] * n_frames for _ in range(bs)]
    total = 0
    for i, n_gt_boxes in enumerate(n_gt_bbox_list):
        split_idx_list[i // n_frames][i % n_frames] = total
        split_p_f_queries[i // n_frames][i % n_frames] = p_f_queries[total:total + n_gt_boxes]
        split_same_person_label[i // n_frames][i % n_frames] = same_person_label[total:total + n_gt_boxes]
        # split_idx_list[i // bs][i % n_frames] = total
        # split_p_f_queries[i // bs][i % n_frames] = p_f_queries[total:total + n_gt_boxes]
        # split_same_person_label[i // bs][i % n_frames] = same_person_label[total:total + n_gt_boxes]
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


def calc_sim(same_person_lists, frame_p_f_queries, frame_same_person_label, frame_split_idx, th=0.5):
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

    return diff_person_score, same_person_score, total_score
