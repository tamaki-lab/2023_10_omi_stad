import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import (
    miners,
    losses,
    reducers,
    distances,
)


class PersonEncoder(nn.Module):
    def __init__(self, d=256, out_d=256):
        super().__init__()
        self.psn_query_embed = nn.Linear(d, out_d)

    def forward(self, p_queries):
        p_queries = self.psn_query_embed(p_queries)
        return p_queries


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

    def make_posneg_label(self, indices_ex, bs, n_frames):
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
        return loss


# distance = distances.CosineSimilarity()
# reducer = reducers.MeanReducer()
# loss_func = losses.NTXentLoss(temperature=0.07, distance=distance, reducer=reducer)
# mining_func = miners.BatchEasyHardMiner(
#     pos_strategy=miners.BatchEasyHardMiner.ALL,
#     neg_strategy=miners.BatchEasyHardMiner.ALL,
#     distance=distance
# )

# t0_q = torch.Tensor([[10, 10], [20, 20]])
# t1_q = torch.Tensor([[10, 10], [20, 20]])
# q = torch.cat((t0_q, t1_q), dim=0)

# t0_l = torch.Tensor([0, 1])
# t1_l = torch.Tensor([1, 0])
# l = torch.cat((t0_l, t1_l), dim=0)

# miner_out = mining_func(q, l)
# print(miner_out)

# print(loss_func(q, indices_tuple=miner_out))
