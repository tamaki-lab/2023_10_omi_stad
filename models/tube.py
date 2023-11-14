from typing import Tuple
import torch
import numpy as np


class ActionTube:
    def __init__(self, video_name=None, sim_th=0.7, end_consecutive_frames=8, ano=None):
        self.video_name = video_name
        self.tubes = []
        self.end_idx = set()
        self.sim_th = sim_th
        self.k = end_consecutive_frames
        self.ano = ano

    def update(self, d_queries, p_queries, frame_idx, psn_indices, psn_boxes):
        diff_list = [frame_idx - tube["idx_of_p_queries"][-1][0] for tube in self.tubes]
        for list_idx, d in enumerate(diff_list):
            if d >= self.k:
                self.end_idx.add(list_idx)

        if len(self.tubes) == 0:
            for idx, (d_query, p_query) in enumerate(zip(d_queries, p_queries)):
                query_idx = psn_indices[idx].item()
                self.tubes.append({"d_query": [d_query],
                                   "p_query": [p_query],
                                   "idx_of_p_queries": [(frame_idx, query_idx)],
                                   "bbox": [psn_boxes[idx]]})
        else:
            sim_scores = self.sim_scores(self.tubes, p_queries)
            indices_generator = self.find_max_indices(sim_scores.cpu())
            for i, j in indices_generator:
                query_idx = psn_indices[i].item()
                if (sim_scores[i, j] > self.sim_th) and (j != -1):
                    self.tubes[j]["d_query"].append(d_queries[i])
                    self.tubes[j]["p_query"].append(p_queries[i])
                    self.tubes[j]["idx_of_p_queries"].append((frame_idx, query_idx))
                    self.tubes[j]["bbox"].append(psn_boxes[i])
                else:
                    self.tubes.append({"d_query": [d_queries[i]],
                                       "p_query": [p_queries[i]],
                                       "idx_of_p_queries": [(frame_idx, query_idx)],
                                       "bbox": [psn_boxes[i]]})

    def sim_scores(self, tubes, p_embedding):
        final_queries = torch.stack([tube["p_query"][-1] for tube in tubes])
        dot_product = torch.mm(p_embedding, final_queries.t())
        norm_frame_p_f_queries = torch.norm(p_embedding, dim=1).unsqueeze(1)
        norm_final_queries = torch.norm(final_queries, dim=1).unsqueeze(0)
        sim_scores = dot_product / (norm_frame_p_f_queries * norm_final_queries)
        return sim_scores

    def find_max_indices(self, tensor: torch.Tensor):
        used_i_list = []
        for j in self.end_idx:
            tensor[:, j] = -1
        for _ in range(tensor.size(0)):
            i, j = np.unravel_index(np.argmax(tensor), tensor.shape)
            if tensor[i, j] != -1:
                used_i_list.append(i)
                tensor[i, :] = -1
                tensor[:, j] = -1
                yield i, j
            else:
                break  # そのフレームにおいて全てのリストにクエリが割り当てられた場合は強制的に新しい人物とする
        not_used_i_list = [x for x in range(tensor.size(0)) if x not in used_i_list]
        for i in not_used_i_list:
            yield i, -1

    def filter(self, filter_length=16):
        """ exclude short tube """
        self.tubes = [tube for tube in self.tubes if len(tube["idx_of_p_queries"]) > filter_length]
        for tube in self.tubes:
            tube["p_query"] = [x.cpu() for x in tube["p_query"]]
            tube["d_query"] = [x.cpu() for x in tube["d_query"]]

    def extract(self, tube, indices=None):
        if indices is None:
            frame_indices = [frame_idx for frame_idx, query_idx in tube["idx_of_p_queries"]]
            boxes = [box for box in tube["bbox"]]
        else:
            indices = torch.where(indices)[0]
            frame_indices = [tube["idx_of_p_queries"][i][0] for i in indices]
            boxes = [tube["bbox"][i] for i in indices]
        return {frame_idx: bbox for frame_idx, bbox in zip(frame_indices, boxes)}

    def split(self):
        """ split tube based on predicted action id """
        new_tubes = []
        for tube in self.tubes:
            new_tubes.extend([(
                self.video_name,
                {"class": i.item(),
                 "score": tube["action_score"][tube["action_id"] == i].mean().item(),
                 "boxes": self.extract(tube, tube["action_id"] == i)})
                for i in tube["action_id"].unique()])
        self.tubes = new_tubes
