import math
from webdataset import ShardWriter
from multiprocessing.managers import BaseManager
from pathlib import Path
import json
import numpy as np


def bytes2kmg(size: int) -> str:
    GB = 1024 * 1024 * 1024
    MB = 1024 * 1024
    kB = 1024
    if size > GB:
        return '{:.2f}GB'.format(size / GB)
    elif size > MB:
        return '{:d}MB'.format(size // MB)
    elif size > kB:
        return '{:d}kB'.format(size // kB)
    else:
        return str(size)


def short_side(w, h, size):
    if min(w, h) <= size:
        return w, h  # do not resize for smaller frame size

    # https://github.com/facebookresearch/pytorchvideo/blob/a77729992bcf1e43bf5fa507c8dc4517b3d7bc4c/pytorchvideo/transforms/functional.py#L118
    if w < h:
        new_h = int(math.floor((float(h) / w) * size))
        new_w = size
    else:
        new_h = size
        new_w = int(math.floor((float(w) / h) * size))
    return new_w, new_h


class MyManager(BaseManager):
    # https://docs.python.org/ja/3/library/multiprocessing.html#customized-managers
    pass


class MyShardWriter(ShardWriter):

    def __init__(self, pattern, maxcount=100000, maxsize=3e9, post=None, start_shard=0, **kw):
        super().__init__(pattern, maxcount, maxsize, post, start_shard)
        self.set_verbose(False)
        self.reset_counter()

    def get_size(self):
        return self.size

    def get_shards(self):
        return self.shard

    def set_verbose(self, verbose):
        self.verbose = verbose

    def reset_counter(self):
        self.counter = 0

    def write(self, obj):
        self.counter += 1
        super().write(obj)

    def get_counter(self):
        return self.counter


def info_from_json(shard_path):
    json_file = Path(shard_path).glob('*.json')
    json_file = str(next(json_file))  # get the first json file
    with open(json_file, 'r') as f:
        info_dic = json.load(f)
    dataset_size = info_dic['dataset size']
    n_classes = info_dic['n_classes']
    return dataset_size, n_classes


def xyxy2cxcywh(box: np.ndarray):
    x0, y0, x1, y1 = np.split(box, box.shape[0], axis=0)
    box = [float((x0 + x1) / 2), float((y0 + y1) / 2), float(x1 - x0), float(y1 - y0)]
    return box


def box_normalize(box: np.ndarray, size):
    img_h, img_w = size
    box = box / np.array([img_w, img_h, img_w, img_h])
    return box
