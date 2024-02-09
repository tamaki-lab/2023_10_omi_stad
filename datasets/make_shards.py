from io import BytesIO
import pathlib
from pathlib import Path
import os
import os.path as osp
import random
import json
from multiprocessing import Process, Manager
import queue
import argparse
from typing import Tuple
import pandas as pd
from PIL import Image
from tqdm import tqdm
import yaml
import av

from utils import bytes2kmg, MyManager, MyShardWriter


def worker(q, lock, pbar, sink, quality, class_to_idx, pos, ano):
    while True:
        try:
            video_file_path = q.get(timeout=1)
        except queue.Empty:
            return
        if video_file_path is None:
            return

        extenstions = [".png", ".jpg"]
        img_file_paths = [
            path
            for path in Path(video_file_path).glob("**/*.*")
            if path.suffix in extenstions
        ]
        img_file_paths = sorted(img_file_paths)
        n_frames = len(img_file_paths)

        jpg_byte_list = []
        width = Image.open(img_file_paths[0]).width
        height = Image.open(img_file_paths[0]).height

        with tqdm(
            img_file_paths,
            total=n_frames,
            position=pos + 1,
            leave=False,
            mininterval=0.5,
        ) as frame_pbar:
            frame_pbar.set_description(f"worker {pos:02d}")
            for img_path in frame_pbar:
                img = Image.open(img_path)
                with BytesIO() as buffer:
                    img.save(buffer, format="JPEG", quality=quality)
                    jpg_byte_list.append(buffer.getvalue())

        category_name = video_file_path.split("/")[-2]
        video_name = video_file_path.split("/")[-1]
        label = class_to_idx[category_name]
        key_str = category_name + "/" + video_name

        video_stats_dict = {
            "__key__": key_str,
            "video_id": video_name,
            "category": category_name,
            "label": label,
            "label_bbox": ano[key_str],
            "width": width,
            "height": height,
            "n_frames": n_frames,
        }

        with lock:
            video_stats_dict["shard"] = sink.get_shards()

            sample_dic = {
                "__key__": key_str,
                "video.pickle": (jpg_byte_list, json.dumps(video_stats_dict)),
            }

            sink.write(sample_dic)
            pbar.update(1)
            pbar.set_postfix_str(
                f"shard {sink.get_shards()}, " f"size {bytes2kmg(sink.get_size())}"
            )


def make_shards(args, params):
    ano_dict = read_ano(args, params)

    video_file_paths = [name for name in ano_dict.keys()]
    if args.shuffle:
        random.shuffle(video_file_paths)
    n_samples = len(video_file_paths)

    # https://github.com/pytorch/vision/blob/a8bde78130fd8c956780d85693d0f51912013732/torchvision/datasets/folder.py#L36
    class_list = sorted(
        entry.name for entry in os.scandir(params["dataset_path"]) if entry.is_dir()
    )
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_list)}

    shard_dir_path = Path(params[args.subset + "_shards_path"])
    shard_dir_path.mkdir(exist_ok=True, parents=True)
    shard_filename = str(shard_dir_path / f"{args.dataset.upper()}-%05d.tar")

    # https://qiita.com/tttamaki/items/96b65e6555f9d255ffd9
    MyManager.register("Tqdm", tqdm)
    MyManager.register("Sink", MyShardWriter)

    with MyManager() as my_manager, Manager() as manager:

        # prepare manager objects
        q = manager.Queue()
        lock = manager.Lock()
        pbar = my_manager.Tqdm(
            total=n_samples,
            position=0,
        )
        pbar.set_description("Main process")
        sink = my_manager.Sink(
            pattern=shard_filename,
            maxsize=int(args.max_size_gb * 1000**3),
            maxcount=args.max_count,
        )

        # start workers
        p_all = [
            Process(
                target=worker,
                args=(q, lock, pbar, sink, args.quality, class_to_idx, i, ano_dict),
            )
            for i in range(args.num_workers)
        ]
        [p.start() for p in p_all]

        for item in video_file_paths:
            q.put(os.path.join(params["dataset_path"], item))
        for _ in range(args.num_workers):
            q.put(None)

        # wait workers, then close
        [p.join() for p in p_all]
        [p.close() for p in p_all]

        dataset_size_filename = str(
            shard_dir_path / f"{args.dataset.upper()}-dataset-size.json"
        )
        with open(dataset_size_filename, "w") as fp:
            json.dump(
                {
                    "dataset size": sink.get_counter(),
                    "n_classes": len(class_list),
                },
                fp,
            )

        sink.close()
        pbar.close()


def pre_worker(untrimmed_q, clip_q, lock, pre_pbar, sink, pos, ano_dict, clip_dur=10):
    while True:
        try:
            untrimmed_video = untrimmed_q.get(timeout=1)
            untrimmed_video = UntrimmedVideo(untrimmed_video)
        except queue.Empty:
            return

        iter = 900 // clip_dur  # TODO change for videos of any length
        with tqdm(
            range(iter),
            position=pos + 1,
            leave=False,
            mininterval=0.5,
        ) as clip_pbar:
            for _ in clip_pbar:
                clip_name, n_frames = untrimmed_video()
                clip_pbar.set_description(f"worker {pos:02d}")

                try:
                    with open(clip_name, "rb") as f:
                        movie_binary = f.read(-1)
                    container = av.open(clip_name)
                    os.remove(clip_name)
                except Exception as e:
                    print(clip_name)
                    print(e)
                    continue

                video_stream_id = 0  # default
                stream = container.streams.video[video_stream_id]

                # unnormlize bbox
                width = stream.codec_context.width
                height = stream.codec_context.height

                clip_key = clip_name.split(".")[0]
                for time, ano in ano_dict[clip_key].copy().items():
                    for obj_id, bbox in ano.items():
                        ano_dict[clip_key][time][obj_id][0] = int(
                            ano_dict[clip_key][time][obj_id][0] * width
                        )
                        ano_dict[clip_key][time][obj_id][2] = int(
                            ano_dict[clip_key][time][obj_id][2] * width
                        )
                        ano_dict[clip_key][time][obj_id][1] = int(
                            ano_dict[clip_key][time][obj_id][1] * height
                        )
                        ano_dict[clip_key][time][obj_id][3] = int(
                            ano_dict[clip_key][time][obj_id][3] * height
                        )

                key_str = clip_key
                video_stats_dict = {
                    "__key__": key_str,
                    "suffix": untrimmed_video.file_path.suffix[1:],
                    "label_bbox": ano_dict[clip_key],
                    "width": stream.codec_context.width,
                    "height": stream.codec_context.height,
                    "fps": float(stream.base_rate),
                    "n_frames": n_frames,
                    "duraion": float(container.duration) / av.time_base,
                    "shard": sink.get_shards(),
                }

                sample_dic = {
                    "__key__": key_str,
                    "video.pickle": (movie_binary, json.dumps(video_stats_dict)),
                }

                with lock:
                    clip_q.put(sample_dic)

        with lock:
            pre_pbar.update(1)


def write_worker(clip_q, write_pbar, sink):
    while True:
        try:
            sample_dic = clip_q.get(timeout=1)
        except queue.Empty:
            continue
        if sample_dic is None:
            return

        sink.write(sample_dic)
        write_pbar.update(1)
        write_pbar.set_postfix_str(
            f"shard {sink.get_shards()}, " f"size {bytes2kmg(sink.get_size())}"
        )


class UntrimmedVideo:
    """untrimmed videoからtrimmed videoを作成するクラス"""

    def __init__(self, file_path: pathlib.PosixPath, trimmed_duration=10):
        """
        Args:
            file_path (pathlib.PosixPath): untrimmed video path
            trimmed_duration (int, optional): length of trimmed video. Defaults to 10.
        """
        self.file_path = file_path
        self.trimmed_duration = trimmed_duration

        self.container = av.open(str(file_path))
        self.stream = self.container.streams.video[0]

        self.current_realtime = 0
        self.current_time = 0
        self.width = self.stream.width
        self.height = self.stream.height
        self.codec = self.stream.codec_context.name
        self.base_rate = self.stream.base_rate
        self.spf = 1 / self.stream.average_rate
        self.pix_fmt = self.stream.pix_fmt

        self.container.seek(
            offset=900 // self.stream.time_base,
            any_frame=False,
            backward=True,
            stream=self.stream,
        )
        for frame in self.container.decode(video=0):
            if frame.time + self.spf < 900:
                continue
            else:
                self.current_realtime = frame.time
                self.current_time = 900
                break

    def __call__(self) -> Tuple[str, int]:
        clip_dur = str(self.current_time) + "-" + str(self.current_time + 10)
        created_filename = self.file_path.stem + "__" + clip_dur + self.file_path.suffix
        new_container = av.open(created_filename, mode="w")
        new_stream = new_container.add_stream(self.codec, rate=self.base_rate)
        new_stream.width = self.width
        new_stream.height = self.height
        new_stream.pix_fmt = self.pix_fmt

        for frame in self.container.decode(video=0):
            if frame.time + self.spf < self.current_realtime + self.trimmed_duration:
                frame = frame.to_ndarray(format="rgb24")
                frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                for packet in new_stream.encode(frame):
                    new_container.mux(packet)
            else:
                self.current_realtime = frame.time
                self.current_time += self.trimmed_duration
                break
        for packet in new_stream.encode():
            new_container.mux(packet)
        new_container.close()

        return created_filename, new_stream.frames


def make_untrimmed_videl_shards(args):
    ano_dict = read_ava_ano(args.subset)

    video_base_dir = osp.join(params["dataset_path"], args.subset)
    video_file_paths = [p for p in Path(video_base_dir).glob("*")]

    shard_dir_path = Path(params[args.subset + "_shards_path"])
    shard_dir_path.mkdir(exist_ok=True, parents=True)
    shard_filename = str(shard_dir_path / f"{args.dataset.upper()}-%05d.tar")

    MyManager.register("Tqdm", tqdm)
    MyManager.register("Sink", MyShardWriter)

    with MyManager() as my_manager, Manager() as manager:
        untrimmed_q = manager.Queue()
        clip_q = manager.Queue()
        lock = manager.Lock()
        pre_pbar = my_manager.Tqdm(
            total=len(video_file_paths),
            position=0,
        )
        pre_pbar.set_description("Prepare process")
        write_pbar = tqdm(
            total=len(ano_dict.keys()),
            position=args.num_workers,
        )
        write_pbar.set_description("Write process")
        sink = my_manager.Sink(
            pattern=shard_filename,
            maxsize=int(args.max_size_gb * 1000**3),
            maxcount=args.max_count,
        )

        p_all = [
            Process(
                target=pre_worker,
                args=(untrimmed_q, clip_q, lock, pre_pbar, sink, i, ano_dict),
            )
            for i in range(args.num_workers - 1)
        ]

        for path in video_file_paths:
            untrimmed_q.put(path)
        p_all.append(Process(target=write_worker, args=(clip_q, write_pbar, sink)))
        [p.start() for p in p_all]

        [p.join() for p in p_all[:-1]]
        clip_q.put(None)
        p_all[-1].join()
        [p.close() for p in p_all]

        dataset_size_filename = str(
            shard_dir_path / f"{args.dataset.upper()}-dataset-size.json"
        )
        with open(dataset_size_filename, "w") as fp:
            json.dump(
                {
                    "dataset size": sink.get_counter(),
                    "n_classes": 60,  # TODO change
                },
                fp,
            )

        sink.close()
        pre_pbar.close()


def read_ucf_ano(subset, params):
    video_list_txt = params[subset + "_videos_file_path"]
    with open(video_list_txt) as f:
        video_list = f.readlines()
    video_list = [video.replace("\n", "") for video in video_list]

    file_path = params["annotation_file_path"]
    df = pd.read_json(file_path)

    ano_dict = {}
    for video_name in video_list:
        ano_dict[video_name] = {}
        for obj_id in range(len(df[video_name]["annotations"])):
            sf = df[video_name]["annotations"][obj_id]["sf"]
            bbox_list = df[video_name]["annotations"][obj_id]["boxes"]
            cls_id = df[video_name]["annotations"][obj_id]["label"]
            for j, box in enumerate(bbox_list):
                box[2] = box[0] + box[2]
                box[3] = box[1] + box[3]
                if j + sf not in ano_dict[video_name]:
                    ano_dict[video_name][j + sf] = {}
                ano_dict[video_name][j + sf][obj_id] = box
                ano_dict[video_name][j + sf][obj_id].append(cls_id)

    return ano_dict


def read_jhmdb_ano(subset, params):
    video_list_txt = params[subset + "_videos_file_path"]
    with open(video_list_txt) as f:
        video_list = f.readlines()
    video_list = [video.replace("\n", "") for video in video_list]

    json_load = json.load(open(params["annotation_file_path"]))
    class_list = sorted(os.listdir(params["dataset_path"]))[1:]
    cls_to_idx = {cls: i for i, cls in enumerate(class_list)}

    ano_dict = {}
    for video_name in video_list:
        cls = video_name.split("/")[0]
        cls_id = cls_to_idx[cls]
        ano_dict[video_name] = {}
        for i, box in enumerate(json_load["gttubes"][video_name][str(cls_id)][0]):
            ano_dict[video_name][i] = {}
            ano_dict[video_name][i][0] = box[1:]
            ano_dict[video_name][i][0].append(cls_id)

    return ano_dict


def read_ava_ano(subset, params, clip_duration=10):
    org_label_list_file = params["org_label_list_file_path"]
    label_list_file = params["label_list_file_path"]
    action_id2id = {}  # csv annofile id to classification id(only used class)
    org_label_list = []
    with open(org_label_list_file) as f:
        for line in f:
            if "name" in line:
                label_name = line.split('name: "')[1].replace('"\n', "")
                org_label_list.append(label_name)
    tgt_label_list = []
    with open(label_list_file) as f:
        for line in f:
            if "name" in line:
                label_name = line.split('name: "')[1].replace('"\n', "")
                tgt_label_list.append(label_name)
    for i, label in enumerate(org_label_list):
        if label not in tgt_label_list:
            continue
        tgt_id = tgt_label_list.index(label)
        action_id2id[i + 1] = tgt_id

    cd = clip_duration

    df = pd.read_csv(
        params[subset + "_annotation_file_path"],
        header=None,
        names=["video_name", "sec", "x1", "y1", "x2", "y2", "cls_id", "psn_id"],
    )
    ano_dict = {}
    video_name_list = list(set(df["video_name"].to_list()))
    for video_name in video_name_list:
        for i in range(900, 1800, cd):
            clip_duration_str = str(i) + "-" + str(i + cd)
            ano_dict[video_name + "__" + clip_duration_str] = {}

    for index, row in tqdm(df.iterrows(), total=len(df)):
        video_name = row["video_name"]
        sec = row["sec"]
        psn_id = row["psn_id"]
        sec_in_clip = sec % cd
        start_sec = sec - sec_in_clip
        end_sec = start_sec + cd
        clip_name = video_name + "__" + str(start_sec) + "-" + str(end_sec)
        if sec_in_clip not in ano_dict[clip_name]:
            ano_dict[clip_name][sec_in_clip] = {}
        x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
        cls_id = row["cls_id"]
        if cls_id not in action_id2id:
            continue
        cls_id = action_id2id[cls_id]
        # 補完のために同一人物かつ同一行動を確認するためのキー
        key_name = str(psn_id) + "-" + str(cls_id)
        ano_dict[clip_name][sec_in_clip][key_name] = [x1, y1, x2, y2, cls_id]
        # clip分割の切れ目のアノテーションを分割したclipのどちらにも書くための処理
        if sec_in_clip == 0:
            clip_name = video_name + "__" + str(start_sec - cd) + "-" + str(start_sec)
            if cd not in ano_dict[clip_name]:
                ano_dict[clip_name][cd] = {}
            ano_dict[clip_name][cd][key_name] = [x1, y1, x2, y2, cls_id]

    return ano_dict


def read_ano(args, params) -> dict:
    """
    Returns:
        dict[video_name][frame_id][object_id]=[x1, y1, x2, y2, cls_id]
    """
    if args.dataset == "ucf101-24":
        return read_ucf_ano(args.subset, params)
    elif args.dataset == "jhmdb21":
        return read_jhmdb_ano(args.subset, params)
    else:
        raise NameError("invalide dataset name")


def arg_factory():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", "--dataset", default="ucf101-24", choices=["ucf101-24", "jhmdb21", "ava"]
    )
    parser.add_argument("-sub", "--subset", type=str, default="train")
    parser.add_argument(
        "--max_size_gb",
        type=float,
        default=10.0,
        help="Max size [GB] of each shard tar file. " "default to 10.0 [GB].",
    )
    parser.add_argument(
        "--max_count",
        type=int,
        default=100000,
        help="Max number of entries in each shard tar file. " "default to 100,000.",
    )
    parser.add_argument(
        "--shuffle", dest="shuffle", action="store_true", help="use shuffle"
    )
    parser.add_argument(
        "--no_shuffle", dest="shuffle", action="store_false", help="do not use shuffle"
    )
    parser.set_defaults(shuffle=True)

    parser.add_argument("-w", "--num_workers", type=int, default=8)

    # following auguments are about images (not video)
    parser.add_argument(
        "-ss",
        "--short_side_size",
        type=int,
        default=360,
        help="Shorter side of resized frames. " "default to 360.",
    )
    parser.add_argument(
        "-q",
        "--quality",
        type=int,
        default=80,
        help="Qualify factor of JPEG file. " "default to 80.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_factory()

    params = yaml.safe_load(open(f"../projects/{args.dataset}.yml"))

    make_shards(args, params)
    # make_untrimmed_videl_shards(args)
