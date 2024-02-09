# 時空間アクション検出のための人物クエリのマッチングによるアクションチューブ生成

# (Action tube generation by person query matching for spatio-temporal action detection)

## コード

- datasets/dataset.py: データセットを読み込むためのファイル．動画をシーケンスに読み込む．評価時などに用いる．
- datasets/make_shrads.py: shard生成コード．
- datasets/use_shrads.py: shard読込コード．学習時に用いる．
- models/{backbone, transformer, detr, matcher, position_encoding}.py: [DETR](https://github.com/facebookresearch/detr)のコード．不要なものを除外して改良．
- models/person_encoder.py: 人物特徴抽出器(f_c)のコード．学習に必要なN-Pair Lossもここに書かれている．
- models/action_head.py: Action Headのコード
- models/tube.py:
