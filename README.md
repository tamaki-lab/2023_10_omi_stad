# 時空間アクション検出のための人物クエリのマッチングによるアクションチューブ生成

<!-- # (Action tube generation by person query matching for spatio-temporal action detection) -->

![overview_1-crop.pdf](https://github.com/tamaki-lab/2023_10_omi_stad/files/14218198/overview_1-crop.pdf)

## 準備

[comet](https://www.comet-ml.com/docs/)でログを残すためにアカウント作成

以下のコマンドでcometのapiをホームディレクトリ以下に置けばデフォルトでそれを参照する

```bash
comet init --api-key
```

<br>

環境は[こちら](https://hub.docker.com/r/tttamaki/docker-ssh/)のDockerイメージからコンテナ作成．（バージョンは[これ](https://hub.docker.com/layers/tttamaki/docker-ssh/2023.08.24.4/images/sha256-9817e7b4844e6b0185f16e7048885db79e0fb2f14a9131c7ed3af8608344e99b?context=explore)）

必要に応じて以下のコマンドで実行環境をそろえる

```bash
pip install -r requirements.txt
```

## コード

- datasets/dataset.py: データセットを読み込むためのファイル．動画をシーケンスに読み込む．評価時などに用いる．
- datasets/make_shrads.py: shard生成コード．
- datasets/use_shrads.py: shard読込コード．学習時に用いる．
- models/{backbone, transformer, detr, matcher, position_encoding}.py: [DETR](https://github.com/facebookresearch/detr)のコード．不要なものを除外して改良．
- models/person_encoder.py: 人物特徴抽出器 $f_p$ のコード．学習に必要なN-Pair Lossのコードも．
- models/action_head.py: Action Headのコード．
- models/tube.py: tubeのコード．
- utils/box_ops.py: IoUの計算などが書かれたコード．モーションカテゴリを得るコードも．
- utils/gt_tubes.py: 真値tubeを得るコード．
- utils/misc.py: よく使う関数をまとめたコード．179行目以降はDETRの元コード．
- utils/plot_utils.py: 予測の可視化コード．
- utils/video_map.py: video-mAPを計算するコード．
- train_detr.py: DETRを学習するコード．
- train_qmm.py: 人物特徴抽出器 $f_p$ を学習するコード．
- train_action_head.py: Action Headを学習するコード．
- eval_qmm.py: QMM（人物特徴抽出器）を評価するコード．真値tubeの領域のみ（アクション無視）に対するRecallを計算．
- eval_map.py: Action Headを評価するコード．video-mAPを計算．

## 実行

人物特徴抽出器 $f_p$ の学習．

```bash
python train_qmm.py
```

#### オプション

- `-epochs`： エポック数
- `-device`： GPU番号
- `-qmm_name`： 実験名．checkpoint/`dataset`/`qmm_name`/encoder以下に学習した人物特徴抽出器のpthファイルが保存．
- `--shards_path`: 学習に用いるshardパス
- `--dataset`: データセット名．
- `--batch_size`: バッチサイズ．
- `--n_frames`: 学習時のクリップのフレーム数．
- `--sampling_rate`: 学習時のクリップのサンプリングレート．
- `--lr_en`: 学習率．
- `--psn_score_th`:　学習に用いるクエリの人物スコア閾値．（$Q^t$から$Q^{t\prime}$の閾値）
- `iou_th`: クエリが担当している人物があるかどうかを判断する時の（クエリの検出領域と真値領域の）IoU閾値．
- `--is_skip`: 人物特徴抽出器 $f_p$ にスキップ接続を用いるかどうか．

<br>

人物特徴抽出器 $f_p$ の評価．

真値tubeの領域のみに対するRecallを求める（3DIoUの計算はラベルのあるフレームのみで行う）．

同時にAction Headの学習のためにアクションラベルを付与した同一人物クエリをtarファイルに保存．

```bash
python eval_qmm.py
```

#### オプション

- `-device`, `--dataset`, `--n_frames`, `--psn_score_th`, `--iou_th`, `--is_skip`: train_qmm.pyと同じ．
- `--qmm_name`: 評価する人物特徴抽出器 $f_p$の名前．train_qmm.pyのオプションで指定したものと同じにすればよい．
- `--n_frames`: 並列で実行するフレーム数．
- `--subset`: 評価するサブセット．
- `--load_epoch`: 読み込む人物特徴抽出器 $f_p$ のエポック
- `--link_cues`: リンクに用いるもの．人物特徴量なら"feature"，IoUなら"iou"．
- `--sim_th`: リンクする類似度スコア閾値．`--link_cues`で"iou"を指定したならリンクするIoU閾値．
- `--tiou_th`: 予測の正誤を判断する3DIoU閾値．
- `--filter_length`: 予測から除外する長さ．リンクされたクエリ数がこの値未満なら除かれる．
- 注意: 一度評価したものはtarファイルに書き込まれるので，再度評価したい時や3DIoU閾値を変更して評価したい際は333-334行目をコメントアウトして高速に実行可能．また，337行目や350-352行目で評価方法を変更可能．詳細はコードを参照．

<br>

Action Headの学習．

```bash
python train_action_head.py
```

#### オプション

- `epochs`, `-device`, `dataset`, `qmm_name`, `--load_epoch`, `--psn_score_th`, `--sim_th`, `--filter_length`: train_qmm.py, eval_qmm.pyと同じ．
- `--head_name`: 実験名．checkpoint/`dataset`/`qmm_name`/head/`head_name`以下に学習したAction Headのpthファイルが保存．
- `--head_type`: Action Headのアーキテクチャを指定．"vanilla"がデフォルト．"time_ecd:add"は時刻情報を表すベクトルを入力に加算．"time_ecd:cat"はベクトルを連結．"x3d"はHead内のquery-key-valueアテンションにおけるqueryにx3d_xsから得られる特徴量を使用．論文の実験結果におけるtime encoding, global featureに相当．
- 注意：time encodingを用いる場合はデータセットに合わせてtime encodingの周期を変更（models/action_head.pyの37行目）．

<br>

video-mAPによるAction Headの評価．

```bash
python eval_map.py
```

#### オプション

- `device`, `dataset`, `n_frames`, `subset`, `qmm_name`, `head_type`, `head_name`, `psn_score_th`, `sim_th`, `tiou_th`, `filter_length`: train_qmm.py, eval_qmm.py, train_action_head.pyと同じ．
- `--metric`: 評価指標．"v-mAP" or "motion-AP"
- `--topk`: 予測スコアのtopkまで考慮して予測tubeを作成．詳しくは論文参照．
- `--load_epoch_qmm`: 読み込む人物特徴抽出器 $f_p$ のエポック
- `--load_epoch_head`: 読み込むAction Headのエポック

## メモ

- train_detr.pyでUCF101-24を学習させるのはオリジナルのDETRの検出精度が悪いから．ロスの重み（クラス，領域）はmodels/detr.pyの336行目を直接変更．
- eval_qmm.pyで"train"に対しても評価を行うのはQMMの出力でAction Headを学習させるため．
- 評価時に関数`tube_iou()`で3DIoUの計算方法を変えられる．ラベルのあるフレームだけで3DIoUを計算したいならば`label_centric=True`，フレーム単位のIoUがある値`a`を超えら1としたい場合（3DIoUが低い場合に検出領域のミスまたはリンクのミスかを簡易なデバッグで判断）は`frame_iou_set=(True, a)`．
- Action Headの学習時に毎エポックQMMの出力を得るのは効率が悪いのでeval_qmm.pyで結果をtarファイルに保存してそれを読み込む．
- 論文にのせた実験を行った際のオプションやチェックポイントは[修論overleaf](https://www.overleaf.com/project/6500fba815cd279a9e3af773)の表付近のコメントアウト参照．seedを固定しているため再現性あり．
