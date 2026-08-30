# 数据变换设计：6 类官方变换 + manifest 字段

> Owner：数据变换（feat/data-transforms 分支）。变换算子、档位参数、manifest
> schema 以本文件 + `src/transforms/spec.py` 为准，改动需同步两处。官方演示集
> `data/val/`（COCO val2017 + DALL·E Advanced）带 `DO_NOT_TRAIN` 标记：作
> `--split train` 索引会被拒绝，评测索引放行（见 [data.md](data.md)）。
>
> <!-- 2026-08-29, tianqi, eval is an adapter, not a second op implementation -->
> 评测 `src/eval/transforms.py` 只做薄封装：条件名、种子、像素都走 spec/ops。
> jitter 用一档独立采样 ±20%（方案 A），没有 `jitter_m20`。
> <!-- end -->

## 1. 官方档位（冻结，勿改参数）

题面 6 类变换共 **14 个档位**，加 clean 共 **15 个评测条件**：

| transform_key | op | 参数 | 题面原文 |
|---|---|---|---|
| `jpeg_q90 / q70 / q50 / q30` | jpeg | quality | 90 / 70 / 50 / 30 |
| `blur_s05 / s10 / s20` | blur | sigma | 0.5 / 1.0 / 2.0 |
| `resize_s05 / s025` | resize | scale，降采样后放回原尺寸 | 0.5× / 0.25× then upscale |
| `noise_s002 / s005 / s010` | noise | sigma（[0,1] 像素尺度） | 0.02 / 0.05 / 0.10 |
| `jitter_p20` | jitter | brightness/contrast/saturation 各自均匀采样 ±20% | ±20% |
| `crop_p80` | crop | 保留中心 80%×80%，**不放大回原尺寸** | crop 80% |

- 建议评估口径（与评测侧对齐）：**robust AUC = 14 个变换档位 AUC 的宏平均**，clean 单列。
- 实现：Pillow + NumPy。JPEG 用 PIL 重编码（默认 4:2:0 色度子采样）；blur 用
  `ImageFilter.GaussianBlur`（其 radius 即 σ）；noise 加在 [0,1] 浮点像素上再截断回
  uint8；jitter 用 `ImageEnhance.Brightness/Contrast/Color` 串联（顺序：亮度→对比度→饱和度）；
  resize 双向与 crop 的 `--crop-resize-back` 均用双线性插值（`Resampling.BILINEAR`）。

## 2. 题面歧义的处理决策

| 歧义 | 决策 | 理由 / 逃生口 |
|---|---|---|
| crop 80% 后是否放大回原尺寸 | **不放大**，输出边长 ×0.8 | 题面只对 resize 写了 "then upscale"，头像裁剪场景本来就是小图；`build.py --crop-resize-back` 一键切换 |
| jitter ±20% 的取值 | 每张图三因子在 [0.8, 1.2] 独立均匀采样，**实际采样值写入 manifest 的 `params`** | 题面给的是范围而非定值；种子固定 ⇒ 可复现 |
| noise σ 的尺度 | [0,1] 归一化像素 | 该尺度下 0.02→轻度、0.10→重度，符合题意 |
| 原图已是 jpg 的 JPEG 档位 | 照样再编码一次 | 社交再压缩本来就是二次压缩（data.md） |
| 非 JPEG 档位的落盘格式 | PNG 无损 | 保证除目标变换外无额外退化；JPEG 档位把编码字节直接写盘，**不做二次压缩** |
| 变换在哪个分辨率上做 | **原图原生分辨率**；模型侧 resize 归训练/评估管线的预处理 | JPEG/模糊/下采样的破坏程度与分辨率强相关，原生分辨率才可比 |

## 3. 复现性（种子规则）

```
seed = sha1("{image_id}|{transform_key}|v2") 的前 7 字节（56 位大端整数，恒为非负且在带符号 int64 内，下游 pandas/SQL 不会溢出）
```

同一份 source manifest + 同一 spec 版本 + **相同版本的 numpy/Pillow**，任何人任何
机器重建都得到**逐字节相同**的评测集。JPEG 档位的编码字节随 Pillow 打包的 libjpeg
版本可能不同，跨机器/跨版本重建前先固定依赖版本。spec 参数或种子规则若变更，把盐
`v2` 升 `v3` 并在本文件记录变更原因。

## 4. 目录布局

```
data/
├── manifests/                      # JSONL（小文件；按 data.md 不进 git）
│   ├── source_*.jsonl              # build_source.py 产物
│   └── transforms_eval.jsonl       # build.py 产物
└── transforms/                     # 派生图（gitignore）
    └── <transform_key>/<image_id前2位>/<image_id>.<png|jpg>
```

## 5. manifest 字段

**source**（`build_source.py` 产出，每张原图一行）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `image_id` | str | 默认 sha1(`"{相对路径}:{文件大小}"`)；`--hash-content` 时为文件内容 sha1 |
| `path` | str | 相对仓库根的 posix 路径 |
| `label` | int | 1 = AIGC，0 = real |
| `source_dataset` | str | cifake / sid_set / wildfake / flux_gen / … |
| `generator` | str\|null | 假图的生成器（如 `sd14`、`flux1-dev`），真图为 null |
| `split` | str | train / val / test / unseen |
| `width` / `height` | int | 原生尺寸 |
<!-- 2026-08-30, tianqi, ablation + leak-audit columns; optional in old JSONL -->
| `family` | str\|null | 假图 `gan` / `diffusion`；真图 **null** |
| `arch` | str\|null | `unet` / `dit` / `flow` / `pixel`；真图 null；GAN 也用 null |
| `generation_type` | str\|null | 假图 `t2i` / `i2i`；真图 null（与 family 分开，见 DATA_ABLATION_PLAN.md） |
| `content_type` | str | `real` / `full_synthetic` / `partial_manipulation`。缺省时由 label 推断。**tampered 标第三种，默认不进 train** |
| `original_format` | str\|null | 落盘后缀（`jpeg`→`jpg`）。审计 DDA 格式捷径，不是训练目标 |
| `phash` | str\|null | 64-bit DCT perceptual hash，16 位 hex。去重、以及 SID train vs 官方 val 的 **整图拷贝** 碰撞。局部篡改通常对不上 COCO 原图，那些靠 `content_type` 排除 |
<!-- end -->

**transform**（`build.py` 产出，每张派生图一行；source 字段反范式带入，评估免 join）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `row_id` | str | `{image_id}_{transform_key}` |
| `source_image_id` / `source_path` | str | 回链原图 |
| `transform` / `transform_key` | str | op 名 / 档位名 |
| `params` | object | 实际参数（jitter 为采样到的三因子） |
| `seed` | int | 派生种子 |
| `path` | str | 派生图路径 |
| `label` / `source_dataset` / `generator` / `split` | | 从 source 复制 |
| `width` / `height` | int | 派生后尺寸（crop 变小，其余不变） |
<!-- 2026-08-30, tianqi, transform rows copy source audit columns -->
| `family` / `arch` / `generation_type` / `content_type` / `original_format` / `phash` | | 从 source 复制（旧 transform JSONL 缺这几列也能读） |
<!-- end -->

校验：core 字段缺失/多余/取值非法会在构建时报错。`family` 等新列可缺省为 null。
旧 CIFAKE source JSONL（没有 phash）仍然能 `read_jsonl`。

训练过滤（`filter_train_rows`）：

```python
from src.transforms.manifest import read_jsonl, filter_train_rows

train = read_jsonl("data/manifests/source_sid_train.jsonl", "source")
holdout = read_jsonl("data/manifests/source_demo_val.jsonl", "source")
kept, leaks = filter_train_rows(train, holdout=holdout)
# kept: split=train 且不是 partial_manipulation，且 phash 未撞上 holdout
```

`build_source` 会写 phash / original_format。假图示例：
`--generator sd15 --family diffusion --arch unet --generation-type t2i`。
真图的 generator/family/arch/generation_type 一律写成 null。

## 6. 用法

## 6. 用法

```bash
# 0) 一次性环境
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# 1) 目录树 → source manifest（label 按父目录；官方演示集按 real/fake 命名即可）
python -m src.transforms.build_source --root data/cifake/test \
  --dataset cifake --split test \
  --out data/manifests/source_cifake_test.jsonl

# 1b) 官方演示集（带 DO_NOT_TRAIN）：评测索引放行，作 train 会被拒绝
python -m src.transforms.build_source --root data/val \
  --dataset demo_wildfake --split val \
  --out data/manifests/source_demo_val.jsonl

# 2) 冒烟：2 个档位 × 每档 200 张（CLI 默认 splits=val,test,unseen，train 不参与）
python -m src.transforms.build \
  --source-manifest data/manifests/source_cifake_test.jsonl \
  --out-manifest data/manifests/transforms_eval.jsonl \
  --settings blur_s10,jpeg_q50 --limit-per-setting 200

# 3) 全量：14 档位 × 默认 splits 全部 source（同种子 ⇒ 逐字节可复现）
python -m src.transforms.build \
  --source-manifest data/manifests/source_demo_val.jsonl,data/manifests/source_cifake_test.jsonl \
  --out-manifest data/manifests/transforms_eval.jsonl
```

`build_source` 常用参数：`--generator flux1-dev`、`--label 1`（强制整树标签，
Flux 出图用）、`--hash-content`（跨数据集去重时用内容哈希）。

注意 `--limit-per-setting` 取的是 split 过滤后的**前 N 行**（CIFAKE 目录按
REAL/FAKE 分层排列，冒烟集可能只有单一标签）——它只用于快速冒烟，不要把
冒烟产物当评估集用；全量构建不用这个参数。

## 7. 与训练 / 评估的接口

- **评估侧**：从 `transforms_eval.jsonl` 取 14 个档位条件、从 source manifest 取
  clean 条件，逐条件算 AUC；bad case 按 `transform_key × generator × label` 落 JSONL。
- **训练侧**：**不要预生成训练增强**——用 `random_augment` 在线随机施加（官方要求
  "apply these randomly during training, they simulate the real-world
  redistribution pipeline"）：

  ```python
  from src.transforms.augment import random_augment
  img, info = random_augment(img, rng, p_clean=0.2)  # info 记录实际施加的 op/参数
  ```

  默认在官方离散档位里随机（与评测网格同分布），`continuous=True` 改为范围内
  连续采样；`p_clean` 保留一部分无变换样本；`op_weights` 可对 6 类算子重新配比；
  `chain_jpeg_p` 可在任意变换后追加一次随机质量 JPEG 重编码（更贴近社媒再压缩
  管线，默认关闭）。真/假图都要加增强，避免 "真=有损、假=无损" 的捷径
  （data.md 同款提醒）。增强在**原生分辨率**上做，模型侧 resize 放在它之后。
- **数据侧**：source manifest 是唯一入口，选数据的同学按第 5 节字段产出即可接入。

## 8. 数据集落库注意（给选数据的同学）

- **CIFAKE**：32×32。blur σ2 / JPEG q30 在这个分辨率上破坏力比大图更狠，属正常，
  不要为它单独放宽参数。目录 `train/{REAL,FAKE}`、`test/{REAL,FAKE}`，label 按父目录。
- **SID_Set**（HF parquet，~140GB）：三分类 0=real / 1=fully synthetic / 2=tampered。
  入库时 `label` 仍是 0/1；tampered 用 `content_type=partial_manipulation`，
  **不要当主力 fake**。SID tampered 底图来自 COCO val：phash **不一定**撞上
  `data/val/real`（局部编辑会改 hash），所以排除靠 content_type，外加
  `filter_train_rows(train, holdout=official_val)` 抓整图拷贝。
- **Flux 产物**：`--dataset flux_gen --split unseen --generator flux --family diffusion --generation-type t2i --arch flow`；
  目录与训练数据物理分开；是否拿少量当补充训练集由算法侧决定（data.md：可补充，不可当主训练集）。
- **演示集** `data/val/`（COCO val2017 + DALL·E Advanced）：带 `DO_NOT_TRAIN`——
  当 `--split train` 索引会被拒绝；评测索引（其他 split）放行并打印 stderr 提示。
  该目录绝不能用于训练，训练 loader 也不得扫描。
