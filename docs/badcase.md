# Bad case 收集与统计管线

> Owner：数据变换/评测（feat/badcase-pipeline）。对接群任务「评估 2：搭建收集统计
> bad case 的管线，为了后续分析工作做准备」。产物直接服务交付物 5（误差分析笔记）
> 和风控的阈值/FPR 叙事。

## 它解决什么

`run_eval.py score` 只能倒排 **top-50** FP/FN，且不带生成器信息。本管线补齐三件事：

1. **全量**落盘每一个 FP/FN（不截断）；
2. 每条错误带上 **generator / source_dataset / condition** 元数据（来自
   `src.transforms.build_source` 产的 manifest，见 [transforms.md](transforms.md) §5）；
3. 按 **condition × generator × source_dataset** 聚合错误率，直接回答
   「模型在哪类生成器/哪个变换档位上翻车」——误差分析笔记的核心素材。

## 用法

```bash
# 官方演示集 + generator 元数据（推荐：predict 先跑 official_val，再喂进来）
python predict.py data/val outputs/pred.json --ckpt checkpoints/best.pt
python scripts/run_badcase.py --pred outputs/pred.json --split official_val \
    --manifest data/manifests/source_demo_val.jsonl

# 任意已物化的变换目录（condition 手动标注）
python scripts/run_badcase.py --pred outputs/pred_evalgen.json \
    --image-dir data/evalgen --condition clean

# 关键参数：--threshold（阈值归风控拍板）--max-images（平衡抽样）--worst-k（summary 里保留的极错样本数）
```

## 产物 schema（默认 `outputs/badcases/<split>/`）

**badcases.jsonl** —— 每行一个错误：

| 字段 | 说明 |
|---|---|
| `error_type` | FP（真图被判为AI）/ FN（AI图被判为真） |
| `image_path` / `pred` / `label` | 定位与得分 |
| `condition` | transform_key（有 manifest 时）或 `--condition` 默认值 |
| `generator` / `source_dataset` | manifest 元数据；真图 `generator` 固定为 `real`（按定义无生成器），假图元数据缺失时为 `unknown` |

**badcase_stats.csv** —— 长表：`group_type ∈ {overall, condition, generator, source_dataset}`，
每行 `n_images / n_real / n_fake / n_fp / n_fn / fpr / fnr`，可直接透视/贴进 robustness 表附录。
分母与 `src.eval.metrics.binary_metrics` 同名同义：**`fpr = n_fp / n_real`（只算真图）、
`fnr = n_fn / n_fake`（只算假图）**；generator 桶里全是假图，`fpr` 恒为 0 属正常。

**badcase_summary.json** —— 总量与比率（同上口径）、三组聚合、`worst_fp`（置信度最高的真图误报，
最伤用户信任的那批）与 `worst_fn`（置信度最低的漏检漏网），外加 `binary_metrics`
（acc/auroc/precision/recall/fpr，复用 `src.eval.metrics`）。

## 条件命名对照（manifest transform_key ↔ robustness condition）

manifest 侧（`src/transforms/spec.py` 的 transform_key）与 robustness 表侧
（`src/eval/transforms.py` 的 condition）目前是**两套命名**。本管线 `by_condition`
取的是 manifest 的 transform_key；jpeg 四档与 `jitter_p20` 两边一致，其余跨表引用时
按下表换算（统一命名是后续工程）：

| transform_key（本管线） | robustness condition |
|---|---|
| `blur_s05 / blur_s10 / blur_s20` | `blur_s0.5 / blur_s1.0 / blur_s2.0` |
| `resize_s05 / resize_s025` | `resize_x0.5 / resize_x0.25` |
| `noise_s002 / noise_s005 / noise_s010` | `noise_s0.02 / noise_s0.05 / noise_s0.10` |
| `crop_p80` | `center_crop_80` |
| `jitter_p20` | `jitter_p20`（eval 侧另有 `jitter_m20`） |

## 与其他模块的关系

- 预测来源：`predict.py`（官方契约），本管线不碰模型。
- 标签来源：`src.eval.labels`（目录标签，与 `run_eval.py score` 完全同一套 join 语义）。
- 元数据来源：`build_source.py` 产的 source manifest —— **想让统计带 generator 维度，
  数据入库时就要跑 manifest**；没有 manifest 也能跑（generator=unknown）。
- 阈值：`--threshold` 默认 0.5 只是占位，正式口径由风控定（roles.md）。

## 局限

- 元数据 join 把 pred 路径与 manifest 路径各自归一到绝对形式（原样 + 相对
  `predict_root`/`src_root` 解析）后再匹配，因此 `predict.py` 用绝对路径跑、
  manifest 是相对路径（或反过来）也能挂上；同文件但路径标识不同（硬链/跨挂载）
  的极端情形匹配不到，计入 `unmatched_metadata` 并标 `unknown`——该计数同时
  打在 CLI 汇总行里，比例异常时先查 manifest 路径形式。
- 目前不带 per-image 可视化（Grad-CAM 之类），如误差分析需要再议。
