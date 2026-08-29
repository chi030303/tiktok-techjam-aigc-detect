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
| `generator` / `source_dataset` | manifest 元数据，缺失时为 `unknown` |

**badcase_stats.csv** —— 长表：`group_type ∈ {overall, condition, generator, source_dataset}`，
每行 `n_images / n_fp / n_fn / fp_rate / fn_rate`，可直接透视/贴进 robustness 表附录。

**badcase_summary.json** —— 总量与比率、三组聚合、`worst_fp`（置信度最高的真图误报，
最伤用户信任的那批）与 `worst_fn`（置信度最低的漏检漏网），外加 `binary_metrics`
（acc/auroc/precision/recall/fpr，复用 `src.eval.metrics`）。

## 与其他模块的关系

- 预测来源：`predict.py`（官方契约），本管线不碰模型。
- 标签来源：`src.eval.labels`（目录标签，与 `run_eval.py score` 完全同一套 join 语义）。
- 元数据来源：`build_source.py` 产的 source manifest —— **想让统计带 generator 维度，
  数据入库时就要跑 manifest**；没有 manifest 也能跑（generator=unknown）。
- 阈值：`--threshold` 默认 0.5 只是占位，正式口径由风控定（roles.md）。

## 局限

- 元数据 join 按 path 字符串精确匹配（posix 归一化后），manifest 覆盖不到的图计入
  `unmatched_metadata` 并标 `unknown`，不会失败也不会静默混入假分组。
- 目前不带 per-image 可视化（Grad-CAM 之类），如误差分析需要再议。
