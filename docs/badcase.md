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
4. `badcase_gallery.py` 把错误渲染成可直接翻看的 **HTML 画册**（见下节）。

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

## 画册：肉眼直接看 bad case（`scripts/badcase_gallery.py`）

把 pred JSON 渲染成**自包含 HTML 画册**——FP/FN 分区、按置信度排序（最糟的
排最前）、缩略图 base64 内嵌；单文件产物，scp 回本地浏览器打开即可，不需要
服务器在线。

<!-- 2026-08-31, tianqi, current high-AUC galleries + do not open on Vast Jupyter -->
组员看画册：把 `outputs/tables/badcase_galleries/` 和 `badcase_compare/` **下到本机**，打开 `index.html`。Vast Jupyter / 文件浏览器点不开。步骤见 [badcase-galleries.md](badcase-galleries.md)。当前入口含 fuse 400、last4、D3 mix 等。
<!-- end -->

```bash
# 服务器实战：compare_spec 里某模型 clean 条件的 bad case（注意 --predict-root）
python scripts/badcase_gallery.py \
    --pred /workspace/experiments/compare_spec/official_val/sid_dinov2/pred_clean.json \
    --split official_val \
    --predict-root /workspace/experiments/compare_spec/official_val/images/clean \
    --condition clean --max-per-type 60 \
    --out /workspace/experiments/yun_eval/galleries/sid_dinov2_clean.html

# 拉回本地（服务器地址一律用占位符，真实 IP/端口/账号不进仓库）
scp -P <ssh端口> <user>@<gpu服务器>:/workspace/experiments/yun_eval/galleries/*.html .
```

| 参数 | 说明 |
|---|---|
| `--pred` / `--out` | 预测 JSON / 输出 HTML（必填） |
| `--split` / `--image-dir` | 标签来源，二选一（与 run_badcase 同一套） |
| `--predict-root` | **最常见坑**：pred 路径的根目录——predict.py 跑在哪个输入目录就指哪个，不传默认 = split 根目录。对 run_eval 物化树（`images/jpeg_q50/…`）跑的预测**必须**传物化目录，否则 `no predictions matched labels`。缩略图也按它做候选回退（raw → `predict_root`/rel → split 根/rel） |
| `--manifest` | source manifest JSONL，传了卡片带 generator 元数据 |
| `--condition` / `--threshold` | 条件标签 / FP-FN 判定阈值（默认 0.5） |
| `--max-images` / `--seed` | 平衡抽样（正整数；0 或负数直接报错，两个 badcase CLI 语义一致） |
| `--max-per-type` / `--thumb` | 每类最多画几张（默认 60）/ 缩略图最长边（默认 384） |

产物：单文件 HTML。FP 区（真图误判为 AI，红框，置信度最高的排最前）+ FN 区
（假图漏检，橙框，置信度最低的排最前），卡片带 pred、标签、条件、generator、
完整路径；超出 `--max-per-type` 的只计数不渲染，计数显示在区块标题。

**缩略图缺失不会静默**：某张图打不开（物化树被清理、路径失效等）就渲染占位符，
stdout 与 HTML meta 行都带 `thumbnails ok/shown` 计数；**全部缺失时额外打 stderr
告警**，先检查 `--predict-root` 与文件是否还在。

## 从画册生成 Error Analysis Note

`analyze_badcase_galleries.py` 读取两个或更多全量 gallery，计算 FP/FN、错误分数分布和
跨模型错误重合度，同时输出可审计的 JSON 与 Markdown 初稿：

```bash
python scripts/analyze_badcase_galleries.py \
  --gallery 'clipb16_sid=<gallery-dir>/clipb16_sid_full_clean/index.html' \
  --gallery 'clipl14_sid=<gallery-dir>/clipl14_sid_full_clean/index.html' \
  --auroc clipb16_sid=0.9655 --auroc clipl14_sid=0.9766 \
  --n-real 5000 --n-fake 8843 \
  --out-md docs/error_analysis_backbone_ablation.md \
  --out-json outputs/tables/badcase_backbone_ablation.json
```

gallery 只含错例，没有完整预测分布，因此 `n_real/n_fake` 必须显式提供；AUROC 必须来自
同一数据、同一 condition 的评测表。脚本不会根据图片自动猜视觉归因，Markdown 中的
人工标注协议用于后续肉眼复核。拿到 last4/fuse gallery 后只需替换 `--gallery` 和
`--auroc` 即可重跑最终版。

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
- 画册（上节）提供原始图浏览；模型可解释性可视化（Grad-CAM/频域图）尚未做，
  误差归因目前靠人工看图 + 分组统计。
