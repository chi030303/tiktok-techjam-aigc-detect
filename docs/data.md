# 数据与权重（不进主仓）

官方没有计分测试集。我们自己训、自己做变换表；主办方可能拿私有图跑 `predict.py`。

## 共享盘 vs 个人 clone（Vast）

图和骨干只放一份，不要拷进 `/workspace/<who>/`：

| 路径 | 内容 |
|---|---|
| `/workspace/data` | `sid_set` / `cifake` / `val` / `evalgen` / `wildfake`（全量，可选） |
| `/workspace/models` | CLIP / ResNet / DINOv2 |
| `/workspace/experiments` | 各实验的 ckpt、log（按实验名） |
| `/workspace/<who>/tiktok-techjam-aigc-detect` | 代码 + `experiments/<name>/recipe.yaml` |

`DATA_ROOT` / `MODELS_ROOT` / `EXP_ROOT` 见 `.env.example`。新实验：复制 `experiments/_template/`，改名，写 `recipe.yaml`，然后 `python scripts/run_experiment.py experiments/<name>/recipe.yaml`。

## 禁止训练

| 集合 | 内容 | 本地路径 |
|---|---|---|
# 2026-08-29, tianqi, official demo val path is data/val not demo_wildfake
| 演示验证集（题面） | COCO val2017 ~4998 真图 + DALL·E Advanced 8843 假图 | `data/val/`（`real/` + `fake/`） |
# end
| EvalGEN | Flux / GoT / Infinity / OmniGen / NOVA，约 5.5 万张 | `data/evalgen/` |

`data/val` **就是题面那份 Demonstration 集**：不算分、只用来看迭代和 robustness 表。**不是** 主办方隐藏打分测试集。目录内 `DO_NOT_TRAIN`。训练 loader 不得扫描 `data/val/`、`data/evalgen/`、`data/demo_wildfake/`。

# 2026-08-30, tianqi, full WildFake is a train source; only the demo subset stays hold-out
**WildFake 全量**不是演示集，**可以训**，但必须先丢掉和 `data/val` 重叠的部分（COCO val2017、DALL·E Advanced、phash）。不要给全量树打 `DO_NOT_TRAIN`。EvalGEN 仍然禁止训练。

# 2026-08-30, tianqi, do not snapshot the whole WildFake; teammates own ADM/DDPM/Imagen
按维度下，不要再跑 `scripts/download_wildfake.py`（会整库 snapshot，和同学抢带宽/盘）。共享目录：

| 维度 | 状态 | 路径 |
|---|---|---|
| ADM | 已就位 ~15.5 万张 | `data/wildfake/cross_arch/adm/` |
| DDPM | 已就位 ~7.7 万张 | `data/wildfake/cross_arch/ddpm/` |
| Imagen / UNet 等 | 同学在下，不要重复拉 | 等他们写进 `data/wildfake/` |

训之前仍要排除 `data/val` 重叠。
# end

```bash
python scripts/download_official_val.py          # -> data/val  or /workspace/data/val
python scripts/download_evalgen.py              # extra hold-out, not official val
python scripts/download_backbones.py             # CLIP-B/16, CLIP-L/14, ResNet-50, DINOv2-L/14
```

## 建议训练 / 冒烟

| 数据 | 用途 | 下载 | 本地 |
|---|---|---|---|
| CIFAKE | 冒烟、短实验 | [Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) | `data/cifake/` |
| SID_Set | 主训练（优先） | [HF saberzl/SID_Set](https://huggingface.co/datasets/saberzl/SID_Set) | `data/sid_set/` |
| WildFake 分维 | 训练（排除 `data/val` 重叠后） | 同学在下；不要再 snapshot 全库 | `data/wildfake/cross_arch/` |

公开、有授权即可。标签只有 **真 / 假**，不要物体类别。人货背景混在一起训。

## JPEG 不是「只收 jpg」

题面 **JPEG Compression** = 把图按质量 90/70/50/30 **再编码一遍**（社交再压缩），不是要求原文件必须是 `.jpg`。

训练：在线随机 JPEG（真假都做），避免真=有损、假=无损的捷径。  
评测：同一批图按题面档位 **生成一次、固定下来**（`data/transforms/` 可 gitignore）。

其它变换同样是像素操作：blur / resize / noise / jitter / center crop 80%。

<!-- 2026-08-29, tianqi, point at run_eval.py for the robustness deliverable -->
## 评估管线

官方接口仍是 `predict.py`（目录 → JSON）。评测脚本只 **读** `data/val` / `data/evalgen` / `data/cifake/test`，不会进 train loader。

```bash
# 已有 pred.json：对标签打分（acc / AUROC / FPR / FP·FN）
python scripts/run_eval.py score --pred outputs/pred.json --split official_val

# 日常粗表（clean vs JPEG-50 vs crop 80%），先用子集
python scripts/run_eval.py robustness --split official_val --conditions daily --max-images 400 --ckpt checkpoints/best.pt

# 提交用的全档位表（题面全部 JPEG / blur / resize / noise / jitter / crop）
python scripts/run_eval.py robustness --split official_val --conditions full --ckpt checkpoints/best.pt

# 把变换图冻到盘上（可选；robustness 默认写到实验 work dir）
python scripts/run_eval.py materialize --split official_val --conditions daily --max-images 400
```

# 2026-08-29, tianqi, eval condition names now match src/transforms/spec.py
`--conditions daily` = `clean,jpeg_q50,crop_p80`（[ops.md](ops.md) 每天那张粗表）。`--conditions full` = clean + 官方 14 档（`docs/transforms.md`）。像素算子和种子以 `src/transforms` 为准，不要再用 `center_crop_80` / `jitter_m20`。表写到 `outputs/tables/*.csv`（可进 git）和 `*.md`。EvalGEN 几乎全是假图，AUROC 会是空的，看 recall / mean_pred。
# end

# 2026-08-30, tianqi, stream full-val + EvalGEN real-pool (do not copy 14k x 15 to disk)
全量官方 val / EvalGEN 用 `scripts/run_full_eval.py`：变换在 Dataset 里做，不落 15 份拷贝。默认打 CLIP-B/L SID aug 的 `best.pt`。

```bash
# 全量 official_val，先 clean（~1.4 万/模型），再决定要不要 14 档
CUDA_VISIBLE_DEVICES=1 python scripts/run_full_eval.py --split official_val --conditions clean
CUDA_VISIBLE_DEVICES=1 python scripts/run_full_eval.py --split official_val --conditions full

# EvalGEN：假图来自 data/evalgen，真图默认 SID validation（on-the-fly parquet）
CUDA_VISIBLE_DEVICES=1 python scripts/run_full_eval.py --split evalgen --reals sid_val --conditions clean
# 真图改用 COCO val（和官方 val 同一半）或 WildFake Real（会丢掉 val2017 重叠）
python scripts/run_full_eval.py --split evalgen --reals coco --conditions clean
python scripts/run_full_eval.py --split evalgen --reals wildfake --max-fakes-per-gen 200
```

EvalGEN 的 AUROC 是「该生成器假图 + 所选真图池」。SID val 真图和 CLIP-SID 同域；COCO 真图方便和官方 val 对比。不要用 SID **train** 真图（泄漏）。
# end
<!-- end -->

## 权重

| 模型 | 限制 | 存放 |
|---|---|---|
| CLIP / ResNet 等骨干 + 分类头 | 提交检测器 **&lt;2B** | `models/` 或 `checkpoints/`（gitignore） |
| Flux 等生成器 | 只允许当 **造样本工具**，不能当提交模型 | 不要进 git |

线性头 / ResNet 权重用网盘或 HF 私有仓库同步，README 写下载命令。Flux 出的图若使用，当作补充数据，**不要当主训练集**（否则只认 Flux）。

## 进 git 的数据相关文件

- 下载脚本、路径约定、小 CSV 表（如 `outputs/tables/*.csv` 可择要提交）
- 不要：原图、演示集、成百张 FP/FN 大图（网盘链接即可）
