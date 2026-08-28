# 数据与权重（不进主仓）

官方没有计分测试集。我们自己训、自己做变换表；主办方可能拿私有图跑 `predict.py`。

## 禁止训练

| 集合 | 内容 | 本地路径 |
|---|---|---|
| 演示验证集 | COCO val2017 4998 真图 + DALL·E Advanced 8843 假图 | `data/demo_wildfake/` |

该集合 **不算分**，只用来看迭代、出 robustness 表、录 demo。目录内放空文件 `DO_NOT_TRAIN`。训练 loader 不得扫描此路径。

## 建议训练 / 冒烟

| 数据 | 用途 | 下载 | 本地 |
|---|---|---|---|
| CIFAKE | 冒烟、短实验 | [Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) | `data/cifake/` |
| SID_Set | 主训练（优先） | [HF saberzl/SID_Set](https://huggingface.co/datasets/saberzl/SID_Set) | `data/sid_set/` |
| WildFake 全量 | 可选，很大 | [ModelScope](https://modelscope.cn/datasets/hy2628982280/WildFake/summary) | 不要和演示子集混 |

公开、有授权即可。标签只有 **真 / 假**，不要物体类别。人货背景混在一起训。

## JPEG 不是「只收 jpg」

题面 **JPEG Compression** = 把图按质量 90/70/50/30 **再编码一遍**（社交再压缩），不是要求原文件必须是 `.jpg`。

训练：在线随机 JPEG（真假都做），避免真=有损、假=无损的捷径。  
评测：同一批图按题面档位 **生成一次、固定下来**（`data/transforms/` 可 gitignore）。

其它变换同样是像素操作：blur / resize / noise / jitter / center crop 80%。

## 权重

| 模型 | 限制 | 存放 |
|---|---|---|
| CLIP / ResNet 等骨干 + 分类头 | 提交检测器 **&lt;2B** | `models/` 或 `checkpoints/`（gitignore） |
| Flux 等生成器 | 只允许当 **造样本工具**，不能当提交模型 | 不要进 git |

线性头 / ResNet 权重用网盘或 HF 私有仓库同步，README 写下载命令。Flux 出的图若使用，当作补充数据，**不要当主训练集**（否则只认 Flux）。

## 进 git 的数据相关文件

- 下载脚本、路径约定、小 CSV 表（如 `outputs/tables/*.csv` 可择要提交）
- 不要：原图、演示集、成百张 FP/FN 大图（网盘链接即可）
