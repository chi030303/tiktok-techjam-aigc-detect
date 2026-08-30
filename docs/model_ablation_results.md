# 模型线消融结果(Model-Track Ablation)

2026-08-30, yun。对应 `DATA_ABLATION_PLAN.md` 第十三节定义的"模型线"——固定同一份 SID 0/1 数据集，只改模型侧变量（backbone 解冻、consistency、dual branch、336 分辨率），与数据线严格解耦。

本文档是这条线的主表 + 每格结论，对应数据线文档里 Master Ablation Table 的角色。

## 一、任务范围

- **任务1**：CLIP ViT-B/16、CLIP ViT-L/14 在 CIFAKE 全量数据（10万train/2万test）上训练线性探测头（linear probe），backbone 冻结。
- **任务2**：CLIP-B/16 在 SID_Set 全量数据（14万train/2万val）上的四组模型消融，共同固定：CLIP-B backbone、SID 0/1 标签、在线官方6种变换随机增强（除 Exp2 外）、3 epoch、输入 RGB（除 Exp3 的频域分支外）。

## 二、代码与数据位置

| 用途 | 路径 |
|---|---|
| SID 在线增强数据加载器 | `src/data/sid.py` |
| 频域变换（highpass / FFT） | `src/data/freq.py` |
| paired-view 数据加载（Exp2） | `src/data/sid_paired.py` |
| 双路数据加载（Exp3） | `src/data/sid_dual.py` |
| 支持 head_kind / 336插值 的探测头模型 | `src/models/sid_linear_probe.py` |
| 部分解冻模型（Exp1） | `src/models/partial_unfreeze_probe.py` |
| 双路融合模型（Exp3） | `src/models/dual_branch_probe.py` |
| baseline / Exp4 训练循环 | `src/train/sid_online.py` |
| Exp1 训练循环（不依赖 feat-cache） | `src/train/partial_unfreeze_train.py` |
| Exp2 一致性损失训练循环 | `src/train/consistency_train.py` |
| Exp3 双路训练循环 | `src/train/dual_branch_train.py` |
| 模型线专用 predictor（image_size / 部分解冻 ckpt） | `src/sid_infer.py` |
| 入口脚本 | `scripts/run_experiment_sid.py`（baseline/Exp4）、`scripts/run_experiment_finetune.py`（Exp1）、`scripts/run_experiment_consistency.py`（Exp2）、`scripts/run_experiment_dualbranch.py`（Exp3） |
| 鲁棒性对比评测 | `scripts/run_ablation_compare.py`（单视图 ckpt）、`scripts/eval_dualbranch.py`（Exp3 双视图专用） |
| Recipe | `experiments/clipb16_linear_sid_aug/`（baseline）、`experiments/clipb16_linear_sid_res336/`（Exp4）、`experiments/clipb16_linear_sid_unfreeze{2,4}/`（Exp1）、`experiments/clipb16_linear_sid_consistency/`（Exp2）、`experiments/clipb16_linear_sid_dualbranch/`（Exp3） |
| 主表 | `outputs/tables/model_ablation_master.{csv,md,json}` |
| CIFAKE 鲁棒性表（任务1） | `outputs/tables/cifake_test_task1_cifake_compare.{csv,md}` |

**为什么另建 `sid_online.py` / `sid_linear_probe.py` / `sid_infer.py`，而不是直接扩展 `src/train/loop.py` / `src/models/linear_probe.py` / `src/infer.py`？**
这条分支要求不改动其他实验已经依赖的现有文件，所以把 SID 在线增强训练所需的能力（`image_size`、`head_kind`、位置编码插值、按 source/input_mode 命名的 feature cache）整体放进平行的新模块，只新增文件、零改动。两边 API 基本一致，后续如果要合并回主线，直接把 `sid_online.py` 的能力合进 `loop.py` 即可。

## 三、任务1结果：CLIP-B/16 vs CLIP-L/14 @ CIFAKE

Backbone 冻结、线性头、**无在线增强**（纯 clean 图片训练）。

| 模型 | clean acc | jpeg_q30 | blur_s20 | resize_s025 | crop_p80 |
|---|---|---|---|---|---|
| CLIP-B/16 | 0.940 | 0.870 | **0.680** | **0.684** | 0.862 |
| CLIP-L/14 | 0.956 | 0.876 | **0.552** | **0.540** | 0.916 |

**结论**：两个模型 clean 准确率都不错（94~96%），CLIP-L/14 全面略优于 B/16。但**两者在 blur/resize 这类分辨率退化的攻击下鲁棒性都很差**（准确率跌到55~68%），且模型越大掉得越狠（L/14 在 blur_s20/resize_s025 上反而比 B/16 更差）。原因很直接：这两个 checkpoint 训练时**完全没有做在线数据增强**，只见过干净图片。对比后面 SID 线用了在线6种官方变换增强的 baseline（blur_s20 86.8%、resize_s025 86.2%），差距非常悬殊——这本身就是一个值得写进最终报告的发现：**在线鲁棒性增强比堆大模型更能决定 blur/resize 抗性**。

## 四、任务2结果：CLIP-B + SID 四组模型消融

共同 baseline：`clipb16_linear_sid_aug`（CLIP-B/16、SID 14万/2万、在线6种官方变换随机增强、3 epoch、线性头、224分辨率）。Clean SID 验证集 acc = 99.27%。

鲁棒性评测统一用 official_val（COCO真图+DALL·E假图，与训练域不同，检验泛化），500张均衡采样，15种条件（clean+14种官方变换）。**只跑了一次、没有多种子重复，±1~2pt 的差异大概率是采样噪声**，只有量级明显更大或多个条件同向一致的信号才可信。

完整数据见 `outputs/tables/model_ablation_master.md`；下表摘录 official_val 上最关键的几档：

| 模型 | clean | jpeg_q30 | blur_s20 | resize_s025 | crop_p80 |
|---|---|---|---|---|---|
| baseline | .900/.970 | .872/.965 | .868/.979 | .862/.975 | .826/.959 |
| Exp4 336分辨率 | .898/.975 | .890/.967 | .828/.986 | .814/.980 | .830/.963 |
| Exp1a 解冻末2层 | .886/.985 | .886/.979 | .854/.992 | .848/.990 | .846/.981 |
| Exp1b 解冻末4层 | .862/.990 | .870/.985 | .810/.989 | .804/.986 | .772/.989 |
| Exp2 一致性损失 | .890/.967 | .880/.964 | .878/.970 | .878/.970 | .842/.956 |
| Exp3 频域双路 | .888/.970 | .870/.966 | .856/.976 | .842/.972 | .816/.960 |

（每格 `accuracy/AUROC`）

### Exp4 —— 分辨率 224 vs 336：价值有限

假设：336 多一些 patch，能在 crop/resize 之后保留更多细节。

结果：AUROC 几乎全面小涨（+0.3~1.4pt），但按判断标准重点看的三档：jpeg_q30 accuracy 涨了（+1.8pt），**resize_s025 反而掉了4.8pt，blur_s20 掉了4pt**。

**结论**：336 提升了排序能力（AUROC），但没有解决"先破坏分辨率再攻击"这类场景的实际判别准确率，价值有限，不建议作为默认分辨率。

### Exp1 —— 部分解冻 + 差分学习率：不是过拟合，是阈值/校准漂移

假设：冻结特征对 JPEG/模糊这类非语义痕迹不够敏感，解冻末几层能让表示适应这个任务。

结果：两个变体的 AUROC 都大幅上涨（解冻4层达到0.990，全场最高），但 accuracy（固定0.5阈值）在几乎所有条件上**同步下降**，尤其解冻4层在 crop_p80 上掉了5.4pt。层数越多（4层vs2层），accuracy 掉得越多。

**结论**：这不是"clean涨robust掉"的经典过拟合信号（clean 也没涨），而是微调后模型输出的概率分布整体偏移，0.5 阈值不再是最优判定点。排序能力（AUROC）确实变强了，但需要在验证集上重新选阈值才能把这个增益兑现成准确率提升。**建议**：若要采用解冻，必须配一次阈值重新校准（threshold recalibration），否则线上按 0.5 判定反而会变差。

### Exp2 —— Paired-view 一致性损失（λ=0.3）：四组里唯一完全达标

假设：强迫同一张图的 clean/corrupted 两种视角输出接近，防止模型靠"是否被处理过"这类捷径分开学习。

结果：clean accuracy 基本持平（-1pt，在"可接受"范围内），而 jpeg_q30、blur_s20、resize_s025、crop_p80 **全部同步小幅上涨**（+0.8~+1.6pt）。clean 与最差条件（crop_p80）的落差从基线的 7.4pt 收窄到 4.8pt。

**结论**：完全符合预设的成功标准（clean 持平、robust 涨、worst-case 落差收窄）。四组实验里最干净的正向结果，建议优先考虑吸收进主训练配方。

### Exp3 —— RGB + 频域（highpass）双路：没有踩捷径，但也没有实质收益

假设：生成痕迹部分存在于频谱/高通残差里，加一路频域分支能补充语义塔看不到的信息。

**捷径检测结论**：判定标准是"blur/resize 上 AUROC 暴涨、jpeg/crop 不涨 = 踩了和旧 FFT 实验一样的捷径"。实测：所有15个条件的涨跌幅度都在 baseline 的 ±2pt 以内，没有出现 blur/resize 独涨而 jpeg/crop 不动的分裂模式——**没有踩捷径**。

但代价是：所有指标都和 baseline 几乎无统计可辨的差异（clean acc 88.8% vs 90.0%，其余条件同理），加的这个 highpass 分支目前**没有带来可衡量的增益**，浅层CNN可能没学到超出 CLIP 语义特征之外的有效信号，或者 SID 的生成痕迹本身不在 highpass 残差这个频段里。

**结论**：安全（未踩捷径）但无效。如果要继续这个方向，下一步应该换成 fft_mag 分支（`src/data/freq.py` 里已有 `fft_mag_rgb`）或加大浅层CNN容量，而不是直接放弃频域信号这条路。

## 五、跨实验对比：谁最值得采纳

按"clean 是否保持 + robust 是否真涨（看 accuracy 不只看 AUROC）"排序：

1. **Exp2 一致性损失**——唯一 accuracy 口径下四项 robust 条件同步上涨、clean 基本不掉的实验，建议采纳。
2. **Exp1（解冻）**——AUROC 显著提升但需要配阈值重新校准才能兑现，暂不建议直接上线，可作为"配合校准"的后续方向。
3. **Exp4（336分辨率）**——价值有限，不建议作为默认改动。
4. **Exp3（频域双路）**——安全但当前配置无效，需要换频域表示或加大容量才有继续投入的价值。

## 六、已知局限

- 鲁棒性评测都是 n=500 单次跑，没有多种子重复，小幅度差异（<2pt）不完全可信。
- Exp2 只跑了 λ=0.3 一个值，没有做 λ∈{0.1,0.5} 的扫描对比。
- Exp1 的阈值漂移现象没有做重新校准实验验证"校准后 accuracy 能否真的追上/超过 baseline"，这是一个明确的后续 TODO。
- 任务1（CIFAKE）的两个 checkpoint 没有做在线增强训练，不能和 SID 线的 baseline 直接比较模型能力，只能说明"无增强训练"本身鲁棒性差这一点。
