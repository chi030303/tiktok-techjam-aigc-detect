# Error Analysis Note（交付物 5）

<!-- 2026-08-31, tianqi, DINOv2 note is last-night ablation; contest ranking is CLIP-B last4 / fuse -->
> **Contest ranking (31 Aug):** submit CLIP-B last-4 (official formula **0.990** / full 0.989) or fuse last4+D3 (**0.993**). This note still documents frozen **SID DINOv2** (formula ~0.90) as a backbone ablation. High-AUC galleries: `outputs/tables/badcase_galleries/index.html` — download and open locally, see [badcase-galleries.md](badcase-galleries.md).
<!-- end -->

> 评测对象：`sid_dinov2`（冻结 DINOv2 ViT-L/14 + 1,025 参数线性头，SID_Set 训练）。
> 数据：官方演示集（COCO val2017 真图 200 + DALL·E Advanced 假图 200，balanced）。
> 阈值 0.5（占位口径，最终由风控拍板）。复现命令见文末。分析人：Zyun，2026-08-31。

## 1. 总体表现

| 条件 | AUROC | FP@0.5 | FN@0.5 | FPR | Recall(fake) |
|---|---|---|---|---|---|
| clean | 0.904 | 10 / 200 | 88 / 200 | 0.050 | 0.560 |
| jpeg q50（最弱档） | 0.872 | 17 / 200 | 90 / 200 | 0.085 | 0.550 |
| **15 条件宏平均** | **0.897**（14 变换档） | — | — | — | — |

15 条件全表（AUROC）：clean 0.904；q90 0.911、q70 0.881、q50 0.872、q30 0.889；
blur σ0.5/1.0/2.0 = 0.905/0.905/0.905；resize 0.5×/0.25× = 0.902/0.872；
noise σ0.02/0.05/0.10 = 0.911/0.899/0.887；jitter 0.905；crop 0.902。
**最低档位 0.872（q50 与 rx0.25），波动带仅 ±0.02**——全档位平坦是本模型的核心竞争力。

## 2. 错误结构：这是一个「低误杀、高漏检」的模型

threshold=0.5 时 FP:FN = **10:88**。对照其他模型（同一批 400 张、同一阈值）：

| 模型 | FP | FN | 形态 |
|---|---|---|---|
| **sid_dinov2（提交模型）** | 10 | 88 | 漏检型 |
| sid_resnet | 28 | 90 | 漏检型 |
| dinov2_clean | 144 | 7 | **误杀型** |
| resnet_clean | 198 | 2 | **严重误杀型** |
| clipb16_aug | 143 | 62 | 混乱（见 §6） |
| dinov2_fft | 31 | 153 | 漏检型 |

**发现（阈值校准）**：`resnet_clean` AUROC 0.701 并不差，但 FP:FN=198:2——它的分数整体
偏高，0.5 的阈值把一半真图划成了 AI。**排序能力（AUROC）与决策点（阈值）是两件事**：
这类问题不需要重训，按业务误杀预算选阈值即可大幅改善 acc/FPR。`predict.py` 输出的
`pred` 是连续分数，下游应按自身 FPR 预算选工作点；若评审固定按 0.5 判，建议对 head 的
bias 做一次校准（排序不变、0.5 语义变优）。

## 3. 极端错例（top offenders）

- **最自信的漏检（FN，pred≈0.00）**：`ab99a9de651d…jpg`、`a864d2982818…jpg`、
  `9240403efbf4…jpg`（DALL·E Advanced）——模型**绝对确信是真图**的盲区，属「特征层面
  完全没有识别出生成痕迹」的硬错误，不是阈值问题。人工看图入口：
  `galleries/sid_dinov2_clean.html` FN 区（按置信度升序 = 最糟在最前）。
- **最自信的误杀（FP，pred≥0.92）**：`000000105264.jpg`(0.962)、`000000175387.jpg`(0.961)、
  `000000088040.jpg`(0.920)（COCO val2017）——均为真实摄影图，需人工归因（猜测与图像
  纹理的规则性/后处理有关），见 `galleries/sid_dinov2_clean.html` FP 区。

## 4. 条件维度：哪里变弱、哪里反而更好

- **弱档**：`jpeg_q50` 与 `resize_0.25×`（均 0.872，相对均值 -0.025）。中等压缩与最狠
  下采样会抹掉部分生成痕迹。
- **反直觉现象**：多个模型在轻度退化下 AUROC **高于 clean**（sid_dinov2 的 q90/n0.02
  达 0.911 > 0.904；resnet_clean 的 rx0.25 0.776 > 0.701）。解释假设：COCO 真图自带的
  锐利高频细节在 clean 时是干扰源，轻度退化「抹平」真图高频后真假更可分——即模型的
  判别线索主要在中低频段。该现象与模型协作验证过，可作为分析亮点写入报告。

## 5. 未见生成器泛化（EvalGEN，5 个 held-out 家族）

真图池：SID val reals（10,000）；假图：每家 400 张，共 12,000。clean 条件：

**总体（AUROC / fake recall）**：clipl14_sid 0.9950 / 0.895；clipb16_sid 0.9913 / 0.902；
unfreeze2 0.9911 / 0.842；unfreeze4 0.9888 / 0.782；
**sid_dinov2（提交模型，全量 55,298 假图 + 10,000 真图池）：AUROC 0.964 / recall 0.888 / FPR 0.091**。

**分生成器 recall（越低越危险）**：

| 模型 | Flux | GoT | Infinity | **Nova** | OmniGen |
|---|---|---|---|---|---|
| unfreeze4（400 张抽样） | 0.970 | 0.938 | 0.588 | **0.450** | 0.963 |
| unfreeze2（400 张抽样） | 0.988 | 0.980 | 0.740 | **0.545** | 0.958 |
| clipl14_sid（400 张抽样） | 0.975 | 0.973 | 0.920 | **0.635** | 0.970 |
| clipb16_sid（400 张抽样） | 0.985 | 0.985 | 0.880 | **0.688** | 0.973 |
| **sid_dinov2（全量 11,060 张/类）** | 0.946 | 0.928 | 0.827 | **0.777** | 0.959 |

- **提交模型 sid_dinov2 在最难的 Nova 家族上全场最佳**（0.777，且是在 11,059 张全量上
  测得，其余模型为 400 张抽样、不可直接比较但量级差距明显）；Infinity 上 CLIP-L14
  抽样值更高（0.920）值得注意。
- **Nova 是所有模型的共同盲区**（recall 0.45–0.78），Infinity 次之——跨生成器泛化的
  短板非常集中，指向「数据侧补 Nova/Infinity 风格假图」这一条最明确的改进路径。
- Flux/GoT/OmniGen 上所有模型 recall ≥ 0.93：对多数未见扩散生成器泛化良好。

## 6. 附加诊断（团队内部）

- `clipb16_aug / aug6` 多档位 AUROC < 0.5（最低 0.445）：方向性错误，疑似训练崩溃或
  标签对调，建议复查训练曲线与 pred 分布（画册可见其 pred 挤在同一侧）。
- `dinov2_fft`：clean 0.647 但 blur σ2.0 达 0.913——频域特征实现存疑，建议单独复查。

## 7. 权衡与改进建议

1. **阈值政策**（风控）：演示平台类场景优先低 FPR → sid_dinov2 @0.5 已是 FPR 5%；
   若要求更低，代价是 recall 进一步下降（见画册 FN 分布）。
2. **数据补充**（优先级从高到低）：① Nova/Infinity 风格假图；② i2i（重建式）假图——
   目前训练以 t2i 为主；③ 更多样的真图域（降低 FP）。
3. **鲁棒化**：q50/rx0.25 弱档建议训练增强向这两档加权（`src/transforms/augment.py`
   已支持 op_weights/continuous）。
4. **若强制 0.5 阈值**：对 head bias 做温度校准后重导出（不改变排序）。

## 8. 复现与产物

```bash
# 三件套（错误清单/聚合表/summary）
python scripts/run_badcase.py --pred <pred.json> --split official_val \
    --predict-root <物化目录> --condition clean --out-dir outputs/badcases/<model>
# 画册（肉眼复核）
python scripts/badcase_gallery.py --pred <pred.json> --split official_val \
    --predict-root <物化目录> --out gallery.html
# EvalGEN 分生成器（流式，无需物化）
python scripts/run_full_eval.py --split evalgen --reals sid_val --conditions clean \
    --ckpt sid_dinov2=/workspace/experiments/dinov2l_linear_sid/ckpts/best.pt
```

产物：`outputs/badcases/*`、`outputs/tables/*`、画册 `*.html`（服务器
`/workspace/experiments/yun_eval/galleries/`，7 份）。
