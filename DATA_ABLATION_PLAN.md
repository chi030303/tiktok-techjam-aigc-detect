# 数据线计划：Generator Coverage & Generalisation Ablation

## 一、数据线目标

当前 SID baseline 已经能够较好匹配官方 demonstration validation，因此数据侧不再以：

> “继续提高官方 validation AUC”

作为主要目标。

数据侧核心目标调整为：

> **以 SID 作为稳定 baseline，补充 SID 没覆盖或覆盖不足的生成器 / 生成架构，通过严格控制变量的数据消融实验，验证新增数据是否能够提高 unseen-generator generalisation，同时尽量不损伤已有的官方 validation 表现。**

最终需要回答三个问题：

1. **SID 缺什么？**
   - t2i / i2i 是否存在明显差异？
   - 是否需要 GAN？
   - Diffusion 内部不同 architecture 是否值得覆盖？
   - 自建数据是否真正增加了泛化能力？

2. **补这些数据有没有用？**
   - 官方 validation 是否基本保持？
   - unseen generator 是否提高？
   - robustness worst-case 是否改善？

3. **模型到底在学习生成痕迹还是 dataset shortcut？**
   - 内容
   - 长宽比
   - 格式
   - 色彩
   - 水印 / 黑边
   - dataset source

这些 bias 优先作为 evaluation slice 分析，而不是一开始人为把所有数据“平衡”掉。

---

## 二、硬约束

### 2.1 禁止进入训练的数据

以下数据严格隔离：

#### 官方 demonstration validation

`data/val/`

包括：

- COCO val2017
- DALL·E Advanced

只能用于 evaluation。

不得：

- 进入训练
- 用于 hard-negative mining
- 根据 badcase 反复针对 DALL·E 调训练数据
- 作为自建数据的 prompt / image source

#### EvalGEN

EvalGEN 全量保留作为 unseen-generator evaluation：

- FLUX
- GoT
- Infinity
- OmniGen
- NOVA

不得混入任何 train split。

#### SID tampered

SID：

`label = 2`

代表 partial manipulation / 局部重绘。

当前比赛核心是 image-level AIGC detection，因此：

**SID tampered 不作为 fake=1 加入当前主训练。**

尤其不能把 SID tampered 当作 i2i 数据来做 t2i/i2i 消融。

---

## 三、数据使用原则

### 3.1 标签

模型训练目标始终只有：

```text
label = 0  # real
label = 1  # fake
```

不使用：

- 人
- 猫
- 狗
- 车
- 风景

等物体类别作为训练目标。

generator / architecture 等信息全部作为 metadata。

---

### 3.2 Real 数据

新增实验中的 real pool 使用：

- OpenImages
- Flickr / 合规开放数据
- 自采自然图片
- 其他许可合适的自然图像

避免使用：

**COCO val2017**

以防和官方 demonstration validation 发生数据泄漏。

原则上不同消融格尽可能使用**同一池 real images**。

这样才能保证：

> 改变的是 fake generator，而不是 real domain。

---

### 3.3 Fake 数据

主要来源：

- SID full synthetic
- WildFake ADM / DDPM
- SD1.5 ≥512
- PixArt-Sigma
- Hunyuan-DiT
- GAN 数据
- 自建整图 i2i

优先补 SID 没覆盖的生成方式。

不为了扩大数据量重复生成 SID 已经充分覆盖的 FLUX。

---

## 四、Manifest 先行

在继续堆数据之前，先冻结 manifest schema。

**没有 metadata 的图片不进入正式实验池。**

现有字段：

```text
path
label
source_dataset
generator
split
width
height
```

至少补充：

```text
family
arch
```

推荐最终 schema：

```text
path
label

source_dataset
generator

family
arch

generation_type
split

width
height

original_format
phash
```

---

### family

这里建议不要把两个维度混在一个字段。

使用：

```text
gan
diffusion
```

而 t2i / i2i 单独放：

```text
generation_type
```

例如：

```text
generator = sd15
family = diffusion
arch = unet
generation_type = t2i
```

而不是把：

`t2i / i2i / gan / diffusion`

全部塞进 family。

这样后面做消融会干净很多。

---

### arch

主要取：

```text
unet
dit
flow
pixel
```

真图：

```text
null
```

GAN 如果后面需要进一步细分，可以记录具体 architecture；当前实验只需要能够区分 GAN family 即可。

---

### split

固定：

```text
train
val
test
unseen
```

其中：

> **unseen = 训练阶段完全没有出现过的 generator。**

一旦某 generator 被指定为 unseen，就不能再因为“训练数据不够”把其中一部分塞回 train。

---

## 五、实验基本协议

为了让数据消融能够真正说明问题，所有数据实验固定：

### Backbone

**CLIP-B**

数据侧不参与 backbone search。

---

### Training

第一阶段：

**Frozen CLIP-B + Linear Probe**

优先 feature cache，快速完成大量数据消融。

只有数据格胜出以后，才进入：

**online augmentation + 正式训练**

---

### 数据规模

每个主要实验格：

**8k real + 8k fake**

不同实验：

- 使用同一 real pool
- fake 总量保持一致
- 只改变当前研究的数据轴

generator-level 小实验：

**每 generator 约 1k–2k fake**

用于快速判断 generator / architecture 是否值得进入最终 mix。

---

### SID 全量

约 14 万规模 SID 不用于每个消融实验。

只用于最后：

**1–2 个最终数据配比。**

否则数据实验成本过高，而且变量难以控制。

---

## 六、统一 Evaluation Protocol

每个消融格至少输出三组指标。

### 6.1 官方 Validation

记录：

```text
AUC_clean
AUC_robust
formula
```

数据 mix 的基本要求：

> 相比 SID baseline，官方 validation 允许轻微下降，但目标控制在 ≤ 0.005 左右。

重点不是把 0.97 继续刷到更高，而是：

**不要为了 unseen improvement 明显破坏已有能力。**

---

### 6.2 Unseen Generator

两部分：

#### EvalGEN

由于 EvalGEN 是预留的生成器集合：

主要看：

```text
Recall
```

并按 generator 分开：

```text
FLUX
GoT
Infinity
OmniGen
NOVA
```

不要只看 aggregate。

#### 自建 unseen set

如果能够同时构造 real/fake：

记录：

```text
AUROC
```

核心问题：

> 新增 generator coverage 是否真正提高了 unseen generalisation？

---

### 6.3 Robustness

不用每次都把所有结果塞进主表。

完整评测仍然跑题面 transformation。

主消融表重点记录：

> **14 个 transformation conditions 中表现最差的 3 个。**

例如：

```text
JPEG30
Blur σ=2
Noise σ=0.10
```

这样可以直接观察：

新增数据到底是在提高 generalisation，还是同时改变了 transformation robustness。

---

## 七、核心消融 A：t2i vs i2i

目的：

> 判断整图 t2i 和整图 i2i 是否存在不同 detection signal，以及混合训练是否提高泛化。

保持：

```text
real = 同一 8k
fake = 8k
```

只改变 fake composition。

| 实验 | Fake |
|---|---|
| A1 t2i-only | 8k t2i |
| A2 i2i-only | 8k 整图 i2i |
| A3 t2i+i2i | 4k t2i + 4k i2i |

<!-- 2026-08-31, tianqi, 60-triplet hard set is a diagnostic, not 8k A2 -->
当前只有 **60 组三元组**（真图 + Gemini 重建 + Codex 重建，`source_id` 绑定）。
这不是计划里的 8k A2。做法：先对已有 SID/T2I ckpt 做 **配对打分**
`P(重建 > 同场景真图)`，再决定要不要小学习率微调。不要把 180 张当独立样本混进 D4。
# end

<!-- 2026-08-31, tianqi, scaled A-grid: A1=D2 t2i, A2=60+120 i2i, A3=D2+i2i fakes -->
可跑的 A 轴（frozen CLIP-B, feat cache, `scripts/a_grid.sh`，排在 D5 + `a2_i2i.sh` 之后）：

| 实验 | 实际规模 | 指标 |
|---|---|---|
| A1 t2i-only | D1 8k real + 自建 t2i（= D2 jsonl） | official 400 + EvalGEN clean + pair_acc |
| A2 i2i-only | 60 real + 120 i2i（hard60） | 同上；official 会差，看 pair_acc |
| A3 t2i+i2i | A1 再混入 120 i2i fake | 看相对 A1 的 official / EvalGEN / pair_acc |

不是 8k vs 8k。A2 只有 60 场景，不能当 submit。主看 **pair_acc** 和 EvalGEN，不要只看 Acc@0.5。
# end

### t2i

可以使用：

- SID full synthetic
- 或同规模自建 t2i

### i2i

必须是：

**whole-image img2img**

例如：

- SD img2img
- FLUX img2img

不能使用：

- local inpainting
- SID tampered

来冒充 i2i。

### 输出结论

最终回答：

> i2i signal 是否与 t2i 明显不同？

以及：

> mix 是否在基本保持官方 val 的同时提高 unseen？

---

## 八、核心消融 B：Diffusion vs GAN

目的：

> 判断增加少量 GAN 是否能够扩大 detector 的 generator coverage。

不建议做纯 GAN 主训练。

实验：

| 实验 | Fake |
|---|---|
| B1 | Diffusion only |
| B2 | Diffusion + 10% GAN |
| B3 | Diffusion + 20% GAN |

另一种更干净的实验：

> GAN 完全作为 unseen generator。

即：

**Train：Diffusion only**

**Test：GAN**

回答：

> 一个完全没有见过 GAN 的 detector 到底会掉多少？

这个结果本身就可以成为很好的 generalisation analysis。

---

## 九、核心消融 C：Diffusion Architecture Coverage

这是当前数据侧最重要的一组实验。

SID 已经覆盖 FLUX，因此重点不是继续增加 FLUX，而是补 architecture hole。

### C1 UNet

目标：

**SD1.5 ≥512 resolution**

来源：

- 开源已有数据
- 或自建 500–1000 / 1k–2k

注意：

**CIFAKE 不作为这一格。**

CIFAKE 32×32 与真实使用场景差异过大，只保留 smoke run 用途。

不能把 CIFAKE 当作 SD1.5 / UNet architecture coverage 的正式代表。

---

### C2 DiT

优先：

- PixArt-Sigma 0.6B
- Hunyuan-DiT 1.5B

使用已有开源数据或自建。

目标：

**1k–2k fake**

不需要为了这一格生成数万张。

---

### C3 Flow

使用：

**SID 中现有 FLUX subset。**

目标：

1k–2k fake。

不重新生成大量 FLUX。

GPU 资源优先用于 SID 当前没有覆盖的架构。

---

### C4 Pixel-space Diffusion

来源：

**WildFake ADM / DDPM**

目标：

1k–2k fake。

优先考虑作为：

- architecture coverage
- unseen evaluation

而不是大规模主训练来源。

---

### Architecture 小实验

保持：

```text
同一 8k real pool
```

每种 architecture 先使用：

```text
1k–2k fake
```

快速训练 frozen CLIP-B linear probe。

重点观察：

1. 官方 val transfer
2. EvalGEN transfer
3. cross-architecture transfer

然后再决定哪些 architecture 值得进入最终 8k fake mix。

---

## 十、核心消融 D：SID × 自建数据

这一组直接回答：

> 我们自己补的数据到底有没有价值？

设置三个格。

### D1 SID-only

当前 baseline。

作为 reference：

> 官方 demonstration 已约 0.97。

---

### D2 Self-built only

```text
8k open-source real
+
8k self-built fake
```

用于观察自建数据独立训练时的能力。

预期：

**官方 val 低于 SID 是正常的。**

不需要为了让 self-built 看起来更强而针对 DALL·E 调数据。

---

### D3 SID + Self-built

SID 作为 base。

新增：

**2k–8k fake mix-in**

重点补：

- SD1.5 / UNet
- DiT
- 必要时 pixel diffusion
- i2i

而不是重复补 FLUX。

#### 成功标准

理想结果不是：

`Self-built > SID on official val`

而是：

> **SID + Self-built ≈ SID on official val**

同时：

> **SID + Self-built > SID on unseen generators**

这就是数据线最重要的 positive result。

---

### D4 SID + nano_banana + PixArt + SDXL + GPT

# 2026-08-31, tianqi, D4 includes GPT; PixArt stands in for DiT; no Hunyuan
SID 作为 base，mixin 换成：

- **全部** nano_banana（Vertex 已落盘 ~1500）
- PixArt Sigma（`data_new/output/pixart_sigma_quality_v2` ~1500）——**潜空间 + DiT 的替代**，不使用混元
- SDXL full-refiner（`data_new/output/sdxl_full_refiner_v1` ~1500）+ ComfyUI 里额外的 sdxl
- GPT Image（`data_new/output/GPT` ~1500）

**不进 Hunyuan**：造数据同学明确混元输出不可用于训练。GPT 和官方 val DALL·E 同属 OpenAI 文生图，官方分可能偏乐观；**unseen 以 EvalGEN 为准，Nova 是硬家族**。不进 EvalGEN 图、不进 `data/val`。等权替换 SID FLUX。协议与 D3 相同：冻结 CLIP-B + online aug。

成功标准同 D3：官方 val 不掉、EvalGEN（尤其 Nova）改善。
# end

---

### D5 SID + D3 ∪ D4

# 2026-08-31, tianqi, D5 is the combined mix after D4 probe
D3 开源骨架（flux2 / sd35 / WildFake UNet 4k / ADM / DDPM）加上 D4 新 T2I（全量 nano、PixArt、自建 SDXL、GPT）。文件名去重。无 Hunyuan。协议同 D3/D4。D4 出分后再训。
# end

---

### D6 SID + D5 ∪ i2i fakes

# 2026-08-31, tianqi, D6 = D5 plus 118 whole-image i2i fakes after Nova eval
D5 mixin 再加上 A2 完整 triplet 的 i2i fake（Codex + nano reconstruction，约 118 张，不含配对 real）。等权替换 SID FLUX。协议同 D5：冻结 CLIP-B + online aug。Nova 评完再训。对照 D5 看官方 400 / EvalGEN clean / i2i pair_acc。118 / 14 万 ≈ 0.08%，预期官方分几乎不动，pair_acc 是否相对 D5 的 0.79 有增益才是这轮问题。

# 2026-09-01, tianqi, D6 finished numbers
结果：官方 400 **0.977**（D5 0.975，D3 0.978），EvalGEN clean 0.994，pair_acc **0.805**（D5 0.79）。fuse last4+D6 = **0.9929**，不赢 last4+D3 0.9930。不提交 D6。
# end

---

## 十一、Bias：先评测，不急着“修”

Dataset bias 不再作为：

> “看到不平衡 → 马上删除/补齐”

处理。

首先做 evaluation slicing。

### E1 Content

检查：

- real 是否大量风景
- fake 是否大量人物 / 人脸

关注：

```text
FPR
FNR
```

---

### E2 Aspect Ratio

检查：

- fake 是否大量 1:1
- real 是否大量自然长宽比

判断模型是否学习：

> square = fake

---

### E3 Color

检查：

- saturation
- brightness
- color cast

避免模型依赖生成器特定调色风格。

---

### E4 Watermark / Border

检查：

- watermark
- logo
- black border
- UI artifact

防止模型把角标等当成 AIGC fingerprint。

---

### Bias 的处理原则

只有当：

> 某一个 slice 的 FPR/FNR 明显异常

并且：

> 认为 hidden test 很可能存在类似分布

才进行：

- balanced sampling
- augmentation
- targeted data addition

否则：

**记录，不强行修。**

这些结果直接进入最终：

- error analysis
- limitations
- presentation

作为比赛材料。

---

## 十二、Augmentation 原则

继续使用模型侧已有的：

**online augmentation**

并且：

**real / fake 同时 augmentation。**

不要提前把：

`SID × 14 transformations`

全部落盘。

原因：

1. 浪费磁盘
2. 产生大量重复数据
3. manifest 复杂
4. 后续实验不灵活
5. online aug 已足够支持 controlled robustness experiment

数据侧负责提供：

**clean source images + metadata。**

augmentation 由 dataloader / training pipeline 在线完成。

---

## 十三、与模型线严格解耦

为了保证最后能解释“提升到底来自数据还是模型”，两条线不要混实验。

### 数据线

固定：

**CLIP-B + Frozen Backbone + Linear Head**

固定训练协议。

只改变：

**training data composition。**

研究：

- t2i/i2i
- GAN
- architecture
- SID/self-built

---

### 模型线

固定：

**同一份 SID 0/1 dataset**

研究：

- backbone 解冻
- consistency
- dual branch
- 336 resolution
- 其他模型创新

---

如果：

数据 + 模型

同时变化，就无法回答最终提升来自哪里。

---

## 十四、实验优先级

时间有限，不需要把所有格完整跑到底。

### P0：必须完成

#### C Architecture Coverage

重点：

- UNet
- DiT
- Flow
- Pixel

回答：

> SID 缺失 architecture 是否值得补。

#### D SID × Self-built

这是最终数据故事的核心。

必须得到：

```text
SID-only
vs
Self-built-only
vs
SID + Self-built
```

---

### P1：尽量完成

#### A t2i / i2i

尤其值得看：

```text
t2i-only
vs
t2i+i2i
```

如果时间不足，可以弱化 i2i-only。

#### E Bias Slicing

至少完成：

- aspect ratio
- content
- watermark/border

---

### P2：有时间再做

#### GAN mix ratio

如果 GAN 数据准备成本较高，可以直接：

> GAN 留作 unseen evaluation

而不做完整 10% / 20% mix grid。

---

## 十五、服务器资源使用优先级

SID 已经有 FLUX，因此：

**不要继续花 GPU 大规模生成 FLUX。**

GPU 优先用于补：

1. **SD1.5 ≥512 / UNet**
2. **PixArt-Sigma / Hunyuan-DiT**
3. **whole-image i2i**
4. 必要的其他 architecture hole

Pixel diffusion：

优先直接从 WildFake 抽 ADM/DDPM。

GAN：

优先寻找已有合规数据。

---

## 十六、最终训练数据

消融结束后，不需要提交很多版本。

最终最多保留：

### Final Mix A

**SID baseline**

### Final Mix B

**SID + 少量补洞数据**

例如：

```text
SID
+
SD1.5 / UNet
+
PixArt / Hunyuan DiT
+
少量 i2i
+
可选 ADM/DDPM
```

具体比例根据消融结果决定，而不是现在凭经验拍脑袋确定。

SID 全量 14 万级数据只在最终 1–2 个配置中使用。

---

## 十七、最终核心输出：Data Ablation Master Table

数据线是否完成，以这张表为准。

# 2026-09-01, tianqi, fill A/D master table after D6
| Experiment | Data Change | Official Formula | Δ vs SID | Unseen AUROC | EvalGEN Recall | Worst Robust #1 | #2 | #3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SID baseline | SID only, frozen CLIP-B | 0.970 | — | EvalGEN 0.992 | Nova rec 0.71 | JPEG-30 | crop | — |
| A1 | t2i only, no SID | 0.595 | −0.375 | 0.760 | — | — | — | — |
| A2 | 59-triplet i2i only | 0.443 | −0.527 | 0.810 | pair_acc 0.42 | — | — | — |
| A3 | t2i + 118 i2i, no SID | 0.786 | −0.184 | 0.786 | pair_acc 0.59 | — | — | — |
| B1 | Diffusion | | | | | | | |
| B2 | Diffusion+GAN | skipped (contest/EvalGEN are not GAN) | | | | | | |
| C-UNet | SD1.5 | | | | | | | |
| C-DiT | PixArt/Hunyuan | PixArt in D4; Hunyuan unused | | | | | | |
| C-Flow | SID-FLUX | | | | | | | |
| C-Pixel | ADM/DDPM | 0.649 full | −0.321 | never fires on DALL·E | | | | |
| D1 | SID 8k only | 0.734 full | | | | | | |
| D2 | Self-built 8k | | | | | | | |
| D3 | SID + UNet/pixel/flux2/sd35 (~9.6k) | **0.978** | +0.008 | EvalGEN **0.995**, Nova **0.988** | Nova rec **0.86** | JPEG-30 0.944 | ×0.25 0.958 | — |
| D4 | SID + nano/PixArt/SDXL/GPT (~6k) | 0.973 | +0.003 | EvalGEN 0.989, Nova 0.963 | Nova rec 0.71 | — | — | — |
| D5 | SID + D3 ∪ D4 (~15k) | 0.975 | +0.005 | EvalGEN 0.995, Nova 0.986 | Nova rec 0.85 | — | — | — |
| D6 | SID + D5 ∪ 118 i2i fakes | 0.977 | +0.007 | EvalGEN 0.994 | pair_acc **0.805** | — | — | — |
# end

最终数据侧必须能用这张表回答：

> **增加哪些数据真正改善了 generalisation？**

而不是：

> **我们一共收集了多少万张图片？**

---

## 十八、Bias Analysis 附表

单独输出：

| Slice | N real | N fake | AUC | FPR | FNR | Observation |
|---|---:|---:|---:|---:|---:|---|
| Portrait | | | | | | |
| Landscape | | | | | | |
| Square | | | | | | |
| Non-square | | | | | | |
| High saturation | | | | | | |
| Watermark | | | | | | |
| Border | | | | | | |

不一定需要解决每个 bias。

重点是知道：

> **模型在哪里失败，以及为什么可能失败。**

---

## 十九、数据线最终 Deliverables

最终交付以下五项。

### 1. Manifest

```text
train_manifest.csv
val_manifest.csv
unseen_manifest.csv
```

每张图片都有完整 generator / family / arch / generation_type metadata。

### 2. Data Pools

整理好的：

```text
real_pool
sid_pool
unet_pool
dit_pool
flow_pool
pixel_pool
i2i_pool
gan_pool
unseen_pool
```

### 3. Ablation Configs

每个实验格保存明确的数据 ID / manifest：

```text
A_t2i.csv
A_i2i.csv
A_mix.csv

B_diffusion.csv
B_diffusion_gan.csv

C_unet.csv
C_dit.csv
C_flow.csv
C_pixel.csv

D_sid.csv
D_self.csv
D_mix.csv
```

保证任何结果都可以重新复现。

### 4. Master Ablation Table

这是数据线的核心实验结果。

### 5. Bias Slice Report

作为最终：

- error analysis
- limitations
- Devpost
- presentation

的素材。

---

## 二十、数据线 Definition of Done

数据线完成不是：

> “WildFake / SID / CIFAKE / AutoSplice 都下载好了。”

也不是：

> “我们已经有几十万张图片。”

而是：

### 数据资产

- manifest 完整
- train / unseen / official val 严格隔离
- generator / family / arch 可追踪
- 每个实验格可复现

### Controlled Ablation

至少完成：

- Architecture coverage
- SID vs Self-built vs Mix

尽量完成：

- t2i vs i2i
- GAN unseen/mix

### Evaluation

每个核心实验都有：

- Official AUC_clean
- Official AUC_robust
- Formula
- Δ vs SID
- Unseen AUROC / EvalGEN recall
- Worst robustness conditions

### Bias

至少完成几个关键 evaluation slices，并记录 FPR/FNR。

### 最终能够给出一句明确的数据结论

例如：

> Adding a small amount of architecture-diverse synthetic data improved unseen-generator detection while preserving performance on the provided demonstration benchmark.

或者如果实验没有提升，也应该能够明确得到：

> Increasing generator diversity did not consistently improve generalisation; data-domain alignment had a larger effect than generator count.

**只要没有最终的 ablation master table，就不能认为数据需求已经完成。**
