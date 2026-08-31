# Demo 视频脚本（≤3 分钟）

<!-- 2026-08-31, tianqi, last4 0.990 / fuse 0.993; no Streamlit; no DINOv2 0.90 as submit -->
上传 **YouTube 公开**，链接贴进 Devpost。OBS 录 1080p。先在本机跑通命令再录；终端字体 ≥16pt。

**不要做 Streamlit。** 用 4 页 slide + 终端 + 本机打开 `outputs/tables/badcase_galleries/index.html`。  
**不要念 DINOv2 0.90。** 提交分是 CLIP-B last4 **0.990**，双 ckpt 则 fuse **0.993**。
# end

屏幕上的英文标题可以照抄（评委看 YouTube）。

## 分镜 0（0:00–0:25）问题

**画面：** 一页 slide。

**标题：** Robust AIGC detection under JPEG / blur / crop

**台词：**
> 社交流转后的图会被压缩、模糊、裁剪。赛题分数是 0.50×干净 AUC + 0.50×14 种变换后的 AUC，不是 0.5 阈值准确率。
> 我们用约 8600 万参数的 CLIP-B：解冻最后 4 层，在官方 400 子集上公式分 0.990；如果允许两个权重，和 D3 数据混合模型做 logit 平均，到 0.993。

## 分镜 1（0:25–1:15）端到端推理

**画面：** 终端。

```bash
pip install -r requirements.txt
pip install torch torchvision transformers
python scripts/download_backbones.py --only clip-vit-base-patch16

python predict.py data/val/fake out.json \
  --ckpt experiments/clipb16_linear_sid_unfreeze4/ckpts/best.pt
python -c "import json; print(json.load(open('out.json'))[:3])"
```

可选再跑一行 fuse（如果录双模）：

```bash
python predict.py data/val/fake out_fuse.json \
  --ckpt experiments/clipb16_linear_sid_unfreeze4/ckpts/best.pt \
  --ckpt-b experiments/clipb16_linear_sid_d3_mix/ckpts/best.pt
```

**台词：**
> 官方接口：一个图片目录进，JSON 出。每条是 image_path 和 pred，pred 是 AIGC 的置信度。
> 演示集 COCO + DALL·E 没有进训练。模型远小于 2B。

## 分镜 2（1:15–2:15）鲁棒性 + 错例

**画面：** `docs/robustness.md` 或 README 的 15 档表；然后打开  
`outputs/tables/badcase_galleries/fuse_u4_d3_400_clean.html` 或 `unfreeze4_400_clean.html`。  
先 FN（漏检 DALL·E），再 FP（几乎没有）。

**台词：**
> 14 种官方变换下 last4 的 AUROC 都在 0.98 以上。
> 0.5 阈值时 fuse 只有 1 张误杀、44 张漏检——低误杀、偏保守。分数高不代表 0.5 就是好工作点。

## 分镜 3（2:15–2:40）未见生成器

**画面：** robustness 里 EvalGEN / Nova 那张小表。

**台词：**
> EvalGEN 五家生成器都没进训练。Nova 最难。D3 混合数据把 Nova 召回拉上去；fuse 保住官方 DALL·E 分数，同时 Nova AUC 到 0.988。

## 分镜 4（2:40–3:00）复现

**画面：** README Reproduce 四行命令。

**台词：**
> 仓库公开。按 README 装 CLIP-B、下权重、对任意目录跑 predict.py 即可复现。

## 拍摄前

- [ ] README / robustness 数字是 last4 0.990、fuse 0.993
- [ ] 本机有 `data/val` 或一小撮图，不要用带商标的素材
- [ ] 画册已下载到本机（Vast Jupyter 打不开）
- [ ] 不要入镜第三方 logo
- [ ] 上传 YouTube → Public → 链接进 Devpost
