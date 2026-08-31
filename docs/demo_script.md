# Demo 视频脚本（≤3 分钟）

> 上传 YouTube 公开，链接贴进 Devpost。录屏工具：OBS（录 1080p）。
> 建议先在本机跑通所有命令再录；终端字体调大（≥16pt）。
> 分镜负责人：Demo/视频角色；台词可直接照念。
>
> <!-- 2026-08-31, tianqi, no Streamlit; slides + local HTML gallery -->
> **不要做 Streamlit。** 3 分钟视频用 slides + 必要时本机打开 `outputs/tables/badcase_galleries/index.html`。数字以 README Results 为准（CLIP-B last4 0.990 / fuse 0.993），不要念 DINOv2 0.90 当提交分。
> <!-- end -->

## 分镜 0（0:00–0:20）问题与方案

**画面**：一页 slide（标题 + 三行要点），或直接停在 README Results 表。

**台词**：
> AI 生成的图片在社交媒体上会被压缩、裁剪、加噪——检测器必须在这些"现实世界退化"下依然有效。
> 我们做了一个 0.3B 参数的检测器：冻结的 DINOv2 骨干加一个 1025 参数的线性头，
> 在官方全部 14 种退化档位下 AUROC 稳定在 0.87 到 0.91，clean 0.90。

## 分镜 1（0:20–1:10）端到端推理演示

**画面**：终端。依次执行并让输出停留 2 秒：

```bash
# 1) 环境与权重（已就绪，展示命令即可）
pip install -r requirements.txt
python scripts/download_backbones.py --only dinov2-vit-large-patch14
gh release download v0.1-model -R chi030303/tiktok-techjam-aigc-detect -p sid_dinov2_best.pt

# 2) 官方契约：目录进、JSON 出
python predict.py data/val/fake out_pred.json --ckpt checkpoints/sid_dinov2_best.pt
cat out_pred.json | head -12
```

**台词**：
> 这是官方要求的推理接口：一个图片目录进，一个 JSON 出，pred 是 AI 生成的置信度。
> checkpoint 只有 5.95 KB——因为我们用的是线性探针：骨干完全冻结，只训练这一层。
> 整个模型 0.3B 参数，远低于 2B 的限制。

## 分镜 2（1:10–2:10）鲁棒性评估 + 画册看错例

**画面**：切到 `outputs/tables/official_val_spec_full400_pivot.md`（或截图），高亮 sid_dinov2 行；
然后打开 `sid_dinov2_clean.html` 画册，滚到 FN 区点开 1–2 张、再回 FP 区看 1 张。

**台词**：
> 这是 15 个评测条件下的鲁棒性表：clean、四档 JPEG 压缩、三档模糊、两档缩放、
> 三档噪声、色彩抖动、中心裁剪——全部 0.87 以上。
> 更重要的是知道它错在哪。我们的 bad case 管线把每一张分类错误的图找出来、
> 按置信度排序生成了这个画册：橙色是漏检的 AI 图，红色是误杀的真图。
> 可以看到我们的模型是"低误杀"型的——200 张真图只误杀了 10 张，但漏掉了 88 张假图。

## 分镜 3（2:10–2:40）未见生成器泛化

**画面**：EvalGen 分生成器表（docs/error_analysis.md §5），高亮 Nova 一列。

**台词**：
> 检测器真正的考验是没见过的生成器。在 5 个从未参与训练的生成器家族上——
> Flux、GoT、Infinity、Nova、OmniGen——多数家族召回率超过 0.93；
> Nova 是所有模型的共同盲区，这也指出了我们下一步的数据方向。

## 分镜 4（2:40–3:00）总结

**画面**：回 README，滚动过 Reproduce 四步命令。

**台词**：
> 总结：一套可复现的完整管线——官方变换、防泄漏的数据清单、线性探针训练、
> 15 条件鲁棒性评估、bad case 归因。0.3B 参数、5.95 KB 权重、单卡可跑。
> 仓库和 checkpoint 全部公开，欢迎按 README 复现。

---

## 录制清单（拍摄前检查）

- [ ] `git pull` 最新 main；`check.sh` 绿
- [ ] 终端：字体 16pt+、深色主题、窗口 1280×720 以上
- [ ] 画册 HTML 已在本地（`badcase_galleries/sid_dinov2_clean.html`）
- [ ] pivot 表已打开备用
- [ ] 录完上传 YouTube（公开），链接填入 Devpost
