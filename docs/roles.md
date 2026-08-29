# 角色与目录边界

避免五个人改同一个 `predict.py`。合并前仍走 [SOP-git.md](SOP-git.md)。上机操作见 [dev.md](dev.md)。

| 角色 | 主目录 / 文件 | 可以改 | 不要直接改 |
|---|---|---|---|
| 技术（训练） | `src/train/`、`checkpoints` 导出脚本 | 增强、骨干、融合 | 随便改 JSON 字段名 |
| 评测 / 产品 | `src/eval/`、`docs/`、robustness 表 | 变换集脚本、FP/FN 笔记 | 训练超参（先开 issue） |
| 风控 | 阈值、FPR–召回 | `src/eval/metrics.py`、误差归类 | 换骨干 |
| Demo / 视频 | `docs/demo.md`、录屏脚本 | 叙事、截图表 | 训练代码 |

`predict.py` 的 schema（`image_path`, `pred`）冻结。要改必须在 PR 里写明，并由 kiki Approve。
