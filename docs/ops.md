# 协作节奏（72 小时）

赛程：8/29 12:00 – 9/1 12:00（GMT+8）提交截止。

## 沟通

- 代码：GitHub PR（不要用微信传 `.py` 当主路径）
- 阻塞 / 占卡：即时通讯，一句话说清「卡、命令、还要多久」
- 决策：技术拍模型；产品拍范围和表；风控拍阈值与误杀叙事
- 上 GPU 的操作规范：[dev.md](dev.md)

## 每天结束

- `main` 上 `bash scripts/check.sh` 为绿
- 一张粗表：clean vs JPEG-50 vs crop（即使数字还差）
- 群里报：明天谁开哪条分支

## 冻结

| 时间 | 冻结什么 |
|---|---|
| Day 2 晚 | 骨干不再换 |
| Day 3 中午 | 不再开新训练 |
| 9/1 08:00 | 只修崩溃和 README；仓库改 public |

## 提交物对照

- Devpost 英文说明
- 公开 GitHub + README 可复现
- `predict.py` 目录 → JSON
- YouTube ≤3 min
- Robustness 表 + 误差分析
