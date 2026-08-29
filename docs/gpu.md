# GPU：Vast

<!-- 2026-08-29, tianqi, team GPU is Vast only; campus A100 not in team docs -->
组里训练 / 评测只走这台 Vast。日常怎么连、tmux、venv、占卡：[dev.md](dev.md)。
<!-- end -->

检测器（CLIP 头 / ResNet-50）单张 **24GB 4090 足够**。

## 这台机

- 2× RTX 4090。双卡用来 **两件事并行**（例如 GPU0 出图 / GPU1 训练），不是把一个 CLIP 训快一倍。
- 选 **On-Demand**，不要 Interruptible（中断会丢本地盘）。
- Container = **系统盘**（镜像里的 PyTorch），不是内存。数据一律放 Volume：`/workspace`。
- Flux 只可作少量补充样本，不要当主数据。Flux 和训练 **不要同时占一张卡**。
- Checkpoint 很小，优先从 `/workspace/experiments/<name>/` 拷出来。实例不用了再 Destroy。

<!-- 2026-08-29, tianqi, shared volume layout; personal clones are code-only -->
## 目录

```text
/workspace/data/           共享图
/workspace/models/         共享骨干
/workspace/experiments/    共享跑出来的 ckpt（按 recipe name）
/workspace/<who>/...       个人 git clone，不要在这里再下一份 SID/CLIP
```
<!-- end -->

## 约定

- 群里同步：「谁在占用 GPU0 / GPU1」。
- 开训前：`CUDA_VISIBLE_DEVICES=<0或1>`，`nvidia-smi` 看显存；细则见 [dev.md](dev.md)。
- 作业写成可断点（ckpt）。
