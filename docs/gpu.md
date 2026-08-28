# GPU：校园卡 vs Vast

检测器（CLIP 头 / ResNet-50）单张 **24GB 4090 足够**。不必为训练租 A100。

## 校园 A100

- 账号 **不要把密码发给组员**（学校/超算协议通常禁止转借）。
- 同一账号往往能多人 SSH，但 **GPU 配额只有一份**：同一时刻只跑一个训练 job。
- 组员改代码走 GitHub；由账号本人 `sbatch` / 启动训练。

## Vast.ai（插队、并行选型）

- 选 **On-Demand**，不要 Interruptible（中断会丢本地盘，传文件更烦）。
- Container = **磁盘**（建议 64GB），不是内存。RAM 看机器卡片（建议 ≥32GB，Flux 同机建议 64GB）。
- Volume：只训公开数据 100–200GB；还要存 Flux 出图则 300GB+。
- 1×4090 训检测器。双卡 **不是** 把 CLIP 训快一倍，而是 GPU0 出图 / GPU1 训练。
- Flux 只可作少量补充样本，不要当主数据。Flux 和训练 **不要同时占一张 24GB 卡**。
- 训完 Destroy。Checkpoint 很小，优先 `scp` 出来而不是实例开满 75 小时。

## 约定

- 群里同步：「谁在占用哪张卡 / 哪个 Vast instance」。
- 作业写成可断点（ckpt），PSB 排队不稳时用短 job。
