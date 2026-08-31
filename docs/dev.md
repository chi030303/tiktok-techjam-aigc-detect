# 开发规范（Vast + 本机）

<!-- 2026-08-29, tianqi, team handbook: git pointer, tmux, venv, GPUs, SSH -->
进 GPU 干活之前先读这一页。Git 细则仍以 [SOP-git.md](SOP-git.md) 为准，这里只写**机器上怎么活着、怎么不互相踩**。
<!-- end -->

| 文档 | 内容 |
|---|---|
| 本页 | SSH、tmux、venv、显卡、目录 |
| [SOP-git.md](SOP-git.md) | 分支、PR、谁合 `main` |
| [badcase-galleries.md](badcase-galleries.md) | 误差画册：本机打开，不要在 Vast 上点 HTML |
| [data.md](data.md) | 数据路径、禁止训练 |
| [gpu.md](gpu.md) | Vast 这台机 |
| [experiments/README.md](../experiments/README.md) | 一个实验一个 recipe |

## 0. 目录：共享盘 vs 个人 clone

五个人 **不要** 共用同一份 git 工作树，也 **不要** 在自己的 clone 里再下一份 SID / CLIP。

```text
/workspace/data/              共享图（sid_set, cifake, val, evalgen）
/workspace/models/            共享骨干
/workspace/experiments/      共享 ckpt / log（按实验名）
/workspace/<who>/tiktok-techjam-aigc-detect/
                              你的 git clone（代码 + recipe）
```

在 clone 里跑脚本时，路径会自动认 `/workspace/data` 和 `/workspace/models`（见 `src/paths.py`）。也可以在 `.env` 里写死 `DATA_ROOT` / `MODELS_ROOT` / `EXP_ROOT`。

**官方 `data/val`** = 题面 Demonstration 集（不算分、不能训）。EvalGEN 是额外 hold-out，也不是官方打分集。

## 1. SSH 上这台 Vast

- **必须带端口**，默认 22 会失败。端口以控制台 **Advanced Connection Options** 为准，重建实例后会变。
- `-L 8080:localhost:8080` 只是把 Jupyter 转到本机，**不参与登录**。
- 账号密码 **不要** 发给队友。每人用自己的 SSH 公钥；只传 `.pub`，私钥留在自己电脑。
- 公钥必须是 **完整一行**（`ssh-ed25519 AAAA... 注释`）。两把钥匙粘在同一行，SSH 会整文件作废。
- 本机连：

```bash
ssh -p <PORT> -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes root@<IP>
```

Cursor：**Remote-SSH** 打开 `/workspace/<who>/tiktok-techjam-aigc-detect` 或 `/workspace`，不要打开 `/root`（会只看到 `.bashrc`）。

每人第一次：

```bash
cd /workspace
git clone <repo-url> /workspace/<who>/tiktok-techjam-aigc-detect
cd /workspace/<who>/tiktok-techjam-aigc-detect
```

## 2. tmux（Vast 会自动挂上）

Vast 的 SSH 往往已经在一个 tmux 里（`$TMUX` 有值）。Cursor 多开几个终端，经常是 **同一个 session 的镜像**，不是三个独立壳。

**不要** 在已经进 tmux 时再执行 `tmux new -s train`。会报：

```text
sessions should be nested with care, unset $TMUX to force
```

不要 `unset TMUX` 去套娃。

| 你想做 | 做法 |
|---|---|
| 当前 session 再开一个窗口 | `tmux new-window -n train` 或 `Ctrl+B` 再按 `c` |
| 后台单独建 session | `tmux new-session -d -s train` 然后 `tmux switch-client -t train` |
| 列 session / 窗口 | `tmux ls` / `tmux list-windows` |
| 切窗口 | `Ctrl+B` 再 `n` / `p`，或 `Ctrl+B` 再 `0` `1` `2` |
| 改窗口名 | `Ctrl+B` 再 `,` |
| 脱离（进程继续跑） | `Ctrl+B` 再 `d` |
| 重新附上 | `tmux attach -t train` 或 `tmux attach` |

约定：

- 长任务（下载、训练）放在 **具名窗口** 里：`sid`、`evalgen`、`train`、`flux`。
- **一次只粘一行命令。** 多行粘进 tmux 会被拆成 tmux 指令（之前出现过 `pipe-pane: too many arguments`）。
- 不要对正在下 SID_Set / EvalGEN 的窗口乱按 `Ctrl+C`。
- 看别人占没占窗口：`tmux ls` + `nvidia-smi`。

前缀键默认是 `Ctrl+B`：先按组合键，**松开**，再按后面那个字母。

## 3. 虚拟环境

这台机用 **venv，不用 conda**（Vast PyTorch 镜像已经带 CUDA）。

每人在 **自己的 clone** 里建一份，不要往系统 `pip` 装包，也不要五个人共用一个 `.venv`（容易把依赖拧乱）。

```bash
cd /workspace/<who>/tiktok-techjam-aigc-detect
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# 训练还需要 torch/timm/open_clip 等：镜像里若已有 torch，就不要再装一套 CPU 版
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

每次新开 shell / tmux 窗口都要重新 `source .venv/bin/activate`。提示符前面应有 `(.venv)`。

加依赖：先在自己分支改 `requirements.txt`，再 `pip install`，不要只装在本机。

`.venv/` 已 gitignore。不要把 venv 推进 GitHub。

## 4. 显卡

这台是 **2× RTX 4090**。检测器单卡够；双卡用来 **两件事并行**，不是把一个 CLIP 训快一倍。

```bash
nvidia-smi          # 谁在占哪张卡、显存
nvidia-smi -L       # UUID
```

开训 / 出图前 **锁死可见卡**，不要让进程默认占满两张：

```bash
# 卡 0：Flux 出图（若做）
CUDA_VISIBLE_DEVICES=0 python ...

# 卡 1：训练 / 抽特征
CUDA_VISIBLE_DEVICES=1 python scripts/run_experiment.py experiments/clipb16_linear_sid/recipe.yaml
```

Recipe 里的 `gpu:` 字段应和上面一致。群里报一声：「我占用 GPU1，大概 N 小时」。

| 可以 | 不要 |
|---|---|
| GPU0 Flux、GPU1 训练 | 同一张 24GB 卡上 Flux 驻留 + 同时训练（会 OOM） |
| 冻 CLIP 线性头 / ResNet 全微调 | 五个人同一工作树、同一 `runs/` 互相覆盖 |
| 实验产物写 `/workspace/experiments/<name>/` | 把 ckpt 写进别人的 clone |
| 训完 `nvidia-smi` 确认进程没了 | 丢下一个占满显存的 python 过夜不说 |

显存粗估：冻 CLIP 头大约 8–16GB；ResNet-50 大约 4–8GB。OOM 时先降 `batch_size`，不要改去抢另一张已有人的卡。

## 5. 实验怎么开

1. 从最新 `main` 拉自己的 `feat/<who>-...`（见 SOP）。
2. 复制 `experiments/_template/` → `experiments/<name>/`，改 `recipe.yaml`。
3. 命名：`<骨干>_<头>_<数据>_<技巧>`，例如 `clipb16_linear_sid`。
4. `python scripts/run_experiment.py experiments/<name>/recipe.yaml` 先 dry-run，确认 `data` / `models` 指到 `/workspace/...`。
5. `train.forbid` 必须含 `val`、`evalgen`、`demo_wildfake`。
<!-- 2026-08-29, tianqi, eval is a separate script; val stays out of train -->
6. 训完评测：`python scripts/run_eval.py robustness --split official_val --conditions daily --max-images 400 --experiment <name> --ckpt .../ckpts/best.pt`。不要拿 `data/val` 开训。
<!-- end -->

## 6. 日常不要做

- 在 `main` 上直接改；用微信传 `.py` 当主同步。
- 把 `data/`、`models/`、`.env`、密钥推进 GitHub。
- 用 `data/val` 或 `data/evalgen` 训练。
- tmux 套娃；conda 再装一套和镜像冲突的 PyTorch。
- 不设 `CUDA_VISIBLE_DEVICES` 就开两个大进程。
- 改 `authorized_keys` 时把手动换行插进公钥中间。

卡住时群里一句话：**哪张卡、哪条命令、还要多久**。
<!-- end -->
