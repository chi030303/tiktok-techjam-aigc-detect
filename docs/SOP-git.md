# SOP：GitHub 分支工作流

每人在自己的分支上干活，**本地 `bash scripts/check.sh` 通过后再开 PR 合进 `main`。** `main` 必须随时能跑 `predict.py`。

不要把数据、权重、`.env` 推进 GitHub。

## 0. 第一次（每人做一次）

```bash
git clone git@github.com:<ORG>/<REPO>.git
cd <REPO>
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

本地身份（可只对这个仓库）：

```bash
git config user.name "Your Name"
git config user.email "you@example.com"
```

不要改别人的文件时顺手 `git config --global`。

## 1. 开工前更新 main

```bash
git checkout main
git pull origin main
git checkout -b feat/<who>-<thing>
```

命名：

| 例子 | 含义 |
|---|---|
| `feat/kiki-train` | 训练与增强 |
| `feat/product-eval` | 变换集、robustness 表 |
| `feat/risk-metrics` | FPR / 阈值 |
| `docs/kiki-readme` | 只改文档时也可用 `docs/` 前缀 |

规则：

- 一律从 **最新 `main`** 开分支，不要从别人的 `feat/` 再开。
- 一人一条活分支；做完就合，**不要隔夜囤 PR**。
- 禁止 `git push --force` 到 `main`。只有没人用过的自己的分支才能 force。
- 72 小时内用 **merge**，不要 rebase。

## 2. 小步提交

```bash
git add <files>
git commit -m "train: add jpeg quality sampler"
git push -u origin HEAD
```

前缀：`train:` `eval:` `infer:` `docs:` `data:` `chore:`

一次提交只做一类事。不要把训练脚本和 README 和视频脚本捆在一起。

## 3. 合入 main 之前必须过的检查

在 **你的分支** 上：

```bash
source .venv/bin/activate
bash scripts/check.sh
```

当前检查包含：

1. `predict.py` 对 `fixtures/sample_images` 写出合法 JSON（`image_path` + `pred`）
2. 没有把 `data/`、`*.pth`、`.env` 加进暂存区（靠 `.gitignore` + 自己看 `git status`）

有真模型之后，把「演示集 clean 上能跑通、不读 `data/demo_wildfake` 当 train」补进 `scripts/check.sh`。

**检查不过：不准开 PR。** 自己修到绿。

## 4. 开 PR → 合并

1. GitHub 开 Pull request：`feat/...` → `main`
2. 用默认模板填：改了什么、怎么测的、有没有动 `predict.py` schema
3. 指定 reviewer：
   - 动 `predict.py` / 训练入口 → 技术负责人 Approve
   - 评测表 / 文档 → 产品或风控看一眼即可
4. 讨论里说一声；**Approve 后再按 Merge**（Create a merge commit）
5. 合完：GitHub 删远程分支；本地：

```bash
git checkout main
git pull origin main
git branch -d feat/<who>-<thing>
```

冲突：以 **脚本** 为准。不要手合 `.pth`。权重文件直接丢，重新导出。

## 5. 标签

| Tag | 何时 |
|---|---|
| `v0-skeleton` | 空预测能跑（本 SOP 落地时） |
| `v1-baseline` | Day 1 锁定的骨干 |
| `v2-submit` | 交 Devpost 那一版 |

```bash
git checkout main && git pull
git tag -a v2-submit -m "TechJam submit"
git push origin v2-submit
```

## 6. 出事

| 情况 | 做法 |
|---|---|
| 训练挂了，`predict.py` 还好 | 不要回滚 `main` |
| `predict.py` 坏了 | `git revert` 该 commit，或回到上一 tag |
| 误推密钥 / 数据 | 立刻轮换密钥；不要只 `git rm`。必要时 **新建空仓库** |
| 9/1 前 | Settings → 把仓库改成 **Public** |

## 7. 不要做

- 直接在 `main` 上 commit（热修也先开 `hotfix/<who>-<thing>` 再 PR）
- 把 SID_Set / 演示集 / Flux 出图推进 GitHub
- 用 fork（本队全部 Collaborator，同一仓库）
