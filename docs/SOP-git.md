# SOP：GitHub 分支工作流

<!-- 2026-08-29, tianqi, personal-branch-first; kiki merges main -->
默认：**尽量长时间待在自己的分支上开发。** 日常修改只 commit / push 到个人 `feat/<who>-...`，**不要直接改 `main`。** 一整块功能在自己分支上测通（`bash scripts/check.sh` + 你负责的那部分能跑）之后再开 PR，由 **kiki（仓库 owner）合进 `main`。**

`main` 必须随时能跑 `predict.py`。不要把数据、权重、`.env` 推进 GitHub。

机器上的 tmux / venv / 显卡：[dev.md](dev.md)。
<!-- end -->

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
- 一人可以同时有 1～2 条活分支（例如训练一条、文档一条），但同一文件不要两条分支一起改。
- **日常工作都留在个人分支**：写完就 `commit` + `push` 到自己的 remote 分支，避免只存在笔记本上。
- 禁止 `git push --force` 到 `main`。只有没人用过的自己的分支才能 force。
- 72 小时内用 **merge**，不要 rebase。
- 每天至少一次：`git fetch origin && git merge origin/main` 进你的分支，避免最后 PR 爆冲突。

<!-- 2026-08-29, tianqi, when to PR vs stay on branch -->
开 PR 的时机：**一块完整功能**测完，而不是每一个 typo。例如「JPEG 增强 + 能出一张表」可以一个 PR；不要把三天的训练+评测+视频捆成一个巨型 PR。功能已经能跑就提，**不要隔夜囤着已经测完的代码**。
<!-- end -->

## 2. 小步提交（都提交到个人分支）

```bash
git add <files>
git commit -m "feat: add jpeg quality sampler"
git push -u origin HEAD
```

<!-- 2026-08-29, tianqi, commit types are feat/fix/docs/chore/debug not train/data -->
前缀用 **类型**，不要用模块名：

| 前缀 | 何时 |
|---|---|
| `feat:` | 新功能（下载脚本、recipe、训练循环） |
| `fix:` | 修 bug |
| `docs:` | 只改文档 |
| `chore:` | 依赖、gitignore、检查脚本、杂务 |
| `debug:` | 临时排查，合 `main` 前尽量清掉 |

分支名仍是 `feat/<who>-<thing>`，和 commit 前缀不是一回事。
<!-- end -->

一次提交只做一类事。不要把训练脚本和 README 和视频脚本捆在一起。

**大改动必须带注释和文档（自己写或 AI 写都可以，但要进同一个 PR）：**

- 代码：改动处附近用简短注释说明 *为什么*（尤其是增强参数、阈值、路径约定）
- 文档：更新 `docs/data.md` / `README` / 评测说明里对应小节，别等交稿前一次性补
- PR 描述写清：改了什么、怎么测的、组员需要知道的新命令

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

## 4. 开 PR → 由 kiki 合进 main

个人分支上可以有很多 commit；**一个 PR = 一块已测完的功能**，不是每个 commit 都开 PR。

1. GitHub 开 Pull request：`feat/...` → `main`
2. 用默认模板填：改了什么、怎么测的、有没有动 `predict.py` schema；大改动附上文档/注释位置
3. Assignee / reviewer：指定 **kiki**（`chi030303`）合并
4. 群里说一声；**只有 kiki 按 Merge**（Create a merge commit）。其他人不要自己合 `main`
5. 合完：GitHub 删远程分支；本地：

```bash
git checkout main
git pull origin main
git branch -d feat/<who>-<thing>
```

然后若还有下一件活：再从最新 `main` 开新的 `feat/<who>-<next>`。

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

- 直接在 `main` 上 commit（热修也先开 `hotfix/<who>-<thing>` 再 PR，仍由 kiki 合）
- 把 SID_Set / 演示集 / Flux 出图推进 GitHub
- 用 fork（本队全部 Collaborator，同一仓库）
- 用微信传 `.py` 当主同步方式（必须在个人分支上，别人才能 PR）
- 等「全部做完」再提一个巨型 PR（按功能切片提）
