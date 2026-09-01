# Archive: ComfyUI / Vertex batch generation (not the detector)

This folder is **not** required to run `predict.py`. It is the Vast automation used to build the self-built mix later released as [Kaggle aigctrace-mix](https://www.kaggle.com/datasets/wwjjames/aigctrace-mix). Kept on branch `feat/kiki-comfy-automation` so contest `main` stays the detector.

| In this folder | Not included |
|---|---|
| API workflows (`workflows/`, `input/*.json`), prompt bank, batch scripts, tests | ComfyUI checkpoints / VAE / text encoders |
| UI graphs in `comfy-user-workflows/` (FLUX.2, PixArt, SD3.5) | Hunyuan graphs and custom nodes (never used in train) |
| Nano Banana Vertex scripts (keys stay in env, not in git) | `state/`, `logs/`, generated PNGs |

Detector + reproduce: repo root README. Images: Kaggle, not this tree.

---

# Vast / ComfyUI 双模型批量生成

这套脚本使用同一个 ComfyUI 实例，按 `flux2`、`sd35` 两个模型组顺序执行。默认队列上限为 1，因此不会在单 GPU 上并行推理；切换模型组时会调用 `/free` 释放模型和显存。

## 已核对的输入

- prompt 文件是含 1500 个对象的 JSON 数组；1500 个 `prompt_id` 全部唯一，`rendered_prompt` 均非空。
- FLUX.2：提示词 `76.value`，seed `75:73.noise_seed`，尺寸 `75:68.value` / `75:69.value`，保存节点 `9`。
- SD 3.5：提示词 `16.text`，负向提示词 `40.text`，seed `3.seed`，尺寸 `53.width` / `53.height`，保存节点 `9`。
- 两个工作流当前均为 1024×1024、batch size 1，且没有 Mac 本地绝对路径。
- 脚本启动时会重新自动识别并验证以上入口，不依赖这些节点号作为唯一依据。

prompt 文本里的宽高比有五种，但原始 API 工作流都固定为 1024×1024。脚本默认忠实保留工作流尺寸；如需统一覆盖，可显式传入 `--width` 和 `--height`。不要仅凭 prompt 中的自然语言宽高比自动改变训练数据尺寸。

## 部署

在 Mac 上把目录和 prompt 文件传到 Vast（请将 `YOUR_VAST_HOST` 替换为自己的 SSH 主机别名）：

```bash
ssh YOUR_VAST_HOST 'mkdir -p /workspace/data_new/automation/input'
rsync -av automation/ YOUR_VAST_HOST:/workspace/data_new/automation/
rsync -av data/prompts/t2i_prompt_bank_1500.json \
  YOUR_VAST_HOST:/workspace/data_new/automation/input/prompts.json
```

脚本只使用 Python 标准库，不需要 `pip install`。Vast 上先检查模型及服务：

```bash
cd /workspace/data_new/automation
curl -sS http://127.0.0.1:18188/system_stats
nvidia-smi
df -h /workspace
ls -lh /workspace/data_new/ComfyUI/models/diffusion_models/flux-2-klein-base-4b.safetensors
ls -lh /workspace/data_new/ComfyUI/models/text_encoders/qwen_3_4b.safetensors
ls -lh /workspace/data_new/ComfyUI/models/vae/flux2-vae.safetensors
ls -lh /workspace/data_new/ComfyUI/models/checkpoints/sd3.5_large_fp8_scaled.safetensors
sha256sum /workspace/data_new/ComfyUI/models/checkpoints/sd3.5_large_fp8_scaled.safetensors
```

SD 3.5 checkpoint 的预期 SHA256 是：

```text
5ad94d6f951556b1ab6b75930fd4effbafaf3130fe9df440e7f2d05a220dd1be
```

## 验证与 smoke test

纯本地校验，不连接 ComfyUI、不写状态：

```bash
python3 batch_generate.py \
  --prompts input/prompts.json \
  --model all \
  --smoke-test 2 \
  --dry-run
```

在 ComfyUI 上为每个模型各生成 2 张：

```bash
python3 batch_generate.py \
  --prompts input/prompts.json \
  --model all \
  --smoke-test 2 \
  --resume
```

检查 `ComfyUI/output/flux2/`、`ComfyUI/output/sd35/`、`state/summary.json`、`state/missing_files.json` 和 SQLite 状态库。确认图片正确后再全量运行。

## 全量运行

在独立 tmux 会话中运行：

```bash
tmux new -s james_t2i
cd /workspace/data_new/automation
python3 -u batch_generate.py \
  --prompts input/prompts.json \
  --model all \
  --resume
```

按 `Ctrl-b`、再按 `d` 离开 tmux；用 `tmux attach -t james_t2i` 恢复。日志同时实时写入 `logs/`。

也可明确分批或分模型运行：

```bash
python3 -u batch_generate.py --prompts input/prompts.json --model flux2 --start 0 --limit 100 --resume
python3 -u batch_generate.py --prompts input/prompts.json --model flux2 --start 100 --limit 100 --resume
python3 -u batch_generate.py --prompts input/prompts.json --model sd35 --resume
```

`--limit` 对每个选中的模型生效。稳定 seed 由 `sample_id + model + seed salt` 生成，因此分批、重启或改变数组顺序不会改变 seed。

## 恢复与错误记录

- `state/jobs.sqlite3` 使用 WAL 和同步提交，保存每个 `(sample_id, model)` 的最新状态。
- `attempts` 表追加保存每次提交、完成或失败事件；HTTP 400、节点验证、执行失败、OOM、超时、网络错误和输出缺失会分类记录。
- `--resume` 只跳过状态成功且文件确实存在的任务；文件丢失会重新生成。
- 默认严格要求完整输入文件有 1500 条；其他数据集可用 `--expected-count N`，或以 `--expected-count 0` 关闭数量约束。
- 对中断时已提交的任务，恢复时先查 `/history` 和 `/queue`；仍存在则继续等待，服务器重启导致的陈旧任务会重新提交。
- 默认初次尝试后最多重试 2 次。`Ctrl-C`/SIGTERM 会停止继续提交，远端已提交任务保留为可恢复状态。
- 输出前缀为 `flux2/<sample_id>_flux2` 或 `sd35/<sample_id>_sd35`，实际文件名由 ComfyUI 添加序号，完整路径写回 SQLite。

常用参数可通过 `python3 batch_generate.py --help` 查看。多人共用服务器时，请在提交前确认 GPU 没有被队友占满，不要结束不属于自己的进程。
