# Vertex AI Nano Banana 1500 张批量生成

主脚本：`nano_banana_batch_generate.py`  
辅助模块：`nano_banana_generate.py`（主脚本会导入其中的 prompt 校验和文件写入函数，因此两个文件都要上传）

主脚本只走真正的 Google Cloud Vertex AI 通道：

- API 域名：`aiplatform.googleapis.com`
- 认证：Application Default Credentials（ADC）或任务服务账号
- 输入：上传 JSONL 到 Cloud Storage
- 批处理：Vertex AI `batchPredictionJobs`
- 输出：Vertex 写回 Cloud Storage，脚本随后下载图片到 Vast
- 模型：`gemini-3.1-flash-lite-image`
- 地区：`global`

脚本不会读取 `GEMINI_API_KEY`，也不会请求 `generativelanguage.googleapis.com`。

## 功能

- 校验完整输入为 1500 条、ID 唯一、prompt 非空以及 `prompt_sha256`。
- 从每条 prompt 读取自己的宽高比；当前 v2 表包含 1:1、4:5、3:2、9:16、16:9，各 300 条。
- 默认每个 Vertex 任务 50 条，同时最多运行两个任务。
- 本地 SQLite WAL 记录逐图状态、Vertex 任务名、GCS 输入/输出位置和失败原因。
- 支持任务提交恢复、失败重提一次、结果原子写入、断点续跑、小批量测试和预算上限。
- 每 30 秒显示本地完成数及 Vertex `completionStats`。Vertex 有时不会在任务结束前持续更新逐条计数，此时仍会显示任务状态，完成后再显示最终成功/失败数。

## 1. 上传脚本

在 Mac 执行：

```bash
cd "/Users/james/Documents/ChatGPT/TikTok Jam"

rsync -av \
  automation/nano_banana_generate.py \
  automation/nano_banana_batch_generate.py \
  automation/NANO_BANANA_README.md \
  vast-team:/workspace/data_new/automation/
```

如果服务器上的 prompt 还不是 v2：

```bash
rsync -av \
  "data/prompts/v2/remote/t2i_prompt_bank_1500_remote_v2.json" \
  vast-team:/workspace/data_new/automation/input/t2i_prompt_bank_1500.json
```

## 2. Google Cloud 一次性准备

需要已有 Google Cloud 项目和启用结算。下面的命令在已登录 `gcloud` 的可信电脑或 Cloud Shell 执行。

```bash
export GOOGLE_CLOUD_PROJECT="你的项目ID"
export BUCKET_NAME="一个全球唯一的 bucket 名称"

gcloud config set project "$GOOGLE_CLOUD_PROJECT"
gcloud services enable aiplatform.googleapis.com storage.googleapis.com
gcloud storage buckets create "gs://$BUCKET_NAME" --location=US --uniform-bucket-level-access
```

如果 bucket 已存在，不要重复创建，直接使用现有名称。

### 任务服务账号

```bash
export SERVICE_ACCOUNT_NAME="nano-banana-batch"
export SERVICE_ACCOUNT_EMAIL="$SERVICE_ACCOUNT_NAME@$GOOGLE_CLOUD_PROJECT.iam.gserviceaccount.com"

gcloud iam service-accounts create "$SERVICE_ACCOUNT_NAME" \
  --display-name="Nano Banana Vertex Batch"

gcloud projects add-iam-policy-binding "$GOOGLE_CLOUD_PROJECT" \
  --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
  --role="roles/aiplatform.user"

gcloud storage buckets add-iam-policy-binding "gs://$BUCKET_NAME" \
  --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
  --role="roles/storage.objectAdmin"
```

Vertex AI 的服务代理也需要读写该 bucket。尤其 bucket 在另一个项目或启用了严格 bucket 级权限时，请显式授权：

```bash
export PROJECT_NUMBER="$(gcloud projects describe "$GOOGLE_CLOUD_PROJECT" --format='value(projectNumber)')"
export VERTEX_SERVICE_AGENT="service-$PROJECT_NUMBER@gcp-sa-aiplatform.iam.gserviceaccount.com"

gcloud storage buckets add-iam-policy-binding "gs://$BUCKET_NAME" \
  --member="serviceAccount:$VERTEX_SERVICE_AGENT" \
  --role="roles/storage.objectAdmin"
```

## 3. Vast 认证

推荐优先使用短期 ADC：

```bash
gcloud auth application-default login --no-launch-browser
gcloud auth application-default set-quota-project "你的项目ID"
```

如果必须使用服务账号 JSON，请在可信电脑创建任务专用密钥，再安全上传到 Vast。不要提交到 Git，也不要放在 `automation/input`：

```bash
gcloud iam service-accounts keys create nano-banana-batch-key.json \
  --iam-account="$SERVICE_ACCOUNT_EMAIL"

scp nano-banana-batch-key.json \
  vast-team:/workspace/data_new/credentials/nano-banana-batch-key.json
```

然后在 Vast：

```bash
chmod 600 /workspace/data_new/credentials/nano-banana-batch-key.json
export GOOGLE_APPLICATION_CREDENTIALS=/workspace/data_new/credentials/nano-banana-batch-key.json
```

任务结束后删除并在 Google Cloud IAM 中撤销该密钥。多人共用的远程 `root` 账号无法对其他 root 用户隐藏密钥。

## 4. Vast 环境变量和依赖

```bash
cd /workspace/data_new/automation
python3 -m pip install -U google-auth google-cloud-storage

export GOOGLE_CLOUD_PROJECT="你的项目ID"
export GOOGLE_CLOUD_LOCATION=global
export NANO_BANANA_GCS_URI="gs://你的bucket/nano-banana/remote-v2"
```

这里不再设置 `GEMINI_API_KEY`。

## 5. Dry-run

Dry-run 不连接 Google Cloud，也不会产生费用：

```bash
python3 nano_banana_batch_generate.py \
  --prompts input/t2i_prompt_bank_1500.json \
  --dry-run
```

结果应明确显示：

```text
backend: vertex-ai
api_domain: aiplatform.googleapis.com
authentication: ADC/service-account OAuth
prompt_count: 1500
batch_count: 30
estimated_image_cost_usd: 25.2
```

## 6. 三张真实测试

```bash
python3 -u nano_banana_batch_generate.py \
  --prompts input/t2i_prompt_bank_1500.json \
  --smoke-test 3 \
  --chunk-size 3 \
  --max-active-batches 1 \
  --max-cost-usd 1 \
  --resume
```

生成图片默认保存到：

```text
/workspace/data_new/output/nano_banana_vertex_batch/remote_v2/
```

状态数据库默认保存到：

```text
/workspace/data_new/automation/state/nano_banana_vertex_batch_remote_v2/jobs.sqlite3
```

这是新的 Vertex 状态目录，不会读取旧 Developer API Batch 的状态。

## 7. 正式运行

Smoke test 的 3 张成功后，在同一个已设置环境变量的 shell 中：

```bash
tmux new -s nano_banana_vertex_v2

cd /workspace/data_new/automation
python3 -u nano_banana_batch_generate.py \
  --prompts input/t2i_prompt_bank_1500.json \
  --chunk-size 50 \
  --max-active-batches 2 \
  --max-cost-usd 30 \
  --resume
```

Smoke test 已完成的图片会自动跳过。停止后使用相同命令重新运行即可续跑。

另一个已配置相同 ADC 和环境变量的终端可以查看一次状态：

```bash
cd /workspace/data_new/automation
python3 nano_banana_batch_generate.py \
  --prompts input/t2i_prompt_bank_1500.json \
  --status
```

也可以在 Google Cloud Console 的 Vertex AI Batch Inference 页面查看任务状态，在 Cloud Storage 中查看 JSONL 输入和输出。
