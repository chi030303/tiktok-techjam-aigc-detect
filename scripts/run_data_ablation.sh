#!/usr/bin/env bash
# 2026-08-30, tianqi, wait for GPU0 full-eval then train D1 / C-Flow / C-Pixel
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export DATA_ROOT="${DATA_ROOT:-/workspace/data}"
export MODELS_ROOT="${MODELS_ROOT:-/workspace/models}"
export EXP_ROOT="${EXP_ROOT:-/workspace/experiments}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="${PY:-/venv/main/bin/python}"
LOG="${EXP_ROOT}/full_eval/data_ablation_train.log"

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "log=$LOG"

wait_gpu0() {
  echo "wait until run_full_eval is gone..."
  while pgrep -f "scripts/run_full_eval.py" >/dev/null 2>&1; do
    sleep 30
  done
}

train_one() {
  local recipe="$1"
  echo "==== train ${recipe} ===="
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PY" -u scripts/run_experiment.py \
    "experiments/${recipe}/recipe.yaml" --train
}

wait_gpu0
train_one clipb16_linear_D1_sid_only
train_one clipb16_linear_C_flow_sid
train_one clipb16_linear_C_pixel

echo "==== official_val full eval of ablation ckpts ===="
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PY" -u scripts/run_full_eval.py \
  --split official_val --conditions full \
  --ckpt D1_sid=/workspace/experiments/clipb16_linear_D1_sid_only/ckpts/best.pt \
  --ckpt C_flow=/workspace/experiments/clipb16_linear_C_flow_sid/ckpts/best.pt \
  --ckpt C_pixel=/workspace/experiments/clipb16_linear_C_pixel/ckpts/best.pt \
  --workers 4 --batch 32 \
  --experiment ablation_official_val \
  --stem official_val_ablation_clipb \
  --out-dir "$ROOT/outputs/tables"

echo "ablation queue done"
# end
