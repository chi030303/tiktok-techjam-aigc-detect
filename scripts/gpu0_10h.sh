#!/usr/bin/env bash
# 2026-08-31, tianqi, GPU0 10h: 400+evalgen-full for existing ckpts, then unfreeze4+336 train
set -u
REPO="${REPO:-/workspace/kiki/tiktok-techjam-aigc-detect}"
export DATA_ROOT="${DATA_ROOT:-/workspace/data}"
export MODELS_ROOT="${MODELS_ROOT:-/workspace/models}"
export EXP_ROOT="${EXP_ROOT:-/workspace/experiments}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="${PY:-/venv/main/bin/python}"
LOG="${EXP_ROOT}/kiki_gpu0_10h.log"
OUT="${REPO}/outputs/tables"

cd "$REPO"
mkdir -p "$OUT" "$(dirname "$LOG")"
# shellcheck disable=SC1091
source "$REPO/scripts/eval_400_evalgen.sh"
exec >>"$LOG" 2>&1
echo "==== gpu0 10h start $(date -u) ===="
echo "repo=$REPO gpu=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true

E="${EXP_ROOT}"
eval_400_evalgen unfreeze4 "${E}/clipb16_linear_sid_unfreeze4/ckpts/best.pt" 32
eval_400_evalgen clipl14_sid "${E}/clipl14_linear_sid_aug/ckpts/best.pt" 16
eval_400_evalgen clipb16_sid "${E}/clipb16_linear_sid_aug/ckpts/best.pt" 32
eval_400_evalgen res336 "${E}/clipb16_linear_sid_res336/ckpts/best.pt" 16

echo "==== train CLIP-B unfreeze4+res336 $(date -u) ===="
if "$PY" -u scripts/run_experiment.py experiments/clipb16_linear_sid_unfreeze4_res336/recipe.yaml --train; then
  echo "combo train ok"
  eval_400_evalgen unfreeze4_res336 "${E}/clipb16_linear_sid_unfreeze4_res336/ckpts/best.pt" 16
else
  echo "combo train FAILED (likely GPU0 VRAM vs ComfyUI); GPU1 fallback will pick it up"
fi

echo "==== gpu0 10h done $(date -u) ===="
# end
