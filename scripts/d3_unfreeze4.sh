#!/usr/bin/env bash
# 2026-08-31, tianqi, D3 mix + unfreeze4, then 400×15 + EvalGEN full
set -u
REPO="${REPO:-/workspace/kiki/tiktok-techjam-aigc-detect}"
export DATA_ROOT="${DATA_ROOT:-/workspace/data}"
export MODELS_ROOT="${MODELS_ROOT:-/workspace/models}"
export EXP_ROOT="${EXP_ROOT:-/workspace/experiments}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PY="${PY:-/venv/main/bin/python}"
LOG="${EXP_ROOT}/kiki_d3_unfreeze4.log"
OUT="${REPO}/outputs/tables"
CKPT="${EXP_ROOT}/clipb16_linear_d3_unfreeze4/ckpts/best.pt"

cd "$REPO"
mkdir -p "$OUT"
# shellcheck disable=SC1091
source "$REPO/scripts/eval_400_evalgen.sh"
exec >>"$LOG" 2>&1
echo "==== d3 unfreeze4 start $(date -u) ===="
"$PY" -u scripts/run_experiment.py experiments/clipb16_linear_d3_unfreeze4/recipe.yaml --train
echo "train exit=$?"
if [[ -f "$CKPT" ]]; then
  eval_400_evalgen d3_unfreeze4 "$CKPT" 32
else
  echo "FAIL missing $CKPT"
fi
echo "==== d3 unfreeze4 done $(date -u) ===="
# end
