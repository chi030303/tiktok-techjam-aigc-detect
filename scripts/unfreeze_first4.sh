#!/usr/bin/env bash
# 2026-08-31, tianqi, CLIP-B first-4 unfreeze on SID, then 400×15 + EvalGEN full
set -u
REPO="${REPO:-/workspace/kiki/tiktok-techjam-aigc-detect}"
export DATA_ROOT="${DATA_ROOT:-/workspace/data}"
export MODELS_ROOT="${MODELS_ROOT:-/workspace/models}"
export EXP_ROOT="${EXP_ROOT:-/workspace/experiments}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PY="${PY:-/venv/main/bin/python}"
LOG="${EXP_ROOT}/kiki_unfreeze_first4.log"
OUT="${REPO}/outputs/tables"
CKPT="${EXP_ROOT}/clipb16_linear_sid_unfreeze_first4/ckpts/best.pt"

cd "$REPO"
mkdir -p "$OUT"
# shellcheck disable=SC1091
source "$REPO/scripts/eval_400_evalgen.sh"
exec >>"$LOG" 2>&1
echo "==== unfreeze first4 start $(date -u) ===="
"$PY" -u scripts/run_experiment.py experiments/clipb16_linear_sid_unfreeze_first4/recipe.yaml --train
echo "train exit=$?"
if [[ -f "$CKPT" ]]; then
  eval_400_evalgen unfreeze_first4 "$CKPT" 32
else
  echo "FAIL missing $CKPT"
fi
echo "==== unfreeze first4 done $(date -u) ===="
# end
