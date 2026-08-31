#!/usr/bin/env bash
# 2026-08-31, tianqi, D5 after D4: SID mix D3∪D4, then official 400 + evalgen clean
set -u
REPO="${REPO:-/workspace/kiki/tiktok-techjam-aigc-detect}"
export DATA_ROOT="${DATA_ROOT:-/workspace/data}"
export MODELS_ROOT="${MODELS_ROOT:-/workspace/models}"
export EXP_ROOT="${EXP_ROOT:-/workspace/experiments}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PY="${PY:-/venv/main/bin/python}"
LOG="${EXP_ROOT}/kiki_d5_mix.log"
OUT="${REPO}/outputs/tables"
CKPT="${EXP_ROOT}/clipb16_linear_sid_d5_mix/ckpts/best.pt"

cd "$REPO"
mkdir -p "$OUT" "$(dirname "$LOG")"
# shellcheck disable=SC1091
source "$REPO/scripts/eval_400_evalgen.sh"
exec >>"$LOG" 2>&1
echo "==== wait D4 $(date -u) ===="
while pgrep -f "scripts/d4_mix.sh" >/dev/null 2>&1; do sleep 30; done
echo "==== D5 start $(date -u) ===="
"$PY" -u scripts/build_d5_mixin.py
"$PY" -u scripts/run_experiment.py experiments/clipb16_linear_sid_d5_mix/recipe.yaml --train
echo "train exit=$?"
if [[ -f "$CKPT" ]]; then
  eval_400_evalgen D5_mix "$CKPT" 32
else
  echo "FAIL missing $CKPT"
fi
echo "==== D5 done $(date -u) ===="
# end
