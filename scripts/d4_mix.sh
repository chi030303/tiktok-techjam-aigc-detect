#!/usr/bin/env bash
# 2026-08-31, tianqi, D4 after D2: SID mix nano+PixArt+SDXL+GPT (no Hunyuan), then 400+evalgen clean
set -u
REPO="${REPO:-/workspace/kiki/tiktok-techjam-aigc-detect}"
export DATA_ROOT="${DATA_ROOT:-/workspace/data}"
export MODELS_ROOT="${MODELS_ROOT:-/workspace/models}"
export EXP_ROOT="${EXP_ROOT:-/workspace/experiments}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PY="${PY:-/venv/main/bin/python}"
LOG="${EXP_ROOT}/kiki_d4_mix.log"
OUT="${REPO}/outputs/tables"
CKPT="${EXP_ROOT}/clipb16_linear_sid_d4_mix/ckpts/best.pt"

cd "$REPO"
mkdir -p "$OUT" "$(dirname "$LOG")"
# shellcheck disable=SC1091
source "$REPO/scripts/eval_400_evalgen.sh"
exec >>"$LOG" 2>&1
# 2026-08-31, tianqi, SKIP_WAIT=1 starts D4 immediately (evalgen phase2 paused)
if [[ "${SKIP_WAIT:-0}" != 1 ]]; then
  echo "==== wait evalgen then D2 $(date -u) ===="
  while pgrep -f "scripts/evalgen_robust.sh" >/dev/null 2>&1; do sleep 30; done
  while pgrep -f "scripts/d2_selfbuilt.sh" >/dev/null 2>&1; do sleep 30; done
fi
# end
echo "==== D4 start $(date -u) ===="
"$PY" -u scripts/build_d4_mixin.py
"$PY" -u scripts/run_experiment.py experiments/clipb16_linear_sid_d4_mix/recipe.yaml --train
echo "train exit=$?"
if [[ -f "$CKPT" ]]; then
  eval_400_evalgen D4_mix "$CKPT" 32
else
  echo "FAIL missing $CKPT"
fi
echo "==== D4 done $(date -u) ===="
# end
