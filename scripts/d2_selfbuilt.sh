#!/usr/bin/env bash
# 2026-08-31, tianqi, after EvalGEN robust: D2 8k-probe train + official 400 + evalgen clean
set -u
REPO="${REPO:-/workspace/kiki/tiktok-techjam-aigc-detect}"
export DATA_ROOT="${DATA_ROOT:-/workspace/data}"
export MODELS_ROOT="${MODELS_ROOT:-/workspace/models}"
export EXP_ROOT="${EXP_ROOT:-/workspace/experiments}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PY="${PY:-/venv/main/bin/python}"
LOG="${EXP_ROOT}/kiki_d2_selfbuilt.log"
OUT="${REPO}/outputs/tables"
CKPT="${EXP_ROOT}/clipb16_linear_D2_selfbuilt/ckpts/best.pt"

cd "$REPO"
mkdir -p "$OUT"
# shellcheck disable=SC1091
source "$REPO/scripts/eval_400_evalgen.sh"
exec >>"$LOG" 2>&1
echo "==== wait kiki-egen-rob $(date -u) ===="
while tmux has-session -t kiki-egen-rob 2>/dev/null; do sleep 30; done
echo "==== D2 start $(date -u) ===="
"$PY" -u scripts/build_d2_selfbuilt.py
"$PY" -u scripts/run_experiment.py experiments/clipb16_linear_D2_selfbuilt/recipe.yaml --train
echo "train exit=$?"
if [[ -f "$CKPT" ]]; then
  eval_400_evalgen D2_selfbuilt "$CKPT" 32
else
  echo "FAIL missing $CKPT"
fi
echo "==== D2 done $(date -u) ===="
# end
