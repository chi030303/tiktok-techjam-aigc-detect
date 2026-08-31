#!/usr/bin/env bash
# 2026-08-31, tianqi, D6 after Nova: SID mix D5 + i2i fakes, then 400 + evalgen clean + pair_acc
set -u
REPO="${REPO:-/workspace/kiki/tiktok-techjam-aigc-detect}"
export DATA_ROOT="${DATA_ROOT:-/workspace/data}"
export MODELS_ROOT="${MODELS_ROOT:-/workspace/models}"
export EXP_ROOT="${EXP_ROOT:-/workspace/experiments}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PY="${PY:-/venv/main/bin/python}"
LOG="${EXP_ROOT}/kiki_d6_mix.log"
OUT="${REPO}/outputs/tables"
CKPT="${EXP_ROOT}/clipb16_linear_sid_d6_mix/ckpts/best.pt"
A2_MANIFEST="${DATA_ROOT}/manifests/ablation/A2_i2i_hard60.jsonl"

cd "$REPO"
mkdir -p "$OUT" "$(dirname "$LOG")"
# shellcheck disable=SC1091
source "$REPO/scripts/eval_400_evalgen.sh"
exec >>"$LOG" 2>&1

if [[ "${SKIP_WAIT:-0}" != 1 ]]; then
  echo "==== wait fuse/nova $(date -u) ===="
  while pgrep -f "scripts/fuse_d45_nova.sh" >/dev/null 2>&1; do
    echo "wait nova $(date -u)"
    sleep 30
  done
fi
echo "==== D6 start GPU${CUDA_VISIBLE_DEVICES} $(date -u) ===="
"$PY" -u scripts/build_d6_mixin.py
if [[ ! -f "${DATA_ROOT}/manifests/ablation/D6_sid_mixin.jsonl" ]]; then
  echo "FAIL missing D6 jsonl"
  exit 1
fi
"$PY" -u scripts/run_experiment.py experiments/clipb16_linear_sid_d6_mix/recipe.yaml --train
echo "train exit=$?"
if [[ -f "$CKPT" ]]; then
  eval_400_evalgen D6_mix "$CKPT" 32
  echo "==== pair_acc D6 $(date -u) ===="
  "$PY" -u scripts/eval_i2i_triplets.py \
    --manifest "$A2_MANIFEST" \
    --ckpt "D6_mix=${CKPT}" \
    --out "$OUT/i2i_pair_D6_mix.json"
else
  echo "FAIL missing $CKPT"
fi
echo "==== D6 done $(date -u) ===="
# end
