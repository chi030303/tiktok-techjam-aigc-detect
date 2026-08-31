#!/usr/bin/env bash
# 2026-08-31, tianqi, A2 i2i-only on GPU0; A3 stays on GPU1
set -u
REPO="${REPO:-/workspace/kiki/tiktok-techjam-aigc-detect}"
export DATA_ROOT="${DATA_ROOT:-/workspace/data}"
export MODELS_ROOT="${MODELS_ROOT:-/workspace/models}"
export EXP_ROOT="${EXP_ROOT:-/workspace/experiments}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="${PY:-/venv/main/bin/python}"
LOG="${EXP_ROOT}/kiki_a2_gpu0.log"
OUT="${REPO}/outputs/tables"
A2_CKPT="${EXP_ROOT}/clipb16_linear_A2_i2i_hard60/ckpts/best.pt"
A2_MANIFEST="${DATA_ROOT}/manifests/ablation/A2_i2i_hard60.jsonl"

cd "$REPO"
mkdir -p "$OUT" "$(dirname "$LOG")"
# shellcheck disable=SC1091
source "$REPO/scripts/eval_400_evalgen.sh"
exec >>"$LOG" 2>&1

if [[ ! -f "$A2_MANIFEST" ]]; then
  "$PY" -u scripts/build_a2_i2i.py --min-triplets 50 --out "$A2_MANIFEST"
fi

echo "==== A2 train GPU0 $(date -u) ===="
"$PY" -u scripts/run_experiment.py experiments/clipb16_linear_A2_i2i_hard60/recipe.yaml --train
echo "A2 train exit=$?"
if [[ -f "$A2_CKPT" ]]; then
  eval_400_evalgen A2_i2i_hard60 "$A2_CKPT" 32
  echo "==== pair_acc A2 $(date -u) ===="
  "$PY" -u scripts/eval_i2i_triplets.py \
    --manifest "$A2_MANIFEST" \
    --ckpt "A2_i2i_hard60=${A2_CKPT}" \
    --out "$OUT/i2i_pair_A2_i2i_hard60.json"
else
  echo "FAIL missing $A2_CKPT"
fi
echo "==== A2 GPU0 done $(date -u) ===="
# end
