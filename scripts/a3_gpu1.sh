#!/usr/bin/env bash
# 2026-08-31, tianqi, GPU1: finish A1 eval if needed, then A3 mix
set -u
REPO="${REPO:-/workspace/kiki/tiktok-techjam-aigc-detect}"
export DATA_ROOT="${DATA_ROOT:-/workspace/data}"
export MODELS_ROOT="${MODELS_ROOT:-/workspace/models}"
export EXP_ROOT="${EXP_ROOT:-/workspace/experiments}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PY="${PY:-/venv/main/bin/python}"
LOG="${EXP_ROOT}/kiki_a3_gpu1.log"
OUT="${REPO}/outputs/tables"
A1_CKPT="${EXP_ROOT}/clipb16_linear_A1_t2i_only/ckpts/best.pt"
A3_CKPT="${EXP_ROOT}/clipb16_linear_A3_t2i_i2i_mix/ckpts/best.pt"
A2_MANIFEST="${DATA_ROOT}/manifests/ablation/A2_i2i_hard60.jsonl"

cd "$REPO"
mkdir -p "$OUT" "$(dirname "$LOG")"
# shellcheck disable=SC1091
source "$REPO/scripts/eval_400_evalgen.sh"
exec >>"$LOG" 2>&1

if [[ ! -f "$A2_MANIFEST" ]]; then
  "$PY" -u scripts/build_a2_i2i.py --min-triplets 50 --out "$A2_MANIFEST"
fi
"$PY" -u scripts/build_d2_selfbuilt.py
"$PY" -u scripts/build_a3_mix.py --a2 "$A2_MANIFEST"
# 2026-08-31, tianqi, do not train A3 if mixin jsonl failed to build
if [[ ! -f "${DATA_ROOT}/manifests/ablation/A3_t2i_i2i_mix.jsonl" ]]; then
  echo "FAIL missing A3 jsonl"
  exit 1
fi
# end

if [[ -f "$A1_CKPT" ]]; then
  if [[ ! -f "$OUT/evalgen_full_A1_t2i_only_formula.json" ]]; then
    echo "==== finish A1 eval GPU1 $(date -u) ===="
    eval_400_evalgen A1_t2i_only "$A1_CKPT" 32
  fi
  if [[ ! -f "$OUT/i2i_pair_A1_t2i_only.json" ]]; then
    echo "==== pair_acc A1 $(date -u) ===="
    "$PY" -u scripts/eval_i2i_triplets.py \
      --manifest "$A2_MANIFEST" \
      --ckpt "A1_t2i_only=${A1_CKPT}" \
      --out "$OUT/i2i_pair_A1_t2i_only.json"
  fi
fi

echo "==== A3 train GPU1 $(date -u) ===="
"$PY" -u scripts/run_experiment.py experiments/clipb16_linear_A3_t2i_i2i_mix/recipe.yaml --train
echo "A3 train exit=$?"
if [[ -f "$A3_CKPT" ]]; then
  eval_400_evalgen A3_t2i_i2i_mix "$A3_CKPT" 32
  echo "==== pair_acc A3 $(date -u) ===="
  "$PY" -u scripts/eval_i2i_triplets.py \
    --manifest "$A2_MANIFEST" \
    --ckpt "A3_t2i_i2i_mix=${A3_CKPT}" \
    --out "$OUT/i2i_pair_A3_t2i_i2i_mix.json"
else
  echo "FAIL missing $A3_CKPT"
fi
echo "==== A3 GPU1 done $(date -u) ===="
# end
