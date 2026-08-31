#!/usr/bin/env bash
# 2026-08-31, tianqi, A-axis: A1 t2i-only / A2 i2i-only / A3 mix, after D5 + pair eval
set -u
REPO="${REPO:-/workspace/kiki/tiktok-techjam-aigc-detect}"
export DATA_ROOT="${DATA_ROOT:-/workspace/data}"
export MODELS_ROOT="${MODELS_ROOT:-/workspace/models}"
export EXP_ROOT="${EXP_ROOT:-/workspace/experiments}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PY="${PY:-/venv/main/bin/python}"
LOG="${EXP_ROOT}/kiki_a_grid.log"
OUT="${REPO}/outputs/tables"
I2I_NANO="/workspace/data_new/output/i2i/i2i_nano_banana"
I2I_CODEX="/workspace/data_new/output/i2i/i2i_codex"
A1_CKPT="${EXP_ROOT}/clipb16_linear_A1_t2i_only/ckpts/best.pt"
A2_CKPT="${EXP_ROOT}/clipb16_linear_A2_i2i_hard60/ckpts/best.pt"
A3_CKPT="${EXP_ROOT}/clipb16_linear_A3_t2i_i2i_mix/ckpts/best.pt"
A2_MANIFEST="${DATA_ROOT}/manifests/ablation/A2_i2i_hard60.jsonl"

cd "$REPO"
mkdir -p "$OUT" "$(dirname "$LOG")"
# shellcheck disable=SC1091
source "$REPO/scripts/eval_400_evalgen.sh"
exec >>"$LOG" 2>&1

echo "==== wait D4/D5/A2-eval $(date -u) ===="
while pgrep -f "scripts/d4_mix.sh" >/dev/null 2>&1 \
   || pgrep -f "scripts/d5_mix.sh" >/dev/null 2>&1 \
   || pgrep -f "scripts/a2_i2i.sh" >/dev/null 2>&1; do
  echo "A-grid waiting for D4/D5/A2-eval $(date -u)"
  sleep 60
done

# 2026-08-31, tianqi, A1/A2 may run on GPU0; this script only fills gaps then A3
if pgrep -f "scripts/a2_gpu0.sh" >/dev/null 2>&1 \
   || pgrep -f "scripts/a3_gpu1.sh" >/dev/null 2>&1; then
  echo "A-grid skip: a2_gpu0/a3_gpu1 already running $(date -u)"
  exit 0
fi
# end

echo "==== wait i2i upload $(date -u) ===="
while true; do
  n_nano=$(find "$I2I_NANO" -type f \( -iname '*.png' -o -iname '*.jpg' \) 2>/dev/null | wc -l | tr -d ' ')
  n_codex=$(find "$I2I_CODEX" -type f \( -iname '*.png' -o -iname '*.jpg' \) 2>/dev/null | wc -l | tr -d ' ')
  echo "nano=$n_nano codex=$n_codex $(date -u)"
  if [[ "${n_nano}" -ge 50 && "${n_codex}" -ge 50 ]]; then
    break
  fi
  sleep 60
done

"$PY" -u scripts/build_d2_selfbuilt.py
"$PY" -u scripts/build_a2_i2i.py --min-triplets 50 --out "$A2_MANIFEST"
"$PY" -u scripts/build_a3_mix.py --a2 "$A2_MANIFEST"

pair_eval() {
  local name="$1"
  local ckpt="$2"
  if [[ ! -f "$ckpt" ]]; then
    echo "skip pair_acc missing $name $ckpt"
    return 0
  fi
  echo "==== pair_acc $name $(date -u) ===="
  "$PY" -u scripts/eval_i2i_triplets.py \
    --manifest "$A2_MANIFEST" \
    --ckpt "${name}=${ckpt}" \
    --out "$OUT/i2i_pair_${name}.json"
}

echo "==== A1 t2i-only $(date -u) ===="
"$PY" -u scripts/run_experiment.py experiments/clipb16_linear_A1_t2i_only/recipe.yaml --train
echo "A1 train exit=$?"
if [[ -f "$A1_CKPT" ]]; then
  eval_400_evalgen A1_t2i_only "$A1_CKPT" 32
  pair_eval A1_t2i_only "$A1_CKPT"
else
  echo "FAIL missing $A1_CKPT"
fi

echo "==== A2 i2i-only $(date -u) ===="
"$PY" -u scripts/run_experiment.py experiments/clipb16_linear_A2_i2i_hard60/recipe.yaml --train
echo "A2 train exit=$?"
if [[ -f "$A2_CKPT" ]]; then
  eval_400_evalgen A2_i2i_hard60 "$A2_CKPT" 32
  pair_eval A2_i2i_hard60 "$A2_CKPT"
else
  echo "FAIL missing $A2_CKPT"
fi

echo "==== A3 t2i+i2i mix $(date -u) ===="
"$PY" -u scripts/run_experiment.py experiments/clipb16_linear_A3_t2i_i2i_mix/recipe.yaml --train
echo "A3 train exit=$?"
if [[ -f "$A3_CKPT" ]]; then
  eval_400_evalgen A3_t2i_i2i_mix "$A3_CKPT" 32
  pair_eval A3_t2i_i2i_mix "$A3_CKPT"
else
  echo "FAIL missing $A3_CKPT"
fi

echo "==== A-grid done $(date -u) ===="
# end
