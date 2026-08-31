#!/usr/bin/env bash
# 2026-08-31, tianqi, overnight: wait yun eval, train D3 full-SID mix, then official_val+evalgen
set -u
REPO="${REPO:-/workspace/kiki/tiktok-techjam-aigc-detect}"
export DATA_ROOT="${DATA_ROOT:-/workspace/data}"
export MODELS_ROOT="${MODELS_ROOT:-/workspace/models}"
export EXP_ROOT="${EXP_ROOT:-/workspace/experiments}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PY="${PY:-/venv/main/bin/python}"
LOG="${EXP_ROOT}/kiki_d3_night.log"
OUT="${REPO}/outputs/tables"
YUN_STEM="${EXP_ROOT}/yun_model_ablation"
CKPT="${EXP_ROOT}/clipb16_linear_sid_d3_mix/ckpts/best.pt"

cd "$REPO"
mkdir -p "$OUT" "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "==== d3 night start $(date -u) ===="
echo "repo=$REPO gpu=$CUDA_VISIBLE_DEVICES"

wait_yun() {
  echo "wait until yun official_val_yun_clipb eval is gone..."
  while pgrep -f "official_val_yun_clipb" >/dev/null 2>&1; do
    sleep 60
  done
  echo "yun eval gone $(date -u)"
}

wait_yun

# copy yun formula if it landed
if ls "$OUT"/official_val_yun_clipb_formula.json >/dev/null 2>&1; then
  echo "yun formula already in $OUT"
elif ls "$YUN_STEM"/*formula*.json >/dev/null 2>&1; then
  cp -v "$YUN_STEM"/*formula*.json "$OUT"/ || true
fi
if [[ -f /workspace/kiki/tiktok-techjam-aigc-detect/outputs/tables/official_val_yun_clipb_formula.json ]]; then
  echo "yun formula json present"
fi
grep 'formula=' "${EXP_ROOT}/yun_model_ablation/official_val_full.log" | tail -20 || true

echo "==== build D3 mixin ===="
"$PY" -u scripts/build_d3_mixin.py

echo "==== train D3 $(date -u) ===="
"$PY" -u scripts/run_experiment.py experiments/clipb16_linear_sid_d3_mix/recipe.yaml --train
echo "train exit=$?"

if [[ ! -f "$CKPT" ]]; then
  echo "FAIL: missing $CKPT"
  echo "==== d3 night abort $(date -u) ===="
  exit 1
fi

echo "==== official_val full D3 $(date -u) ===="
"$PY" -u scripts/run_full_eval.py \
  --split official_val --conditions full --workers 4 --batch 32 \
  --ckpt D3_mix="$CKPT" \
  --experiment clipb16_linear_sid_d3_mix \
  --stem official_val_d3_mix \
  --out-dir "$OUT"

echo "==== evalgen D3 $(date -u) ===="
"$PY" -u scripts/run_full_eval.py \
  --split evalgen --reals sid_val --conditions clean --workers 4 --batch 32 \
  --max-fakes-per-gen 400 \
  --ckpt D3_mix="$CKPT" \
  --experiment clipb16_linear_sid_d3_mix \
  --stem evalgen_d3_mix \
  --out-dir "$OUT"

echo "==== d3 night done $(date -u) ===="
# end
