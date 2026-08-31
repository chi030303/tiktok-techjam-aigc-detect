#!/usr/bin/env bash
# 2026-08-31, tianqi, GPU1 10h: wait yun, D3 mix train, CLIP-L unfreeze4, 400+evalgen-full
set -u
REPO="${REPO:-/workspace/kiki/tiktok-techjam-aigc-detect}"
export DATA_ROOT="${DATA_ROOT:-/workspace/data}"
export MODELS_ROOT="${MODELS_ROOT:-/workspace/models}"
export EXP_ROOT="${EXP_ROOT:-/workspace/experiments}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PY="${PY:-/venv/main/bin/python}"
LOG="${EXP_ROOT}/kiki_gpu1_10h.log"
OUT="${REPO}/outputs/tables"
D3_CKPT="${EXP_ROOT}/clipb16_linear_sid_d3_mix/ckpts/best.pt"
L4_CKPT="${EXP_ROOT}/clipl14_linear_sid_unfreeze4/ckpts/best.pt"

cd "$REPO"
mkdir -p "$OUT" "$(dirname "$LOG")"
# shellcheck disable=SC1091
source "$REPO/scripts/eval_400_evalgen.sh"
exec >>"$LOG" 2>&1
echo "==== gpu1 10h start $(date -u) ===="
echo "repo=$REPO gpu=$CUDA_VISIBLE_DEVICES"

wait_yun() {
  echo "wait until yun official_val_yun_clipb eval is gone..."
  while pgrep -f "official_val_yun_clipb" >/dev/null 2>&1; do
    sleep 60
  done
  echo "yun eval gone $(date -u)"
}

wait_yun
grep 'formula=' "${EXP_ROOT}/yun_model_ablation/official_val_full.log" | tail -20 || true
cp -v "$OUT"/official_val_yun_clipb_formula.json "$OUT"/ 2>/dev/null || true

echo "==== build D3 mixin ===="
"$PY" -u scripts/build_d3_mixin.py

echo "==== train D3 $(date -u) ===="
"$PY" -u scripts/run_experiment.py experiments/clipb16_linear_sid_d3_mix/recipe.yaml --train
echo "D3 train exit=$?"
if [[ -f "$D3_CKPT" ]]; then
  eval_400_evalgen D3_mix "$D3_CKPT" 32
else
  echo "FAIL: missing D3 ckpt"
fi

echo "==== train CLIP-L unfreeze4 $(date -u) ===="
"$PY" -u scripts/run_experiment.py experiments/clipl14_linear_sid_unfreeze4/recipe.yaml --train
echo "L4 train exit=$?"
if [[ -f "$L4_CKPT" ]]; then
  eval_400_evalgen clipl14_unfreeze4 "$L4_CKPT" 16
else
  echo "FAIL: missing CLIP-L unfreeze4 ckpt"
fi

COMBO="${EXP_ROOT}/clipb16_linear_sid_unfreeze4_res336/ckpts/best.pt"
if [[ ! -f "$COMBO" ]] && ! pgrep -f "clipb16_linear_sid_unfreeze4_res336" >/dev/null 2>&1; then
  echo "==== GPU1 fallback train unfreeze4+res336 $(date -u) ===="
  "$PY" -u scripts/run_experiment.py experiments/clipb16_linear_sid_unfreeze4_res336/recipe.yaml --train || true
fi
if [[ -f "$COMBO" ]]; then
  eval_400_evalgen unfreeze4_res336 "$COMBO" 16
fi

echo "==== gpu1 10h done $(date -u) ===="
# end
