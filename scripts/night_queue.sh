#!/bin/bash
# 2026-08-29, tianqi, GPU1 overnight queue; runs on Vast tmux, no local confirm
set -u
REPO=/workspace/kiki/tiktok-techjam-aigc-detect
LOG=/workspace/experiments/kiki_night.log
OUT=/workspace/experiments/compare_four/tables
PY=/venv/main/bin/python
export CUDA_VISIBLE_DEVICES=1
cd "$REPO"
mkdir -p "$OUT"
echo "night start $(date -u)" >> "$LOG"

wait_aug6() {
  echo "waiting for aug6 EXIT $(date -u)" >> "$LOG"
  while ! grep -q '^EXIT=' /workspace/experiments/kiki_cifake_aug6.log 2>/dev/null; do
    sleep 30
  done
  echo "aug6 done $(date -u)" >> "$LOG"
}

run_one() {
  local recipe="$1"
  echo "==== $recipe $(date -u)" >> "$LOG"
  if $PY -u scripts/run_experiment.py "$recipe" --train >> "$LOG" 2>&1; then
    echo "ok $recipe" >> "$LOG"
  else
    echo "FAIL $recipe" >> "$LOG"
  fi
}

wait_aug6

# CIFAKE: MLP encoder + FFT spectrum (feat-cacheable)
run_one experiments/dinov2l_linear_cifake_mlp/recipe.yaml
run_one experiments/dinov2l_linear_cifake_fft/recipe.yaml

# SID_Set: domain shift, random online 6-op (not 7x expand)
run_one experiments/resnet50_linear_sid_aug/recipe.yaml
run_one experiments/dinov2l_linear_sid_aug/recipe.yaml

CKPTS=()
add_ckpt() {
  local name="$1"
  local path="$2"
  if [ -f "$path" ]; then
    CKPTS+=(--ckpt "${name}=${path}")
  fi
}
add_ckpt mlp /workspace/experiments/dinov2l_linear_cifake_mlp/ckpts/best.pt
add_ckpt fft /workspace/experiments/dinov2l_linear_cifake_fft/ckpts/best.pt
add_ckpt sid_resnet /workspace/experiments/resnet50_linear_sid_aug/ckpts/best.pt
add_ckpt sid_dinov2 /workspace/experiments/dinov2l_linear_sid_aug/ckpts/best.pt
add_ckpt cifake_dinov2_aug6 /workspace/experiments/dinov2l_linear_cifake_aug6/ckpts/best.pt

if [ ${#CKPTS[@]} -gt 0 ]; then
  echo "==== eval official_val $(date -u)" >> "$LOG"
  $PY -u scripts/run_eval.py compare --split official_val --conditions daily --max-images 400 \
    --experiment compare_night --out-dir "$OUT" --stem official_val_night_daily400 \
    "${CKPTS[@]}" >> "$LOG" 2>&1 || echo "FAIL official_val eval" >> "$LOG"
  echo "==== eval cifake_test $(date -u)" >> "$LOG"
  $PY -u scripts/run_eval.py compare --split cifake_test --conditions daily --max-images 400 \
    --experiment compare_night --out-dir "$OUT" --stem cifake_test_night_daily400 \
    "${CKPTS[@]}" >> "$LOG" 2>&1 || echo "FAIL cifake eval" >> "$LOG"
fi

echo "NIGHT_EXIT=0 $(date -u)" >> "$LOG"
# end
