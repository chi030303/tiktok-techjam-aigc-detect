#!/bin/bash
# 2026-08-30, tianqi, GPU1: re-eval existing ckpts on official 14-setting grid, then CLIP mlp/fft/SID
set -u
REPO=/workspace/kiki/tiktok-techjam-aigc-detect
LOG=/workspace/experiments/kiki_spec_clip.log
OUT=/workspace/experiments/compare_spec/tables
WORK=/workspace/experiments/compare_spec
PY=/venv/main/bin/python
export CUDA_VISIBLE_DEVICES=1
cd "$REPO"
mkdir -p "$OUT"
echo "start $(date -u)" >> "$LOG"

run_one() {
  local recipe="$1"
  echo "==== train $recipe $(date -u)" >> "$LOG"
  if $PY -u scripts/run_experiment.py "$recipe" --train >> "$LOG" 2>&1; then
    echo "ok $recipe" >> "$LOG"
  else
    echo "FAIL $recipe" >> "$LOG"
  fi
}

CKPTS=()
add_ckpt() {
  local name="$1"
  local path="$2"
  if [ -f "$path" ]; then
    CKPTS+=(--ckpt "${name}=${path}")
  else
    echo "skip missing ckpt $name $path" >> "$LOG"
  fi
}

# Existing runs (clean / random-aug / six-op / night). Skip if a teammate deleted a ckpt.
add_ckpt resnet_clean /workspace/experiments/resnet50_linear_cifake/ckpts/best.pt
add_ckpt dinov2_clean /workspace/experiments/dinov2l_linear_cifake/ckpts/best.pt
add_ckpt clipb16_clean /workspace/experiments/clipb16_linear_cifake_full/ckpts/best.pt
add_ckpt clipl14_clean /workspace/experiments/clipl14_linear_cifake_full/ckpts/best.pt
add_ckpt resnet_aug /workspace/experiments/resnet50_linear_cifake_aug/ckpts/best.pt
add_ckpt dinov2_aug /workspace/experiments/dinov2l_linear_cifake_aug/ckpts/best.pt
add_ckpt clipb16_aug /workspace/experiments/clipb16_linear_cifake_aug/ckpts/best.pt
add_ckpt clipl14_aug /workspace/experiments/clipl14_linear_cifake_aug/ckpts/best.pt
add_ckpt resnet_aug6 /workspace/experiments/resnet50_linear_cifake_aug6/ckpts/best.pt
add_ckpt dinov2_aug6 /workspace/experiments/dinov2l_linear_cifake_aug6/ckpts/best.pt
add_ckpt clipb16_aug6 /workspace/experiments/clipb16_linear_cifake_aug6/ckpts/best.pt
add_ckpt clipl14_aug6 /workspace/experiments/clipl14_linear_cifake_aug6/ckpts/best.pt
add_ckpt dinov2_mlp /workspace/experiments/dinov2l_linear_cifake_mlp/ckpts/best.pt
add_ckpt dinov2_fft /workspace/experiments/dinov2l_linear_cifake_fft/ckpts/best.pt
add_ckpt sid_resnet /workspace/experiments/resnet50_linear_sid_aug/ckpts/best.pt
add_ckpt sid_dinov2 /workspace/experiments/dinov2l_linear_sid_aug/ckpts/best.pt

eval_split() {
  local split="$1"
  local stem="$2"
  echo "==== eval $split full400 $(date -u)" >> "$LOG"
  $PY -u scripts/run_eval.py compare --split "$split" --conditions full --max-images 400 \
    --experiment compare_spec --work-dir "$WORK/$split" --out-dir "$OUT" --stem "$stem" \
    "${CKPTS[@]}" >> "$LOG" 2>&1 || echo "FAIL eval $split" >> "$LOG"
}

if [ ${#CKPTS[@]} -gt 0 ]; then
  eval_split official_val official_val_spec_full400
  eval_split cifake_test cifake_test_spec_full400
else
  echo "FAIL no ckpts for spec eval" >> "$LOG"
fi

echo "SPEC_EVAL_DONE $(date -u)" >> "$LOG"

# CLIP fill-in: mlp/fft (feat-cache) then SID online aug (B then L)
run_one experiments/clipb16_linear_cifake_mlp/recipe.yaml
run_one experiments/clipl14_linear_cifake_mlp/recipe.yaml
run_one experiments/clipb16_linear_cifake_fft/recipe.yaml
run_one experiments/clipl14_linear_cifake_fft/recipe.yaml
run_one experiments/clipb16_linear_sid_aug/recipe.yaml
run_one experiments/clipl14_linear_sid_aug/recipe.yaml

CLIP_CKPTS=()
add_clip() {
  local name="$1"
  local path="$2"
  if [ -f "$path" ]; then
    CLIP_CKPTS+=(--ckpt "${name}=${path}")
  else
    echo "skip missing clip ckpt $name" >> "$LOG"
  fi
}
add_clip clipb16_mlp /workspace/experiments/clipb16_linear_cifake_mlp/ckpts/best.pt
add_clip clipl14_mlp /workspace/experiments/clipl14_linear_cifake_mlp/ckpts/best.pt
add_clip clipb16_fft /workspace/experiments/clipb16_linear_cifake_fft/ckpts/best.pt
add_clip clipl14_fft /workspace/experiments/clipl14_linear_cifake_fft/ckpts/best.pt
add_clip clipb16_sid /workspace/experiments/clipb16_linear_sid_aug/ckpts/best.pt
add_clip clipl14_sid /workspace/experiments/clipl14_linear_sid_aug/ckpts/best.pt

if [ ${#CLIP_CKPTS[@]} -gt 0 ]; then
  echo "==== eval CLIP fill-in official_val $(date -u)" >> "$LOG"
  $PY -u scripts/run_eval.py compare --split official_val --conditions full --max-images 400 \
    --experiment compare_spec --work-dir "$WORK/official_val" --out-dir "$OUT" \
    --stem official_val_clip_full400 \
    "${CLIP_CKPTS[@]}" >> "$LOG" 2>&1 || echo "FAIL clip official_val eval" >> "$LOG"
  echo "==== eval CLIP fill-in cifake_test $(date -u)" >> "$LOG"
  $PY -u scripts/run_eval.py compare --split cifake_test --conditions full --max-images 400 \
    --experiment compare_spec --work-dir "$WORK/cifake_test" --out-dir "$OUT" \
    --stem cifake_test_clip_full400 \
    "${CLIP_CKPTS[@]}" >> "$LOG" 2>&1 || echo "FAIL clip cifake eval" >> "$LOG"
fi

echo "CLIP_QUEUE_EXIT=0 $(date -u)" >> "$LOG"
# end
