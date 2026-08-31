#!/bin/bash
# 2026-08-30, tianqi, GPU1 after CLIP queue: SID clean vs clean+robust for linear/mlp/fft
set -u
REPO=/workspace/kiki/tiktok-techjam-aigc-detect
LOG=/workspace/experiments/kiki_sid_mlp_fft.log
CLIP_LOG=/workspace/experiments/kiki_spec_clip.log
OUT=/workspace/experiments/compare_spec/tables
WORK=/workspace/experiments/compare_spec
PY=/venv/main/bin/python
export CUDA_VISIBLE_DEVICES=1
cd "$REPO"
mkdir -p "$OUT"
echo "sid mlp/fft wait $(date -u)" >> "$LOG"

while ! grep -q '^CLIP_QUEUE_EXIT=' "$CLIP_LOG" 2>/dev/null; do
  sleep 30
done
echo "clip queue done, start SID $(date -u)" >> "$LOG"

run_one() {
  local recipe="$1"
  echo "==== train $recipe $(date -u)" >> "$LOG"
  if $PY -u scripts/run_experiment.py "$recipe" --train >> "$LOG" 2>&1; then
    echo "ok $recipe" >> "$LOG"
  else
    echo "FAIL $recipe" >> "$LOG"
  fi
}

# Clean first so RGB feat cache is shared by linear then MLP.
run_one experiments/dinov2l_linear_sid/recipe.yaml
run_one experiments/dinov2l_linear_sid_mlp/recipe.yaml
run_one experiments/dinov2l_linear_sid_fft/recipe.yaml
run_one experiments/dinov2l_linear_sid_mlp_aug/recipe.yaml
run_one experiments/dinov2l_linear_sid_fft_aug/recipe.yaml

CKPTS=()
add_ckpt() {
  local name="$1"
  local path="$2"
  if [ -f "$path" ]; then
    CKPTS+=(--ckpt "${name}=${path}")
  else
    echo "skip missing ckpt $name" >> "$LOG"
  fi
}
add_ckpt sid_dino_clean /workspace/experiments/dinov2l_linear_sid/ckpts/best.pt
add_ckpt sid_dino_aug /workspace/experiments/dinov2l_linear_sid_aug/ckpts/best.pt
add_ckpt sid_dino_mlp /workspace/experiments/dinov2l_linear_sid_mlp/ckpts/best.pt
add_ckpt sid_dino_mlp_aug /workspace/experiments/dinov2l_linear_sid_mlp_aug/ckpts/best.pt
add_ckpt sid_dino_fft /workspace/experiments/dinov2l_linear_sid_fft/ckpts/best.pt
add_ckpt sid_dino_fft_aug /workspace/experiments/dinov2l_linear_sid_fft_aug/ckpts/best.pt

if [ ${#CKPTS[@]} -gt 0 ]; then
  echo "==== eval SID official_val full400 $(date -u)" >> "$LOG"
  $PY -u scripts/run_eval.py compare --split official_val --conditions full --max-images 400 \
    --experiment compare_spec --work-dir "$WORK/official_val" --out-dir "$OUT" \
    --stem official_val_sid_head_full400 \
    "${CKPTS[@]}" >> "$LOG" 2>&1 || echo "FAIL official_val eval" >> "$LOG"
  echo "==== eval SID cifake_test full400 $(date -u)" >> "$LOG"
  $PY -u scripts/run_eval.py compare --split cifake_test --conditions full --max-images 400 \
    --experiment compare_spec --work-dir "$WORK/cifake_test" --out-dir "$OUT" \
    --stem cifake_test_sid_head_full400 \
    "${CKPTS[@]}" >> "$LOG" 2>&1 || echo "FAIL cifake eval" >> "$LOG"
fi

echo "SID_HEAD_EXIT=0 $(date -u)" >> "$LOG"
# end
