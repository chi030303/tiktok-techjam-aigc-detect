#!/usr/bin/env bash
# 2026-08-31, tianqi, 400-subset clean badcase galleries for representative ckpts
set -u
REPO=/workspace/kiki/tiktok-techjam-aigc-detect
export DATA_ROOT=/workspace/data
export MODELS_ROOT=/workspace/models
export EXP_ROOT=/workspace/experiments
export CUDA_VISIBLE_DEVICES=1
PY=/venv/main/bin/python
LOG="${EXP_ROOT}/kiki_badcase400.log"
OUT_G="${EXP_ROOT}/kiki_eval/galleries"
OUT_B="${EXP_ROOT}/kiki_eval/badcases"
MANIFEST="${DATA_ROOT}/manifests/source_demo_val.jsonl"
cd "$REPO"
mkdir -p "$OUT_G" "$OUT_B"
exec >>"$LOG" 2>&1
echo "==== badcase400 start $(date -u) ===="

echo "==== predict 400 clean ===="
"$PY" -u scripts/run_full_eval.py \
  --split official_val --conditions clean --max-images 400 --seed 0 \
  --workers 4 --batch 16 --save-preds \
  --experiment kiki_eval400 \
  --stem official_val400_badcase \
  --out-dir "$REPO/outputs/tables" \
  --ckpt unfreeze4="${EXP_ROOT}/clipb16_linear_sid_unfreeze4/ckpts/best.pt" \
  --ckpt D3_mix="${EXP_ROOT}/clipb16_linear_sid_d3_mix/ckpts/best.pt" \
  --ckpt clipl14_unfreeze4="${EXP_ROOT}/clipl14_linear_sid_unfreeze4/ckpts/best.pt" \
  --ckpt unfreeze4_res336="${EXP_ROOT}/clipb16_linear_sid_unfreeze4_res336/ckpts/best.pt" \
  --ckpt clipb16_sid="${EXP_ROOT}/clipb16_linear_sid_aug/ckpts/best.pt" \
  --ckpt clipl14_sid="${EXP_ROOT}/clipl14_linear_sid_aug/ckpts/best.pt"

PRED_ROOT="${EXP_ROOT}/kiki_eval400/official_val"
for name in unfreeze4 D3_mix clipl14_unfreeze4 unfreeze4_res336 clipb16_sid clipl14_sid; do
  pred="${PRED_ROOT}/${name}/pred_clean.json"
  if [[ ! -f "$pred" ]]; then
    echo "missing $pred"
    continue
  fi
  bdir="${OUT_B}/${name}_400_clean"
  html="${OUT_G}/${name}_400_clean.html"
  mkdir -p "$bdir"
  echo "==== gallery $name ===="
  "$PY" scripts/run_badcase.py \
    --pred "$pred" --split official_val --condition clean \
    --manifest "$MANIFEST" --max-images 400 --seed 0 \
    --out-dir "$bdir"
  "$PY" scripts/badcase_gallery.py \
    --pred "$pred" --split official_val --condition clean \
    --manifest "$MANIFEST" --max-images 400 --seed 0 \
    --max-per-type 60 \
    --out "$html" \
    --title "${name} official_val 400 clean"
done
echo "==== badcase400 done $(date -u) ===="
# end
