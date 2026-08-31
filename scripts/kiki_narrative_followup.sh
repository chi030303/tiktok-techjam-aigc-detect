#!/usr/bin/env bash
# 2026-08-31, tianqi, after first-4: logit fuse, threshold sweep, photoreal/stylized FN slice
set -u
REPO="${REPO:-/workspace/kiki/tiktok-techjam-aigc-detect}"
export DATA_ROOT="${DATA_ROOT:-/workspace/data}"
export MODELS_ROOT="${MODELS_ROOT:-/workspace/models}"
export EXP_ROOT="${EXP_ROOT:-/workspace/experiments}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PY="${PY:-/venv/main/bin/python}"
LOG="${EXP_ROOT}/kiki_narrative_followup.log"
OUT="${REPO}/outputs/tables"
U4="${EXP_ROOT}/clipb16_linear_sid_unfreeze4/ckpts/best.pt"
D3="${EXP_ROOT}/clipb16_linear_sid_d3_mix/ckpts/best.pt"
D3U4="${EXP_ROOT}/clipb16_linear_d3_unfreeze4/ckpts/best.pt"
WORK="${EXP_ROOT}/narrative_followup"

cd "$REPO"
mkdir -p "$OUT" "$WORK"
exec >>"$LOG" 2>&1
echo "==== wait first4 $(date -u) ===="
while tmux has-session -t kiki-first4 2>/dev/null; do sleep 30; done
echo "==== narrative followup start $(date -u) ===="

echo "---- fuse unfreeze4 + D3 mix, official 400 ----"
"$PY" -u scripts/run_full_eval.py \
  --split official_val --conditions full --max-images 400 --seed 0 \
  --workers 4 --batch 32 --fuse --fuse-weight 0.5 \
  --ckpt "unfreeze4=${U4}" --ckpt "D3_mix=${D3}" \
  --experiment narrative_fuse --stem official_val400_fuse_u4_d3 --out-dir "$OUT"

echo "---- fuse evalgen full + save preds ----"
"$PY" -u scripts/run_full_eval.py \
  --split evalgen --reals sid_val --conditions clean \
  --workers 4 --batch 32 --fuse --fuse-weight 0.5 --save-preds \
  --ckpt "unfreeze4=${U4}" --ckpt "D3_mix=${D3}" \
  --experiment narrative_fuse --stem evalgen_full_fuse_u4_d3 --out-dir "$OUT"

echo "---- evalgen preds for threshold: D3 unfreeze4 ----"
"$PY" -u scripts/run_full_eval.py \
  --split evalgen --reals sid_val --conditions clean \
  --workers 4 --batch 32 --save-preds \
  --ckpt "d3_unfreeze4=${D3U4}" \
  --experiment narrative_thresh --stem evalgen_full_d3u4_preds --out-dir "$OUT"

echo "---- official 400 clean preds for style slice (unfreeze4) ----"
"$PY" -u scripts/run_full_eval.py \
  --split official_val --conditions clean --max-images 400 --seed 0 \
  --workers 4 --batch 32 --save-preds \
  --ckpt "unfreeze4=${U4}" \
  --experiment narrative_style --stem official_val400_u4_cleanpreds --out-dir "$OUT"

FUSE_PRED="${EXP_ROOT}/narrative_fuse/evalgen_sidreals/fuse_unfreeze4_D3_mix/pred_clean.json"
D3U4_PRED="${EXP_ROOT}/narrative_thresh/evalgen_sidreals/d3_unfreeze4/pred_clean.json"
U4_PRED="${EXP_ROOT}/narrative_style/official_val/unfreeze4/pred_clean.json"
# fallback glob if split name differs
if [[ ! -f "$FUSE_PRED" ]]; then
  FUSE_PRED=$(find "${EXP_ROOT}/narrative_fuse" -name 'pred_clean.json' | head -1)
fi
if [[ ! -f "$D3U4_PRED" ]]; then
  D3U4_PRED=$(find "${EXP_ROOT}/narrative_thresh" -name 'pred_clean.json' | head -1)
fi
if [[ ! -f "$U4_PRED" ]]; then
  U4_PRED=$(find "${EXP_ROOT}/narrative_style" -name 'pred_clean.json' | head -1)
fi

echo "---- threshold sweep D3+unfreeze4 evalgen ----"
if [[ -n "${D3U4_PRED}" && -f "$D3U4_PRED" ]]; then
  "$PY" -u scripts/eval_threshold_style.py --preds "$D3U4_PRED" --out-dir "$OUT" \
    --stem thresh_d3_unfreeze4 --fake-root "${DATA_ROOT}/evalgen"
fi
echo "---- threshold sweep fused evalgen ----"
if [[ -n "${FUSE_PRED}" && -f "$FUSE_PRED" ]]; then
  "$PY" -u scripts/eval_threshold_style.py --preds "$FUSE_PRED" --out-dir "$OUT" \
    --stem thresh_fuse_u4_d3 --fake-root "${DATA_ROOT}/evalgen"
fi
echo "---- style slice official val 400 unfreeze4 ----"
if [[ -n "${U4_PRED}" && -f "$U4_PRED" ]]; then
  "$PY" -u scripts/eval_threshold_style.py --preds "$U4_PRED" --out-dir "$OUT" \
    --stem style_u4_val400 --style-slice
fi

echo "==== narrative followup done $(date -u) ===="
# end
