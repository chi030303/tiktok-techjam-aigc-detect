#!/usr/bin/env bash
# 2026-08-31, tianqi, fuse 400 save-preds + galleries + pairwise FN/FP vs SID/unfreeze4/D3
set -u
REPO="${REPO:-/workspace/kiki/tiktok-techjam-aigc-detect}"
export DATA_ROOT="${DATA_ROOT:-/workspace/data}"
export MODELS_ROOT="${MODELS_ROOT:-/workspace/models}"
export EXP_ROOT="${EXP_ROOT:-/workspace/experiments}"
PY="${PY:-/venv/main/bin/python}"
LOG="${EXP_ROOT}/kiki_error_galleries.log"
MANIFEST="${DATA_ROOT}/manifests/source_demo_val.jsonl"
OUT_G="${REPO}/outputs/tables/badcase_galleries"
OUT_C="${REPO}/outputs/tables/badcase_compare"
U4="${EXP_ROOT}/clipb16_linear_sid_unfreeze4/ckpts/best.pt"
D3="${EXP_ROOT}/clipb16_linear_sid_d3_mix/ckpts/best.pt"
PRED400="${EXP_ROOT}/kiki_eval400/official_val"
FUSE_EXP="${EXP_ROOT}/narrative_fuse400"

cd "$REPO"
mkdir -p "$OUT_G" "$OUT_C"
exec >>"$LOG" 2>&1
echo "==== error galleries start $(date -u) ===="

echo "---- pairwise compare existing 400 preds (CPU) ----"
"$PY" -u scripts/compare_badcases.py --html \
  --base "${PRED400}/clipb16_sid/pred_clean.json" --base-name clipb16_sid \
  --new "${PRED400}/unfreeze4/pred_clean.json" --new-name unfreeze4 \
  --split official_val --out-dir "$OUT_C"
"$PY" -u scripts/compare_badcases.py --html \
  --base "${PRED400}/clipb16_sid/pred_clean.json" --base-name clipb16_sid \
  --new "${PRED400}/D3_mix/pred_clean.json" --new-name D3_mix \
  --split official_val --out-dir "$OUT_C"
"$PY" -u scripts/compare_badcases.py --html \
  --base "${PRED400}/unfreeze4/pred_clean.json" --base-name unfreeze4 \
  --new "${PRED400}/D3_mix/pred_clean.json" --new-name D3_mix \
  --split official_val --out-dir "$OUT_C"

if [[ -f "$U4" && -f "$D3" ]]; then
  echo "---- fuse official 400 clean save-preds $(date -u) ----"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" -u scripts/run_full_eval.py \
    --split official_val --conditions clean --max-images 400 --seed 0 \
    --workers 4 --batch 16 --fuse --fuse-weight 0.5 --save-preds \
    --ckpt "unfreeze4=${U4}" --ckpt "D3_mix=${D3}" \
    --experiment narrative_fuse400 --stem official_val400_fuse_u4_d3_preds \
    --out-dir "$REPO/outputs/tables"
fi

FUSE_PRED="${FUSE_EXP}/official_val/fuse_unfreeze4_D3_mix/pred_clean.json"
if [[ ! -f "$FUSE_PRED" ]]; then
  FUSE_PRED=$(find "${FUSE_EXP}" -name 'pred_clean.json' 2>/dev/null | head -1)
fi
if [[ -n "${FUSE_PRED:-}" && -f "$FUSE_PRED" ]]; then
  echo "---- fuse gallery ----"
  "$PY" scripts/badcase_gallery.py \
    --pred "$FUSE_PRED" --split official_val --condition clean \
    --manifest "$MANIFEST" --max-images 400 --seed 0 --max-per-type 60 \
    --out "${OUT_G}/fuse_u4_d3_400_clean.html" \
    --title "fuse last4+D3 official_val 400 clean"
  "$PY" scripts/run_badcase.py \
    --pred "$FUSE_PRED" --split official_val --condition clean \
    --manifest "$MANIFEST" --max-images 400 --seed 0 \
    --out-dir "${OUT_G}/jsonl/fuse_u4_d3_400_clean"
  "$PY" -u scripts/compare_badcases.py --html \
    --base "${PRED400}/clipb16_sid/pred_clean.json" --base-name clipb16_sid \
    --new "$FUSE_PRED" --new-name fuse_u4_d3 \
    --split official_val --out-dir "$OUT_C"
  "$PY" -u scripts/compare_badcases.py --html \
    --base "${PRED400}/unfreeze4/pred_clean.json" --base-name unfreeze4 \
    --new "$FUSE_PRED" --new-name fuse_u4_d3 \
    --split official_val --out-dir "$OUT_C"
  "$PY" -u scripts/compare_badcases.py --html \
    --base "${PRED400}/D3_mix/pred_clean.json" --base-name D3_mix \
    --new "$FUSE_PRED" --new-name fuse_u4_d3 \
    --split official_val --out-dir "$OUT_C"
fi
echo "==== error galleries done $(date -u) ===="
# end
