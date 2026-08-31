#!/usr/bin/env bash
# 2026-08-30, tianqi, dump all FP/FN + HTML galleries after main gallery merge
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export DATA_ROOT="${DATA_ROOT:-/workspace/data}"
export MODELS_ROOT="${MODELS_ROOT:-/workspace/models}"
export EXP_ROOT="${EXP_ROOT:-/workspace/experiments}"
PY="${PY:-/venv/main/bin/python}"

COMPARE="${EXP_ROOT}/compare_spec/official_val"
MANIFEST="${DATA_ROOT}/manifests/source_demo_val.jsonl"
OUT_G="${EXP_ROOT}/kiki_eval/galleries"
OUT_B="${EXP_ROOT}/kiki_eval/badcases"
PRED_FULL="${EXP_ROOT}/kiki_eval/preds"
mkdir -p "$OUT_G" "$OUT_B" "$PRED_FULL"

dump_and_gallery() {
  local pred="$1" name="$2" condition="$3" predict_root="$4" extra_split="${5:-}"
  local bdir="${OUT_B}/${name}_${condition}"
  local html="${OUT_G}/${name}_${condition}.html"
  mkdir -p "$bdir"
  echo "==== badcase ${name} ${condition} ===="
  # shellcheck disable=SC2086
  $PY scripts/run_badcase.py \
    --pred "$pred" \
    --predict-root "$predict_root" \
    --condition "$condition" \
    --manifest "$MANIFEST" \
    --out-dir "$bdir" \
    $extra_split
  $PY scripts/badcase_gallery.py \
    --pred "$pred" \
    --predict-root "$predict_root" \
    --condition "$condition" \
    --manifest "$MANIFEST" \
    --max-per-type 60 \
    --out "$html" \
    --title "${name} ${condition}" \
    $extra_split
}

echo "==== 400-subset compare_spec (CPU) ===="
HEADLINE=(clipb16_sid clipl14_sid sid_dinov2 dinov2_clean clipb16_clean)
for model in "${HEADLINE[@]}"; do
  pred="${COMPARE}/${model}/pred_clean.json"
  if [[ -f "$pred" ]]; then
    dump_and_gallery "$pred" "${model}_400" clean "${COMPARE}/images/clean" "--image-dir ${COMPARE}/images/clean"
  fi
done
# remaining models: jsonl dump only (all FP/FN), no HTML
while IFS= read -r pred; do
  model="$(basename "$(dirname "$pred")")"
  case "$model" in
    clipb16_sid|clipl14_sid|sid_dinov2|dinov2_clean|clipb16_clean) continue ;;
  esac
  bdir="${OUT_B}/${model}_400_clean"
  mkdir -p "$bdir"
  echo "==== dump ${model} clean400 ===="
  $PY scripts/run_badcase.py \
    --pred "$pred" \
    --image-dir "${COMPARE}/images/clean" \
    --predict-root "${COMPARE}/images/clean" \
    --condition clean \
    --manifest "$MANIFEST" \
    --out-dir "$bdir"
done < <(find "$COMPARE" -mindepth 2 -maxdepth 2 -name pred_clean.json | sort)

for cond in jpeg_q50 crop_p80; do
  for model in clipb16_sid clipl14_sid; do
    pred="${COMPARE}/${model}/pred_${cond}.json"
    [[ -f "$pred" ]] || continue
    dump_and_gallery "$pred" "${model}_400" "$cond" "${COMPARE}/images/${cond}" "--image-dir ${COMPARE}/images/${cond}"
  done
done

echo "==== full official_val predict (GPU0) ===="
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
declare -A CKPTS=(
  [clipb16_sid]="${EXP_ROOT}/clipb16_linear_sid_aug/ckpts/best.pt"
  [clipl14_sid]="${EXP_ROOT}/clipl14_linear_sid_aug/ckpts/best.pt"
  [D1_sid]="${EXP_ROOT}/clipb16_linear_D1_sid_only/ckpts/best.pt"
  [C_flow]="${EXP_ROOT}/clipb16_linear_C_flow_sid/ckpts/best.pt"
  [C_pixel]="${EXP_ROOT}/clipb16_linear_C_pixel/ckpts/best.pt"
)
VAL="${DATA_ROOT}/val"
for name in clipb16_sid clipl14_sid D1_sid C_flow C_pixel; do
  ckpt="${CKPTS[$name]}"
  pred="${PRED_FULL}/${name}_official_val_clean.json"
  if [[ ! -f "$pred" ]]; then
    echo "==== predict ${name} ===="
    $PY -u predict.py "$VAL" "$pred" --ckpt "$ckpt"
  else
    echo "skip predict ${name}, exists"
  fi
  dump_and_gallery "$pred" "${name}_full" clean "$VAL" "--split official_val"
done

echo "badcase queue done"
# end
