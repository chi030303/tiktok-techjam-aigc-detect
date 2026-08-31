#!/usr/bin/env bash
# 2026-08-31, tianqi, official_val 400-subset (15 cond) + EvalGEN full fakes, clean
eval_400_evalgen() {
  local name="$1"
  local ckpt="$2"
  local batch="${3:-32}"
  if [[ ! -f "$ckpt" ]]; then
    echo "skip missing ckpt $name $ckpt"
    return 0
  fi
  echo "==== eval 400 $name $(date -u) ===="
  "$PY" -u scripts/run_full_eval.py \
    --split official_val --conditions full --max-images 400 --seed 0 \
    --workers 4 --batch "$batch" \
    --ckpt "${name}=${ckpt}" \
    --experiment "eval400_${name}" \
    --stem "official_val400_${name}" \
    --out-dir "$OUT"
  echo "==== evalgen FULL $name $(date -u) ===="
  "$PY" -u scripts/run_full_eval.py \
    --split evalgen --reals sid_val --conditions clean \
    --workers 4 --batch "$batch" \
    --ckpt "${name}=${ckpt}" \
    --experiment "evalgen_full_${name}" \
    --stem "evalgen_full_${name}" \
    --out-dir "$OUT"
}
# end
