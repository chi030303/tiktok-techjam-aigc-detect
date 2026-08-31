#!/usr/bin/env bash
# 2026-08-31, tianqi, A2 i2i: wait for 60 triplets + D5, then paired-eval (no train)
set -u
REPO="${REPO:-/workspace/kiki/tiktok-techjam-aigc-detect}"
export DATA_ROOT="${DATA_ROOT:-/workspace/data}"
export MODELS_ROOT="${MODELS_ROOT:-/workspace/models}"
export EXP_ROOT="${EXP_ROOT:-/workspace/experiments}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PY="${PY:-/venv/main/bin/python}"
LOG="${EXP_ROOT}/kiki_a2_i2i.log"
OUT="${REPO}/outputs/tables"
MANIFEST="${DATA_ROOT}/manifests/ablation/A2_i2i_hard60.jsonl"
U4="${EXP_ROOT}/clipb16_linear_sid_unfreeze4/ckpts/best.pt"
D3="${EXP_ROOT}/clipb16_linear_sid_d3_mix/ckpts/best.pt"
SID="${EXP_ROOT}/clipb16_linear_sid_aug/ckpts/best.pt"
D4="${EXP_ROOT}/clipb16_linear_sid_d4_mix/ckpts/best.pt"
D5="${EXP_ROOT}/clipb16_linear_sid_d5_mix/ckpts/best.pt"

cd "$REPO"
mkdir -p "$OUT" "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "==== wait i2i upload then D4 then D5 $(date -u) ===="
while true; do
  n_nano=$(find /workspace/data_new/output/i2i/i2i_nano_banana -type f \( -iname '*.png' -o -iname '*.jpg' \) 2>/dev/null | wc -l | tr -d ' ')
  n_codex=$(find /workspace/data_new/output/i2i/i2i_codex -type f \( -iname '*.png' -o -iname '*.jpg' \) 2>/dev/null | wc -l | tr -d ' ')
  echo "nano=$n_nano codex=$n_codex $(date -u)"
  if [[ "${n_nano}" -ge 50 && "${n_codex}" -ge 50 ]]; then
    break
  fi
  sleep 60
done
while pgrep -f "scripts/evalgen_robust.sh" >/dev/null 2>&1; do sleep 30; done
while pgrep -f "scripts/d2_selfbuilt.sh" >/dev/null 2>&1; do sleep 30; done
while pgrep -f "scripts/d4_mix.sh" >/dev/null 2>&1; do sleep 30; done
while pgrep -f "scripts/d5_mix.sh" >/dev/null 2>&1; do sleep 30; done

echo "==== build A2 manifest $(date -u) ===="
"$PY" -u scripts/build_a2_i2i.py --min-triplets 50 --out "$MANIFEST"

CKPTS=()
[[ -f "$U4" ]] && CKPTS+=(--ckpt "last4=${U4}")
[[ -f "$D3" ]] && CKPTS+=(--ckpt "D3_mix=${D3}")
[[ -f "$SID" ]] && CKPTS+=(--ckpt "clipb16_sid=${SID}")
[[ -f "$D4" ]] && CKPTS+=(--ckpt "D4_mix=${D4}")
[[ -f "$D5" ]] && CKPTS+=(--ckpt "D5_mix=${D5}")
if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "FAIL no ckpts"
  exit 1
fi
echo "==== paired eval $(date -u) ===="
"$PY" -u scripts/eval_i2i_triplets.py --manifest "$MANIFEST" "${CKPTS[@]}" \
  --out "$OUT/A2_i2i_hard60_paired.json"
echo "==== A2 eval done $(date -u)  (no train; decide from pair_acc) ===="
# end
