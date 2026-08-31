#!/usr/bin/env bash
# 2026-08-31, tianqi, last4 fuse with D4/D5 + EvalGEN Nova x15 robust (no retrain)
# LANE=0 GPU0 after A2; LANE=1 GPU1 after A3. D3/D4/D5 official 400 already done.
set -u
LANE="${LANE:-0}"
REPO="${REPO:-/workspace/kiki/tiktok-techjam-aigc-detect}"
export DATA_ROOT="${DATA_ROOT:-/workspace/data}"
export MODELS_ROOT="${MODELS_ROOT:-/workspace/models}"
export EXP_ROOT="${EXP_ROOT:-/workspace/experiments}"
export CUDA_VISIBLE_DEVICES="${LANE}"
PY="${PY:-/venv/main/bin/python}"
LOG="${EXP_ROOT}/kiki_fuse_nova_gpu${LANE}.log"
OUT="${REPO}/outputs/tables"
U4="${EXP_ROOT}/clipb16_linear_sid_unfreeze4/ckpts/best.pt"
D3="${EXP_ROOT}/clipb16_linear_sid_d3_mix/ckpts/best.pt"
D4="${EXP_ROOT}/clipb16_linear_sid_d4_mix/ckpts/best.pt"
D5="${EXP_ROOT}/clipb16_linear_sid_d5_mix/ckpts/best.pt"

cd "$REPO"
mkdir -p "$OUT" "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

echo "==== fuse/nova lane=${LANE} $(date -u) ===="
if [[ "$LANE" == "0" ]]; then
  while pgrep -f "scripts/a2_gpu0.sh" >/dev/null 2>&1; do
    echo "GPU0 wait A2 $(date -u)"
    sleep 30
  done
else
  while pgrep -f "scripts/a3_gpu1.sh" >/dev/null 2>&1; do
    echo "GPU1 wait A3 $(date -u)"
    sleep 30
  done
fi

official_fuse() {
  local tag="$1"
  local other="$2"
  local stem="official_val400_fuse_u4_${tag}"
  if [[ -f "$OUT/${stem}_formula.json" ]]; then
    echo "skip exists $stem"
    return 0
  fi
  echo "==== official 400 fuse last4+${tag} $(date -u) ===="
  "$PY" -u scripts/run_full_eval.py \
    --split official_val --conditions full --max-images 400 --seed 0 \
    --workers 4 --batch 32 --fuse --fuse-weight 0.5 \
    --ckpt "unfreeze4=${U4}" --ckpt "${tag}=${other}" \
    --experiment "eval400_fuse_u4_${tag}" \
    --stem "$stem" --out-dir "$OUT"
}

nova_one() {
  local name="$1"
  shift
  local stem="evalgen_nova_rfull_${name}"
  if [[ -f "$OUT/${stem}_formula.json" ]]; then
    echo "skip exists $stem"
    return 0
  fi
  echo "==== nova robust ${name} $(date -u) ===="
  "$PY" -u scripts/run_full_eval.py \
    --split evalgen --reals sid_val --conditions full --generators nova \
    --workers 4 --batch 32 \
    --experiment "evalgen_nova_${name}" \
    --stem "$stem" --out-dir "$OUT" \
    "$@"
}

if [[ ! -f "$U4" ]]; then
  echo "FAIL missing last4 $U4"
  exit 1
fi

if [[ "$LANE" == "0" ]]; then
  official_fuse D4_mix "$D4"
  nova_one D4_mix --ckpt "D4_mix=${D4}"
  nova_one fuse_u4_d4 --fuse --fuse-weight 0.5 \
    --ckpt "unfreeze4=${U4}" --ckpt "D4_mix=${D4}"
  nova_one fuse_u4_d3 --fuse --fuse-weight 0.5 \
    --ckpt "unfreeze4=${U4}" --ckpt "D3_mix=${D3}"
else
  official_fuse D5_mix "$D5"
  nova_one D5_mix --ckpt "D5_mix=${D5}"
  nova_one fuse_u4_d5 --fuse --fuse-weight 0.5 \
    --ckpt "unfreeze4=${U4}" --ckpt "D5_mix=${D5}"
  nova_one D3_mix --ckpt "D3_mix=${D3}"
fi

echo "==== fuse/nova lane=${LANE} done $(date -u) ===="
# end
