#!/usr/bin/env bash
# 2026-09-01, tianqi, after D6 train: official 400 fuse last4+D6 + Nova 15-cond (D6 and fuse)
# Waits for scripts/d6_mix.sh. GPU1. Skip-if-exists so reruns are safe.
set -u
REPO="${REPO:-/workspace/kiki/tiktok-techjam-aigc-detect}"
export DATA_ROOT="${DATA_ROOT:-/workspace/data}"
export MODELS_ROOT="${MODELS_ROOT:-/workspace/models}"
export EXP_ROOT="${EXP_ROOT:-/workspace/experiments}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PY="${PY:-/venv/main/bin/python}"
LOG="${EXP_ROOT}/kiki_d6_fuse_nova.log"
OUT="${REPO}/outputs/tables"
U4="${EXP_ROOT}/clipb16_linear_sid_unfreeze4/ckpts/best.pt"
D6="${EXP_ROOT}/clipb16_linear_sid_d6_mix/ckpts/best.pt"

cd "$REPO"
mkdir -p "$OUT" "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

echo "==== wait d6_mix $(date -u) ===="
while pgrep -f "scripts/d6_mix.sh" >/dev/null 2>&1; do
  echo "wait d6_mix $(date -u)"
  sleep 30
done

if [[ ! -f "$D6" ]]; then
  echo "FAIL missing D6 ckpt $D6"
  exit 1
fi
if [[ ! -f "$U4" ]]; then
  echo "FAIL missing last4 $U4"
  exit 1
fi

echo "==== D6 fuse/nova GPU${CUDA_VISIBLE_DEVICES} $(date -u) ===="

stem_official="official_val400_fuse_u4_D6_mix"
if [[ -f "$OUT/${stem_official}_formula.json" ]]; then
  echo "skip exists $stem_official"
else
  echo "==== official 400 fuse last4+D6 $(date -u) ===="
  "$PY" -u scripts/run_full_eval.py \
    --split official_val --conditions full --max-images 400 --seed 0 \
    --workers 4 --batch 32 --fuse --fuse-weight 0.5 \
    --ckpt "unfreeze4=${U4}" --ckpt "D6_mix=${D6}" \
    --experiment "eval400_fuse_u4_D6_mix" \
    --stem "$stem_official" --out-dir "$OUT"
fi

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

nova_one D6_mix --ckpt "D6_mix=${D6}"
nova_one fuse_u4_d6 --fuse --fuse-weight 0.5 \
  --ckpt "unfreeze4=${U4}" --ckpt "D6_mix=${D6}"

echo "==== D6 fuse/nova done $(date -u) ===="
# end
