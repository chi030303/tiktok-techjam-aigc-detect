#!/usr/bin/env bash
# 2026-08-31, tianqi, EvalGEN official 14-setting robust (was clean-only); GPU1 after kiki-narr
# Phase 1: 400 fakes/gen + 2000 reals x 15 cond (fast table, all shortlist).
# Phase 2: full n x 15 cond on the three ckpts that might be submitted.
set -u
REPO="${REPO:-/workspace/kiki/tiktok-techjam-aigc-detect}"
export DATA_ROOT="${DATA_ROOT:-/workspace/data}"
export MODELS_ROOT="${MODELS_ROOT:-/workspace/models}"
export EXP_ROOT="${EXP_ROOT:-/workspace/experiments}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PY="${PY:-/venv/main/bin/python}"
LOG="${EXP_ROOT}/kiki_evalgen_robust.log"
OUT="${REPO}/outputs/tables"
U4="${EXP_ROOT}/clipb16_linear_sid_unfreeze4/ckpts/best.pt"
D3="${EXP_ROOT}/clipb16_linear_sid_d3_mix/ckpts/best.pt"
D3U4="${EXP_ROOT}/clipb16_linear_d3_unfreeze4/ckpts/best.pt"
SIDB="${EXP_ROOT}/clipb16_linear_sid_aug/ckpts/best.pt"
SIDL="${EXP_ROOT}/clipl14_linear_sid_aug/ckpts/best.pt"

cd "$REPO"
mkdir -p "$OUT"
exec >>"$LOG" 2>&1
echo "==== wait kiki-narr $(date -u) ===="
while tmux has-session -t kiki-narr 2>/dev/null; do sleep 30; done
echo "==== evalgen robust start $(date -u) ===="

evalgen_grid() {
  local tag="$1"
  local cond="$2"
  shift 2
  echo "---- $tag cond=$cond $(date -u) ----"
  "$PY" -u scripts/run_full_eval.py \
    --split evalgen --reals sid_val --conditions "$cond" \
    --workers 4 --batch 32 \
    --experiment "evalgen_robust" \
    --out-dir "$OUT" \
    "$@"
}

# 400/gen x 15: same official keys as the score formula, unseen generators
GRID=(
  "unfreeze4=${U4}"
  "D3_mix=${D3}"
  "d3_unfreeze4=${D3U4}"
  "clipb16_sid=${SIDB}"
  "clipl14_sid=${SIDL}"
)
for spec in "${GRID[@]}"; do
  name="${spec%%=*}"
  path="${spec#*=}"
  if [[ ! -f "$path" ]]; then
    echo "skip missing $name $path"
    continue
  fi
  evalgen_grid "400_${name}" full \
    --max-fakes-per-gen 400 --max-reals 2000 --seed 0 \
    --ckpt "$spec" \
    --stem "evalgen_r400_${name}"
done

if [[ -f "$U4" && -f "$D3" ]]; then
  evalgen_grid "400_fuse_u4_d3" full \
    --max-fakes-per-gen 400 --max-reals 2000 --seed 0 --fuse --fuse-weight 0.5 \
    --ckpt "unfreeze4=${U4}" --ckpt "D3_mix=${D3}" \
    --stem "evalgen_r400_fuse_u4_d3"
fi

# 2026-08-31, tianqi, phase2 = Nova only x 15; Flux/GoT/Infinity/OmniGen already in r400
echo "==== phase2 Nova-only 15-cond $(date -u) ===="
for spec in "unfreeze4=${U4}" "D3_mix=${D3}" "d3_unfreeze4=${D3U4}"; do
  name="${spec%%=*}"
  path="${spec#*=}"
  if [[ ! -f "$path" ]]; then
    echo "skip missing $name $path"
    continue
  fi
  evalgen_grid "full_nova_${name}" full \
    --generators nova \
    --ckpt "$spec" \
    --stem "evalgen_rfull_nova_${name}"
done
# end

echo "==== evalgen robust done $(date -u) ===="
# end
