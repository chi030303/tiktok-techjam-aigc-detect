#!/usr/bin/env bash
# 2026-08-30, samily, run after WildFake zips are extracted on server
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
export DATA_ROOT="${DATA_ROOT:-/workspace/data}"
cd "$REPO"
MAN=$DATA_ROOT/manifests

echo "=== wildfake DDPM manifest ==="
if [ -d "$DATA_ROOT/wildfake/cross_arch/ddpm" ]; then
  $PY scripts/build_wildfake_manifest.py \
    --root "$DATA_ROOT/wildfake/cross_arch/ddpm" \
    --force-generator ddpm \
    --min-side 256 \
    --max-per-generator 2000 \
    --holdout-manifest "$MAN/source_demo_val.jsonl" \
    --out "$MAN/source_wildfake_ddpm.jsonl"
fi

echo "=== wildfake ADM manifest ==="
if [ -d "$DATA_ROOT/wildfake/cross_arch/adm" ]; then
  $PY scripts/build_wildfake_manifest.py \
    --root "$DATA_ROOT/wildfake/cross_arch/adm" \
    --force-generator adm \
    --min-side 256 \
    --max-per-generator 2000 \
    --holdout-manifest "$MAN/source_demo_val.jsonl" \
    --out "$MAN/source_wildfake_adm.jsonl"
fi

echo "=== ablation C_pixel (if both ready) ==="
if [ -f "$MAN/source_wildfake_ddpm.jsonl" ] && [ -f "$MAN/source_wildfake_adm.jsonl" ]; then
  $PY scripts/build_ablation_manifest.py configs/ablation/C_pixel_adm_ddpm.yaml --out-dir "$MAN/ablation"
fi

echo "=== done ==="
ls -la "$MAN"/source_wildfake_*.jsonl 2>/dev/null || true
ls -la "$MAN/ablation/" 2>/dev/null || true
