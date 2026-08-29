#!/usr/bin/env bash
# 2026-08-29, tianqi, local gate before opening a PR
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python -m pytest -q tests/test_predict_schema.py tests/test_recipe.py
# 2026-08-29, tianqi, recipe dry-run is part of the PR gate
python scripts/run_experiment.py experiments/clipb16_linear_sid/recipe.yaml
# end

OUT="$(mktemp)"
python predict.py "$ROOT/fixtures/sample_images" "$OUT"
python - "$OUT" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
assert rows and all("image_path" in r and "pred" in r for r in rows)
print(f"ok {len(rows)} predictions")
PY

echo "check passed"
# end
