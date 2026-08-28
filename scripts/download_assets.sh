#!/usr/bin/env bash
# 2026-08-29, tianqi, download datasets/weights into gitignored folders
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p data/cifake data/sid_set data/demo_wildfake models checkpoints
touch data/demo_wildfake/DO_NOT_TRAIN

echo "Datasets and weights are NOT stored in git."
echo "Fill data/ and models/ locally. Demo set path: data/demo_wildfake/ (do not train)."
echo

download_models=0
if [[ "${1:-}" == "--models" ]]; then
  download_models=1
fi

if [[ "$download_models" -eq 1 ]]; then
  python - <<'PY'
# 2026-08-29, tianqi, pull CLIP via open_clip if installed (optional)
print("Install open_clip / torchvision then re-run; or let train.py download on first use.")
print("Target dir: models/")
PY
fi

if command -v kaggle >/dev/null 2>&1; then
  echo "CIFAKE: kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images -p data/cifake --unzip"
else
  echo "CIFAKE: install Kaggle CLI, then download into data/cifake/"
  echo "  https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images"
fi

echo "SID_Set: huggingface-cli download saberzl/SID_Set --repo-type dataset --local-dir data/sid_set"
echo "  https://huggingface.co/datasets/saberzl/SID_Set"
echo
echo "Demo (NO TRAIN): place COCO val2017 + DALL·E Advanced subset in data/demo_wildfake/"
echo "WildFake full (optional): https://modelscope.cn/datasets/hy2628982280/WildFake/summary"
# end
