#!/usr/bin/env bash
# 2026-08-29, tianqi, download datasets/weights into gitignored folders
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# 2026-08-29, tianqi, val lives in data/val (old name demo_wildfake still accepted as alias)
mkdir -p data/cifake data/sid_set data/val/real data/val/fake data/evalgen data/demo_wildfake models checkpoints
touch data/val/DO_NOT_TRAIN data/evalgen/DO_NOT_TRAIN data/demo_wildfake/DO_NOT_TRAIN
# end

echo "Datasets and weights are NOT stored in git."
echo "Official val (no train): data/val/{real,fake}. Backbones: models/."
echo

download_models=0
download_val=0
if [[ "${1:-}" == "--models" ]]; then
  download_models=1
fi
if [[ "${1:-}" == "--val" ]] || [[ "${2:-}" == "--val" ]]; then
  download_val=1
fi

if [[ "$download_models" -eq 1 ]]; then
  # 2026-08-29, tianqi, CLIP-B/16 CLIP-L/14 ResNet-50 DINOv2-L/14
  python "$(dirname "$0")/download_backbones.py"
  # end
fi

if [[ "$download_val" -eq 1 ]]; then
  python "$(dirname "$0")/download_official_val.py"
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
echo "Official val (NO TRAIN): python scripts/download_official_val.py  -> data/val/"
echo "  real = COCO val2017, fake = WildFake DALL·E Advanced (not the full 3.5M set)"
echo "EvalGEN (NO TRAIN, extra eval): python scripts/download_evalgen.py -> data/evalgen/"
echo "WildFake full (optional): https://modelscope.cn/datasets/hy2628982280/WildFake/summary"
# end
