#!/usr/bin/env python3
# 2026-08-29, tianqi, EvalGEN hold-out only (NeurIPS DDA paper); do not train
"""Download Junwei-Xi/EvalGEN into data/evalgen. Extra eval, not the official TechJam val."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.paths import data_root

# end

REPO_ID = "Junwei-Xi/EvalGEN"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def count_images(root: Path) -> int:
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()
    root = args.data_root or data_root()
    dest = root / "evalgen"
    tmp = root / "_download_tmp"
    dest.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)

    # 2026-08-29, tianqi, same guard as official val
    (dest / "DO_NOT_TRAIN").write_text(
        "EvalGEN (Junwei-Xi / NeurIPS 2025 DDA paper).\n"
        "Hold-out for cross-generator eval. Do NOT use for training.\n"
        "Official TechJam demo val is data/val (COCO val2017 + DALL·E Advanced).\n"
    )
    # end

    if count_images(dest) >= 1000:
        print(f"skip EvalGEN, already have {count_images(dest)} images in {dest}", flush=True)
        return

    print(f"download {REPO_ID} EvalGEN.zip (~14.5GB) -> {tmp}", flush=True)
    zip_path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename="EvalGEN.zip",
            repo_type="dataset",
            local_dir=str(tmp),
        )
    )
    print(f"unzip {zip_path} -> {dest}", flush=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    n = count_images(dest)
    print(f"EvalGEN images: {n} (paper ~55300)  dir={dest}", flush=True)
    zip_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
