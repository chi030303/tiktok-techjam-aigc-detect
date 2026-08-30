#!/usr/bin/env python3
# 2026-08-30, tianqi, full WildFake corpus for training; NOT the demo val
"""Download the full WildFake tree into data/wildfake.

This is the large corpus (reals + many generators), not the TechJam demonstration
set. data/val (COCO val2017 + DALL·E Advanced) stays DO_NOT_TRAIN.

Before training, exclude overlap with data/val (COCO val2017 filenames / DALL·E
Advanced / phash). EvalGEN stays hold-out.

    python scripts/download_wildfake.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.paths import data_root

# end

WILDFAKE_ID = "hy2628982280/WildFake"

TRAIN_NOTE = """Full WildFake corpus (ModelScope hy2628982280/WildFake).

This directory is allowed for training AFTER you drop anything that overlaps
the official demo val:

  - COCO val2017 (same files as data/val/real)
  - DALL·E Advanced / DALLE3 Advanced (same files as data/val/fake)
  - phash collisions against data/val

Do NOT copy this tree into data/val. data/val and data/evalgen stay DO_NOT_TRAIN.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()
    root = args.data_root or data_root()
    dest = root / "wildfake"
    dest.mkdir(parents=True, exist_ok=True)
    # 2026-08-30, tianqi, full WildFake is a train source; do not stamp DO_NOT_TRAIN
    (dest / "TRAIN_NOTE.txt").write_text(TRAIN_NOTE)
    # end

    print(f"snapshot {WILDFAKE_ID} -> {dest}  (large; resume-safe)", flush=True)
    from modelscope.hub.snapshot_download import snapshot_download

    snapshot_download(
        WILDFAKE_ID,
        repo_type="dataset",
        local_dir=str(dest),
    )
    print(
        f"done {dest}\n"
        "Train only after excluding data/val overlap. "
        "For EvalGEN reals: python scripts/run_full_eval.py --split evalgen --reals wildfake",
        flush=True,
    )


if __name__ == "__main__":
    main()
# end
