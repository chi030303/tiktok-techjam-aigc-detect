#!/usr/bin/env python3
# 2026-08-30, samily, index SID_Set into source manifest JSONL
"""Build ``data/manifests/source_sid_train.jsonl`` from SID_Set on disk.

Prerequisites:
  huggingface-cli download saberzl/SID_Set --repo-type dataset --local-dir data/sid_set

Or place extracted images under ``data/sid_set/extracted/{real,full_synth,tampered}/``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.sid_set import build_sid_manifest
from src.paths import data_root

# end


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=data_root() / "manifests" / "source_sid_train.jsonl",
    )
    p.add_argument("--split", default="train", choices=("train", "val", "test", "unseen"))
    p.add_argument("--max-rows", type=int, default=None, help="parquet smoke limit")
    p.add_argument("--no-phash", action="store_true")
    args = p.parse_args()
    build_sid_manifest(
        args.out,
        split=args.split,
        max_rows=args.max_rows,
        compute_phash=not args.no_phash,
    )


if __name__ == "__main__":
    main()
