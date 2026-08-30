#!/usr/bin/env python3
# 2026-08-30, samily, official demo val manifest (nested DALL·E paths)
"""Build source_demo_val.jsonl from data/val/{real,fake}."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.paths import data_root
from src.transforms.build_source import collect_records
from src.transforms.manifest import write_jsonl

# end


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    root = data_root() / "val"
    out = args.out or (data_root() / "manifests" / "source_demo_val.jsonl")
    reals = collect_records(
        root / "real",
        dataset="demo_wildfake",
        split="val",
        label=0,
    )
    fakes = collect_records(
        root / "fake",
        dataset="demo_wildfake",
        split="val",
        label=1,
        generator="dalle_advanced",
        family="diffusion",
        generation_type="t2i",
    )
    write_jsonl(out, reals + fakes)
    print(f"wrote {len(reals) + len(fakes)} rows -> {out}")


if __name__ == "__main__":
    main()
