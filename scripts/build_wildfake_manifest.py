#!/usr/bin/env python3
# 2026-08-30, samily, index WildFake on disk -> source JSONL manifest
"""Scan extracted WildFake trees using configs/wildfake/generators.yaml.

Examples:
  python scripts/build_wildfake_manifest.py --root data/wildfake/cross_arch/ddpm \\
    --generators ddpm --min-side 512 --max-per-generator 2000 \\
    --out data/manifests/source_wildfake_ddpm.jsonl

  python scripts/build_wildfake_manifest.py --root data/wildfake \\
    --generators adm,ddpm,sd_original --report-unmatched \\
    --out data/manifests/source_wildfake_cross_arch.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.wildfake import load_generators_config, scan_wildfake
from src.paths import data_root
from src.transforms.manifest import filter_train_rows, read_jsonl, write_jsonl

# end


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--config", type=Path, default=REPO / "configs/wildfake/generators.yaml")
    p.add_argument("--generators", default="", help="comma-separated generator keys, e.g. adm,ddpm,sd_original")
    p.add_argument(
        "--force-generator",
        default=None,
        help="assign every image under --root to one configured generator",
    )
    p.add_argument("--split", default="train", choices=("train", "val", "test", "unseen"))
    p.add_argument("--min-side", type=int, default=512, help="C-UNet: require short side >= 512")
    p.add_argument("--max-per-generator", type=int, default=2000)
    p.add_argument("--holdout-manifest", type=Path, default=None, help="demo val JSONL for phash filter")
    p.add_argument("--report-unmatched", action="store_true")
    p.add_argument("--no-phash", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    root = args.root
    if not root.is_dir():
        alt = data_root() / args.root
        if alt.is_dir():
            root = alt
        else:
            raise SystemExit(f"root not found: {args.root}")

    gens = {g.strip() for g in args.generators.split(",") if g.strip()} or None
    cfg = load_generators_config(args.config)
    records, stats = scan_wildfake(
        root,
        cfg,
        split=args.split,
        generators=gens,
        min_side=args.min_side,
        max_per_generator=args.max_per_generator,
        compute_phash=not args.no_phash,
        report_unmatched=args.report_unmatched,
        force_generator=args.force_generator,
    )
    holdout = None
    holdout_path = args.holdout_manifest
    if holdout_path is None:
        default = data_root() / "manifests" / "source_demo_val.jsonl"
        if default.is_file():
            holdout_path = default
    if holdout_path and holdout_path.is_file():
        holdout = read_jsonl(holdout_path, kind="source")
        records, leaks = filter_train_rows(records, holdout=holdout)
        if leaks:
            print(f"dropped {len(leaks)} phash leak(s) vs {holdout_path}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, records)
    print(
        f"wrote {len(records)} rows -> {args.out}  "
        f"per_generator={stats['per_generator']}  unmatched={stats['unmatched']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
