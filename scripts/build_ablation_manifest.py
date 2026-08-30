#!/usr/bin/env python3
# 2026-08-30, samily, build controlled ablation manifests from YAML configs
"""Sample 8k+8k (or custom) ablation grids per DATA_ABLATION_PLAN.md.

Example:
  python scripts/build_ablation_manifest.py configs/ablation/D1_sid_only.yaml
  python scripts/build_ablation_manifest.py configs/ablation/ --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.ablation import run_ablation_config
from src.paths import data_root

# end


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path, nargs="+", help="YAML config or directory")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=data_root() / "manifests" / "ablation",
    )
    p.add_argument("--all", action="store_true", help="with a directory, build every *.yaml")
    args = p.parse_args()

    configs: list[Path] = []
    for item in args.config:
        if item.is_dir():
            configs.extend(sorted(item.glob("*.yaml")))
        else:
            configs.append(item)
    if args.all:
        configs = sorted({c for c in configs if c.suffix == ".yaml"})
    if not configs:
        raise SystemExit("no ablation configs found")

    for cfg in configs:
        out = args.out_dir / f"{cfg.stem}.jsonl"
        run_ablation_config(cfg, out)


if __name__ == "__main__":
    main()
