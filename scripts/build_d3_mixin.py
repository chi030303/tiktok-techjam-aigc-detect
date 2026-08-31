#!/usr/bin/env python3
# 2026-08-31, tianqi, D3 mix-in jsonl: self-built + UNet + a slice of pixel ADM/DDPM
"""Build D3 mixin fakes: self-built + sampled UNet + a little pixel-space ADM/DDPM.

Official val DALL·E is the contest unseen; do not train EvalGEN or data/val.

  DATA_ROOT=/workspace/data python scripts/build_d3_mixin.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.paths import data_root

# end

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SELF_DIRS = (
    ("flux2", "flux2", "flow"),
    ("sd35", "sd35", "flow"),
    ("nano_banana_vertex_batch", "nano_banana", None),
    ("old", "sd35", "flow"),
)


def _list_images(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-root", type=Path, default=Path("/workspace/data_new/output"))
    p.add_argument("--unet-n", type=int, default=4000)
    # 2026-08-31, tianqi, a thin pixel-space slice; DALL·E official val stays unseen
    p.add_argument("--adm-n", type=int, default=1000)
    p.add_argument("--ddpm-n", type=int, default=1000)
    # end
    p.add_argument("--seed", type=int, default=20260831)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: $DATA_ROOT/manifests/ablation/D3_sid_mixin.jsonl",
    )
    args = p.parse_args()

    out = args.out or (data_root() / "manifests" / "ablation" / "D3_sid_mixin.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    seen_names: set[str] = set()
    by_gen: Counter[str] = Counter()

    for folder, generator, arch in SELF_DIRS:
        for img in _list_images(args.self_root / folder):
            key = img.name.lower()
            if key in seen_names:
                continue
            seen_names.add(key)
            rows.append(
                {
                    "path": str(img.resolve()),
                    "label": 1,
                    "generator": generator,
                    "arch": arch,
                    "source": "data_new",
                }
            )
            by_gen[generator] += 1

    rng = random.Random(args.seed)
    wf = data_root() / "wildfake" / "cross_arch"
    packs = (
        ("sd_original", "unet", args.unet_n),
        ("adm", "pixel", args.adm_n),
        ("ddpm", "pixel", args.ddpm_n),
    )
    for folder, arch, n_take in packs:
        pool = _list_images(wf / folder)
        if n_take <= 0 or not pool:
            continue
        take = pool if len(pool) <= n_take else rng.sample(pool, n_take)
        for img in take:
            rows.append(
                {
                    "path": str(img.resolve()),
                    "label": 1,
                    "generator": folder,
                    "arch": arch,
                    "source": "wildfake",
                }
            )
            by_gen[folder] += 1

    rng.shuffle(rows)
    with out.open("w") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"wrote {out} n={len(rows)} by_gen={dict(by_gen)}", flush=True)
    if not rows:
        raise SystemExit("D3 mixin is empty")


if __name__ == "__main__":
    main()
# end
