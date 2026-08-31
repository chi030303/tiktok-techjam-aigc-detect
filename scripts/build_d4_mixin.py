#!/usr/bin/env python3
# 2026-08-31, tianqi, D4 mix-in: nano + PixArt + SDXL + GPT; no Hunyuan (license)
"""Build D4 mixin fakes for full-SID mix-in (replace equal SID FLUX).

Uses disk we already have — does not call ComfyUI:

  nano_banana  /workspace/data_new/output/nano_banana_vertex_batch   (~1500)
  pixart       /workspace/data_new/output/pixart_sigma_quality_v2  (~1500)
               PixArt stands in for latent+DiT (Hunyuan weights are not
               usable for training)
  sdxl         /workspace/data_new/output/sdxl_full_refiner_v1    (~1500)
               + ComfyUI/output *sdxl* (skip output/old)
  GPT          /workspace/data_new/output/GPT                     (~1500)

Hunyuan images are skipped even if they sit in ComfyUI/output. GPT is in;
unseen stay EvalGEN (Nova is the hard family). No EvalGEN images, no data/val,
no i2i/real. Dedup by lowercase filename.

  DATA_ROOT=/workspace/data python scripts/build_d4_mixin.py
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

# (root, generator, arch) — every image under root is that generator
SELF_ROOTS = (
    ("nano_banana_vertex_batch", "nano_banana", None),
    ("pixart_sigma_quality_v2", "pixart", "dit"),
    ("sdxl_full_refiner_v1", "sdxl", "unet"),
    # 2026-08-31, tianqi, GPT in D4; unseen is EvalGEN Nova not official DALL·E
    ("GPT", "gpt_image", None),
    # end
)


def _list_images(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS)


def _classify_comfy(path: Path, comfy_root: Path) -> tuple[str, str | None] | None:
    """PixArt / SDXL from ComfyUI output; skip Hunyuan (not licensed to train) and output/old."""
    try:
        rel = path.relative_to(comfy_root)
    except ValueError:
        return None
    if rel.parts and rel.parts[0] == "old":
        return None
    blob = str(rel).lower()
    # 2026-08-31, tianqi, Hunyuan outputs cannot be used for training
    if "hunyuan" in blob:
        return None
    # end
    if "pixart" in blob:
        return ("pixart", "dit")
    if "sdxl" in blob:
        return ("sdxl", "unet")
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-root", type=Path, default=Path("/workspace/data_new/output"))
    p.add_argument("--comfy-root", type=Path, default=Path("/workspace/data_new/ComfyUI/output"))
    p.add_argument("--seed", type=int, default=20260831)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: $DATA_ROOT/manifests/ablation/D4_sid_mixin.jsonl",
    )
    args = p.parse_args()

    out = args.out or (data_root() / "manifests" / "ablation" / "D4_sid_mixin.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    seen_names: set[str] = set()
    by_gen: Counter[str] = Counter()

    def add(img: Path, generator: str, arch: str | None) -> None:
        key = img.name.lower()
        if key in seen_names:
            return
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

    for folder, generator, arch in SELF_ROOTS:
        n_before = by_gen[generator]
        for img in _list_images(args.self_root / folder):
            add(img, generator, arch)
        if by_gen[generator] == n_before:
            print(f"skip missing or empty {args.self_root / folder}", flush=True)

    for img in _list_images(args.comfy_root):
        hit = _classify_comfy(img, args.comfy_root)
        if hit is None:
            continue
        add(img, hit[0], hit[1])

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    with out.open("w") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"wrote {out} n={len(rows)} by_gen={dict(by_gen)}", flush=True)
    if by_gen["nano_banana"] < 1000:
        raise SystemExit(f"D4 expected ~1500 nano_banana, got {by_gen['nano_banana']}")
    if by_gen["pixart"] < 1000:
        raise SystemExit(f"D4 expected ~1500 pixart (DiT stand-in), got {by_gen['pixart']}")
    if "hunyuan" in by_gen:
        raise SystemExit(f"D4 must not include hunyuan, got {by_gen['hunyuan']}")
    if len(rows) < 2000:
        raise SystemExit("D4 mixin is too small")


if __name__ == "__main__":
    main()
# end
