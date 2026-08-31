#!/usr/bin/env python3
# 2026-08-31, tianqi, D5 = D3 union D4: SID mix of all licensed self-built + WildFake UNet/pixel
"""Build D5 mixin: D3 (flux2/sd35/UNet/ADM/DDPM) + D4 (full nano, PixArt, SDXL, GPT).

No Hunyuan. Dedup by lowercase filename so D3's partial nano is replaced by
the full 1500. Official val / EvalGEN stay out.

  DATA_ROOT=/workspace/data python scripts/build_d5_mixin.py
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

# D3 self-built + D4 new T2I (nano listed once; builder scans the full folder)
SELF_ROOTS = (
    ("flux2", "flux2", "flow"),
    ("sd35", "sd35", "flow"),
    ("old", "sd35", "flow"),
    ("nano_banana_vertex_batch", "nano_banana", None),
    ("pixart_sigma_quality_v2", "pixart", "dit"),
    ("sdxl_full_refiner_v1", "sdxl", "unet"),
    ("GPT", "gpt_image", None),
)


def _list_images(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS)


def _classify_comfy(path: Path, comfy_root: Path) -> tuple[str, str | None] | None:
    try:
        rel = path.relative_to(comfy_root)
    except ValueError:
        return None
    if rel.parts and rel.parts[0] == "old":
        return None
    blob = str(rel).lower()
    if "hunyuan" in blob:
        return None
    if "pixart" in blob:
        return ("pixart", "dit")
    if "sdxl" in blob:
        return ("sdxl", "unet")
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-root", type=Path, default=Path("/workspace/data_new/output"))
    p.add_argument("--comfy-root", type=Path, default=Path("/workspace/data_new/ComfyUI/output"))
    p.add_argument("--unet-n", type=int, default=4000)
    p.add_argument("--adm-n", type=int, default=1000)
    p.add_argument("--ddpm-n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260831)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    out = args.out or (data_root() / "manifests" / "ablation" / "D5_sid_mixin.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    seen_names: set[str] = set()
    by_gen: Counter[str] = Counter()

    def add(img: Path, generator: str, arch: str | None, source: str) -> None:
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
                "source": source,
            }
        )
        by_gen[generator] += 1

    for folder, generator, arch in SELF_ROOTS:
        for img in _list_images(args.self_root / folder):
            add(img, generator, arch, "data_new")

    for img in _list_images(args.comfy_root):
        hit = _classify_comfy(img, args.comfy_root)
        if hit is None:
            continue
        add(img, hit[0], hit[1], "data_new")

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
            add(img, folder, arch, "wildfake")

    rng.shuffle(rows)
    with out.open("w") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"wrote {out} n={len(rows)} by_gen={dict(by_gen)}", flush=True)
    if "hunyuan" in by_gen:
        raise SystemExit(f"D5 must not include hunyuan, got {by_gen['hunyuan']}")
    if by_gen["nano_banana"] < 1000 or by_gen["pixart"] < 1000 or by_gen["gpt_image"] < 1000:
        raise SystemExit(f"D5 missing D4 gens: {dict(by_gen)}")
    if by_gen["sd_original"] < 1000:
        raise SystemExit(f"D5 missing WildFake UNet, got {by_gen['sd_original']}")
    if len(rows) < 8000:
        raise SystemExit(f"D5 mixin too small: {len(rows)}")


if __name__ == "__main__":
    main()
# end
