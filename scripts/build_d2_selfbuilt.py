#!/usr/bin/env python3
# 2026-08-31, tianqi, D2 8k-probe: same SID reals as D1 + all self-built fakes (~6k)
"""D2 self-built-only ablation, aligned with D1's 8k SID reals.

Does not pad with WildFake. Fake count is whatever is under data_new/output
(~6081 including GPT). Official val / EvalGEN are not used.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.paths import data_root
from src.transforms.manifest import SourceRecord, read_jsonl, write_jsonl

# end

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
# folder name -> generator, family, arch
SELF_DIRS = (
    ("flux2", "flux2", "diffusion", "flow"),
    ("sd35", "sd35", "diffusion", "dit"),
    ("nano_banana_vertex_batch", "nano_banana", "diffusion", None),
    ("GPT", "gpt_image", "diffusion", None),
    ("old", "sd35", "diffusion", "dit"),
)


def _image_id(path: Path) -> str:
    return hashlib.sha1(f"{path}:{path.stat().st_size}".encode()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--d1",
        type=Path,
        default=None,
        help="D1 jsonl; default $DATA_ROOT/manifests/ablation/D1_sid_only.jsonl",
    )
    p.add_argument("--self-root", type=Path, default=Path("/workspace/data_new/output"))
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    d1 = args.d1 or (data_root() / "manifests" / "ablation" / "D1_sid_only.jsonl")
    out = args.out or (data_root() / "manifests" / "ablation" / "D2_selfbuilt.jsonl")
    if not d1.is_file():
        raise SystemExit(f"missing D1 manifest {d1}")

    reals = [r for r in read_jsonl(d1, kind="source") if int(r.label) == 0]
    if len(reals) != 8000:
        print(f"warn: D1 reals={len(reals)} expected 8000", flush=True)

    fakes: list[SourceRecord] = []
    seen: set[str] = set()
    by_gen: Counter[str] = Counter()
    for folder, generator, family, arch in SELF_DIRS:
        root = args.self_root / folder
        if not root.is_dir():
            print(f"skip missing {root}", flush=True)
            continue
        for img in sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS):
            key = img.name.lower()
            if key in seen:
                continue
            seen.add(key)
            with Image.open(img) as im:
                w, h = im.size
            fakes.append(
                SourceRecord(
                    image_id=_image_id(img),
                    path=str(img.resolve()),
                    label=1,
                    source_dataset="self_built",
                    generator=generator,
                    split="train",
                    width=int(w),
                    height=int(h),
                    family=family,
                    arch=arch,
                    generation_type="t2i",
                    content_type="full_synthetic",
                    original_format=img.suffix.lower().lstrip("."),
                )
            )
            by_gen[generator] += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, reals + fakes)
    print(f"wrote {out} reals={len(reals)} fakes={len(fakes)} {dict(by_gen)}", flush=True)


if __name__ == "__main__":
    main()
# end
