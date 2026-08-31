#!/usr/bin/env python3
# 2026-08-31, tianqi, A2 i2i-hard: 60 real + Gemini/Codex reconstructions bound by source_id
"""Build the 60-triplet whole-image i2i manifest (not the 8k A2 pool).

Expects:

  <root>/real/<source_id>.jpg
  <root>/i2i_codex/<source_id>_codex.png
  <root>/i2i_nano_banana/<source_id>_nano.png

Only complete triplets (real + both fakes) go into the jsonl. Do not treat
the 180 files as independent images: every row shares source_id.

  DATA_ROOT=/workspace/data python scripts/build_a2_i2i.py
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.paths import data_root
from src.transforms.manifest import SourceRecord, write_jsonl

# end

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
# longest suffix first
FAKE_SUFFIXES = (
    ("_nano_banana", "nano_banana"),
    ("_nano", "nano_banana"),
    ("_gemini", "nano_banana"),
    ("_codex", "codex"),
)


def source_id_from_name(path: Path) -> str:
    stem = path.stem
    for suf, _gen in FAKE_SUFFIXES:
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def generator_from_name(path: Path) -> str | None:
    stem = path.stem
    for suf, gen in FAKE_SUFFIXES:
        if stem.endswith(suf):
            return gen
    return None


def _image_id(path: Path) -> str:
    return hashlib.sha1(f"{path}:{path.stat().st_size}".encode()).hexdigest()


def _record(path: Path, label: int, generator: str | None, source_id: str) -> SourceRecord:
    with Image.open(path) as im:
        w, h = im.size
    return SourceRecord(
        image_id=_image_id(path),
        path=str(path.resolve()),
        label=int(label),
        source_dataset="self_built_i2i",
        generator=generator,
        split="train",
        width=int(w),
        height=int(h),
        family=None if label == 0 else "diffusion",
        arch=None,
        generation_type=None if label == 0 else "i2i",
        content_type="real" if label == 0 else "full_synthetic",
        original_format=path.suffix.lower().lstrip("."),
        source_id=source_id,
    )


def collect_groups(root: Path) -> dict[str, dict[str, Path]]:
    """source_id -> {real, nano_banana, codex} paths."""
    groups: dict[str, dict[str, Path]] = defaultdict(dict)
    for img in sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS):
        sid = source_id_from_name(img)
        gen = generator_from_name(img)
        role = "real" if gen is None else gen
        if role in groups[sid] and groups[sid][role] != img:
            print(f"warn: duplicate {role} for {sid}: {groups[sid][role]} vs {img}", flush=True)
            continue
        groups[sid][role] = img
    return groups


def complete_triplets(groups: dict[str, dict[str, Path]]) -> list[str]:
    need = ("real", "nano_banana", "codex")
    return sorted(sid for sid, roles in groups.items() if all(k in roles for k in need))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/workspace/data_new/output/i2i"))
    p.add_argument("--min-triplets", type=int, default=50)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    if not args.root.is_dir():
        raise SystemExit(f"missing i2i root {args.root}")

    groups = collect_groups(args.root)
    ok = complete_triplets(groups)
    n_real = sum(1 for g in groups.values() if "real" in g)
    n_nano = sum(1 for g in groups.values() if "nano_banana" in g)
    n_codex = sum(1 for g in groups.values() if "codex" in g)
    print(
        f"scanned {args.root} ids={len(groups)} real={n_real} nano={n_nano} "
        f"codex={n_codex} complete={len(ok)}",
        flush=True,
    )
    incomplete = [sid for sid in sorted(groups) if sid not in set(ok)]
    for sid in incomplete[:12]:
        print(f"  incomplete {sid} roles={sorted(groups[sid])}", flush=True)
    if len(ok) < args.min_triplets:
        raise SystemExit(
            f"need >= {args.min_triplets} complete triplets, got {len(ok)} "
            "(nano banana still uploading?)"
        )

    rows: list[SourceRecord] = []
    for sid in ok:
        roles = groups[sid]
        rows.append(_record(roles["real"], 0, None, sid))
        rows.append(_record(roles["nano_banana"], 1, "nano_banana", sid))
        rows.append(_record(roles["codex"], 1, "codex", sid))

    out = args.out or (data_root() / "manifests" / "ablation" / "A2_i2i_hard60.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, rows)
    n_fake = sum(r.label == 1 for r in rows)
    print(
        f"wrote {out} n={len(rows)} reals={sum(r.label == 0 for r in rows)} "
        f"fakes={n_fake} triplets={len(ok)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
# end
