# 2026-08-29, tianqi, labeled rows for val / CIFAKE / EvalGEN; never used for training
"""Load (path, y) rows from hold-out folders. y=1 means AIGC."""

from __future__ import annotations

import random
from pathlib import Path

from src.paths import data_root

# end

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
REAL_DIRS = {"real", "REAL"}
FAKE_DIRS = {"fake", "FAKE"}
# EvalGEN zip is generator-named folders of synthetic images (no real half)
EVALGEN_FAKE = {"flux", "got", "infinity", "omnigen", "nova"}

SPLIT_TO_REL = {
    "official_val": ("val",),
    "val": ("val",),
    "evalgen": ("evalgen",),
    "cifake_test": ("cifake", "test"),
    "cifake": ("cifake", "test"),
}


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def split_root(name: str) -> Path:
    rel = SPLIT_TO_REL.get(name)
    if rel is None:
        raise SystemExit(f"unknown split {name!r}; known={sorted(SPLIT_TO_REL)}")
    return data_root().joinpath(*rel)


def _dir_label(parts: tuple[str, ...]) -> int | None:
    names = set(parts)
    if names & FAKE_DIRS:
        return 1
    if names & REAL_DIRS:
        return 0
    lowered = {p.lower() for p in parts}
    if lowered & EVALGEN_FAKE:
        return 1
    return None


def infer_label(path: Path, root: Path) -> int | None:
    try:
        rel_parts = path.relative_to(root).parts[:-1]
    except ValueError:
        rel_parts = path.parts[:-1]
    return _dir_label(rel_parts)


def load_labeled_dir(root: Path, default_fake: bool = False) -> list[tuple[Path, int]]:
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")
    rows: list[tuple[Path, int]] = []
    skipped = 0
    for path in list_images(root):
        y = infer_label(path, root)
        if y is None:
            if default_fake:
                y = 1
            else:
                skipped += 1
                continue
        rows.append((path, y))
    if not rows:
        raise SystemExit(f"no labeled images under {root} (need real/fake or REAL/FAKE)")
    if skipped:
        print(f"skip unlabeled {skipped} files under {root}", flush=True)
    return rows


def load_split(name: str) -> tuple[Path, list[tuple[Path, int]]]:
    root = split_root(name)
    default_fake = name in {"evalgen"}
    return root, load_labeled_dir(root, default_fake=default_fake)


def rel_key(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return rel.with_suffix("").as_posix()


def index_by_rel(rows: list[tuple[Path, int]], root: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for path, y in rows:
        full = path.relative_to(root).as_posix()
        key = rel_key(path, root)
        out[full] = y
        out[key] = y
    return out


def subsample_balanced(
    rows: list[tuple[Path, int]],
    n_total: int,
    seed: int,
) -> list[tuple[Path, int]]:
    rng = random.Random(seed)
    real = [r for r in rows if r[1] == 0]
    fake = [r for r in rows if r[1] == 1]
    if not real or not fake:
        picked = list(rows)
        rng.shuffle(picked)
        return picked[:n_total]
    half = max(1, n_total // 2)
    real_s = real if len(real) <= half else rng.sample(real, half)
    fake_s = fake if len(fake) <= half else rng.sample(fake, half)
    out = real_s + fake_s
    rng.shuffle(out)
    return out
# end
