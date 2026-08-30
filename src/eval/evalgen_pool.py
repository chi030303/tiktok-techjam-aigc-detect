# 2026-08-30, tianqi, pair EvalGEN fakes with SID/WildFake/COCO reals so AUROC is defined
"""EvalGEN is generator folders of fakes only. Pair them with a real pool for AUROC.

Reals, in order of preference for CLIP-SID:
  sid_val  — SID_Set validation label=0 (in-domain photos, not train, not COCO val)
  coco     — data/val/real (same Non-AIGC half as official demo; comparable, not new)
  wildfake — data/wildfake Real/ tree, minus COCO val2017 overlap with data/val

SID validation lives in parquet, so we either score on-the-fly or export once
to data/eval_pools/sid_val_real/real/. That pool is hold-out: do not train on it.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image

from src.eval.labels import EVALGEN_FAKE, list_images, load_labeled_dir, load_split
from src.paths import data_root

# end

SID_VAL_REAL_REL = ("eval_pools", "sid_val_real")


def generator_from_path(path: Path, root: Path | None = None) -> str:
    try:
        parts = Path(path).relative_to(root).parts if root is not None else Path(path).parts
    except ValueError:
        parts = Path(path).parts
    lowered = [p.lower() for p in parts]
    for name in sorted(EVALGEN_FAKE):
        if name in lowered:
            return name
    if any("dalle" in p for p in lowered):
        return "dalle3"
    if "fake" in lowered:
        return "fake"
    if "real" in lowered:
        return "real"
    return "unknown"


def subsample_fakes_per_generator(
    rows: list[tuple[Path, int]],
    root: Path,
    n_per_gen: int | None,
    seed: int,
) -> list[tuple[Path, int]]:
    rng = random.Random(seed)
    reals: list[tuple[Path, int]] = []
    by: dict[str, list[tuple[Path, int]]] = {}
    for path, y in rows:
        if int(y) == 0:
            reals.append((path, y))
            continue
        by.setdefault(generator_from_path(path, root), []).append((path, y))
    out = list(reals)
    for _g, items in sorted(by.items()):
        if n_per_gen is None or len(items) <= n_per_gen:
            out.extend(items)
        else:
            out.extend(rng.sample(items, n_per_gen))
    rng.shuffle(out)
    return out


def load_coco_reals() -> list[tuple[Path, int]]:
    root, rows = load_split("official_val")
    return [(p, 0) for p, y in rows if int(y) == 0]


def _coco_val_basenames() -> set[str]:
    d = data_root() / "val" / "real"
    if not d.is_dir():
        return set()
    return {p.name for p in list_images(d)}


def find_wildfake_real_root() -> Path:
    root = data_root() / "wildfake"
    for rel in (
        ("Images", "Real"),
        ("Real",),
        ("real",),
        ("REAL",),
    ):
        cand = root.joinpath(*rel)
        if cand.is_dir():
            return cand
    raise SystemExit(
        f"no WildFake real tree under {root} "
        "(expected Images/Real, Real/, or real/). "
        "Full WildFake is trainable after excluding data/val overlap; "
        "data/val itself stays DO_NOT_TRAIN."
    )


def load_wildfake_reals(exclude_coco_val: bool = True) -> list[tuple[Path, int]]:
    root = find_wildfake_real_root()
    skip = _coco_val_basenames() if exclude_coco_val else set()
    rows: list[tuple[Path, int]] = []
    for path in list_images(root):
        if exclude_coco_val and "val2017" in path.as_posix():
            continue
        if path.name in skip:
            continue
        rows.append((path, 0))
    if not rows:
        raise SystemExit(f"no WildFake reals under {root} after overlap filter")
    return rows


def sid_val_real_export_dir() -> Path:
    return data_root().joinpath(*SID_VAL_REAL_REL) / "real"


def sid_val_reals_ready() -> bool:
    dest = sid_val_real_export_dir()
    return dest.is_dir() and bool(list_images(dest))


# 2026-08-30, tianqi, eval-only SID parquet load; do not import src.data.sid (train WIP)
def load_sid_hf(split: str):
    from datasets import load_dataset

    root = data_root() / "sid_set"
    if not root.is_dir():
        raise FileNotFoundError(root)
    try:
        return load_dataset(str(root), split=split)
    except Exception:
        data = root / "data"
        files = {
            "train": sorted(str(p) for p in data.glob("train-*.parquet")),
            "validation": sorted(str(p) for p in data.glob("validation-*.parquet")),
        }
        return load_dataset("parquet", data_files=files, split=split)
# end


def load_sid_val_real_files() -> list[tuple[Path, int]]:
    dest = sid_val_real_export_dir()
    if not dest.is_dir():
        raise SystemExit(
            f"SID val reals not exported at {dest}. "
            "Run with --export-sid-reals or --reals sid_val (on-the-fly parquet)."
        )
    return [(p, 0) for p in list_images(dest)]


def export_sid_val_reals(max_images: int | None = None, seed: int = 0) -> Path:
    dest = sid_val_real_export_dir()
    dest.mkdir(parents=True, exist_ok=True)
    pool = dest.parent
    # 2026-08-30, tianqi, SID val reals are a hold-out pool, never a train source
    (pool / "DO_NOT_TRAIN").write_text(
        "SID_Set validation reals (label=0), exported for EvalGEN pairing.\n"
        "Do NOT train on this pool. Train SID uses the train split only.\n"
    )
    # end
    ds = load_sid_hf("validation")
    keep = [i for i, y in enumerate(ds["label"]) if int(y) == 0]
    if max_images is not None and len(keep) > max_images:
        keep = random.Random(seed).sample(keep, max_images)
    meta = pool / "EXPORT_META.json"
    if dest.is_dir() and list_images(dest) and meta.is_file():
        payload = json.loads(meta.read_text(encoding="utf-8"))
        if payload.get("n") == len(keep):
            return dest
    n = 0
    for src_i in keep:
        ex = ds[int(src_i)]
        img = ex["image"]
        if not isinstance(img, Image.Image):
            img = Image.open(img).convert("RGB")
        else:
            img = img.convert("RGB")
        tag = str(ex.get("img_id") or src_i)
        out = dest / f"{tag}.jpg"
        if not out.exists():
            img.save(out, format="JPEG", quality=95)
        n += 1
        if n % 500 == 0:
            print(f"export sid val reals {n}/{len(keep)}", flush=True)
    meta.write_text(json.dumps({"n": n, "split": "validation", "label": 0}, indent=2), encoding="utf-8")
    print(f"exported {n} SID val reals -> {dest}", flush=True)
    return dest


def load_reals(kind: str, reals_dir: Path | None = None) -> tuple[str, list[tuple[Path, int]]]:
    kind = kind.strip().lower()
    if kind in {"dir", "custom"}:
        if reals_dir is None:
            raise SystemExit("--reals dir needs --reals-dir")
        root = Path(reals_dir)
        labeled_dirs = any((root / n).is_dir() for n in ("real", "REAL", "fake", "FAKE"))
        if root.name.lower() == "real" or not labeled_dirs:
            rows = [(p, 0) for p in list_images(root)]
        else:
            rows = [(p, y) for p, y in load_labeled_dir(root) if int(y) == 0]
        if not rows:
            raise SystemExit(f"no reals under {root}")
        return "dir", rows
    if kind in {"coco", "official_val", "val"}:
        return "coco", load_coco_reals()
    if kind in {"wildfake", "wf"}:
        return "wildfake", load_wildfake_reals(exclude_coco_val=True)
    if kind in {"sid_val", "sid"}:
        dest = sid_val_real_export_dir()
        if dest.is_dir() and list_images(dest):
            return "sid_val", load_sid_val_real_files()
        raise SystemExit(
            "SID val reals are parquet-backed. Pass --export-sid-reals once, "
            "or use --reals coco / --reals wildfake."
        )
    raise SystemExit(f"unknown --reals {kind!r}; use sid_val | coco | wildfake | dir")


def pair_evalgen(
    fake_rows: list[tuple[Path, int]],
    real_rows: list[tuple[Path, int]],
) -> list[tuple[Path, int]]:
    fakes = [(p, 1) for p, y in fake_rows if int(y) == 1]
    reals = [(p, 0) for p, y in real_rows if int(y) == 0]
    if not fakes:
        raise SystemExit("EvalGEN pairing has no fakes")
    if not reals:
        raise SystemExit("EvalGEN pairing has no reals")
    return fakes + reals
# end
