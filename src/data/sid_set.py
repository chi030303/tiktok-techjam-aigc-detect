# 2026-08-30, samily, SID_Set manifest builder (real / full_synth / tampered)
"""Build source manifests from SID_Set on disk.

SID_Set labels (HF parquet ``label`` column):
  0 = real
  1 = fully synthetic (FLUX t2i)  -> train as fake=1, arch=flow
  2 = tampered (local edit)       -> content_type=partial_manipulation, excluded from train

Supports:
  1. Extracted PNG tree under ``data/sid_set/extracted/{real,full_synth,tampered}/``
  2. HF download with parquet shards under ``data/sid_set/*.parquet`` (extract-on-index)
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from PIL import Image

from src.paths import data_root
from src.transforms.manifest import SourceRecord, average_phash, write_jsonl

# end

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

SID_LABEL_MAP = {
    0: dict(
        label=0,
        generator=None,
        family=None,
        arch=None,
        generation_type=None,
        content_type="real",
        subdir="real",
    ),
    1: dict(
        label=1,
        generator="flux",
        family="diffusion",
        arch="flow",
        generation_type="t2i",
        content_type="full_synthetic",
        subdir="full_synth",
    ),
    2: dict(
        label=1,
        generator="flux",
        family="diffusion",
        arch="flow",
        generation_type="i2i",
        content_type="partial_manipulation",
        subdir="tampered",
    ),
}


def _image_id(rel_path: str, size: int) -> str:
    return hashlib.sha1(f"{rel_path}:{size}".encode("utf-8")).hexdigest()


def _record_from_image(
    path: Path,
    rel_path: str,
    meta: dict,
    split: str,
    compute_phash: bool,
) -> SourceRecord:
    with Image.open(path) as im:
        width, height = im.size
        phash = average_phash(im) if compute_phash else None
    fmt = path.suffix.lower().lstrip(".")
    return SourceRecord(
        image_id=_image_id(rel_path, path.stat().st_size),
        path=rel_path,
        label=meta["label"],
        source_dataset="sid_set",
        generator=meta["generator"],
        split=split,
        width=width,
        height=height,
        family=meta["family"],
        arch=meta["arch"],
        generation_type=meta["generation_type"],
        content_type=meta["content_type"],
        original_format=fmt or None,
        phash=phash,
    )


def collect_extracted(
    root: Path,
    split: str = "train",
    compute_phash: bool = True,
) -> list[SourceRecord]:
    """Index ``root/{real,full_synth,tampered}/`` image files."""
    records: list[SourceRecord] = []
    counts = {0: 0, 1: 0, 2: 0}
    for sid_label, meta in SID_LABEL_MAP.items():
        folder = root / meta["subdir"]
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*")):
            if not (path.is_file() and path.suffix.lower() in IMAGE_EXTS):
                continue
            try:
                rel = path.relative_to(Path.cwd()).as_posix()
            except ValueError:
                rel = path.resolve().as_posix()
            records.append(
                _record_from_image(path, rel, meta, split, compute_phash)
            )
            counts[sid_label] += 1
    print(
        f"SID extracted: {len(records)} images "
        f"(real={counts[0]} full_synth={counts[1]} tampered={counts[2]})",
        file=sys.stderr,
    )
    return records


def _raw_image_suffix(data: bytes) -> str:
    from io import BytesIO

    with Image.open(BytesIO(data)) as im:
        fmt = (im.format or "png").lower()
    return {"jpeg": "jpg", "tiff": "tif"}.get(fmt, fmt)


def _save_bytes_atomic(data: bytes, dest: Path) -> None:
    """Preserve source bytes; avoids a costly and lossy decode/re-encode pass."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)


def collect_parquet(
    parquet_dir: Path,
    extract_dir: Path,
    split: str = "train",
    max_rows: int | None = None,
    compute_phash: bool = True,
) -> list[SourceRecord]:
    """Extract images from SID parquet shards into ``extract_dir`` and index."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pip install pyarrow to index SID_Set parquet") from exc

    source_split = "validation" if split == "val" else "train"
    shards = sorted(parquet_dir.glob(f"{source_split}-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet under {parquet_dir}")

    records: list[SourceRecord] = []
    n = 0
    for shard in shards:
        # Reading masks roughly doubles I/O and memory while adding no classifier
        # metadata, so request only the two required columns.
        table = pq.read_table(shard, columns=["image", "label"])
        cols = table.column_names
        if "label" not in cols or "image" not in cols:
            raise SystemExit(f"unexpected SID parquet schema in {shard}: {cols}")
        for i in range(table.num_rows):
            if max_rows is not None and n >= max_rows:
                break
            sid_label = int(table.column("label")[i].as_py())
            if sid_label not in SID_LABEL_MAP:
                continue
            # DATA_ABLATION_PLAN: tampered (2) never enters train pool
            if sid_label == 2:
                continue
            meta = SID_LABEL_MAP[sid_label]
            cell = table.column("image")[i].as_py()
            if isinstance(cell, dict):
                raw = cell.get("bytes") or cell.get("path")
            else:
                raw = cell
            if raw is None:
                continue
            if not isinstance(raw, (bytes, bytearray)):
                raise SystemExit(f"unsupported image cell type in {shard}:{i}")
            raw = bytes(raw)
            image_id = hashlib.sha1(f"{shard.name}:{i}".encode()).hexdigest()
            suffix = _raw_image_suffix(raw)
            dest = (
                extract_dir
                / source_split
                / meta["subdir"]
                / image_id[:2]
                / f"{image_id}.{suffix}"
            )
            rel_path = dest.as_posix()
            if not dest.is_file():
                _save_bytes_atomic(raw, dest)
            records.append(
                _record_from_image(dest, rel_path, meta, split, compute_phash)
            )
            n += 1
            if n % 500 == 0:
                print(
                    f"SID {source_split}: indexed {n}"
                    f"{f'/{max_rows}' if max_rows else ''}",
                    file=sys.stderr,
                    flush=True,
                )
        if max_rows is not None and n >= max_rows:
            break
    print(f"SID parquet: indexed {len(records)} rows -> {extract_dir}", file=sys.stderr)
    return records


def build_sid_manifest(
    out: Path,
    *,
    split: str = "train",
    max_rows: int | None = None,
    compute_phash: bool = True,
) -> list[SourceRecord]:
    root = data_root() / "sid_set"
    parquet_dir = root / "data" if (root / "data").is_dir() else root
    extracted = root / "extracted_v2"
    source_split = "validation" if split == "val" else "train"
    if any(parquet_dir.glob(f"{source_split}-*.parquet")):
        records = collect_parquet(
            parquet_dir,
            extract_dir=extracted,
            split=split,
            max_rows=max_rows,
            compute_phash=compute_phash,
        )
    else:
        # Also support manually extracted trees when parquet shards are absent.
        records = collect_extracted(root / "extracted", split=split, compute_phash=compute_phash)
    write_jsonl(out, records)
    print(f"wrote {len(records)} SID rows -> {out}")
    return records
