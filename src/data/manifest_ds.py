# 2026-08-30, tianqi, train from source/ablation JSONL (DATA_ABLATION_PLAN)
"""Load (path, y) rows from a source or ablation manifest."""

from __future__ import annotations

from pathlib import Path

from src.paths import REPO_ROOT, data_root
from src.transforms.manifest import read_jsonl

# end


def resolve_manifest_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_file():
        return path
    if path.parts and path.parts[0] == "data":
        cand = data_root().joinpath(*path.parts[1:])
        if cand.is_file():
            return cand
    cand = data_root() / path
    if cand.is_file():
        return cand
    raise SystemExit(f"manifest not found: {value}")


def resolve_image_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_file():
        return p
    if p.parts and p.parts[0] == "data":
        cand = data_root().joinpath(*p.parts[1:])
        if cand.is_file():
            return cand
    cand = data_root() / p
    if cand.is_file():
        return cand
    cand = REPO_ROOT / p
    if cand.is_file():
        return cand
    return p


def load_source_manifest_rows(path: Path | str) -> list[tuple[Path, int]]:
    manifest = resolve_manifest_path(path)
    records = read_jsonl(manifest, kind="source")
    rows: list[tuple[Path, int]] = []
    missing = 0
    for rec in records:
        img = resolve_image_path(rec.path)
        if not img.is_file():
            missing += 1
            continue
        rows.append((img, int(rec.label)))
    if not rows:
        raise SystemExit(f"no readable images in {manifest} (missing={missing})")
    if missing:
        print(f"skip missing {missing} files in {manifest}", flush=True)
    return rows


def load_mixin_rows(path: Path | str) -> list[tuple[Path, int]]:
    # 2026-08-31, tianqi, D3 mixin jsonl is path/label only (not full SourceRecord)
    import json

    manifest = resolve_manifest_path(path)
    rows: list[tuple[Path, int]] = []
    missing = 0
    for line in manifest.open():
        rec = json.loads(line)
        img = resolve_image_path(rec["path"])
        if not img.is_file():
            missing += 1
            continue
        rows.append((img, int(rec["label"])))
    if not rows:
        raise SystemExit(f"no readable images in mixin {manifest} (missing={missing})")
    if missing:
        print(f"skip missing {missing} mixin files in {manifest}", flush=True)
    return rows
    # end
# end
