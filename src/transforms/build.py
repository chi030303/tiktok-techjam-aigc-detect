# 2026-08-29, zyun, build the frozen transformed eval set from a source manifest
"""Generate the deterministic transformed eval set.

Reads one or more source manifests (build_source.py output), applies the
official settings (spec.py) to each image, writes derived images under
``--out-root`` and a JSONL transform manifest. Re-running with the same
source manifest yields byte-identical results; existing derived images are
skipped unless ``--overwrite``. Run from the repo root so stored paths stay
relative.

Examples:
    # smoke: 2 settings, 200 images per setting
    python -m src.transforms.build \
        --source-manifest data/manifests/source_eval.jsonl \
        --out-manifest data/manifests/transforms_eval.jsonl \
        --settings blur_s10,jpeg_q50 --splits test --limit-per-setting 200

    # full: all 14 official settings (train split excluded by default)
    python -m src.transforms.build \
        --source-manifest data/manifests/source_eval.jsonl \
        --out-manifest data/manifests/transforms_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from .manifest import SourceRecord, TransformRecord, read_jsonl
from .ops import apply_setting, sample_jitter_factors
from .spec import Setting, derive_seed, resolve_settings


def expected_size(setting: Setting, src: SourceRecord, crop_resize_back: bool) -> tuple[int, int]:
    """Derived-image size without opening the file (crop shrinks, others keep)."""
    w, h = src.width, src.height
    if setting.op == "crop" and not crop_resize_back:
        keep = setting.params["keep"]
        return max(1, round(w * keep)), max(1, round(h * keep))
    return w, h


def run_build(
    source_records: list[SourceRecord],
    settings: list[Setting],
    out_root: str | Path,
    out_manifest: str | Path,
    splits: set[str] | None = None,
    crop_resize_back: bool = False,
    overwrite: bool = False,
    limit_per_setting: int = 0,
) -> dict:
    if not source_records:
        raise SystemExit("source manifest is empty; run build_source.py first")
    if splits:
        wanted = set(splits)
        source_records = [r for r in source_records if r.split in wanted]
        if not source_records:
            raise SystemExit(f"no source rows with split in {sorted(wanted)}")

    out_root = Path(out_root)
    out_manifest = Path(out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    rows = made = skipped = 0
    with out_manifest.open("w", encoding="utf-8") as sink:
        for setting in settings:
            ext = "jpg" if setting.op == "jpeg" else "png"
            batch = (
                source_records[:limit_per_setting] if limit_per_setting > 0 else source_records
            )
            s_made = s_skip = 0
            for src in batch:
                seed = derive_seed(src.image_id, setting.key)
                rel_path = (
                    out_root / setting.key / src.image_id[:2] / f"{src.image_id}.{ext}"
                ).as_posix()
                out_path = Path(rel_path)
                if out_path.exists() and not overwrite:
                    # No image work needed; factors/params are seed-derived anyway.
                    if setting.op == "jitter":
                        actual = sample_jitter_factors(
                            np.random.default_rng(seed), setting.params["range"]
                        )
                    else:
                        actual = dict(setting.params)
                    s_skip += 1
                else:
                    with Image.open(src.path) as im:
                        result, actual = apply_setting(im, setting, np.random.default_rng(seed), crop_resize_back)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    if isinstance(result, bytes):  # jpeg: exact encoded bytes
                        out_path.write_bytes(result)
                    else:
                        result.save(out_path, format="PNG")
                    s_made += 1
                width, height = expected_size(setting, src, crop_resize_back)
                rec = TransformRecord(
                    row_id=f"{src.image_id}_{setting.key}",
                    source_image_id=src.image_id,
                    source_path=src.path,
                    transform=setting.op,
                    transform_key=setting.key,
                    params=actual,
                    seed=seed,
                    path=rel_path,
                    label=src.label,
                    source_dataset=src.source_dataset,
                    generator=src.generator,
                    split=src.split,
                    width=width,
                    height=height,
                )
                sink.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                rows += 1
            made += s_made
            skipped += s_skip
            print(f"[{setting.key}] made={s_made} skipped={s_skip}", file=sys.stderr)
    return {"rows": rows, "made": made, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source-manifest",
        required=True,
        help="comma-separated source manifest JSONL files",
    )
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--out-root", default="data/transforms")
    parser.add_argument(
        "--settings", default="all", help="comma-separated keys, or 'all' (14 official)"
    )
    parser.add_argument(
        "--splits",
        default="val,test,unseen",
        help="source splits to transform; schema names only, train excluded by default",
    )
    parser.add_argument("--limit-per-setting", type=int, default=0)
    parser.add_argument(
        "--crop-resize-back",
        action="store_true",
        help="resize cropped output back to source size (off by default)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sources: list[SourceRecord] = []
    for part in args.source_manifest.split(","):
        sources.extend(read_jsonl(Path(part.strip()), kind="source"))
    stats = run_build(
        sources,
        resolve_settings(args.settings),
        args.out_root,
        args.out_manifest,
        splits={s.strip() for s in args.splits.split(",") if s.strip()} or None,
        crop_resize_back=args.crop_resize_back,
        overwrite=args.overwrite,
        limit_per_setting=args.limit_per_setting,
    )
    print(
        f"rows={stats['rows']} made={stats['made']} skipped={stats['skipped']}"
        f" -> {args.out_manifest}"
    )


if __name__ == "__main__":
    main()
# end
