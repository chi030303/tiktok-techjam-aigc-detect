#!/usr/bin/env python3
"""Analyze evaluation-only image patterns against model predictions.

Examples:
    python scripts/analyze_content_patterns.py \
      --split official_val --preds outputs/pred.json \
      --out-dir outputs/content_patterns/official_val

    python scripts/analyze_content_patterns.py \
      --split evalgen --preds outputs/evalgen_pred.json --semantic \
      --out-dir outputs/content_patterns/evalgen
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval.badcase import join_predictions
from src.eval.content_patterns import (
    ClipSemanticExtractor,
    aggregate_groups,
    aggregate_slices,
    extract_low_level,
    load_feature_cache,
    render_candidate_gallery,
    render_report,
    write_feature_cache,
    write_slice_csv,
    write_slice_markdown,
)
from src.eval.evalgen_pool import generator_from_path
from src.eval.labels import load_split
from src.paths import models_root


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: {exc}") from exc
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"{path}: predictions must be a non-empty JSON list")
    for row in rows:
        if not isinstance(row, dict) or "image_path" not in row or "pred" not in row:
            raise SystemExit(
                f"{path}: every prediction needs image_path and pred"
            )
    return rows


def _resolve_path(raw: str, image_root: Path) -> Path:
    path = Path(raw)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend((image_root / path, REPO / path))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def _prepare_predictions(
    predictions: list[dict[str, Any]],
    split: str,
    image_root: Path,
    split_rows: list[tuple[Path, int]],
    threshold: float,
    default_condition: str,
) -> list[dict[str, Any]]:
    if all("y" in row or "label" in row for row in predictions):
        prepared = []
        for row in predictions:
            label = int(row.get("y", row.get("label")))
            if label not in {0, 1}:
                raise SystemExit(f"bad label for {row['image_path']}: {label}")
            score = float(row["pred"])
            if not 0 <= score <= 1:
                raise SystemExit(
                    f"pred must be in [0,1] for {row['image_path']}: {score}"
                )
            prepared.append(
                {
                    **row,
                    "image_path": str(row["image_path"]),
                    "pred": score,
                    "label": label,
                    "condition": str(row.get("condition") or default_condition),
                }
            )
    else:
        result = join_predictions(
            predictions,
            split_rows,
            src_root=image_root,
            predict_root=image_root,
            threshold=threshold,
            default_condition=default_condition,
        )
        if result["unmatched_labels"]:
            print(
                f"warning: skipped {result['unmatched_labels']} predictions "
                "without labels",
                file=sys.stderr,
            )
        prepared = result["joined"]

    for row in prepared:
        resolved = _resolve_path(row["image_path"], image_root)
        row["_resolved_path"] = resolved.as_posix()
        if int(row["label"]) == 0:
            row["generator"] = "real"
        elif row.get("generator") in {None, "", "unknown"}:
            row["generator"] = (
                generator_from_path(resolved, image_root)
                if split == "evalgen"
                else "dalle_advanced"
            )
    return prepared


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_feature_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            public = {key: value for key, value in row.items() if key != "_resolved_path"}
            handle.write(json.dumps(public, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", choices=("official_val", "evalgen"), required=True)
    parser.add_argument("--preds", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--image-root",
        type=Path,
        help="override split root when prediction paths are relative",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--condition",
        default="clean",
        help="condition name when prediction rows do not include one",
    )
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--min-generator-support", type=int, default=20)
    parser.add_argument("--semantic", action="store_true")
    parser.add_argument(
        "--clip-model",
        type=Path,
        default=models_root() / "clip-vit-base-patch16",
    )
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-gallery-patterns", type=int, default=8)
    parser.add_argument("--gallery-per-error", type=int, default=4)
    args = parser.parse_args()

    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    for name in (
        "max_side",
        "min_support",
        "min_generator_support",
        "batch_size",
        "max_gallery_patterns",
        "gallery_per_error",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max-images must be positive")

    split_root, split_rows = load_split(args.split)
    image_root = (args.image_root or split_root).resolve()
    predictions = _load_predictions(args.preds)
    prepared = _prepare_predictions(
        predictions,
        args.split,
        image_root,
        split_rows,
        args.threshold,
        args.condition,
    )
    if args.max_images is not None and len(prepared) > args.max_images:
        prepared = random.Random(args.seed).sample(prepared, args.max_images)

    missing = [
        row["_resolved_path"]
        for row in prepared
        if not Path(row["_resolved_path"]).is_file()
    ]
    if missing:
        examples = "\n  ".join(missing[:5])
        raise SystemExit(
            f"{len(missing)} image paths do not exist; check --image-root. Examples:\n  "
            f"{examples}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.out_dir / "feature_cache.jsonl"
    cache = load_feature_cache(cache_path)
    features_by_path: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(prepared, 1):
        path = Path(row["_resolved_path"])
        key = path.resolve().as_posix()
        features = cache.get(key)
        if features is None or features.get("analysis_max_side") != args.max_side:
            features = extract_low_level(path, max_side=args.max_side)
        features_by_path[key] = features
        if index % 500 == 0:
            print(f"low-level features {index}/{len(prepared)}", flush=True)

    if args.semantic:
        semantic_model_id = args.clip_model.resolve().as_posix()
        semantic_missing = [
            Path(path)
            for path, features in features_by_path.items()
            if features.get("semantic_model") != semantic_model_id
        ]
        if semantic_missing:
            extractor = ClipSemanticExtractor(args.clip_model, device=args.device)
            semantic = extractor.extract(
                semantic_missing, batch_size=args.batch_size
            )
            for path, values in semantic.items():
                features_by_path[path].update(values)
                features_by_path[path]["semantic_model"] = semantic_model_id
            print(
                f"semantic features {len(semantic)}/{len(semantic_missing)}",
                flush=True,
            )

    write_feature_cache(cache_path, features_by_path.values())
    analyzed = []
    for row in prepared:
        features = features_by_path[row["_resolved_path"]]
        analyzed.append({**row, **features})

    slice_rows, overall = aggregate_slices(
        analyzed,
        threshold=args.threshold,
        min_support=args.min_support,
        min_generator_support=args.min_generator_support,
    )
    condition_rows = aggregate_groups(analyzed, "condition", args.threshold)
    generator_rows = aggregate_groups(analyzed, "generator", args.threshold)

    _write_feature_rows(args.out_dir / "image_features.jsonl", analyzed)
    write_slice_csv(args.out_dir / "slice_metrics.csv", slice_rows)
    _write_json(args.out_dir / "slice_metrics.json", slice_rows)
    write_slice_markdown(args.out_dir / "slice_metrics.md", slice_rows)
    _write_json(
        args.out_dir / "group_metrics.json",
        {
            "overall": overall,
            "by_condition": condition_rows,
            "by_generator": generator_rows,
        },
    )
    report = render_report(
        slice_rows,
        overall,
        split=args.split,
        threshold=args.threshold,
    )
    (args.out_dir / "pattern_report.md").write_text(report, encoding="utf-8")
    gallery = render_candidate_gallery(
        analyzed,
        slice_rows,
        max_patterns=args.max_gallery_patterns,
        per_error_type=args.gallery_per_error,
        threshold=args.threshold,
    )
    (args.out_dir / "pattern_gallery.html").write_text(gallery, encoding="utf-8")

    print(
        f"wrote {len(analyzed)} image features and {len(slice_rows)} slices "
        f"to {args.out_dir}",
        flush=True,
    )
    print(
        "evaluation only: do not use official val or EvalGEN for training "
        "or hard-negative mining",
        flush=True,
    )


if __name__ == "__main__":
    main()
