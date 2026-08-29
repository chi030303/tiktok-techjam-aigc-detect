#!/usr/bin/env python3
# 2026-08-29, zyun, bad-case CLI: full FP/FN dump + per-group statistics
"""Collect and aggregate bad cases from a predict.py JSON.

Examples:

    # clean condition on the official demo set, with generator metadata
    python scripts/run_badcase.py --pred outputs/pred.json --split official_val \
        --manifest data/manifests/source_demo_val.jsonl

    # arbitrary folder + fixed condition tag
    python scripts/run_badcase.py --pred outputs/pred.json \
        --image-dir data/transforms_eval/jpeg_q50 --condition jpeg_q50

Outputs under --out-dir (default outputs/badcases/<split>):
- badcases.jsonl       every FP/FN with generator/condition metadata
- badcase_stats.csv    error counts/rates per condition, generator, source dataset
- badcase_summary.json totals + worst-K cases + AUROC from src.eval.metrics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval.badcase import (
    error_rows,
    join_predictions,
    load_manifest_rows,
    summarize,
    write_jsonl,
    write_stats_csv,
)
from src.eval.labels import load_labeled_dir, load_split, subsample_balanced
from src.eval.metrics import binary_metrics

# end


def _load_rows(args) -> tuple[str, Path, list]:
    if args.image_dir is not None:
        root = args.image_dir
        name = args.split or root.name
        return name, root, load_labeled_dir(root, default_fake=name == "evalgen")
    if not args.split:
        raise SystemExit("need --split or --image-dir")
    root, rows = load_split(args.split)
    return args.split, root, rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pred", required=True, type=Path, help="predict.py JSON")
    parser.add_argument("--split", help="official_val | val | evalgen | cifake_test")
    parser.add_argument("--image-dir", type=Path, help="labeled folder (real/fake dirs)")
    parser.add_argument(
        "--condition", default="clean",
        help="condition tag when rows carry no transform_key (default: clean)",
    )
    parser.add_argument(
        "--manifest",
        help="comma-separated manifest JSONLs adding generator/source_dataset metadata",
    )
    parser.add_argument("--predict-root", type=Path, help="root the pred paths resolve against")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-images", type=int, help="balanced subsample before joining")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worst-k", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, help="default: outputs/badcases/<split>")
    args = parser.parse_args()

    name, root, rows = _load_rows(args)
    if args.max_images is not None:
        rows = subsample_balanced(rows, args.max_images, args.seed)
    manifest_rows = load_manifest_rows(
        [Path(x) for x in args.manifest.split(",")] if args.manifest else None
    )
    preds = json.loads(args.pred.read_text(encoding="utf-8"))

    res = join_predictions(
        preds,
        rows,
        src_root=root,
        predict_root=args.predict_root or root,
        threshold=args.threshold,
        manifest_rows=manifest_rows,
        default_condition=args.condition,
    )
    joined = res["joined"]
    metrics = binary_metrics([r["label"] for r in joined], [r["pred"] for r in joined],
                             threshold=args.threshold)
    summary = summarize(joined, threshold=args.threshold, worst_k=args.worst_k)
    summary["metrics"] = metrics
    summary["split"] = name
    summary["default_condition"] = args.condition
    summary["unmatched_labels"] = res["unmatched_labels"]
    summary["unmatched_metadata"] = res["unmatched_metadata"]
    summary["n_manifest_rows"] = res["n_manifest_rows"]

    out_dir = args.out_dir or Path("outputs/badcases") / name
    write_jsonl(out_dir / "badcases.jsonl", error_rows(joined))
    write_stats_csv(out_dir / "badcase_stats.csv", summary)
    (out_dir / "badcase_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"split={name} images={summary['n_images']} fp={summary['n_fp']} "
        f"fn={summary['n_fn']} unmatched_labels={res['unmatched_labels']} "
        f"-> {out_dir}/{{badcases.jsonl,badcase_stats.csv,badcase_summary.json}}"
    )


if __name__ == "__main__":
    main()
# end
