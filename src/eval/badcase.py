# 2026-08-29, zyun, bad-case collection & statistics (full FP/FN dump + group aggregation)
"""Bad-case pipeline: full FP/FN collection with metadata, plus group statistics.

Builds on ``src.eval.score`` (same join semantics via ``_lookup_y``) and enriches
each prediction with optional manifest metadata (generator / source_dataset /
transform_key from ``src.transforms.build_source`` manifests), then aggregates
error rates per condition, generator and source dataset.

Outputs (see docs/badcase.md):
- ``badcases.jsonl``     one row per FP/FN, full metadata — input for error analysis
- ``badcase_stats.csv``  flat per-group counts/rates (condition / generator / dataset)
- ``badcase_summary.json`` totals, rates, per-group stats, worst-K cases
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from src.eval.labels import index_by_rel
from src.eval.score import _lookup_y

# end


def normalize_path(p: str) -> str:
    return Path(p).as_posix()


def load_manifest_rows(paths: list[Path] | None) -> list[dict]:
    """Read generic manifest JSONLs; requires only ``path`` per row."""
    rows: list[dict] = []
    if not paths:
        return rows
    for part in paths:
        with Path(part).open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{part}:{lineno}: {exc}")
                if "path" not in row:
                    raise SystemExit(f"{part}:{lineno}: manifest row missing 'path'")
                rows.append(row)
    return rows


def join_predictions(
    preds: list[dict],
    rows: list[tuple[Path, int]],
    src_root: Path,
    predict_root: Path,
    threshold: float = 0.5,
    manifest_rows: list[dict] | None = None,
    default_condition: str = "clean",
) -> dict:
    """Join every prediction to its label and optional manifest metadata.

    Returns ``{"joined", "unmatched_labels", "unmatched_metadata",
    "n_manifest_rows"}``. Each joined row carries ``error_type`` in
    ``TP / FP / FN / TN`` plus ``condition / generator / source_dataset``
    (manifest values when available, else ``default_condition`` / ``unknown``).
    """
    by_rel = index_by_rel(rows, src_root)
    meta_by_path: dict[str, dict] = {}
    for m in manifest_rows or []:
        meta_by_path.setdefault(normalize_path(str(m["path"])), m)

    joined: list[dict] = []
    unmatched_labels = unmatched_meta = 0
    for row in preds:
        if "image_path" not in row or "pred" not in row:
            raise SystemExit("pred JSON must be a list of {image_path, pred}")
        image_path = str(row["image_path"])
        y = _lookup_y(image_path, predict_root, by_rel)
        if y is None:
            unmatched_labels += 1
            continue
        s = float(row["pred"])
        meta = meta_by_path.get(normalize_path(image_path))
        if meta is None:
            unmatched_meta += 1
            meta = {}
        predicted_positive = s >= threshold
        if y == 1:
            error_type = "TP" if predicted_positive else "FN"
        else:
            error_type = "FP" if predicted_positive else "TN"
        joined.append(
            {
                "image_path": image_path,
                "pred": s,
                "label": y,
                "error_type": error_type,
                "condition": meta.get("transform_key") or default_condition,
                "generator": meta.get("generator") or "unknown",
                "source_dataset": meta.get("source_dataset") or "unknown",
            }
        )
    if not joined:
        raise SystemExit("no predictions matched labels")
    return {
        "joined": joined,
        "unmatched_labels": unmatched_labels,
        "unmatched_metadata": unmatched_meta,
        "n_manifest_rows": len(manifest_rows or []),
    }


def error_rows(joined: list[dict]) -> list[dict]:
    """Only the errors — the badcases.jsonl payload."""
    return [r for r in joined if r["error_type"] in ("FP", "FN")]


def _group_stats(joined: list[dict], key: str) -> dict:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in joined:
        buckets[str(r[key])].append(r)
    stats: dict[str, dict] = {}
    for value in sorted(buckets):
        rs = buckets[value]
        n = len(rs)
        n_fp = sum(r["error_type"] == "FP" for r in rs)
        n_fn = sum(r["error_type"] == "FN" for r in rs)
        stats[value] = {
            "n_images": n,
            "n_fp": n_fp,
            "n_fn": n_fn,
            "fp_rate": round(n_fp / n, 6),
            "fn_rate": round(n_fn / n, 6),
        }
    return stats


def summarize(joined: list[dict], threshold: float, worst_k: int = 20) -> dict:
    """Totals, per-group error rates, and the worst-K FP/FN for quick triage."""
    fps = [r for r in joined if r["error_type"] == "FP"]
    fns = [r for r in joined if r["error_type"] == "FN"]
    n = len(joined)
    return {
        "threshold": threshold,
        "n_images": n,
        "n_fp": len(fps),
        "n_fn": len(fns),
        "fp_rate": round(len(fps) / n, 6) if n else 0.0,
        "fn_rate": round(len(fns) / n, 6) if n else 0.0,
        "by_condition": _group_stats(joined, "condition"),
        "by_generator": _group_stats(joined, "generator"),
        "by_source_dataset": _group_stats(joined, "source_dataset"),
        "worst_fp": sorted(fps, key=lambda r: -r["pred"])[:worst_k],
        "worst_fn": sorted(fns, key=lambda r: r["pred"])[:worst_k],
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_stats_csv(path: Path, summary: dict) -> None:
    """Flat long-format table: one row per (group_type, group_value)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["group_type", "group_value", "n_images", "n_fp", "n_fn", "fp_rate", "fn_rate"])
        w.writerow(
            ["overall", "all", summary["n_images"], summary["n_fp"], summary["n_fn"],
             summary["fp_rate"], summary["fn_rate"]]
        )
        for group_type in ("by_condition", "by_generator", "by_source_dataset"):
            for value, st in summary[group_type].items():
                w.writerow(
                    [group_type.removeprefix("by_"), value, st["n_images"],
                     st["n_fp"], st["n_fn"], st["fp_rate"], st["fn_rate"]]
                )
# end
