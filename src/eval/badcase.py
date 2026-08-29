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
import math
from collections import defaultdict
from pathlib import Path

from src.eval.labels import index_by_rel
from src.eval.score import _lookup_y

# end


def _resolve_forms(p: Path, bases: list[Path]) -> list[str]:
    """Canonical absolute forms of ``p``: as-is, then relative to each base.

    resolve() is non-strict, so lexical ``..``/symlink normalization also
    works for paths that no longer exist on disk. predict.py emits whatever
    path form the caller passed while build_source stores CWD-relative or
    absolute paths, so the metadata join must not rely on raw string equality
    (PR#5 review: absolute pred paths used to silently drop all metadata).
    """
    forms = [p.resolve().as_posix()]
    if not p.is_absolute():
        for base in bases:
            form = (base / p).resolve().as_posix()
            if form not in forms:
                forms.append(form)
    return forms


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
                if not isinstance(row, dict) or "path" not in row:
                    raise SystemExit(
                        f"{part}:{lineno}: manifest row must be a JSON object with 'path'"
                    )
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

    The label join mirrors ``score._lookup_y`` exactly. The metadata join
    canonicalizes both sides to absolute forms (raw plus relative to
    ``src_root`` / ``predict_root``), so a pred JSON written from absolute
    paths still attaches to a CWD-relative manifest and vice versa; only
    same-file-different-path-identity cases (hardlinks, mounts) fall through.

    Returns ``{"joined", "unmatched_labels", "unmatched_metadata",
    "n_manifest_rows"}``. Each joined row carries ``error_type`` in
    ``TP / FP / FN / TN`` plus ``condition / generator / source_dataset``:
    ``condition`` is the manifest ``transform_key`` or ``default_condition``,
    ``generator`` is the manifest value, ``real`` for label-0 images (reals
    have no generator by definition), else ``unknown``.
    """
    if not isinstance(preds, list):
        raise SystemExit("pred JSON must be a list of {image_path, pred}")
    by_rel = index_by_rel(rows, src_root)
    meta_bases = [src_root, predict_root]
    meta_by_path: dict[str, dict] = {}
    for m in manifest_rows or []:
        for form in _resolve_forms(Path(str(m["path"])), meta_bases):
            meta_by_path.setdefault(form, m)

    joined: list[dict] = []
    unmatched_labels = unmatched_meta = 0
    for row in preds:
        if not isinstance(row, dict) or "image_path" not in row or "pred" not in row:
            raise SystemExit("pred JSON must be a list of {image_path, pred}")
        image_path = str(row["image_path"])
        y = _lookup_y(image_path, predict_root, by_rel)
        if y is None:
            unmatched_labels += 1
            continue
        s = float(row["pred"])
        if math.isnan(s):
            raise SystemExit(f"pred for {image_path} is NaN; fix the predictor first")
        meta = None
        for form in _resolve_forms(Path(image_path), meta_bases):
            if form in meta_by_path:
                meta = meta_by_path[form]
                break
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
                "generator": meta.get("generator") or ("real" if y == 0 else "unknown"),
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
        n_real = sum(r["label"] == 0 for r in rs)
        n_fake = n - n_real
        stats[value] = {
            "n_images": n,
            "n_real": n_real,
            "n_fake": n_fake,
            "n_fp": n_fp,
            "n_fn": n_fn,
            # Same denominators as src.eval.metrics: fpr over reals only,
            # fnr over fakes only (max(1, ..) keeps empty halves at 0).
            "fpr": round(n_fp / max(1, n_real), 6),
            "fnr": round(n_fn / max(1, n_fake), 6),
        }
    return stats


def summarize(joined: list[dict], threshold: float, worst_k: int = 20) -> dict:
    """Totals, per-group error rates, and the worst-K FP/FN for quick triage."""
    fps = [r for r in joined if r["error_type"] == "FP"]
    fns = [r for r in joined if r["error_type"] == "FN"]
    n = len(joined)
    n_fp, n_fn = len(fps), len(fns)
    n_real = sum(r["label"] == 0 for r in joined)
    n_fake = n - n_real
    return {
        "threshold": threshold,
        "n_images": n,
        "n_real": n_real,
        "n_fake": n_fake,
        "n_fp": n_fp,
        "n_fn": n_fn,
        "fpr": round(n_fp / max(1, n_real), 6),
        "fnr": round(n_fn / max(1, n_fake), 6),
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
    """Flat long-format table: one row per (group_type, group_value).

    ``fpr`` / ``fnr`` use the same denominators as ``src.eval.metrics``
    (FP over reals, FN over fakes) so the CSV lines up with robustness tables.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["group_type", "group_value", "n_images", "n_real", "n_fake",
             "n_fp", "n_fn", "fpr", "fnr"]
        )
        w.writerow(
            ["overall", "all", summary["n_images"], summary["n_real"],
             summary["n_fake"], summary["n_fp"], summary["n_fn"],
             summary["fpr"], summary["fnr"]]
        )
        for group_type in ("by_condition", "by_generator", "by_source_dataset"):
            for value, st in summary[group_type].items():
                w.writerow(
                    [group_type.removeprefix("by_"), value, st["n_images"],
                     st["n_real"], st["n_fake"], st["n_fp"], st["n_fn"],
                     st["fpr"], st["fnr"]]
                )
# end
