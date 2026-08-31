"""Quantitative analysis helpers for exported bad-case HTML galleries.

The gallery contains only errors, so dataset class counts must be supplied by
the caller.  This module deliberately does not infer visual failure causes:
those require human review of the thumbnails.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any


_CARDS_RE = re.compile(r"const CARDS = (\{.*?\});\s*let kind", re.DOTALL)
_META_RE = re.compile(
    r"threshold=(?P<threshold>[0-9.]+)\s*·\s*"
    r"FP=(?P<fp>\d+)\s*·\s*FN=(?P<fn>\d+)\s*·\s*"
    r"joined=(?P<joined>\d+)"
)


@dataclass(frozen=True)
class Gallery:
    name: str
    path: Path
    threshold: float
    joined: int
    cards: dict[str, list[dict[str, Any]]]


def load_gallery(name: str, path: Path) -> Gallery:
    """Load and validate the JavaScript card payload from a gallery."""
    text = path.read_text(encoding="utf-8")
    cards_match = _CARDS_RE.search(text)
    meta_match = _META_RE.search(html.unescape(text))
    if cards_match is None or meta_match is None:
        raise ValueError(f"{path}: unsupported gallery format")

    cards = json.loads(cards_match.group(1))
    if set(cards) != {"FP", "FN"}:
        raise ValueError(f"{path}: CARDS must contain exactly FP and FN")
    if not all(isinstance(cards[k], list) for k in ("FP", "FN")):
        raise ValueError(f"{path}: FP and FN must be lists")

    expected = {"FP": int(meta_match["fp"]), "FN": int(meta_match["fn"])}
    for error_type, rows in cards.items():
        if len(rows) != expected[error_type]:
            raise ValueError(
                f"{path}: metadata says {expected[error_type]} {error_type}, "
                f"but CARDS contains {len(rows)}"
            )
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"{path}: every card must be an object")
            if row.get("etype") != error_type:
                raise ValueError(f"{path}: card error type does not match its section")
            if "path" not in row or "pred" not in row:
                raise ValueError(f"{path}: card is missing path or pred")

    return Gallery(
        name=name,
        path=path,
        threshold=float(meta_match["threshold"]),
        joined=int(meta_match["joined"]),
        cards=cards,
    )


def image_key(path: str) -> str:
    """Return a root-independent key while preserving subdirectories."""
    parts = Path(path).parts
    for marker in ("real", "fake", "REAL", "FAKE"):
        if marker in parts:
            index = parts.index(marker)
            return "/".join(parts[index:])
    return Path(path).as_posix()


def percentile(values: list[float], quantile: float) -> float | None:
    """Compute a linearly interpolated percentile without extra dependencies."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_gallery(
    gallery: Gallery, n_real: int, n_fake: int, auroc: float | None = None
) -> dict[str, Any]:
    fp = len(gallery.cards["FP"])
    fn = len(gallery.cards["FN"])
    total = n_real + n_fake
    if gallery.joined != total:
        raise ValueError(
            f"{gallery.path}: joined={gallery.joined}, but n_real+n_fake={total}"
        )
    if fp > n_real or fn > n_fake:
        raise ValueError(f"{gallery.path}: error count exceeds class denominator")

    score_distribution = {}
    for error_type in ("FP", "FN"):
        scores = [float(row["pred"]) for row in gallery.cards[error_type]]
        score_distribution[error_type] = {
            key: percentile(scores, q)
            for key, q in (("min", 0), ("q25", 0.25), ("median", 0.5), ("q75", 0.75), ("max", 1))
        }

    return {
        "name": gallery.name,
        "threshold": gallery.threshold,
        "n": total,
        "n_real": n_real,
        "n_fake": n_fake,
        "fp": fp,
        "fn": fn,
        "fpr": fp / n_real,
        "fnr": fn / n_fake,
        "accuracy": 1 - (fp + fn) / total,
        "auroc": auroc,
        "score_distribution": score_distribution,
    }


def pairwise_overlap(left: Gallery, right: Gallery) -> list[dict[str, Any]]:
    """Compare exact error membership for two galleries."""
    result = []
    for error_type in ("FP", "FN"):
        left_keys = {image_key(row["path"]) for row in left.cards[error_type]}
        right_keys = {image_key(row["path"]) for row in right.cards[error_type]}
        intersection = left_keys & right_keys
        union = left_keys | right_keys
        result.append(
            {
                "left": left.name,
                "right": right.name,
                "error_type": error_type,
                "intersection": len(intersection),
                "union": len(union),
                "jaccard": len(intersection) / len(union) if union else 1.0,
                "share_of_left": (
                    len(intersection) / len(left_keys) if left_keys else 0.0
                ),
                "share_of_right": (
                    len(intersection) / len(right_keys) if right_keys else 0.0
                ),
            }
        )
    return result


def build_report(
    galleries: list[Gallery],
    n_real: int,
    n_fake: int,
    aurocs: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable report for two or more galleries."""
    if len(galleries) < 2:
        raise ValueError("at least two galleries are required for an ablation comparison")
    thresholds = {gallery.threshold for gallery in galleries}
    if len(thresholds) != 1:
        raise ValueError("all galleries must use the same threshold")
    aurocs = aurocs or {}
    unknown = set(aurocs) - {gallery.name for gallery in galleries}
    if unknown:
        raise ValueError(f"AUROC supplied for unknown model(s): {sorted(unknown)}")

    return {
        "scope": {
            "condition": "clean",
            "threshold": galleries[0].threshold,
            "n_real": n_real,
            "n_fake": n_fake,
            "n": n_real + n_fake,
        },
        "models": [
            summarize_gallery(gallery, n_real, n_fake, aurocs.get(gallery.name))
            for gallery in galleries
        ],
        "overlaps": [
            row
            for left, right in combinations(galleries, 2)
            for row in pairwise_overlap(left, right)
        ],
    }


def _fmt_rate(value: float) -> str:
    return f"{value * 100:.2f}%"


def _fmt_score(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def render_markdown(
    report: dict[str, Any],
    title: str,
    model_scope_note: str,
) -> str:
    """Render a provisional Error Analysis Note from a report."""
    scope = report["scope"]
    models = report["models"]
    lines = [
        f"# {title}",
        "",
        "> **Status:** Provisional backbone-ablation note. Replace with the "
        "last4/fuse analysis before final submission.",
        f"> **Model scope:** {model_scope_note}",
        f"> **Data scope:** official demonstration validation, clean only; "
        f"{scope['n_real']} real + {scope['n_fake']} fake; "
        f"threshold={scope['threshold']:.2f}.",
        "",
        "## 1. Quantitative summary",
        "",
        "| Model | AUROC | FP | FPR | FN | FNR | Acc@threshold |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        lines.append(
            f"| {model['name']} | {_fmt_score(model['auroc'])} | "
            f"{model['fp']} | {_fmt_rate(model['fpr'])} | "
            f"{model['fn']} | {_fmt_rate(model['fnr'])} | "
            f"{_fmt_rate(model['accuracy'])} |"
        )

    lines += [
        "",
        "AUROC measures ranking quality; FP/FN and accuracy depend on the chosen "
        "threshold. A model can therefore have higher AUROC but lower "
        "Acc@0.5 when its score calibration shifts.",
        "",
        "## 2. Error-score distribution",
        "",
        "| Model | Type | min | p25 | median | p75 | max |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        for error_type in ("FP", "FN"):
            dist = model["score_distribution"][error_type]
            lines.append(
                f"| {model['name']} | {error_type} | "
                + " | ".join(_fmt_score(dist[key]) for key in ("min", "q25", "median", "q75", "max"))
                + " |"
            )

    lines += [
        "",
        "## 3. Cross-model error overlap",
        "",
        "| Pair | Type | Shared errors | Union | Jaccard | Share of left | Share of right |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["overlaps"]:
        lines.append(
            f"| {row['left']} ↔ {row['right']} | {row['error_type']} | "
            f"{row['intersection']} | {row['union']} | "
            f"{row['jaccard']:.3f} | {_fmt_rate(row['share_of_left'])} | "
            f"{_fmt_rate(row['share_of_right'])} |"
        )

    lowest_fpr = min(models, key=lambda model: model["fpr"])
    lowest_fnr = min(models, key=lambda model: model["fnr"])
    lines += [
        "",
        "## 4. Evidence-backed findings",
        "",
        f"- **Lowest false-positive rate:** {lowest_fpr['name']} "
        f"({_fmt_rate(lowest_fpr['fpr'])}).",
        f"- **Lowest false-negative rate:** {lowest_fnr['name']} "
        f"({_fmt_rate(lowest_fnr['fnr'])}).",
        "- The overlap table separates model-specific mistakes from shared hard "
        "cases. Shared FN are the strongest candidates for generator-coverage "
        "analysis; model-specific errors are candidates for ensembling.",
        "- This clean-only gallery cannot support claims about JPEG, blur, resize, "
        "noise, jitter, or crop robustness.",
        "",
        "## 5. Manual visual review protocol",
        "",
        "Review at least the top 50 highest-confidence FP and top 50 "
        "lowest-confidence FN per model. Assign one or more tags:",
        "",
        "- `blur_or_low_detail`",
        "- `stylized_or_artwork`",
        "- `regular_geometry`",
        "- `cinematic_lighting`",
        "- `text_or_sign`",
        "- `unusual_composition`",
        "- `photorealistic_fake`",
        "- `unclear`",
        "",
        "Report tag counts separately for FP and FN. Until those counts exist, "
        "visual patterns must be described as observations, not conclusions.",
        "",
        "## 6. Recommended actions",
        "",
        "1. Calibrate the decision threshold on an allowed validation split; do "
        "not retrain solely to fix Acc@0.5 when AUROC is already strong.",
        "2. Inspect shared high-confidence FN first. They are more likely to "
        "represent a stable blind spot than a backbone-specific error.",
        "3. Test whether model-specific errors are reduced by score/logit fusion.",
        "4. Regenerate this report for the final last4 and fuse models, then add "
        "15-condition robustness and EvalGEN generator slices.",
        "",
        "## 7. Limitations",
        "",
        "- The official demonstration set is evaluation-only and must not be used "
        "for hard-negative mining or training.",
        "- The galleries contain only errors, not all prediction scores; calibration "
        "curves and optimal thresholds require the original prediction JSON.",
        "- No visual category frequencies are claimed before manual tagging.",
        "",
    ]
    return "\n".join(lines)
