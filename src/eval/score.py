# 2026-08-29, tianqi, join predict.py JSON to folder labels; dump ranked FP/FN
from __future__ import annotations

from pathlib import Path

from src.eval.labels import index_by_rel
from src.eval.metrics import binary_metrics

# end


def _lookup_y(
    image_path: str,
    predict_root: Path,
    by_rel: dict[str, int],
) -> int | None:
    p = Path(image_path)
    candidates: list[str] = []
    try:
        rel = p.resolve().relative_to(predict_root.resolve())
        candidates.append(rel.as_posix())
        candidates.append(rel.with_suffix("").as_posix())
    except ValueError:
        pass
    try:
        rel2 = p.relative_to(predict_root)
        candidates.append(rel2.as_posix())
        candidates.append(rel2.with_suffix("").as_posix())
    except ValueError:
        pass
    for c in candidates:
        if c in by_rel:
            return by_rel[c]
    return None


def score_predictions(
    preds: list[dict],
    rows: list[tuple[Path, int]],
    src_root: Path,
    predict_root: Path,
    threshold: float = 0.5,
    max_errors: int = 50,
) -> tuple[dict, dict]:
    by_rel = index_by_rel(rows, src_root)
    y_true: list[int] = []
    scores: list[float] = []
    paired: list[tuple[str, float, int]] = []
    unmatched = 0
    for row in preds:
        if "image_path" not in row or "pred" not in row:
            raise SystemExit("pred JSON must be a list of {image_path, pred}")
        y = _lookup_y(str(row["image_path"]), predict_root, by_rel)
        if y is None:
            unmatched += 1
            continue
        s = float(row["pred"])
        y_true.append(y)
        scores.append(s)
        paired.append((str(row["image_path"]), s, y))
    if not y_true:
        raise SystemExit("no predictions matched labels")
    metrics = binary_metrics(y_true, scores, threshold=threshold)
    metrics["unmatched"] = unmatched
    fps = sorted(
        [{"image_path": p, "pred": s, "y": y} for p, s, y in paired if y == 0 and s >= threshold],
        key=lambda r: -r["pred"],
    )[:max_errors]
    fns = sorted(
        [{"image_path": p, "pred": s, "y": y} for p, s, y in paired if y == 1 and s < threshold],
        key=lambda r: r["pred"],
    )[:max_errors]
    errors = {
        "threshold": threshold,
        "false_positives": fps,
        "false_negatives": fns,
    }
    return metrics, errors
# end
