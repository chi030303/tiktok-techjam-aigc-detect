# 2026-08-29, tianqi, binary metrics for AIGC=1; AUROC skips if one class missing
from __future__ import annotations

from collections.abc import Sequence

# end


def auroc(y_true: Sequence[int], scores: Sequence[float]) -> float | None:
    pos = sorted(float(s) for y, s in zip(y_true, scores) if int(y) == 1)
    neg = sorted(float(s) for y, s in zip(y_true, scores) if int(y) == 0)
    if not pos or not neg:
        return None
    i = 0
    pairs = 0.0
    n_neg = len(neg)
    for p in pos:
        while i < n_neg and neg[i] < p:
            i += 1
        j = i
        while j < n_neg and neg[j] == p:
            j += 1
        pairs += i + 0.5 * (j - i)
    return pairs / (len(pos) * n_neg)


def binary_metrics(
    y_true: Sequence[int],
    scores: Sequence[float],
    threshold: float = 0.5,
) -> dict:
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores length mismatch")
    n = len(y_true)
    tp = fp = tn = fn = 0
    for y, s in zip(y_true, scores):
        pred = 1 if float(s) >= threshold else 0
        y = int(y)
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1 and y == 0:
            fp += 1
        elif pred == 0 and y == 0:
            tn += 1
        else:
            fn += 1
    n_real = tn + fp
    n_fake = tp + fn
    acc = (tp + tn) / max(1, n)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    fpr = fp / max(1, n_real)
    roc = auroc(y_true, scores)
    return {
        "n": n,
        "n_real": n_real,
        "n_fake": n_fake,
        "threshold": threshold,
        "acc": acc,
        "auroc": roc,
        "precision_fake": prec,
        "recall_fake": rec,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "mean_pred": sum(float(s) for s in scores) / max(1, n),
    }
# end
