# 2026-08-29, tianqi, compact clean-vs-transform table for the robustness deliverable
from __future__ import annotations

import csv
import json
from pathlib import Path

# end

COLUMNS = [
    "model",
    "split",
    "condition",
    "n",
    "n_real",
    "n_fake",
    "acc",
    "auroc",
    "precision_fake",
    "recall_fake",
    "fpr",
    "mean_pred",
    "threshold",
]


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_json(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: _fmt(row.get(k)) for k in COLUMNS})


def write_markdown(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    shown = ["model", "condition", "n", "acc", "auroc", "precision_fake", "recall_fake", "fpr"]
    lines = [
        "| " + " | ".join(shown) + " |",
        "| " + " | ".join("---" for _ in shown) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(k)) for k in shown) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
# end
