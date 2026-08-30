#!/usr/bin/env python3
# 2026-08-30, yun, model-track robustness compare: same recipe as scripts/run_eval.py compare,
# but scores checkpoints with src.sid_infer.SidProbePredictor instead of src.infer.ProbePredictor
"""Score several model-track checkpoints across the 15 official_val conditions.

    python scripts/run_ablation_compare.py --split official_val --conditions full --max-images 500 \
        --ckpt baseline=/workspace/experiments/clipb16_linear_sid_aug/ckpts/best.pt \
        --ckpt exp4_res336=/workspace/experiments/clipb16_linear_sid_res336/ckpts/best.pt \
        --experiment ablation_compare

Kept separate from scripts/run_eval.py's cmd_compare (which is wired to
src.infer.ProbePredictor) so this branch can add checkpoint kinds without
touching that shared script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval.labels import load_labeled_dir, load_split, subsample_balanced
from src.eval.robustness import write_condition
from src.eval.score import score_predictions
from src.eval.table import write_csv, write_json, write_markdown
from src.eval.transforms import resolve_conditions
from src.paths import artifact_dir
from src.sid_infer import SidProbePredictor

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


def _maybe_subsample(rows, max_images: int | None, seed: int):
    if max_images is None:
        return rows
    return subsample_balanced(rows, max_images, seed)


def _parse_named_ckpt(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"--ckpt needs name=/path, got {spec!r}")
    name, raw = spec.split("=", 1)
    path = Path(raw)
    if not path.is_file():
        raise SystemExit(f"ckpt not found: {path}")
    return name.strip(), path


def _write_pivot(table: list[dict], path: Path) -> None:
    models, conds = [], []
    seen_m, seen_c = set(), set()
    by = {}
    for row in table:
        m, c = row.get("model") or "", row.get("condition") or ""
        if m not in seen_m:
            models.append(m)
            seen_m.add(m)
        if c not in seen_c:
            conds.append(c)
            seen_c.add(c)
        by[(m, c)] = row
    cols = ["model"] + [f"{c}_acc" for c in conds] + [f"{c}_auroc" for c in conds]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for m in models:
        cells = [m]
        for c in conds:
            acc = by.get((m, c), {}).get("acc")
            cells.append(f"{acc:.4f}" if isinstance(acc, float) else "")
        for c in conds:
            roc = by.get((m, c), {}).get("auroc")
            cells.append(f"{roc:.4f}" if isinstance(roc, float) else "")
        lines.append("| " + " | ".join(cells) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=None)
    ap.add_argument("--image-dir", type=Path, default=None)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--conditions", default="daily", help="daily | full | comma names")
    ap.add_argument("--ckpt", action="append", required=True, help="name=/path/to/best.pt")
    ap.add_argument("--experiment", default="ablation_compare")
    ap.add_argument("--work-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--stem", default=None)
    args = ap.parse_args()

    name, root, rows = _load_rows(args)
    rows = _maybe_subsample(rows, args.max_images, args.seed)
    conditions = resolve_conditions(args.conditions)
    named = [_parse_named_ckpt(s) for s in args.ckpt]
    work = args.work_dir or artifact_dir(args.experiment) / name
    work.mkdir(parents=True, exist_ok=True)
    print(f"compare n={len(rows)} conditions={conditions} models={[n for n, _ in named]}", flush=True)
    for cond in conditions:
        dest = work / "images" / cond
        print(f"materialize {cond} -> {dest}", flush=True)
        write_condition(rows, root, dest, cond)

    table: list[dict] = []
    for exp_name, ckpt in named:
        predictor = SidProbePredictor(ckpt)
        for cond in conditions:
            image_dir = work / "images" / cond
            pred_json = work / exp_name / f"pred_{cond}.json"
            preds = predictor.predict_dir(image_dir)
            pred_json.parent.mkdir(parents=True, exist_ok=True)
            pred_json.write_text(json.dumps(preds, indent=2), encoding="utf-8")
            metrics, _errors = score_predictions(
                preds, rows, src_root=root, predict_root=image_dir, threshold=args.threshold
            )
            metrics["split"] = name
            metrics["condition"] = cond
            metrics["model"] = exp_name
            table.append(metrics)
            roc = metrics["auroc"]
            roc_s = f"{roc:.3f}" if isinstance(roc, float) else "na"
            print(f"  {exp_name} {cond}  acc={metrics['acc']:.3f}  auroc={roc_s}  n={metrics['n']}", flush=True)
        del predictor
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_dir = args.out_dir or (REPO / "outputs" / "tables")
    stem = args.stem or f"{name}_{args.experiment}"
    write_json(table, out_dir / f"{stem}.json")
    write_csv(table, out_dir / f"{stem}.csv")
    write_markdown(table, out_dir / f"{stem}.md")
    _write_pivot(table, out_dir / f"{stem}_pivot.md")
    print(f"wrote {out_dir / (stem + '.csv')}", flush=True)
    print(f"wrote {out_dir / (stem + '_pivot.md')}", flush=True)


if __name__ == "__main__":
    main()
