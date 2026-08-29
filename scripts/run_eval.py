#!/usr/bin/env python3
# 2026-08-29, tianqi, eval CLI: score JSON, robustness table, materialize transforms
"""Evaluate predict.py JSON and build the clean-vs-transform robustness table.

Examples:

    python scripts/run_eval.py score --pred outputs/pred.json --split official_val
    python scripts/run_eval.py robustness --split official_val --conditions daily --max-images 400
    python scripts/run_eval.py materialize --split official_val --conditions jpeg_q50,center_crop_80
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval.labels import load_labeled_dir, load_split, subsample_balanced
from src.eval.robustness import default_predict, robustness_table, write_condition
from src.eval.score import score_predictions
from src.eval.table import write_csv, write_json, write_markdown
from src.eval.transforms import resolve_conditions
from src.paths import artifact_dir, data_root

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


def cmd_score(args) -> None:
    name, root, rows = _load_rows(args)
    rows = _maybe_subsample(rows, args.max_images, args.seed)
    preds = json.loads(args.pred.read_text(encoding="utf-8"))
    metrics, errors = score_predictions(
        preds,
        rows,
        src_root=root,
        predict_root=args.predict_root or root,
        threshold=args.threshold,
    )
    metrics["split"] = name
    metrics["condition"] = args.condition
    print(json.dumps(metrics, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"metrics": metrics, "errors": errors}
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.out}", flush=True)


def cmd_robustness(args) -> None:
    name, root, rows = _load_rows(args)
    rows = _maybe_subsample(rows, args.max_images, args.seed)
    conditions = resolve_conditions(args.conditions)
    work = args.work_dir
    if work is None:
        work = artifact_dir(args.experiment or name) / "eval"
    ckpt = args.ckpt

    def predict_fn(image_dir: Path, out_json: Path) -> None:
        default_predict(image_dir, out_json, ckpt=ckpt)

    table, errors_by = robustness_table(
        rows,
        src_root=root,
        conditions=conditions,
        work_root=work,
        predict_fn=predict_fn,
        threshold=args.threshold,
        split_name=name,
    )
    for row in table:
        row["model"] = args.experiment or name
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = REPO / "outputs" / "tables"
    stem = args.stem or f"{name}_{args.experiment or 'eval'}"
    write_json(table, out_dir / f"{stem}.json")
    write_csv(table, out_dir / f"{stem}.csv")
    write_markdown(table, out_dir / f"{stem}.md")
    err_path = work / "errors.json"
    err_path.write_text(json.dumps(errors_by, indent=2), encoding="utf-8")
    print(f"wrote {out_dir / (stem + '.csv')}", flush=True)
    print(f"wrote {err_path}", flush=True)


def _parse_named_ckpt(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"--ckpt needs name=/path, got {spec!r}")
    name, raw = spec.split("=", 1)
    path = Path(raw)
    if not path.is_file():
        raise SystemExit(f"ckpt not found: {path}")
    return name.strip(), path


def _write_pivot(table: list[dict], path: Path) -> None:
    models = []
    conds = []
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


def cmd_compare(args) -> None:
    # 2026-08-29, tianqi, materialize transforms once, then score every ckpt in-process
    name, root, rows = _load_rows(args)
    rows = _maybe_subsample(rows, args.max_images, args.seed)
    conditions = resolve_conditions(args.conditions)
    named = [_parse_named_ckpt(s) for s in args.ckpt]
    work = args.work_dir or artifact_dir(args.experiment or "compare") / name
    work.mkdir(parents=True, exist_ok=True)
    print(f"compare n={len(rows)} conditions={conditions} models={[n for n,_ in named]}", flush=True)
    for cond in conditions:
        dest = work / "images" / cond
        print(f"materialize {cond} -> {dest}", flush=True)
        write_condition(rows, root, dest, cond)

    from src.infer import ProbePredictor

    table: list[dict] = []
    for exp_name, ckpt in named:
        predictor = ProbePredictor(ckpt)
        for cond in conditions:
            image_dir = work / "images" / cond
            pred_json = work / exp_name / f"pred_{cond}.json"
            preds = predictor.predict_dir(image_dir)
            pred_json.parent.mkdir(parents=True, exist_ok=True)
            pred_json.write_text(json.dumps(preds, indent=2), encoding="utf-8")
            metrics, _errors = score_predictions(
                preds,
                rows,
                src_root=root,
                predict_root=image_dir,
                threshold=args.threshold,
            )
            metrics["split"] = name
            metrics["condition"] = cond
            metrics["model"] = exp_name
            table.append(metrics)
            roc = metrics["auroc"]
            roc_s = f"{roc:.3f}" if isinstance(roc, float) else "na"
            print(
                f"  {exp_name} {cond}  acc={metrics['acc']:.3f}  auroc={roc_s}  n={metrics['n']}",
                flush=True,
            )
        del predictor
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out_dir = args.out_dir or (REPO / "outputs" / "tables")
    stem = args.stem or f"{name}_{args.experiment or 'compare'}"
    write_json(table, out_dir / f"{stem}.json")
    write_csv(table, out_dir / f"{stem}.csv")
    write_markdown(table, out_dir / f"{stem}.md")
    _write_pivot(table, out_dir / f"{stem}_pivot.md")
    print(f"wrote {out_dir / (stem + '.csv')}", flush=True)
    print(f"wrote {out_dir / (stem + '_pivot.md')}", flush=True)
    # end


def cmd_materialize(args) -> None:
    name, root, rows = _load_rows(args)
    rows = _maybe_subsample(rows, args.max_images, args.seed)
    conditions = [c for c in resolve_conditions(args.conditions) if c != "clean"]
    dest_root = args.out_dir or (data_root() / "transforms" / name)
    for cond in conditions:
        dest = dest_root / cond
        print(f"materialize {cond} -> {dest}  n={len(rows)}", flush=True)
        write_condition(rows, root, dest, cond)
    print(f"done {dest_root}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score predictions / robustness table")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_shared(p):
        p.add_argument("--split", default=None, help="official_val | evalgen | cifake_test")
        p.add_argument("--image-dir", type=Path, default=None)
        p.add_argument("--max-images", type=int, default=None)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--threshold", type=float, default=0.5)

    p_score = sub.add_parser("score", help="score a pred.json against labeled folders")
    add_shared(p_score)
    p_score.add_argument("--pred", type=Path, required=True)
    p_score.add_argument("--predict-root", type=Path, default=None)
    p_score.add_argument("--condition", default="clean")
    p_score.add_argument("--out", type=Path, default=None)
    p_score.set_defaults(func=cmd_score)

    p_rob = sub.add_parser("robustness", help="clean vs transforms via predict.py")
    add_shared(p_rob)
    p_rob.add_argument("--conditions", default="daily", help="daily | full | comma names")
    p_rob.add_argument("--ckpt", type=Path, default=None)
    p_rob.add_argument("--experiment", default=None, help="artifact folder name")
    p_rob.add_argument("--work-dir", type=Path, default=None)
    p_rob.add_argument("--out-dir", type=Path, default=None)
    p_rob.add_argument("--stem", default=None)
    p_rob.set_defaults(func=cmd_robustness)

    p_cmp = sub.add_parser("compare", help="same transforms, several ckpts")
    add_shared(p_cmp)
    p_cmp.add_argument("--conditions", default="daily")
    p_cmp.add_argument("--ckpt", action="append", required=True, help="name=/path/to/best.pt")
    p_cmp.add_argument("--experiment", default="compare")
    p_cmp.add_argument("--work-dir", type=Path, default=None)
    p_cmp.add_argument("--out-dir", type=Path, default=None)
    p_cmp.add_argument("--stem", default=None)
    p_cmp.set_defaults(func=cmd_compare)

    p_mat = sub.add_parser("materialize", help="write frozen transform copies under data/transforms")
    add_shared(p_mat)
    p_mat.add_argument("--conditions", default="daily")
    p_mat.add_argument("--out-dir", type=Path, default=None)
    p_mat.set_defaults(func=cmd_materialize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
# end
