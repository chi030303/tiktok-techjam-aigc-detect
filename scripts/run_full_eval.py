#!/usr/bin/env python3
# 2026-08-30, tianqi, full official_val / EvalGEN eval without materializing 15x copies
"""Stream full-val and EvalGEN scoring. Transforms run in the Dataset, not on disk.

Official val (default --conditions clean):

    python scripts/run_full_eval.py --split official_val --conditions clean
    python scripts/run_full_eval.py --split official_val --conditions full

EvalGEN is fakes-only; pair with SID val / COCO / WildFake reals:

    python scripts/run_full_eval.py --split evalgen --reals sid_val --conditions clean
    python scripts/run_full_eval.py --split evalgen --reals coco --max-fakes-per-gen 200

Default ckpts are CLIP-B/L SID aug under EXP_ROOT. GPU1: CUDA_VISIBLE_DEVICES=1.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import random

from src.eval.evalgen_pool import (
    export_sid_val_reals,
    generator_from_path,
    load_reals,
    pair_evalgen,
    sid_val_reals_ready,
    subsample_fakes_per_generator,
)
from src.eval.formula import official_formula
from src.eval.labels import load_labeled_dir, load_split, subsample_balanced
from src.eval.metrics import binary_metrics
from src.eval.score import score_paired
from src.eval.table import write_csv, write_json, write_markdown
from src.eval.transforms import resolve_conditions
from src.paths import artifact_dir, exp_root

# end

DEFAULT_CKPTS = (
    ("clipb16_sid", "clipb16_linear_sid_aug"),
    ("clipl14_sid", "clipl14_linear_sid_aug"),
)


def _parse_named_ckpt(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"--ckpt needs name=/path, got {spec!r}")
    name, raw = spec.split("=", 1)
    path = Path(raw)
    if not path.is_file():
        raise SystemExit(f"ckpt not found: {path}")
    return name.strip(), path


def _default_ckpts() -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    missing: list[str] = []
    for name, exp in DEFAULT_CKPTS:
        path = exp_root() / exp / "ckpts" / "best.pt"
        if path.is_file():
            out.append((name, path))
        else:
            missing.append(str(path))
    if not out:
        raise SystemExit(
            "no default CLIP-SID ckpts; pass --ckpt name=/path. missing:\n  " + "\n  ".join(missing)
        )
    if missing:
        print("skip missing default ckpt:\n  " + "\n  ".join(missing), flush=True)
    return out


def _count(rows: list[tuple[Path, int]]) -> tuple[int, int, int]:
    n_real = sum(1 for _p, y in rows if int(y) == 0)
    n_fake = sum(1 for _p, y in rows if int(y) == 1)
    return len(rows), n_real, n_fake


def _per_generator(preds: list[dict], fake_root: Path | None, threshold: float) -> list[dict]:
    reals = [r for r in preds if int(r["y"]) == 0]
    by: dict[str, list[dict]] = defaultdict(list)
    for row in preds:
        if int(row["y"]) != 1:
            continue
        g = generator_from_path(Path(row["image_path"]), fake_root)
        by[g].append(row)
    table: list[dict] = []
    for gen, fakes in sorted(by.items()):
        mixed = reals + fakes
        y_true = [int(r["y"]) for r in mixed]
        scores = [float(r["pred"]) for r in mixed]
        m = binary_metrics(y_true, scores, threshold=threshold)
        m["generator"] = gen
        m["mean_pred_fake"] = sum(float(r["pred"]) for r in fakes) / max(1, len(fakes))
        table.append(m)
    return table


def _load_official(args) -> tuple[str, Path, list]:
    if args.image_dir is not None:
        root = args.image_dir
        return args.split or root.name, root, load_labeled_dir(root)
    root, rows = load_split(args.split)
    return args.split, root, rows


def _load_evalgen(args) -> tuple[str, Path, Path | None, list, str]:
    if args.image_dir is not None:
        fake_root = args.image_dir
        fake_rows = load_labeled_dir(fake_root, default_fake=True)
    else:
        fake_root, fake_rows = load_split("evalgen")
    fake_rows = [(p, 1) for p, y in fake_rows if int(y) == 1]
    if args.max_fakes_per_gen is not None:
        fake_rows = subsample_fakes_per_generator(fake_rows, fake_root, args.max_fakes_per_gen, args.seed)

    if args.reals in {"sid_val", "sid"}:
        if sid_val_reals_ready():
            real_name, real_rows = load_reals("sid_val", args.reals_dir)
            return "evalgen_sidreals", fake_root, fake_root, pair_evalgen(fake_rows, real_rows), real_name
        return "evalgen_sidreals", fake_root, None, fake_rows, "sid_val_onthefly"

    real_name, real_rows = load_reals(args.reals, args.reals_dir)
    if args.max_reals is not None and len(real_rows) > args.max_reals:
        real_rows = random.Random(args.seed).sample(real_rows, args.max_reals)
    rows = pair_evalgen(fake_rows, real_rows)
    return f"evalgen_{real_name}", fake_root, fake_root, rows, real_name


def _predict_rows(
    predictor,
    rows: list,
    src_root: Path,
    condition: str,
    workers: int,
    sid_reals_otf: bool,
    max_reals: int | None,
    seed: int,
    fuse=None,
) -> list[dict]:
    from src.eval.stream_predict import (
        DualSidValRealConditionDataset,
        SidValRealConditionDataset,
        predict_dataset,
        predict_fused_dataset,
        predict_fused_labeled,
        predict_labeled,
    )

    if fuse is not None:
        pred_a, pred_b, weight = fuse
        if sid_reals_otf:
            fakes = [(p, y) for p, y in rows if int(y) == 1]
            preds = predict_fused_labeled(
                pred_a, pred_b, fakes, src_root, condition, workers=workers, weight=weight
            )
            sid_ds = SidValRealConditionDataset(
                condition,
                input_mode=pred_a.input_mode,
                max_images=max_reals,
                seed=seed,
            )
            preds.extend(predict_fused_dataset(pred_a, pred_b, sid_ds, workers=0, weight=weight))
            return preds
        return predict_fused_labeled(
            pred_a, pred_b, rows, src_root, condition, workers=workers, weight=weight
        )

    if sid_reals_otf:
        fakes = [(p, y) for p, y in rows if int(y) == 1]
        preds = predict_labeled(predictor, fakes, src_root, condition, workers=workers)
        # 2026-08-31, tianqi, dual-branch EvalGEN reals must return rgb+highpass
        if getattr(predictor, "dual", False):
            sid_ds = DualSidValRealConditionDataset(
                condition,
                input_mode=predictor.input_mode,
                max_images=max_reals,
                seed=seed,
                image_size=int(getattr(predictor, "image_size", 224) or 224),
            )
        else:
            sid_ds = SidValRealConditionDataset(
                condition,
                input_mode=predictor.input_mode,
                max_images=max_reals,
                seed=seed,
            )
        # end
        preds.extend(predict_dataset(predictor, sid_ds, workers=0))
        return preds
    return predict_labeled(predictor, rows, src_root, condition, workers=workers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-val / EvalGEN stream eval (no 15x disk copies)")
    parser.add_argument("--split", default="official_val", help="official_val | evalgen")
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--conditions", default="clean", help="clean | daily | full | comma keys")
    parser.add_argument("--ckpt", action="append", default=None, help="name=/path/to/best.pt (repeatable)")
    parser.add_argument("--reals", default="sid_val", help="evalgen only: sid_val | coco | wildfake | dir")
    parser.add_argument("--reals-dir", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=None, help="balanced subsample (official_val)")
    parser.add_argument("--max-fakes-per-gen", type=int, default=None)
    parser.add_argument("--max-reals", type=int, default=None)
    parser.add_argument("--max-errors", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--experiment", default="full_eval")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--stem", default=None)
    parser.add_argument("--save-preds", action="store_true")
    parser.add_argument("--fuse", action="store_true", help="average logits of exactly two --ckpt")
    parser.add_argument("--fuse-weight", type=float, default=0.5, help="weight on the first --ckpt")
    parser.add_argument("--export-sid-reals", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.export_sid_reals:
        export_sid_val_reals(max_images=args.max_reals, seed=args.seed)

    split = args.split
    sid_otf = False
    fake_root: Path | None = None
    if split in {"evalgen"}:
        name, src_root, fake_root, rows, real_name = _load_evalgen(args)
        sid_otf = real_name == "sid_val_onthefly"
        if sid_otf:
            n_fake = sum(1 for _p, y in rows if int(y) == 1)
            n_real = args.max_reals if args.max_reals else "sid_val(all)"
            print(
                f"evalgen fakes={n_fake} reals=sid_val on-the-fly ({n_real})  root={src_root}",
                flush=True,
            )
        else:
            n, n_real, n_fake = _count(rows)
            print(f"{name} n={n} real={n_real} fake={n_fake} reals={real_name} root={src_root}", flush=True)
    else:
        name, src_root, rows = _load_official(args)
        if args.max_images is not None:
            rows = subsample_balanced(rows, args.max_images, args.seed)
        n, n_real, n_fake = _count(rows)
        print(f"{name} n={n} real={n_real} fake={n_fake} root={src_root}", flush=True)

    conditions = resolve_conditions(args.conditions)
    print(f"conditions={conditions}", flush=True)
    if args.dry_run:
        return

    named = [_parse_named_ckpt(s) for s in args.ckpt] if args.ckpt else _default_ckpts()
    print(f"models={[n for n, _ in named]}", flush=True)

    from src.infer import ProbePredictor

    work = artifact_dir(args.experiment) / name
    work.mkdir(parents=True, exist_ok=True)
    out_dir = args.out_dir or (REPO / "outputs" / "tables")
    stem = args.stem or f"{name}_{args.experiment}"

    table: list[dict] = []
    gen_table: list[dict] = []
    formula_by: dict[str, dict] = {}
    errors_by: dict[str, dict] = {}

    # 2026-08-31, tianqi, --fuse: one combined model from two ckpts (logit mean)
    fuse_pack = None
    if args.fuse:
        if len(named) != 2:
            raise SystemExit("--fuse needs exactly two --ckpt name=/path")
        (n0, p0), (n1, p1) = named
        pa = ProbePredictor(p0, batch=args.batch)
        pb = ProbePredictor(p1, batch=args.batch)
        pb.model.to(pa.device)
        fuse_name = f"fuse_{n0}_{n1}"
        named = [(fuse_name, p0)]
        fuse_pack = (pa, pb, float(args.fuse_weight))
        print(f"fuse {n0}*{args.fuse_weight} + {n1}*{1.0 - args.fuse_weight}", flush=True)
    # end

    for exp_name, ckpt in named:
        predictor = fuse_pack[0] if fuse_pack is not None else ProbePredictor(ckpt, batch=args.batch)
        model_rows: list[dict] = []
        for cond in conditions:
            print(f"  {exp_name} {cond} ...", flush=True)
            preds = _predict_rows(
                predictor,
                rows,
                src_root,
                cond,
                workers=args.workers,
                sid_reals_otf=sid_otf,
                max_reals=args.max_reals,
                seed=args.seed,
                fuse=fuse_pack,
            )
            metrics, errors = score_paired(preds, threshold=args.threshold, max_errors=args.max_errors)
            metrics["split"] = name
            metrics["condition"] = cond
            metrics["model"] = exp_name
            table.append(metrics)
            model_rows.append(metrics)
            errors_by[f"{exp_name}/{cond}"] = errors
            if split == "evalgen":
                for row in _per_generator(preds, fake_root or src_root, args.threshold):
                    row["model"] = exp_name
                    row["split"] = name
                    row["condition"] = cond
                    gen_table.append(row)
            if args.save_preds:
                pred_path = work / exp_name / f"pred_{cond}.json"
                pred_path.parent.mkdir(parents=True, exist_ok=True)
                pred_path.write_text(json.dumps(preds), encoding="utf-8")
            roc = metrics["auroc"]
            roc_s = f"{roc:.3f}" if isinstance(roc, float) else "na"
            print(
                f"  {exp_name} {cond}  acc={metrics['acc']:.3f}  auroc={roc_s}  "
                f"fpr={metrics['fpr']:.3f}  fnr={metrics['fn'] / max(1, metrics['n_fake']):.3f}  "
                f"n={metrics['n']}",
                flush=True,
            )
        formula_by[exp_name] = official_formula(model_rows)
        formula_by[exp_name]["model"] = exp_name
        print(f"  {exp_name} formula={formula_by[exp_name]}", flush=True)
        del predictor
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_json(table, out_dir / f"{stem}.json")
    write_csv(table, out_dir / f"{stem}.csv")
    write_markdown(table, out_dir / f"{stem}.md")
    (out_dir / f"{stem}_formula.json").write_text(json.dumps(formula_by, indent=2), encoding="utf-8")
    (work / "errors.json").write_text(json.dumps(errors_by, indent=2), encoding="utf-8")
    if gen_table:
        write_json(gen_table, out_dir / f"{stem}_by_generator.json")
        print(f"wrote {out_dir / (stem + '_by_generator.json')}", flush=True)
    print(f"wrote {out_dir / (stem + '.csv')}", flush=True)
    print(f"wrote {out_dir / (stem + '_formula.json')}", flush=True)


if __name__ == "__main__":
    main()
# end
