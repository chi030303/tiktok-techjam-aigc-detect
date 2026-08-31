#!/usr/bin/env python3
# 2026-08-31, tianqi, paired i2i ranking: P(recon > paired real), not 180 independent Acc
"""Score existing ckpts on A2 i2i triplets.

The metric that matters is paired ranking, not Acc@0.5 on 180 independent
images: for each source_id, the reconstruction should score higher than the
matched real.

    python scripts/eval_i2i_triplets.py \
      --manifest data/manifests/ablation/A2_i2i_hard60.jsonl \
      --ckpt last4=/workspace/experiments/clipb16_linear_sid_unfreeze4/ckpts/best.pt \
      --ckpt D3_mix=/workspace/experiments/clipb16_linear_sid_d3_mix/ckpts/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.manifest_ds import load_source_manifest_rows, resolve_manifest_path
from src.eval.metrics import binary_metrics
from src.eval.stream_predict import predict_labeled
from src.infer import ProbePredictor
from src.transforms.manifest import read_jsonl

# end


def _parse_ckpt(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"--ckpt needs name=/path, got {spec!r}")
    name, raw = spec.split("=", 1)
    path = Path(raw)
    if not path.is_file():
        raise SystemExit(f"ckpt not found: {path}")
    return name.strip(), path


def paired_report(records, preds: list[dict], threshold: float = 0.5) -> dict:
    by_path = {str(Path(r["image_path"]).resolve()): float(r["pred"]) for r in preds}
    groups: dict[str, dict[str, float]] = defaultdict(dict)
    y_true: list[int] = []
    scores: list[float] = []
    for rec in records:
        p = str(Path(rec.path).resolve())
        if p not in by_path:
            continue
        s = by_path[p]
        role = "real" if rec.label == 0 else str(rec.generator)
        groups[str(rec.source_id)][role] = s
        y_true.append(int(rec.label))
        scores.append(s)

    pair_n = 0
    hits: dict[str, int] = defaultdict(int)
    n_by: dict[str, int] = defaultdict(int)
    deltas: dict[str, list[float]] = defaultdict(list)
    for sid, roles in groups.items():
        if "real" not in roles:
            continue
        real_s = roles["real"]
        for gen, fake_s in roles.items():
            if gen == "real":
                continue
            pair_n += 1
            n_by[gen] += 1
            deltas[gen].append(fake_s - real_s)
            if fake_s > real_s:
                hits[gen] += 1

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    pair_acc = {
        gen: (hits[gen] / n_by[gen] if n_by[gen] else float("nan")) for gen in n_by
    }
    overall_n = sum(n_by.values())
    overall_hits = sum(hits.values())
    metrics = binary_metrics(y_true, scores, threshold=threshold) if y_true else {}
    return {
        "n_scored": len(y_true),
        "n_groups": len(groups),
        "n_pairs": overall_n,
        "pair_acc_all": (overall_hits / overall_n) if overall_n else float("nan"),
        "pair_acc_by_gen": pair_acc,
        "mean_delta_fake_minus_real": {g: mean(deltas[g]) for g in deltas},
        "binary_at_0.5": {
            k: metrics[k]
            for k in ("acc", "auroc", "fpr", "recall_fake")
            if k in metrics
        },
        "note": "pair_acc = P(recon_score > paired_real). Acc@0.5 on 180 independent rows is secondary.",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--ckpt", action="append", default=[], help="name=/path (repeat)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if not args.ckpt:
        raise SystemExit("pass at least one --ckpt name=/path")

    manifest = resolve_manifest_path(args.manifest)
    records = read_jsonl(manifest, kind="source")
    rows = load_source_manifest_rows(manifest)
    src_root = Path(records[0].path).parent.parent if records else Path(".")

    out: dict = {"manifest": str(manifest), "models": {}}
    for spec in args.ckpt:
        name, ckpt = _parse_ckpt(spec)
        print(f"---- {name} {ckpt} ----", flush=True)
        pred = ProbePredictor(ckpt)
        scored = predict_labeled(pred, rows, src_root=src_root, condition="clean", workers=args.workers)
        report = paired_report(records, scored)
        out["models"][name] = report
        print(json.dumps(report, indent=2), flush=True)

    dest = args.out or (manifest.parent / "A2_i2i_hard60_paired.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
# end
