#!/usr/bin/env python3
# 2026-08-30, yun, exp3 robustness eval: dual_branch_clip ckpt needs two views, ProbePredictor can't do it
"""Score a dual_branch_clip checkpoint across the 15 official_val conditions.

Mirrors run_eval.py compare's methodology (same split, same balanced subsample,
same official transform grid, same binary_metrics) but builds (rgb, freq) view
pairs in-process instead of going through predict.py's single-view JSON path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.freq import highpass_residual
from src.eval.labels import load_split, subsample_balanced
from src.eval.metrics import binary_metrics
from src.eval.transforms import CONDITIONS, apply_condition, seed_for
from src.models.dual_branch_probe import DualBranchProbe
from src.paths import models_root
from src.train.sid_online import _tfm

# end


def _freq_tfm(size: int):
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", default="official_val")
    ap.add_argument("--max-images", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "tables" / "dualbranch_robustness.json")
    args = ap.parse_args()

    root, rows = load_split(args.split)
    rows = subsample_balanced(rows, args.max_images, args.seed)
    print(f"n_rows={len(rows)}", flush=True)

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if blob.get("kind") != "dual_branch_clip":
        raise SystemExit(f"{args.ckpt} is not a dual_branch_clip ckpt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = int(blob.get("image_size") or 224)
    model = DualBranchProbe(
        models_root() / str(blob["backbone"]), freq_dim=int(blob["freq_dim"]), head_kind=blob.get("head_kind") or "linear"
    ).to(device)
    model.freq_cnn.load_state_dict(blob["freq_cnn_state_dict"])
    model.head.load_state_dict(blob["head"])
    model.eval()
    print(f"loaded {args.ckpt}  image_size={image_size}  device={device}", flush=True)

    rgb_tfm = _tfm(image_size)
    freq_tfm = _freq_tfm(image_size)

    results = []
    for cond in CONDITIONS:
        y_true: list[int] = []
        scores: list[float] = []
        rgb_batch, freq_batch, y_batch = [], [], []

        def flush():
            nonlocal rgb_batch, freq_batch, y_batch
            if not rgb_batch:
                return
            xa = torch.stack(rgb_batch).to(device)
            xb = torch.stack(freq_batch).to(device)
            with torch.no_grad():
                logit = model(xa, xb)
            probs = logit.sigmoid().cpu().tolist()
            scores.extend(probs)
            y_true.extend(y_batch)
            rgb_batch, freq_batch, y_batch = [], [], []

        for path, y in rows:
            rel = path.relative_to(root).as_posix()
            seed = seed_for(rel, cond)
            img = apply_condition(Image.open(path), cond, seed=seed)
            rgb_batch.append(rgb_tfm(img))
            freq_batch.append(freq_tfm(highpass_residual(img)))
            y_batch.append(y)
            if len(rgb_batch) >= args.batch:
                flush()
        flush()

        m = binary_metrics(y_true, scores)
        m["condition"] = cond
        results.append(m)
        roc = m["auroc"]
        roc_s = f"{roc:.4f}" if isinstance(roc, float) else "na"
        print(f"  {cond}  acc={m['acc']:.4f}  auroc={roc_s}  n={m['n']}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
