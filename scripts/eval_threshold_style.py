# 2026-08-31, tianqi, threshold sweep + photoreal/stylized FN slice for the writeup
"""Sweep decision threshold (AUC unchanged) and tag official-val fakes as photo vs stylized."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
from PIL import Image

from src.eval.evalgen_pool import generator_from_path
from src.eval.metrics import auroc, binary_metrics
from src.eval.table import write_csv, write_json, write_markdown
from src.paths import models_root

# end

PHOTO_PROMPTS = [
    "a photograph of a real-world scene",
    "a realistic camera photo",
]
STYLE_PROMPTS = [
    "an anime illustration",
    "a digital painting or poster",
    "graphic design with text overlay",
]


def _load_preds(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"empty preds {path}")
    return rows


def sweep_thresholds(preds: list[dict], thresholds: list[float], fake_root: Path | None) -> list[dict]:
    y_true = [int(r["y"]) for r in preds]
    scores = [float(r["pred"]) for r in preds]
    roc = auroc(y_true, scores)
    out = []
    for t in thresholds:
        m = binary_metrics(y_true, scores, threshold=t)
        m["auroc_fixed"] = roc
        m["threshold"] = t
        out.append(m)
    by_gen: dict[str, list[dict]] = defaultdict(list)
    reals = [r for r in preds if int(r["y"]) == 0]
    if fake_root is not None:
        for row in preds:
            if int(row["y"]) != 1:
                continue
            by_gen[generator_from_path(Path(row["image_path"]), fake_root)].append(row)
        for gen, fakes in sorted(by_gen.items()):
            mixed = reals + fakes
            ys = [int(r["y"]) for r in mixed]
            ss = [float(r["pred"]) for r in mixed]
            roc_g = auroc(ys, ss)
            for t in (0.3, 0.4, 0.5, 0.6):
                m = binary_metrics(ys, ss, threshold=t)
                m["generator"] = gen
                m["auroc_fixed"] = roc_g
                m["threshold"] = t
                out.append(m)
    return out


def _clip_text_features(model, proc, prompts: list[str], device):
    batch = proc(text=prompts, padding=True, truncation=True, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
    with torch.no_grad():
        feat = model.get_text_features(**batch)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.mean(dim=0, keepdim=True)


def style_slice(preds: list[dict], out_dir: Path, stem: str, batch: int = 32) -> None:
    from transformers import CLIPModel, CLIPProcessor

    bb = models_root() / "clip-vit-base-patch16"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(bb, local_files_only=True).to(device)
    proc = CLIPProcessor.from_pretrained(bb, local_files_only=True)
    model.eval()
    photo_f = _clip_text_features(model, proc, PHOTO_PROMPTS, device)
    style_f = _clip_text_features(model, proc, STYLE_PROMPTS, device)

    fakes = [r for r in preds if int(r["y"]) == 1]
    tagged = []
    for i in range(0, len(fakes), batch):
        chunk = fakes[i : i + batch]
        imgs = []
        keep = []
        for r in chunk:
            p = Path(r["image_path"])
            if not p.is_file():
                continue
            with Image.open(p) as im:
                imgs.append(im.convert("RGB"))
            keep.append(r)
        if not imgs:
            continue
        pix = proc(images=imgs, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            vis = model.get_image_features(pixel_values=pix)
            vis = vis / vis.norm(dim=-1, keepdim=True)
            sp = (vis @ photo_f.T).squeeze(-1)
            ss = (vis @ style_f.T).squeeze(-1)
        for r, photo_s, style_s in zip(keep, sp.tolist(), ss.tolist()):
            stylized = float(style_s) > float(photo_s)
            tagged.append(
                {
                    **r,
                    "photo_sim": float(photo_s),
                    "style_sim": float(style_s),
                    "stylized": stylized,
                    "fn_at_05": int(r["y"]) == 1 and float(r["pred"]) < 0.5,
                }
            )

    rows = []
    for label, subset in (
        ("all_fake", tagged),
        ("photoreal", [t for t in tagged if not t["stylized"]]),
        ("stylized", [t for t in tagged if t["stylized"]]),
    ):
        n = len(subset)
        n_fn = sum(1 for t in subset if t["fn_at_05"])
        rows.append(
            {
                "slice": label,
                "n_fake": n,
                "n_fn_at_0.5": n_fn,
                "fnr": n_fn / max(1, n),
                "mean_pred": sum(float(t["pred"]) for t in subset) / max(1, n),
            }
        )
    write_json(rows, out_dir / f"{stem}_style_slice.json")
    write_csv(rows, out_dir / f"{stem}_style_slice.csv")
    write_markdown(rows, out_dir / f"{stem}_style_slice.md")
    print(f"wrote {out_dir / (stem + '_style_slice.csv')}", flush=True)
    for r in rows:
        print(
            f"  {r['slice']:12s} n={r['n_fake']:4d} FN@0.5={r['n_fn_at_0.5']:4d} "
            f"fnr={r['fnr']:.3f} mean_pred={r['mean_pred']:.3f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--fake-root", type=Path, default=None)
    parser.add_argument("--style-slice", action="store_true")
    args = parser.parse_args()
    preds = _load_preds(args.preds)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = [round(x * 0.05, 2) for x in range(1, 20)]
    swept = sweep_thresholds(preds, ts, args.fake_root)
    overall = [r for r in swept if "generator" not in r]
    write_json(swept, args.out_dir / f"{args.stem}_threshold.json")
    write_csv(overall, args.out_dir / f"{args.stem}_threshold.csv")
    write_markdown(overall, args.out_dir / f"{args.stem}_threshold.md")
    roc = overall[0]["auroc_fixed"] if overall else None
    print(f"auroc={roc} (unchanged across thresholds)  wrote {args.out_dir / (args.stem + '_threshold.csv')}", flush=True)
    if args.style_slice:
        style_slice(preds, args.out_dir, args.stem)


if __name__ == "__main__":
    main()
# end
