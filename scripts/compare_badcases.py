#!/usr/bin/env python3
# 2026-08-31, tianqi, pairwise FP/FN compare between two pred JSONs (same image set)
"""Compare errors of a weaker model vs a stronger one on the same labeled split.

Prints what got fixed / remaining / newly introduced at a threshold. Optional
HTML: remaining FN (still missed) and remaining FP (still killing reals).

    python scripts/compare_badcases.py \
      --base pred_sid.json --base-name clipb16_sid \
      --new pred_u4.json --new-name unfreeze4 \
      --split official_val --max-images 400 --seed 0 \
      --out-dir outputs/tables/badcase_compare
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import badcase_gallery as gal
from src.eval.labels import load_split

# end


def _key(path: str) -> str:
    parts = Path(path).parts
    for token in ("fake", "real", "FAKE", "REAL"):
        if token in parts:
            i = parts.index(token)
            return "/".join(parts[i:])
    return Path(path).name


def _load(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text())
    out = {}
    for r in rows:
        out[_key(r["image_path"])] = r
    return out


def _err(row: dict, thr: float) -> str | None:
    y = int(row["y"]) if "y" in row else int(row["label"])
    s = float(row["pred"])
    if y == 0 and s >= thr:
        return "FP"
    if y == 1 and s < thr:
        return "FN"
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True, help="weaker / older pred JSON")
    p.add_argument("--new", type=Path, required=True, help="stronger pred JSON")
    p.add_argument("--base-name", default="base")
    p.add_argument("--new-name", default="new")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--split", default="official_val")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-per-type", type=int, default=60)
    p.add_argument("--thumb", type=int, default=256)
    p.add_argument("--html", action="store_true")
    args = p.parse_args()

    base = _load(args.base)
    new = _load(args.new)
    keys = sorted(set(base) & set(new))
    if not keys:
        raise SystemExit("no overlapping image keys; check pred path formats")

    buckets = Counter()
    remaining_fn: list[dict] = []
    remaining_fp: list[dict] = []
    fixed_fn: list[dict] = []
    new_fn: list[dict] = []
    for k in keys:
        a, b = base[k], new[k]
        ea, eb = _err(a, args.threshold), _err(b, args.threshold)
        y = int(a.get("y", a.get("label")))
        rec = {
            "image_path": b.get("image_path") or a["image_path"],
            "pred_base": float(a["pred"]),
            "pred_new": float(b["pred"]),
            "label": y,
            "error_type": eb or "ok",
            "condition": "clean",
            "generator": "dalle" if y == 1 else "real",
        }
        if ea == "FN" and eb is None:
            buckets["fn_fixed"] += 1
            rec["error_type"] = "FN"
            rec["pred"] = rec["pred_new"]
            fixed_fn.append(rec)
        elif ea == "FN" and eb == "FN":
            buckets["fn_remain"] += 1
            rec["pred"] = rec["pred_new"]
            remaining_fn.append(rec)
        elif ea is None and eb == "FN":
            buckets["fn_new"] += 1
            rec["pred"] = rec["pred_new"]
            new_fn.append(rec)
        elif ea == "FP" and eb is None:
            buckets["fp_fixed"] += 1
        elif ea == "FP" and eb == "FP":
            buckets["fp_remain"] += 1
            rec["pred"] = rec["pred_new"]
            remaining_fp.append(rec)
        elif ea is None and eb == "FP":
            buckets["fp_new"] += 1
            rec["pred"] = rec["pred_new"]
            remaining_fp.append(rec)
        else:
            buckets["ok"] += 1

    n_fake = sum(1 for k in keys if int(base[k].get("y", base[k].get("label"))) == 1)
    n_real = len(keys) - n_fake
    summary = {
        "base": args.base_name,
        "new": args.new_name,
        "n_overlap": len(keys),
        "n_real": n_real,
        "n_fake": n_fake,
        "threshold": args.threshold,
        **dict(buckets),
        "fn_base": buckets["fn_fixed"] + buckets["fn_remain"],
        "fn_new_model": buckets["fn_remain"] + buckets["fn_new"],
        "fp_base": buckets["fp_fixed"] + buckets["fp_remain"],
        "fp_new_model": buckets["fp_remain"] + buckets["fp_new"],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.base_name}_to_{args.new_name}"
    (args.out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)

    remaining_fn.sort(key=lambda r: r["pred"])
    remaining_fp.sort(key=lambda r: -r["pred"])
    if args.html:
        root, _rows = load_split(args.split)
        roots = [root]
        sec_fn, _, _ = gal.render_section(remaining_fn, f"FN remaining in {args.new_name} (still missed fakes)", args.max_per_type, args.thumb, roots)
        sec_fp, _, _ = gal.render_section(remaining_fp, f"FP remaining in {args.new_name} (still killing reals)", args.max_per_type, args.thumb, roots)
        html = gal._PAGE.format(
            title=f"{args.base_name} → {args.new_name}",
            meta=(
                f"overlap={len(keys)}  FN {summary['fn_base']}→{summary['fn_new_model']} "
                f"(fixed {buckets['fn_fixed']}, new {buckets['fn_new']})  "
                f"FP {summary['fp_base']}→{summary['fp_new_model']} "
                f"(fixed {buckets['fp_fixed']}, new {buckets['fp_new']})"
            ),
            sections=sec_fp + sec_fn,
        )
        out_html = args.out_dir / f"{stem}.html"
        out_html.write_text(html)
        print(f"wrote {out_html}", flush=True)


if __name__ == "__main__":
    main()
# end
