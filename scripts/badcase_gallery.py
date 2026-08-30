#!/usr/bin/env python3
# 2026-08-29, zyun, render bad cases into a self-contained HTML gallery
"""Turn a predict.py JSON into a browsable FP/FN gallery (thumbnails embedded).

Examples:

    python scripts/badcase_gallery.py --pred outputs/pred.json \
        --split official_val --condition clean --out gallery.html

    python scripts/badcase_gallery.py --pred pred_jpeg_q50.json \
        --image-dir /workspace/experiments/eval/images/jpeg_q50 \
        --condition jpeg_q50 --max-per-type 60 --out gallery_q50.html

The output HTML is self-contained (base64 thumbnails, no external files):
copy/scp it anywhere and open in a browser.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval.badcase import join_predictions, load_manifest_rows
from src.eval.labels import load_labeled_dir, load_split, subsample_balanced

# end

_CARD = """
<figure class="card {cls}">
  {img}
  <figcaption>
    <span class="badge">{etype}</span>
    <b>pred {pred:.3f}</b> · label {label} · {condition}<br>
    <code>{path}</code>
  </figcaption>
</figure>"""

_PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ background:#111; color:#ddd; font: 14px/1.5 -apple-system, "PingFang SC", sans-serif; margin: 24px; }}
  h1 {{ font-size: 20px; }} h2 {{ font-size: 16px; margin-top: 32px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }}
  .card {{ margin: 0; background: #1c1c1c; border-radius: 8px; overflow: hidden; }}
  .card img {{ width: 100%; display: block; }}
  .card.FP {{ border: 1px solid #b33; }} .card.FN {{ border: 1px solid #d80; }}
  .badge {{ display: inline-block; padding: 0 6px; border-radius: 4px; font-weight: 700; }}
  .FP .badge {{ background: #b33; color: #fff; }} .FN .badge {{ background: #d80; color: #111; }}
  code {{ font-size: 10px; color: #8ab; word-break: break-all; }}
  .meta {{ color: #888; font-size: 12px; }}
</style></head><body>
<h1>{title}</h1><p class="meta">{meta}</p>
{sections}
</body></html>"""


def thumb_b64(path: Path, max_side: int) -> str | None:
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=70)
            return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def render_section(rows: list[dict], max_n: int, max_side: int) -> str:
    cards = []
    for r in rows[:max_n]:
        b64 = thumb_b64(Path(r["image_path"]), max_side)
        img = (
            f'<img src="data:image/jpeg;base64,{b64}" alt="badcase">'
            if b64
            else f'<div class="meta">[image not found: {html.escape(r["image_path"])}]</div>'
        )
        cards.append(
            _CARD.format(
                cls=r["error_type"],
                etype=r["error_type"],
                pred=r["pred"],
                label=r["label"],
                condition=html.escape(str(r.get("condition", ""))),
                path=html.escape(r["image_path"]),
                img=img,
            )
        )
    shown = f"showing {len(cards)}/{len(rows)}"
    title = (
        "FP · 真图误判为 AI（误杀）"
        if rows and rows[0]["error_type"] == "FP"
        else "FN · 假图漏检（判为真）"
    )
    return f"<h2>{html.escape(title)} <span class='meta'>{shown}</span></h2><div class='grid'>{''.join(cards)}</div>"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--split")
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--predict-root", type=Path)
    parser.add_argument("--manifest", help="optional manifest JSONL for generator metadata")
    parser.add_argument("--condition", default="clean")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-per-type", type=int, default=60)
    parser.add_argument("--thumb", type=int, default=384)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--title", default="Bad cases")
    args = parser.parse_args()

    if args.image_dir is not None:
        root, rows = args.image_dir, load_labeled_dir(
            args.image_dir, default_fake=(args.split == "evalgen")
        )
    elif args.split:
        root, rows = load_split(args.split)
    else:
        raise SystemExit("need --split or --image-dir")
    if args.max_images:
        rows = subsample_balanced(rows, args.max_images, args.seed)

    manifest_rows = load_manifest_rows(
        [Path(x) for x in args.manifest.split(",")] if args.manifest else None
    )
    preds = json.loads(args.pred.read_text(encoding="utf-8"))
    res = join_predictions(
        preds, rows, src_root=root, predict_root=args.predict_root or root,
        threshold=args.threshold, manifest_rows=manifest_rows,
        default_condition=args.condition,
    )
    fps = sorted(
        (r for r in res["joined"] if r["error_type"] == "FP"), key=lambda r: -r["pred"]
    )
    fns = sorted(
        (r for r in res["joined"] if r["error_type"] == "FN"), key=lambda r: r["pred"]
    )

    sections = ""
    if fps:
        sections += render_section(fps, args.max_per_type, args.thumb)
    if fns:
        sections += render_section(fns, args.max_per_type, args.thumb)
    meta = (
        f"threshold={args.threshold} · FP={len(fps)} · FN={len(fns)} · "
        f"joined={len(res['joined'])} · unmatched_labels={res['unmatched_labels']}"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        _PAGE.format(title=html.escape(args.title), meta=meta, sections=sections),
        encoding="utf-8",
    )
    print(f"wrote {args.out} (FP {len(fps)} / FN {len(fns)})")


if __name__ == "__main__":
    main()
# end
