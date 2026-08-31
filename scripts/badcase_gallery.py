#!/usr/bin/env python3
# 2026-08-29, zyun, render bad cases into a self-contained HTML gallery
"""Turn a predict.py JSON into a browsable FP/FN gallery.

Examples:

    python scripts/badcase_gallery.py --pred outputs/pred.json \
        --split official_val --condition clean --out gallery.html

    python scripts/badcase_gallery.py --pred pred_jpeg_q50.json \
        --image-dir /workspace/experiments/eval/images/jpeg_q50 \
        --condition jpeg_q50 --max-per-type 60 --out gallery_q50.html

    # full FP/FN without freezing the browser (paged folder, ~36 thumbs/page)
    python scripts/badcase_gallery.py --pred pred.json --split official_val \
        --max-per-type 99999 --layout folder --out galleries/clipl14_full/

    # split an already-embedded giant HTML into the same paged folder
    python scripts/badcase_gallery.py --from-html giant.html --out galleries/clipl14_full/

Small sets (≤80 cards, default) stay a single HTML with base64 thumbs.
Full sets auto-switch to a folder: index.html + thumbs/*.jpg, one page at a time.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import re
import shutil
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
    <b>pred {pred:.3f}</b> · label {label} · {condition} · {generator}<br>
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

# 2026-08-31, tianqi, paged folder so full FP/FN galleries do not freeze the browser
_FOLDER_CAP = 80  # auto layout: more cards than this → folder, not one giant HTML

_PAGED_PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ background:#111; color:#ddd; font: 14px/1.5 -apple-system, "PingFang SC", sans-serif; margin: 24px; }}
  h1 {{ font-size: 20px; }}
  .meta {{ color: #888; font-size: 12px; }}
  .bar {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin: 16px 0; }}
  button, select {{ background:#2a2a2a; color:#ddd; border:1px solid #444; border-radius:6px; padding:6px 12px; cursor:pointer; }}
  button.active {{ background:#b33; color:#fff; border-color:#b33; }}
  button.active.fn {{ background:#d80; color:#111; border-color:#d80; }}
  button:disabled {{ opacity:0.4; cursor:default; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }}
  .card {{ margin: 0; background: #1c1c1c; border-radius: 8px; overflow: hidden; }}
  .card img {{ width: 100%; display: block; background:#222; min-height: 80px; }}
  .card.FP {{ border: 1px solid #b33; }} .card.FN {{ border: 1px solid #d80; }}
  .badge {{ display: inline-block; padding: 0 6px; border-radius: 4px; font-weight: 700; }}
  .FP .badge {{ background: #b33; color: #fff; }} .FN .badge {{ background: #d80; color: #111; }}
  code {{ font-size: 10px; color: #8ab; word-break: break-all; }}
  figcaption {{ padding: 8px; font-size: 13px; }}
</style></head><body>
<h1>{title}</h1>
<p class="meta">{meta}</p>
<p class="meta">每页只加载 {page_size} 张图。把整个文件夹发给别人，用浏览器打开本页（不要只发这一份 HTML）。</p>
<div class="bar">
  <button type="button" id="tab-fp" class="active">FP</button>
  <button type="button" id="tab-fn">FN</button>
  <button type="button" id="prev">上一页</button>
  <span class="meta" id="pageinfo"></span>
  <button type="button" id="next">下一页</button>
  <label class="meta">每页
    <select id="psize">
      <option>24</option>
      <option selected>36</option>
      <option>48</option>
      <option>60</option>
    </select>
  </label>
</div>
<div class="grid" id="grid"></div>
<script>
const CARDS = {cards_json};
let kind = (CARDS.FP && CARDS.FP.length) ? "FP" : "FN";
let page = 0;
let pageSize = {page_size};
const grid = document.getElementById("grid");
const pageinfo = document.getElementById("pageinfo");
function esc(s) {{
  const d = document.createElement("div");
  d.textContent = s == null ? "" : String(s);
  return d.innerHTML;
}}
function render() {{
  const rows = CARDS[kind] || [];
  const pages = Math.max(1, Math.ceil(rows.length / pageSize));
  page = Math.min(Math.max(0, page), pages - 1);
  const slice = rows.slice(page * pageSize, (page + 1) * pageSize);
  grid.innerHTML = slice.map(function (c) {{
    const img = c.src
      ? '<img src="' + esc(c.src) + '" alt="badcase" loading="lazy">'
      : '<div class="meta">[image not found: ' + esc(c.path) + ']</div>';
    return '<figure class="card ' + c.etype + '">' + img +
      '<figcaption><span class="badge">' + esc(c.etype) + '</span> ' +
      '<b>pred ' + Number(c.pred).toFixed(3) + '</b> · label ' + esc(c.label) +
      ' · ' + esc(c.condition) + ' · ' + esc(c.generator) + '<br><code>' +
      esc(c.path) + '</code></figcaption></figure>';
  }}).join("");
  pageinfo.textContent = kind + " · 第 " + (page + 1) + " / " + pages +
    " 页 · " + rows.length + " 张";
  document.getElementById("prev").disabled = page <= 0;
  document.getElementById("next").disabled = page >= pages - 1;
  document.getElementById("tab-fp").className = kind === "FP" ? "active" : "";
  document.getElementById("tab-fn").className = kind === "FN" ? "active fn" : "";
  document.getElementById("tab-fp").textContent = "FP (" + (CARDS.FP || []).length + ")";
  document.getElementById("tab-fn").textContent = "FN (" + (CARDS.FN || []).length + ")";
  window.scrollTo(0, 0);
}}
document.getElementById("tab-fp").onclick = function () {{ kind = "FP"; page = 0; render(); }};
document.getElementById("tab-fn").onclick = function () {{ kind = "FN"; page = 0; render(); }};
document.getElementById("prev").onclick = function () {{ page -= 1; render(); }};
document.getElementById("next").onclick = function () {{ page += 1; render(); }};
document.getElementById("psize").onchange = function (e) {{
  pageSize = parseInt(e.target.value, 10); page = 0; render();
}};
document.addEventListener("keydown", function (e) {{
  if (e.key === "ArrowLeft") {{ page -= 1; render(); }}
  if (e.key === "ArrowRight") {{ page += 1; render(); }}
}});
render();
</script>
</body></html>"""

_STUB = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>{title}</title></head>
<body style="font:16px/1.5 sans-serif;max-width:40em;margin:2rem auto">
<p>全量画册已改成分页目录（单文件会卡死浏览器）。请打开：</p>
<p><a href="{href}">{href}</a></p>
</body></html>"""

_FIG_RE = re.compile(
    r'<figure class="card (FP|FN)">\s*'
    r'(?:<img src="data:image/jpeg;base64,([^"]+)"[^>]*>|'
    r'<div class="meta">\[image not found: ([^\]]+)\]</div>)\s*'
    r'<figcaption>[\s\S]*?<b>pred ([0-9.]+)</b> · label (\S+) · ([^·]+) · ([^<]+)<br>\s*'
    r'<code>([^<]*)</code>',
)


def thumb_jpeg(candidates: list[Path], max_side: int) -> bytes | None:
    """First readable candidate wins; None means every candidate failed."""
    for path in candidates:
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((max_side, max_side))
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=70)
                return buf.getvalue()
        except Exception:
            continue
    return None


def thumb_b64(candidates: list[Path], max_side: int) -> str | None:
    """First readable candidate wins; None means every candidate failed."""
    jpeg = thumb_jpeg(candidates, max_side)
    return base64.b64encode(jpeg).decode("ascii") if jpeg else None
# end


def _thumb_candidates(image_path: str, roots: list[Path]) -> list[Path]:
    """Places the thumbnail may live: raw path first, then root-relative forms.

    Mirrors the label join's candidate logic so pred paths that only resolve
    under ``--predict-root`` still find their files from any CWD.
    """
    p = Path(image_path)
    out = [p]
    if p.is_absolute():
        return out
    for root in roots:
        cand = root / p
        if cand not in out:
            out.append(cand)
    return out


def render_section(
    rows: list[dict], header: str, max_n: int, max_side: int, roots: list[Path]
) -> tuple[str, int, int]:
    """Render one FP/FN section; returns ``(html, n_shown, n_missing)``."""
    cards, missing = [], 0
    for r in rows[:max_n]:
        b64 = thumb_b64(_thumb_candidates(r["image_path"], roots), max_side)
        if b64 is None:
            missing += 1
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
                generator=html.escape(str(r.get("generator", ""))),
                path=html.escape(r["image_path"]),
                img=img,
            )
        )
    shown = f"showing {len(cards)}/{len(rows)}"
    return (
        f"<h2>{html.escape(header)} <span class='meta'>{shown}</span></h2>"
        f"<div class='grid'>{''.join(cards)}</div>",
        len(cards),
        missing,
    )


# 2026-08-31, tianqi, paged folder writer + split giant embedded HTML
def _folder_dir(out: Path) -> Path:
    """`--out foo.html` → `foo/`; `--out foo/` stays a directory."""
    if out.suffix.lower() in {".html", ".htm"}:
        return out.with_suffix("")
    return out


def _card_dict(etype: str, pred, label, condition, generator, path, src: str | None) -> dict:
    try:
        label = int(label)
    except (TypeError, ValueError):
        pass
    return {
        "etype": etype,
        "pred": float(pred),
        "label": label,
        "condition": str(condition).strip(),
        "generator": str(generator).strip(),
        "path": path,
        "src": src,
    }


def write_paged_folder(
    out_dir: Path,
    title: str,
    meta: str,
    cards: dict[str, list[dict]],
    page_size: int,
    stub_html: Path | None = None,
) -> Path:
    """Write `index.html` + `thumbs/` already filled by caller. `cards` has src paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(cards, ensure_ascii=False).replace("<", "\\u003c")
    (out_dir / "index.html").write_text(
        _PAGED_PAGE.format(
            title=html.escape(title),
            meta=html.escape(meta),
            page_size=int(page_size),
            cards_json=payload,
        ),
        encoding="utf-8",
    )
    if stub_html is not None:
        stub_html.parent.mkdir(parents=True, exist_ok=True)
        rel = f"{out_dir.name}/index.html"
        stub_html.write_text(
            _STUB.format(title=html.escape(title), href=html.escape(rel)),
            encoding="utf-8",
        )
    return out_dir / "index.html"


def materialize_thumbs(
    rows: list[dict], kind: str, max_n: int, max_side: int, roots: list[Path], thumb_dir: Path
) -> tuple[list[dict], int]:
    """Write jpeg thumbs; return (cards, n_missing)."""
    thumb_dir.mkdir(parents=True, exist_ok=True)
    cards, missing = [], 0
    for i, r in enumerate(rows[:max_n]):
        jpeg = thumb_jpeg(_thumb_candidates(r["image_path"], roots), max_side)
        name = f"{kind.lower()}_{i:04d}.jpg"
        src = None
        if jpeg is None:
            missing += 1
        else:
            (thumb_dir / name).write_bytes(jpeg)
            src = f"thumbs/{name}"
        cards.append(
            _card_dict(
                r["error_type"],
                r["pred"],
                r["label"],
                r.get("condition", ""),
                r.get("generator", ""),
                r["image_path"],
                src,
            )
        )
    return cards, missing


def split_embedded_html(html_path: Path, out_dir: Path, page_size: int, title: str | None) -> None:
    """Turn a base64-embedded gallery into a paged folder (no re-read of original images)."""
    text = html_path.read_text(encoding="utf-8")
    h1 = re.search(r"<h1>([^<]*)</h1>", text)
    meta_m = re.search(r'<p class="meta">([^<]*)</p>', text)
    page_title = title or (h1.group(1) if h1 else html_path.stem)
    meta = meta_m.group(1) if meta_m else ""
    grouped: dict[str, list[dict]] = {"FP": [], "FN": []}
    if out_dir.exists():
        shutil.rmtree(out_dir)
    thumb_dir = out_dir / "thumbs"
    thumb_dir.mkdir(parents=True)
    n_ok = 0
    for m in _FIG_RE.finditer(text):
        etype, b64, missing_path, pred, label, condition, generator, path = m.groups()
        idx = len(grouped[etype])
        src = None
        if b64:
            name = f"{etype.lower()}_{idx:04d}.jpg"
            (thumb_dir / name).write_bytes(base64.b64decode(b64))
            src = f"thumbs/{name}"
            n_ok += 1
        grouped[etype].append(
            _card_dict(
                etype,
                pred,
                label.strip(),
                condition,
                generator,
                (missing_path or path).strip(),
                src,
            )
        )
    n_shown = len(grouped["FP"]) + len(grouped["FN"])
    if n_shown == 0:
        raise SystemExit(f"no FP/FN cards parsed from {html_path}")
    stub = html_path if out_dir.resolve() == _folder_dir(html_path).resolve() else None
    index = write_paged_folder(out_dir, page_title, meta, grouped, page_size, stub_html=stub)
    print(
        f"wrote {index} (FP {len(grouped['FP'])} / FN {len(grouped['FN'])}; "
        f"thumbnails {n_ok}/{n_shown} as files, page_size={page_size})"
    )


def _write_embed(
    out: Path, title: str, meta_prefix: str, fps, fns, max_per_type, thumb, thumb_roots
) -> tuple[int, int]:
    sections, n_shown, n_missing = "", 0, 0
    if fps:
        sec, shown, missing = render_section(
            fps, "FP · 真图误判为 AI（误杀）", max_per_type, thumb, thumb_roots
        )
        sections += sec
        n_shown += shown
        n_missing += missing
    if fns:
        sec, shown, missing = render_section(
            fns, "FN · 假图漏检（判为真）", max_per_type, thumb, thumb_roots
        )
        sections += sec
        n_shown += shown
        n_missing += missing
    meta = f"{meta_prefix} · thumbnails {n_shown - n_missing}/{n_shown}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        _PAGE.format(title=html.escape(title), meta=meta, sections=sections),
        encoding="utf-8",
    )
    return n_shown, n_missing
# end


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pred", type=Path)
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
    parser.add_argument("--out", type=Path)
    parser.add_argument("--title", default="Bad cases")
    # 2026-08-31, tianqi, auto folder layout when the gallery would freeze a browser
    parser.add_argument(
        "--layout",
        choices=("auto", "embed", "folder"),
        default="auto",
        help="auto: embed ≤80 cards, else paged folder; folder never embeds",
    )
    parser.add_argument("--page-size", type=int, default=36, help="cards per page in folder layout")
    parser.add_argument(
        "--from-html",
        type=Path,
        help="split an existing base64-embedded HTML into a paged folder",
    )
    # end
    args = parser.parse_args()

    if args.from_html:
        src = args.from_html
        dest = args.out if args.out else _folder_dir(src)
        split_embedded_html(src, dest, args.page_size, None if args.title == "Bad cases" else args.title)
        return

    if args.pred is None or args.out is None:
        raise SystemExit("need --pred and --out (or --from-html)")

    if args.image_dir is not None:
        root, rows = args.image_dir, load_labeled_dir(
            args.image_dir, default_fake=(args.split == "evalgen")
        )
    elif args.split:
        root, rows = load_split(args.split)
    else:
        raise SystemExit("need --split or --image-dir")
    if args.max_images is not None:
        if args.max_images <= 0:
            raise SystemExit("--max-images must be a positive integer")
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

    n_fp_show = min(len(fps), args.max_per_type)
    n_fn_show = min(len(fns), args.max_per_type)
    use_folder = args.layout == "folder" or (
        args.layout == "auto" and (n_fp_show + n_fn_show) > _FOLDER_CAP
    )

    thumb_roots = [root] + ([args.predict_root] if args.predict_root else [])
    meta = (
        f"threshold={args.threshold} · FP={len(fps)} · FN={len(fns)} · "
        f"joined={len(res['joined'])} · unmatched_labels={res['unmatched_labels']}"
    )

    if use_folder:
        out_dir = _folder_dir(args.out)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        thumb_dir = out_dir / "thumbs"
        fp_cards, miss_fp = materialize_thumbs(
            fps, "FP", args.max_per_type, args.thumb, thumb_roots, thumb_dir
        )
        fn_cards, miss_fn = materialize_thumbs(
            fns, "FN", args.max_per_type, args.thumb, thumb_roots, thumb_dir
        )
        n_shown = len(fp_cards) + len(fn_cards)
        n_missing = miss_fp + miss_fn
        meta = f"{meta} · thumbnails {n_shown - n_missing}/{n_shown}"
        stub = args.out if args.out.suffix.lower() in {".html", ".htm"} else None
        index = write_paged_folder(
            out_dir, args.title, meta, {"FP": fp_cards, "FN": fn_cards}, args.page_size, stub_html=stub
        )
        print(
            f"wrote {index} (FP {len(fps)} / FN {len(fns)}; "
            f"thumbnails {n_shown - n_missing}/{n_shown} as files, page_size={args.page_size})"
        )
    else:
        n_shown, n_missing = _write_embed(
            args.out, args.title, meta, fps, fns, args.max_per_type, args.thumb, thumb_roots
        )
        print(
            f"wrote {args.out} (FP {len(fps)} / FN {len(fns)}; "
            f"thumbnails {n_shown - n_missing}/{n_shown} embedded)"
        )

    if n_shown and n_missing == n_shown:
        print(
            f"warning: all {n_shown} thumbnails failed to load — pred paths point at "
            f"files this machine/CWD cannot see; check --predict-root and that the "
            f"images still exist",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
# end
