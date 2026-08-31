#!/usr/bin/env python3
"""Compare exported bad-case galleries and generate an Error Analysis Note.

Example:
    python scripts/analyze_badcase_galleries.py \
      --gallery clipb16_sid=badcase/clipb16_sid_full_clean/index.html \
      --gallery clipl14_sid=badcase/clipl14_sid_full_clean/index.html \
      --auroc clipb16_sid=0.9655 --auroc clipl14_sid=0.9766 \
      --n-real 5000 --n-fake 8843 \
      --out-md docs/error_analysis_backbone_ablation.md \
      --out-json outputs/tables/badcase_backbone_ablation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval.gallery_analysis import build_report, load_gallery, render_markdown


def _named_value(raw: str, value_type: type, flag: str):
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"{flag} must use NAME=VALUE")
    name, value = raw.split("=", 1)
    if not name or not value:
        raise argparse.ArgumentTypeError(f"{flag} must use non-empty NAME=VALUE")
    try:
        return name, value_type(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"invalid {flag} value: {raw}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--gallery",
        action="append",
        required=True,
        metavar="NAME=INDEX_HTML",
        help="repeat for each model; at least two are required",
    )
    parser.add_argument(
        "--auroc",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="optional AUROC measured on the same split and condition",
    )
    parser.add_argument("--n-real", type=int, required=True)
    parser.add_argument("--n-fake", type=int, required=True)
    parser.add_argument(
        "--title", default="Backbone Ablation Error Analysis Note"
    )
    parser.add_argument(
        "--model-scope-note",
        default=(
            "Frozen CLIP backbone SID ablation; this is not the final "
            "last4/fuse submission analysis."
        ),
    )
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    if args.n_real <= 0 or args.n_fake <= 0:
        parser.error("--n-real and --n-fake must be positive")

    gallery_pairs = [
        _named_value(raw, Path, "--gallery") for raw in args.gallery
    ]
    names = [name for name, _ in gallery_pairs]
    if len(gallery_pairs) < 2:
        parser.error("at least two --gallery arguments are required")
    if len(names) != len(set(names)):
        parser.error("gallery names must be unique")

    auroc_pairs = [_named_value(raw, float, "--auroc") for raw in args.auroc]
    if len(auroc_pairs) != len(dict(auroc_pairs)):
        parser.error("AUROC model names must be unique")
    for name, value in auroc_pairs:
        if not 0 <= value <= 1:
            parser.error(f"AUROC for {name} must be between 0 and 1")

    try:
        galleries = [load_gallery(name, path) for name, path in gallery_pairs]
        report = build_report(
            galleries,
            n_real=args.n_real,
            n_fake=args.n_fake,
            aurocs=dict(auroc_pairs),
        )
        markdown = render_markdown(
            report, title=args.title, model_scope_note=args.model_scope_note
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(markdown, encoding="utf-8")
    args.out_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out_md}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
