# 2026-08-29, tianqi, submission interface stub (directory in, JSON out)
"""Official-style inference entry: image directory -> JSON list of {image_path, pred}."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def list_images(root: Path) -> list[Path]:
    files = [p for p in sorted(root.rglob("*")) if p.is_file()]
    images = [p for p in files if p.suffix.lower() in IMAGE_EXTS]
    return images if images else files


def predict_paths(image_dir: Path, ckpt: Path | None, ckpt_b: Path | None = None, fuse_weight: float = 0.5) -> list[dict]:
    paths = list_images(image_dir)
    if ckpt is None:
        # 2026-08-29, tianqi, no ckpt = schema stub so local check.sh does not need torch
        return [{"image_path": str(path), "pred": 0.5} for path in paths]
    # 2026-08-31, tianqi, optional second ckpt = last4+D3 logit fuse
    if ckpt_b is not None:
        from src.infer import fuse_predict_dir

        return fuse_predict_dir(ckpt, ckpt_b, image_dir, weight=fuse_weight)
    # end
    from src.infer import ProbePredictor

    return ProbePredictor(ckpt).predict_dir(image_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("out_json", type=Path)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--ckpt-b", type=Path, default=None, help="second ckpt; mean-logit fuse with --ckpt")
    parser.add_argument("--fuse-weight", type=float, default=0.5, help="weight on --ckpt when fusing")
    args = parser.parse_args()
    if not args.image_dir.is_dir():
        raise SystemExit(f"not a directory: {args.image_dir}")
    if args.ckpt_b is not None and args.ckpt is None:
        raise SystemExit("--ckpt-b requires --ckpt")
    rows = predict_paths(args.image_dir, args.ckpt, args.ckpt_b, args.fuse_weight)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} rows -> {args.out_json}")


if __name__ == "__main__":
    main()
# end
