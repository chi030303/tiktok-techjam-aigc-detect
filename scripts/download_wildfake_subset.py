#!/usr/bin/env python3
# 2026-08-30, samily, download WildFake zip subsets from ModelScope
"""Download and extract WildFake cross-architecture packs.

Examples:
  python scripts/download_wildfake_subset.py --list
  python scripts/download_wildfake_subset.py C_pixel_ddpm C_pixel_adm
  python scripts/download_wildfake_subset.py C_unet_sd_original
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.paths import data_root

# end

WILDFAKE_ID = "hy2628982280/WildFake"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def default_cfg() -> Path:
    return REPO / "configs" / "wildfake" / "subsets.yaml"


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def count_images(root: Path) -> int:
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def download_file(dataset_id: str, file_path: str, local_dir: Path) -> Path:
    from modelscope.hub.file_download import dataset_file_download

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"downloading {file_path} ...", flush=True)
    dataset_file_download(dataset_id=dataset_id, file_path=file_path, local_dir=str(local_dir))
    found = list(local_dir.rglob(Path(file_path).name))
    if not found:
        raise SystemExit(f"after download, {Path(file_path).name} not found under {local_dir}")
    return found[0]


def extract_zip(zip_path: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if not m.endswith("/")]
        print(f"extract {zip_path.name} -> {dest} ({len(members)} members)", flush=True)
        for name in members:
            info = zf.getinfo(name)
            if info.is_dir():
                continue
            if Path(name).suffix.lower() not in IMAGE_EXTS:
                continue
            # Preserve the full archive-relative path so repeated basenames are
            # never silently dropped. Ignore traversal components.
            parts = [
                part
                for part in Path(name.replace("\\", "/")).parts
                if part not in ("", ".", "..", "/")
            ]
            out = dest.joinpath(*parts)
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists() and out.stat().st_size > 0:
                n += 1
                continue
            with zf.open(info) as src, open(out, "wb") as fh:
                shutil.copyfileobj(src, fh)
            n += 1
            if n % 500 == 0:
                print(f"  extracted {n}", flush=True)
    return n


def run_subset(name: str, spec: dict, cfg: dict, data: Path, tmp: Path) -> None:
    dataset_id = cfg.get("dataset_id", WILDFAKE_ID)
    extract_under = spec.get("extract_under", name)
    dest = data / "wildfake" / extract_under
    complete = dest / ".complete"
    if complete.is_file():
        print(f"skip {name}: verified complete ({complete.read_text().strip()})", flush=True)
        return
    extracted = 0
    for rel in spec.get("files") or []:
        zip_path = download_file(dataset_id, rel, tmp)
        extracted += extract_zip(zip_path, dest)
        zip_path.unlink(missing_ok=True)
    total = count_images(dest)
    if total == 0 or extracted == 0:
        raise SystemExit(f"{name}: extraction produced no images")
    complete.write_text(f"images={total}\n")
    print(f"done {name}: {total} images in {dest}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("subset", nargs="*", help="subset keys from subsets.yaml")
    p.add_argument("--config", type=Path, default=default_cfg())
    p.add_argument("--list", action="store_true")
    p.add_argument("--data-root", type=Path, default=None)
    args = p.parse_args()

    cfg = load_cfg(args.config)
    subsets = cfg.get("subsets") or {}
    if args.list:
        for key, spec in subsets.items():
            desc = spec.get("description", "")
            files = spec.get("files") or []
            print(f"{key}: {desc}")
            for f in files:
                print(f"  - {f}")
        return

    if not args.subset:
        raise SystemExit("pass subset name(s) or --list")

    data = args.data_root or data_root()
    tmp = data / "_download_tmp" / "wildfake"
    tmp.mkdir(parents=True, exist_ok=True)
    (data / "wildfake").mkdir(parents=True, exist_ok=True)

    for name in args.subset:
        if name not in subsets:
            raise SystemExit(f"unknown subset {name!r}; --list for keys")
        run_subset(name, subsets[name], cfg, data, tmp)


if __name__ == "__main__":
    main()
