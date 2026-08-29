#!/usr/bin/env python3
# 2026-08-29, tianqi, official demo val only: COCO val2017 real + WildFake DALL·E Advanced (no train)
"""Build data/val/{real,fake} for the TechJam demonstration set. Do not use for training."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

# end

COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
# 2026-08-29, tianqi, DALLE.zip is Typical+Advanced; we only unzip Advanced (dalle3, 8843)
DALLE_ZIP_REL = "Images/Diffusion_based/DALLE.zip"
WILDFAKE_ID = "hy2628982280/WildFake"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# end


def default_data_root() -> Path:
    shared = Path("/workspace/data")
    if shared.is_dir():
        return shared
    return Path(__file__).resolve().parents[1] / "data"


def wget_c(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["wget", "-c", "-O", str(dest), url]
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def count_images(root: Path) -> int:
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def download_coco_real(val_root: Path, tmp: Path) -> None:
    # 2026-08-29, tianqi, official Non-AIGC half is COCO val2017 (listed 4998; zip has 5000)
    real_dir = val_root / "real"
    if count_images(real_dir) >= 4998:
        print(f"skip COCO, already have {count_images(real_dir)} images in {real_dir}", flush=True)
        return
    zip_path = tmp / "val2017.zip"
    wget_c(COCO_VAL_URL, zip_path)
    real_dir.mkdir(parents=True, exist_ok=True)
    print(f"unzip {zip_path} -> {real_dir}", flush=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)
    src = tmp / "val2017"
    if not src.is_dir():
        raise SystemExit(f"expected {src} after unzip")
    for p in src.iterdir():
        if p.is_file():
            target = real_dir / p.name
            if not target.exists():
                shutil.move(str(p), str(target))
    shutil.rmtree(src, ignore_errors=True)
    zip_path.unlink(missing_ok=True)
    n = count_images(real_dir)
    print(f"COCO real images: {n} (challenge text says 4998; val2017.zip is 5000)", flush=True)
    # end


def download_dalle_advanced(val_root: Path, tmp: Path) -> None:
    # 2026-08-29, tianqi, only extract Diffusion_based/DALLE/Advanced (do not keep Typical/DALLE2)
    fake_dir = val_root / "fake"
    if count_images(fake_dir) >= 8843:
        print(f"skip DALL·E Advanced, already have {count_images(fake_dir)} images in {fake_dir}", flush=True)
        return

    zip_path = tmp / "DALLE.zip"
    if not zip_path.exists() or zip_path.stat().st_size < 1_000_000_000:
        print(f"download WildFake {DALLE_ZIP_REL} (~25GB, then keep Advanced only)", flush=True)
        from modelscope.hub.file_download import dataset_file_download

        dataset_file_download(
            dataset_id=WILDFAKE_ID,
            file_path=DALLE_ZIP_REL,
            local_dir=str(tmp),
        )
        # 2026-08-29, tianqi, modelscope may nest Images/Diffusion_based/DALLE.zip under local_dir
        if not zip_path.exists():
            found = list(tmp.rglob("DALLE.zip"))
            if not found:
                raise SystemExit(f"DALLE.zip not found under {tmp}")
            shutil.move(str(found[0]), str(zip_path))
        # end
    print(f"zip size GB: {zip_path.stat().st_size / 1e9:.2f}", flush=True)

    fake_dir.mkdir(parents=True, exist_ok=True)
    print("extract Advanced/* only (keep subdirs; basenames collide)", flush=True)
    n_ok = 0
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            name
            for name in zf.namelist()
            if "/Advanced/" in name.replace("\\", "/") and not name.endswith("/")
        ]
        print(f"zip members matching Advanced: {len(members)}", flush=True)
        for name in members:
            info = zf.getinfo(name)
            if info.is_dir():
                continue
            # 2026-08-29, tianqi, keep DALLE3/dalle3/<folder>/file.jpg so hashes do not collide
            parts = Path(name.replace("\\", "/")).parts
            if "Advanced" in parts:
                dest = fake_dir.joinpath(*parts[parts.index("Advanced") + 1 :])
            else:
                dest = fake_dir / Path(name).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and dest.stat().st_size > 0:
                n_ok += 1
                continue
            with zf.open(info) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
            n_ok += 1
            if n_ok % 500 == 0:
                print(f"  extracted {n_ok}/{len(members)}", flush=True)
            # end
    n = count_images(fake_dir)
    print(f"DALL·E Advanced images: {n} (expected 8843)", flush=True)
    if n < 8843:
        raise SystemExit(f"too few fake images: {n}; keeping {zip_path} for retry")
    zip_path.unlink(missing_ok=True)
    # end


def write_guard(val_root: Path) -> None:
    # 2026-08-29, tianqi, training loaders must skip this tree
    guard = val_root / "DO_NOT_TRAIN"
    note = (
        "Official TechJam demonstration / reference val set.\n"
        "Does NOT count toward the score. Do NOT use for training.\n"
        "real/ = COCO val2017 (Non-AIGC)\n"
        "fake/ = WildFake DALL·E Advanced / DALLE3 (AIGC, 8843)\n"
    )
    guard.write_text(note)
    # end


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--skip-coco", action="store_true")
    parser.add_argument("--skip-dalle", action="store_true")
    args = parser.parse_args()

    val_root = args.data_root / "val"
    tmp = args.data_root / "_download_tmp"
    val_root.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    write_guard(val_root)

    if not args.skip_coco:
        download_coco_real(val_root, tmp)
    if not args.skip_dalle:
        download_dalle_advanced(val_root, tmp)

    print(
        f"done  real={count_images(val_root / 'real')}  fake={count_images(val_root / 'fake')}  dir={val_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
