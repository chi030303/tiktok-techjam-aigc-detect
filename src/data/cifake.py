# 2026-08-29, tianqi, CIFAKE folder labels REAL=0 FAKE=1; smoke subsample only
"""CIFAKE image-folder loader. FAKE is the AIGC class (pred target = 1)."""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from src.paths import data_root

# end

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# FAKE = AIGC
LABEL_DIR = {"REAL": 0, "FAKE": 1}


def _list_split(cifake_root: Path, split: str, label_name: str, sort: bool = True) -> list[Path]:
    folder = cifake_root / split / label_name
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    # 2026-08-29, tianqi, skip sort on full split; sorting 1M paths is wasted CPU
    out = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    if sort:
        out.sort()
    return out
    # end


def subsample_balanced(real: list[Path], fake: list[Path], n_total: int, seed: int) -> list[tuple[Path, int]]:
    # 2026-08-29, tianqi, smoke table is 200-500 images, keep classes even
    rng = random.Random(seed)
    half = max(1, n_total // 2)
    real_s = real if len(real) <= half else rng.sample(real, half)
    fake_s = fake if len(fake) <= half else rng.sample(fake, half)
    rows = [(p, 0) for p in real_s] + [(p, 1) for p in fake_s]
    rng.shuffle(rows)
    return rows
    # end


def load_cifake_rows(split: str, max_images: int | None, seed: int) -> list[tuple[Path, int]]:
    root = data_root() / "cifake"
    real = _list_split(cifake_root=root, split=split, label_name="REAL", sort=max_images is not None)
    fake = _list_split(cifake_root=root, split=split, label_name="FAKE", sort=max_images is not None)
    # 2026-08-29, tianqi, None = full split (100k train / 20k test)
    if max_images is None:
        rows = [(p, 0) for p in real] + [(p, 1) for p in fake]
        rng = random.Random(seed)
        rng.shuffle(rows)
        return rows
    # end
    return subsample_balanced(real, fake, max_images, seed)


class ImagePathDataset(Dataset):
    def __init__(self, rows: list[tuple[Path, int]], transform):
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        path, y = self.rows[idx]
        # 2026-08-29, tianqi, close the file handle; 1M-scale workers leak FDs otherwise
        with Image.open(path) as im:
            img = im.convert("RGB")
        x = self.transform(img)
        return x, y, str(path)
        # end
