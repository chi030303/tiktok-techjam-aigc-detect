# 2026-08-30, yun, exp2: two views of the same SID image per step (clean + one random official op)
"""Paired-view dataset for the consistency ablation.

view A is always clean; view B is one official transform forced non-clean
(p_clean=0) so the pair never collapses to two identical clean images.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.sid import BINARY_OK, _balanced_keep, load_sid_hf
from src.transforms.augment import random_augment

# end


class SidPairedViewDataset(Dataset):
    def __init__(self, split: str, transform, seed: int = 0, max_images: int | None = None):
        hf_split = "train" if split in ("train", "train_set") else "validation"
        self.ds = load_sid_hf(hf_split)
        labels = [int(y) for y in self.ds["label"]]
        keep = [i for i, y in enumerate(labels) if y in BINARY_OK]
        if max_images is not None:
            keep = _balanced_keep(labels, max_images, seed)
        self.keep = keep
        self.transform = transform
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.keep)

    def __getitem__(self, idx: int):
        src_i = int(self.keep[idx])
        ex = self.ds[src_i]
        img = ex["image"]
        if not isinstance(img, Image.Image):
            img = Image.open(img).convert("RGB")
        else:
            img = img.convert("RGB")
        y = int(ex["label"])
        worker = torch.utils.data.get_worker_info()
        wid = 0 if worker is None else worker.id
        rng = np.random.default_rng((self.seed, self.epoch, idx, wid))
        view_a = img.convert("RGB")
        view_b, _info = random_augment(img, rng, p_clean=0.0, continuous=True, chain_jpeg_p=0.3)
        xa = self.transform(view_a)
        xb = self.transform(view_b)
        return xa, xb, y
# end
