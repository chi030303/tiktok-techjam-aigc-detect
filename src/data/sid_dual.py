# 2026-08-30, yun, exp3: RGB view + highpass-residual view of the SAME augmented image
"""Both branches must see the same aug draw -- otherwise the model can learn to
tell branches apart by which transform each one got, instead of real/fake.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.freq import highpass_residual
from src.data.sid import BINARY_OK, _balanced_keep, load_sid_hf
from src.transforms.augment import random_augment

# end


class SidDualBranchDataset(Dataset):
    def __init__(
        self,
        split: str,
        rgb_transform,
        freq_transform,
        augment: bool = True,
        seed: int = 0,
        max_images: int | None = None,
    ):
        hf_split = "train" if split in ("train", "train_set") else "validation"
        self.ds = load_sid_hf(hf_split)
        labels = [int(y) for y in self.ds["label"]]
        keep = [i for i, y in enumerate(labels) if y in BINARY_OK]
        if max_images is not None:
            keep = _balanced_keep(labels, max_images, seed)
        self.keep = keep
        self.rgb_transform = rgb_transform
        self.freq_transform = freq_transform
        self.augment = augment
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
        if self.augment:
            worker = torch.utils.data.get_worker_info()
            wid = 0 if worker is None else worker.id
            rng = np.random.default_rng((self.seed, self.epoch, idx, wid))
            img, _info = random_augment(img, rng, p_clean=0.2, continuous=True, chain_jpeg_p=0.3)
        else:
            img = img.convert("RGB")
        # 2026-08-30, yun, same (possibly augmented) image feeds both branches
        rgb_view = self.rgb_transform(img)
        freq_view = self.freq_transform(highpass_residual(img))
        return rgb_view, freq_view, y
# end
