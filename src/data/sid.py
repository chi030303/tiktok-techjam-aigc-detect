# 2026-08-29, tianqi, SID_Set parquet loader; label 0=real 1=full synthetic; drop tampered=2
"""SID_Set (HF parquet). Binary AIGC: keep real vs fully synthetic, drop tampered."""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from src.paths import data_root

# end

BINARY_OK = {0, 1}


def load_sid_hf(split: str):
    from datasets import load_dataset

    root = data_root() / "sid_set"
    if not root.is_dir():
        raise FileNotFoundError(root)
    try:
        return load_dataset(str(root), split=split)
    except Exception:
        data = root / "data"
        files = {
            "train": sorted(str(p) for p in data.glob("train-*.parquet")),
            "validation": sorted(str(p) for p in data.glob("validation-*.parquet")),
        }
        return load_dataset("parquet", data_files=files, split=split)


def _balanced_keep(labels: list[int], n_total: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    real = [i for i, y in enumerate(labels) if y == 0]
    fake = [i for i, y in enumerate(labels) if y == 1]
    half = max(1, n_total // 2)
    real_s = real if len(real) <= half else rng.sample(real, half)
    fake_s = fake if len(fake) <= half else rng.sample(fake, half)
    keep = real_s + fake_s
    rng.shuffle(keep)
    return keep


class SidHfDataset(Dataset):
    def __init__(
        self,
        split: str,
        transform,
        augment: bool = False,
        expand_ops: bool = False,
        seed: int = 0,
        max_images: int | None = None,
        input_mode: str = "rgb",
        extra_rows: list[tuple[Path, int]] | None = None,
        replace_sid_fakes: bool = True,
    ):
        hf_split = "train" if split in ("train", "train_set") else "validation"
        self.ds = load_sid_hf(hf_split)
        labels = [int(y) for y in self.ds["label"]]
        keep = [i for i, y in enumerate(labels) if y in BINARY_OK]
        if max_images is not None:
            keep = _balanced_keep(labels, max_images, seed)
        self.keep = keep
        self.transform = transform
        self.augment = augment
        self.expand_ops = expand_ops
        self.seed = seed
        self.input_mode = input_mode or "rgb"
        self.epoch = 0
        if expand_ops:
            from src.transforms.augment import OFFICIAL_OPS

            self.views = (None,) + tuple(OFFICIAL_OPS)
        else:
            self.views = None
        # 2026-08-31, tianqi, D3: mix disk fakes into full SID; drop equal SID FLUX
        self.extra_rows: list[tuple[Path, int]] = list(extra_rows or [])
        self.n_sid_fakes_dropped = 0
        if self.extra_rows and replace_sid_fakes:
            n_drop = sum(1 for _p, y in self.extra_rows if int(y) == 1)
            fake_idx = [i for i in self.keep if labels[i] == 1]
            real_idx = [i for i in self.keep if labels[i] == 0]
            rng = random.Random(int(seed) ^ 0xD30831)
            rng.shuffle(fake_idx)
            n_drop = min(n_drop, len(fake_idx))
            fake_keep = fake_idx[n_drop:]
            self.keep = real_idx + fake_keep
            rng.shuffle(self.keep)
            self.n_sid_fakes_dropped = n_drop
            print(
                f"  sid-mix replace  drop_sid_fakes={n_drop} extra={len(self.extra_rows)} "
                f"sid_keep={len(self.keep)}",
                flush=True,
            )
        # end

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        n = len(self.keep) + len(self.extra_rows)
        if self.expand_ops:
            return n * len(self.views)
        return n

    def __getitem__(self, idx: int):
        if self.expand_ops:
            n_views = len(self.views)
            row_i = idx // n_views
            view = self.views[idx % n_views]
        else:
            row_i = idx
            view = None
        n_sid = len(self.keep)
        # 2026-08-31, tianqi, extra mixin rows are disk paths after SID parquet keep
        if row_i >= n_sid:
            path, y = self.extra_rows[row_i - n_sid]
            with Image.open(path) as im:
                img = im.convert("RGB")
            tag = str(path)
        else:
            src_i = int(self.keep[row_i])
            ex = self.ds[src_i]
            img = ex["image"]
            if not isinstance(img, Image.Image):
                img = Image.open(img).convert("RGB")
            else:
                img = img.convert("RGB")
            y = int(ex["label"])
            tag = str(ex.get("img_id") or src_i)
        # end
        if self.expand_ops:
            import numpy as np
            import torch

            from src.transforms.augment import apply_one_op

            if view is not None:
                worker = torch.utils.data.get_worker_info()
                wid = 0 if worker is None else worker.id
                rng = np.random.default_rng((self.seed, self.epoch, idx, wid))
                img, _info = apply_one_op(img, rng, op=view, continuous=False, chain_jpeg_p=0.0)
        elif self.augment:
            import numpy as np
            import torch

            from src.transforms.augment import random_augment

            worker = torch.utils.data.get_worker_info()
            wid = 0 if worker is None else worker.id
            rng = np.random.default_rng((self.seed, self.epoch, idx, wid))
            img, _info = random_augment(img, rng, p_clean=0.2, continuous=True, chain_jpeg_p=0.3)
        if self.input_mode and self.input_mode not in ("rgb", "clean"):
            from src.data.freq import apply_input_mode

            img = apply_input_mode(img, self.input_mode)
        x = self.transform(img)
        return x, y, tag
# end
