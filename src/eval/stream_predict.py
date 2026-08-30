# 2026-08-30, tianqi, on-the-fly official transforms so full-val eval does not copy 15x disk
"""Score labeled rows without materializing transform copies.

Used by scripts/run_full_eval.py: 14k official val x 15 settings would be
~200k files if we reused robustness.write_condition. Apply the spec op in
the Dataset instead, with the same seed_for(rel, key) rule as eval.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.eval.transforms import apply_condition, seed_for
from src.infer import ProbePredictor, _tfm

# end


class ConditionPathDataset(Dataset):
    def __init__(
        self,
        rows: list[tuple[Path, int]],
        src_root: Path,
        condition: str,
        input_mode: str = "rgb",
    ):
        self.rows = rows
        self.src_root = Path(src_root)
        self.condition = condition
        self.input_mode = input_mode or "rgb"
        self.transform = _tfm()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        path, y = self.rows[idx]
        path = Path(path)
        with Image.open(path) as im:
            img = im.convert("RGB")
        try:
            rel = path.resolve().relative_to(self.src_root.resolve()).as_posix()
        except ValueError:
            rel = path.name
        img = apply_condition(img, self.condition, seed=seed_for(rel, self.condition))
        if self.input_mode not in ("rgb", "clean"):
            # 2026-08-30, tianqi, freq views only for fft/highpass ckpts
            from src.data.freq import apply_input_mode

            img = apply_input_mode(img, self.input_mode)
        return self.transform(img), int(y), str(path)


def predict_dataset(
    predictor: ProbePredictor,
    dataset: Dataset,
    workers: int = 4,
) -> list[dict]:
    if len(dataset) == 0:
        return []
    loader = DataLoader(
        dataset,
        batch_size=predictor.batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=predictor.device.type == "cuda",
    )
    amp = predictor.device.type == "cuda"
    out: list[dict] = []
    predictor.model.eval()
    with torch.no_grad():
        for x, _y, paths_b in loader:
            x = x.to(predictor.device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp):
                logit = predictor.model(x)
            prob = logit.float().sigmoid().cpu()
            for path, p, y in zip(paths_b, prob.tolist(), _y.tolist()):
                out.append({"image_path": path, "pred": float(p), "y": int(y)})
    return out


def predict_labeled(
    predictor: ProbePredictor,
    rows: list[tuple[Path, int]],
    src_root: Path,
    condition: str = "clean",
    workers: int = 4,
) -> list[dict]:
    if not rows:
        return []
    ds = ConditionPathDataset(rows, src_root, condition, input_mode=predictor.input_mode)
    return predict_dataset(predictor, ds, workers=workers)


# 2026-08-30, tianqi, SID val reals stay in parquet; pair with EvalGEN without dumping 20k jpegs
class SidValRealConditionDataset(Dataset):
    """On-the-fly SID validation reals; no disk export required."""

    def __init__(
        self,
        condition: str,
        input_mode: str = "rgb",
        max_images: int | None = None,
        seed: int = 0,
    ):
        import random

        from src.eval.evalgen_pool import load_sid_hf

        ds = load_sid_hf("validation")
        keep = [i for i, y in enumerate(ds["label"]) if int(y) == 0]
        if max_images is not None and len(keep) > max_images:
            keep = random.Random(seed).sample(keep, max_images)
        self.ds = ds
        self.keep = keep
        self.condition = condition
        self.input_mode = input_mode or "rgb"
        self.transform = _tfm()

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
        tag = str(ex.get("img_id") or src_i)
        rel = f"sid_val_real/{tag}.jpg"
        img = apply_condition(img, self.condition, seed=seed_for(rel, self.condition))
        if self.input_mode not in ("rgb", "clean"):
            from src.data.freq import apply_input_mode

            img = apply_input_mode(img, self.input_mode)
        return self.transform(img), 0, f"sid://validation/{tag}"
# end
