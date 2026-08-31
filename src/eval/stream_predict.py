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
from src.infer import ProbePredictor, _freq_tfm, _tfm

# end


class ConditionPathDataset(Dataset):
    def __init__(
        self,
        rows: list[tuple[Path, int]],
        src_root: Path,
        condition: str,
        input_mode: str = "rgb",
        image_size: int = 224,
    ):
        self.rows = rows
        self.src_root = Path(src_root)
        self.condition = condition
        self.input_mode = input_mode or "rgb"
        # 2026-08-30, tianqi, match ckpt image_size (yun exp4 res336 is 336)
        self.image_size = int(image_size)
        self.transform = _tfm(self.image_size)
        # end

    def __len__(self) -> int:
        return len(self.rows)

    def _open_conditioned(self, idx: int):
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
            from src.data.freq import apply_input_mode

            img = apply_input_mode(img, self.input_mode)
        return img, int(y), str(path)

    def __getitem__(self, idx: int):
        img, y, path = self._open_conditioned(idx)
        return self.transform(img), y, path


# 2026-08-30, tianqi, yun exp3 dual-branch: RGB + highpass residual per image
class DualBranchConditionDataset(ConditionPathDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq_transform = _freq_tfm(self.image_size)

    def __getitem__(self, idx: int):
        from src.data.freq import highpass_residual

        img, y, path = self._open_conditioned(idx)
        return self.transform(img), self.freq_transform(highpass_residual(img)), y, path
# end


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
    dual = bool(getattr(predictor, "dual", False))
    with torch.no_grad():
        for batch in loader:
            # 2026-08-30, tianqi, dual-branch batches are (rgb, freq, y, path)
            if dual:
                rgb, freq, _y, paths_b = batch
                rgb = rgb.to(predictor.device, non_blocking=True)
                freq = freq.to(predictor.device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=amp):
                    logit = predictor.model(rgb, freq)
            else:
                x, _y, paths_b = batch
                x = x.to(predictor.device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=amp):
                    logit = predictor.model(x)
            # end
            prob = logit.float().sigmoid().cpu()
            for path, p, y in zip(paths_b, prob.tolist(), _y.tolist()):
                out.append({"image_path": path, "pred": float(p), "y": int(y)})
    return out


# 2026-08-31, tianqi, average logits of two frozen/unfreeze CLIP probes (not dual-branch)
def predict_fused_dataset(
    pred_a: ProbePredictor,
    pred_b: ProbePredictor,
    dataset: Dataset,
    workers: int = 4,
    weight: float = 0.5,
) -> list[dict]:
    if getattr(pred_a, "dual", False) or getattr(pred_b, "dual", False):
        raise SystemExit("logit fusion is for RGB probes, not dual_branch_clip")
    if len(dataset) == 0:
        return []
    loader = DataLoader(
        dataset,
        batch_size=min(pred_a.batch, pred_b.batch),
        shuffle=False,
        num_workers=workers,
        pin_memory=pred_a.device.type == "cuda",
    )
    amp = pred_a.device.type == "cuda"
    w = float(weight)
    out: list[dict] = []
    pred_a.model.eval()
    pred_b.model.eval()
    with torch.no_grad():
        for x, _y, paths_b in loader:
            x = x.to(pred_a.device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp):
                logit = w * pred_a.model(x) + (1.0 - w) * pred_b.model(x)
            prob = logit.float().sigmoid().cpu()
            for path, p, y in zip(paths_b, prob.tolist(), _y.tolist()):
                out.append({"image_path": path, "pred": float(p), "y": int(y)})
    return out


def predict_fused_labeled(
    pred_a: ProbePredictor,
    pred_b: ProbePredictor,
    rows: list[tuple[Path, int]],
    src_root: Path,
    condition: str = "clean",
    workers: int = 4,
    weight: float = 0.5,
) -> list[dict]:
    if not rows:
        return []
    size = int(getattr(pred_a, "image_size", 224) or 224)
    ds = ConditionPathDataset(
        rows=rows,
        src_root=src_root,
        condition=condition,
        input_mode=pred_a.input_mode,
        image_size=size,
    )
    return predict_fused_dataset(pred_a, pred_b, ds, workers=workers, weight=weight)
# end


def predict_labeled(
    predictor: ProbePredictor,
    rows: list[tuple[Path, int]],
    src_root: Path,
    condition: str = "clean",
    workers: int = 4,
) -> list[dict]:
    if not rows:
        return []
    size = int(getattr(predictor, "image_size", 224) or 224)
    kw = dict(
        rows=rows,
        src_root=src_root,
        condition=condition,
        input_mode=predictor.input_mode,
        image_size=size,
    )
    if getattr(predictor, "dual", False):
        ds = DualBranchConditionDataset(**kw)
    else:
        ds = ConditionPathDataset(**kw)
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


# 2026-08-31, tianqi, EvalGEN sid-val reals for dual-branch ckpts
class DualSidValRealConditionDataset(SidValRealConditionDataset):
    def __init__(self, *args, image_size: int = 224, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_size = int(image_size)
        self.transform = _tfm(self.image_size)
        self.freq_transform = _freq_tfm(self.image_size)

    def __getitem__(self, idx: int):
        from src.data.freq import highpass_residual

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
        return (
            self.transform(img),
            self.freq_transform(highpass_residual(img)),
            0,
            f"sid://validation/{tag}",
        )
# end
