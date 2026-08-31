# 2026-08-29, tianqi, cache frozen backbone feats; 10x data must not re-extract each epoch
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.paths import feat_cache_path

# end


def cache_path(
    backbone: str,
    split: str,
    n: int,
    seed: int,
    size: int = 224,
    source: str = "cifake",
    input_mode: str = "rgb",
) -> Path:
    return feat_cache_path(backbone, split, n, seed, size, source=source, input_mode=input_mode)


def load_cached(path: Path) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not path.is_file():
        return None
    blob = torch.load(path, map_location="cpu", weights_only=True)
    return blob["feat"], blob["y"]


def save_cached(path: Path, feat: torch.Tensor, y: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"feat": feat.contiguous(), "y": y.contiguous()}, path)


@torch.no_grad()
def extract_features(
    model,
    rows: list,
    transform,
    batch: int,
    workers: int,
    device: torch.device,
    tag: str,
    input_mode: str = "rgb",
    dataset=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from src.data.cifake import ImagePathDataset

    ds = dataset if dataset is not None else ImagePathDataset(rows, transform, input_mode=input_mode)
    loader_kw: dict = dict(
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
    )
    if workers > 0:
        loader_kw["prefetch_factor"] = 4
    loader = DataLoader(ds, **loader_kw)
    amp = device.type == "cuda"
    feats: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    n_batches = max(1, len(loader))
    model.eval()
    for step, (x, y, _) in enumerate(loader, 1):
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            f = model.encode(x)
        feats.append(f.detach().to(dtype=torch.float16, device="cpu"))
        labels.append(y.cpu())
        if step == 1 or step % 100 == 0 or step == n_batches:
            print(f"  extract {tag} {step}/{n_batches}", flush=True)
    return torch.cat(feats, 0), torch.cat(labels, 0)
