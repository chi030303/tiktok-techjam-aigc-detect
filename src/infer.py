# 2026-08-29, tianqi, load frozen linear-probe ckpt and score a folder
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.cifake import ImagePathDataset
from src.models.linear_probe import FrozenLinearProbe
from src.paths import models_root

# end

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224


def _tfm():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class ProbePredictor:
    def __init__(self, ckpt: Path, device: torch.device | None = None, batch: int = 32):
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        if not isinstance(blob, dict) or "head" not in blob or "backbone" not in blob:
            raise SystemExit(f"ckpt must be {{head, backbone}}: {ckpt}")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch = batch
        bb = models_root() / str(blob["backbone"])
        self.model = FrozenLinearProbe(bb).to(self.device)
        self.model.head.load_state_dict(blob["head"])
        self.model.eval()
        self.backbone = str(blob["backbone"])
        print(f"loaded {ckpt}  backbone={self.backbone}  device={self.device}", flush=True)

    @torch.no_grad()
    def predict_dir(self, image_dir: Path) -> list[dict]:
        paths = sorted(
            p
            for p in image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
        )
        if not paths:
            return []
        rows = [(p, 0) for p in paths]
        loader = DataLoader(
            ImagePathDataset(rows, _tfm()),
            batch_size=self.batch,
            shuffle=False,
            num_workers=4,
            pin_memory=self.device.type == "cuda",
        )
        amp = self.device.type == "cuda"
        out: list[dict] = []
        for x, _y, paths_b in loader:
            x = x.to(self.device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp):
                logit = self.model(x)
            prob = logit.float().sigmoid().cpu()
            for path, p in zip(paths_b, prob.tolist()):
                out.append({"image_path": path, "pred": float(p)})
        return out
