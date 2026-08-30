# 2026-08-30, yun, model-track predictor: image_size + partial_unfreeze_clip ckpt support
"""Load a model-track checkpoint (baseline/exp4/exp1/exp2 style) and score a folder.

Separate from src/infer.py (which CIFAKE recipes' checkpoints use) so this
branch adds checkpoint kinds -- {image_size}-aware resize/pos-emb interpolation,
and partial_unfreeze_clip's fine-tuned backbone weights -- without touching the
predictor other experiments already depend on.

dual_branch_clip checkpoints need two views (rgb + highpass) per image and are
out of scope here; see scripts/eval_dualbranch.py.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.cifake import ImagePathDataset
from src.models.partial_unfreeze_probe import PartialUnfreezeClipProbe
from src.models.sid_linear_probe import FrozenLinearProbe
from src.paths import models_root

# end

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224


def _tfm(size: int = IMAGE_SIZE):
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class SidProbePredictor:
    def __init__(self, ckpt: Path, device: torch.device | None = None, batch: int = 32):
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        if not isinstance(blob, dict) or "head" not in blob or "backbone" not in blob:
            raise SystemExit(f"ckpt must be {{head, backbone}}: {ckpt}")
        if blob.get("kind") == "dual_branch_clip":
            raise SystemExit(
                f"{ckpt} is a dual_branch_clip ckpt; needs two views per image, "
                "use scripts/eval_dualbranch.py instead"
            )
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch = batch
        bb = models_root() / str(blob["backbone"])
        head_kind = blob.get("head_kind") or "linear"
        self.image_size = int(blob.get("image_size") or IMAGE_SIZE)
        interp = self.image_size != IMAGE_SIZE
        if blob.get("kind") == "partial_unfreeze_clip":
            self.model = PartialUnfreezeClipProbe(
                bb,
                n_unfreeze=int(blob["n_unfreeze"]),
                head_kind=head_kind,
                interpolate_pos_encoding=interp,
            ).to(self.device)
            self.model.backbone.load_state_dict(blob["backbone_state_dict"])
        else:
            self.model = FrozenLinearProbe(
                bb, head_kind=head_kind, interpolate_pos_encoding=interp
            ).to(self.device)
        self.model.head.load_state_dict(blob["head"])
        self.model.eval()
        self.backbone = str(blob["backbone"])
        print(
            f"loaded {ckpt}  backbone={self.backbone}  head={head_kind}  "
            f"image_size={self.image_size}  device={self.device}",
            flush=True,
        )

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
            ImagePathDataset(rows, _tfm(self.image_size)),
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
# end
