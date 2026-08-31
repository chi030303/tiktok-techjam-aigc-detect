# 2026-08-29, tianqi, load frozen linear-probe ckpt and score a folder
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.cifake import ImagePathDataset
from src.models.dual_branch_probe import DualBranchProbe
from src.models.linear_probe import FrozenLinearProbe
from src.models.partial_unfreeze_probe import PartialUnfreezeClipProbe
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


def _freq_tfm(size: int = IMAGE_SIZE):
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])


class ProbePredictor:
    def __init__(self, ckpt: Path, device: torch.device | None = None, batch: int = 32):
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        if not isinstance(blob, dict) or "head" not in blob or "backbone" not in blob:
            raise SystemExit(f"ckpt must be {{head, backbone}}: {ckpt}")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch = batch
        bb = models_root() / str(blob["backbone"])
        head_kind = blob.get("head_kind") or "linear"
        self.input_mode = blob.get("input_mode") or "rgb"
        # 2026-08-30, tianqi, yun exp1/3/4 ckpts: image_size, unfreeze, dual-branch
        self.image_size = int(blob.get("image_size") or IMAGE_SIZE)
        self.kind = str(blob.get("kind") or "frozen_linear")
        self.dual = self.kind == "dual_branch_clip"
        interp = self.image_size != IMAGE_SIZE
        if self.dual:
            self.model = DualBranchProbe(
                bb, freq_dim=int(blob.get("freq_dim") or 128), head_kind=head_kind
            ).to(self.device)
            self.model.freq_cnn.load_state_dict(blob["freq_cnn_state_dict"])
        elif self.kind == "partial_unfreeze_clip":
            self.model = PartialUnfreezeClipProbe(
                bb,
                n_unfreeze=int(blob["n_unfreeze"]),
                head_kind=head_kind,
                interpolate_pos_encoding=interp,
                unfreeze_from=str(blob.get("unfreeze_from") or "last"),
            ).to(self.device)
            self.model.backbone.load_state_dict(blob["backbone_state_dict"])
        else:
            self.model = FrozenLinearProbe(
                bb, head_kind=head_kind, interpolate_pos_encoding=interp
            ).to(self.device)
        # end
        self.model.head.load_state_dict(blob["head"])
        self.model.eval()
        self.backbone = str(blob["backbone"])
        print(
            f"loaded {ckpt}  backbone={self.backbone}  head={head_kind} "
            f"input={self.input_mode}  image_size={self.image_size}  kind={self.kind}  "
            f"device={self.device}",
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
            ImagePathDataset(rows, _tfm(self.image_size), input_mode=self.input_mode),
            batch_size=self.batch,
            shuffle=False,
            num_workers=4,
            pin_memory=self.device.type == "cuda",
        )
        if self.dual:
            raise SystemExit("dual_branch_clip: use scripts/run_full_eval.py, not predict.py")
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
