# 2026-08-30, yun, model-track extension of FrozenLinearProbe: head_kind + input resolution
"""Frozen HF backbone + a configurable head (linear or small MLP).

Kept separate from src/models/linear_probe.py (which the CIFAKE recipes use
as-is) so the model-ablation branch adds capability without touching code
other experiments already depend on. Same CLIP-vision-tower fix as
linear_probe.py: a full CLIP checkpoint loads as CLIPModel via AutoModel,
whose forward requires text input_ids we don't have, so CLIP loads its
image tower directly instead.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, CLIPVisionModel

# end


def _pool_features(model_type: str, outputs) -> torch.Tensor:
    if model_type in ("resnet", "clip_vision_model"):
        return outputs.pooler_output.flatten(1)
    if hasattr(outputs, "last_hidden_state"):
        h = outputs.last_hidden_state
        if h.dim() == 4:
            return h.mean(dim=(2, 3))
        return h[:, 0]
    raise RuntimeError(f"cannot pool features for {model_type}")


def build_head(hidden: int, head_kind: str = "linear") -> nn.Module:
    kind = (head_kind or "linear").lower()
    if kind == "linear":
        return nn.Linear(hidden, 1)
    if kind == "mlp":
        return nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
    raise SystemExit(f"unknown head {head_kind!r}")


class FrozenLinearProbe(nn.Module):
    def __init__(
        self,
        backbone_dir: Path,
        head_kind: str = "linear",
        interpolate_pos_encoding: bool = False,
    ):
        super().__init__()
        top_config = AutoConfig.from_pretrained(backbone_dir, local_files_only=True)
        if top_config.model_type == "clip":
            self.backbone = CLIPVisionModel.from_pretrained(backbone_dir, local_files_only=True)
        else:
            self.backbone = AutoModel.from_pretrained(backbone_dir, local_files_only=True)
        self.config = self.backbone.config
        # 2026-08-30, yun, exp4: feed >native-resolution input; CLIP interpolates its own pos-emb
        self._interpolate_pos = bool(interpolate_pos_encoding) and self.config.model_type == "clip_vision_model"
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        hidden = getattr(self.config, "hidden_size", None)
        if not hidden:
            sizes = getattr(self.config, "hidden_sizes", None)
            hidden = sizes[-1] if sizes else None
        if not hidden:
            raise SystemExit(f"no hidden size in {backbone_dir}")
        self.head = build_head(hidden, head_kind)
        self.head_kind = head_kind

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self._interpolate_pos:
                out = self.backbone(pixel_values=pixel_values, interpolate_pos_encoding=True)
            else:
                out = self.backbone(pixel_values=pixel_values)
        return _pool_features(self.config.model_type, out)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(pixel_values)).squeeze(-1)
# end
