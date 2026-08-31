# 2026-08-29, tianqi, frozen HF backbone + one linear logit (AIGC = 1)
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, CLIPVisionModel

# end


def _pool_features(model_type: str, outputs) -> torch.Tensor:
    # 2026-08-29, tianqi, HF ResNet pooler is [B, C, 1, 1]; CLIP vision pooler is [B, D]
    if model_type in ("resnet", "clip_vision_model"):
        return outputs.pooler_output.flatten(1)
    if hasattr(outputs, "last_hidden_state"):
        h = outputs.last_hidden_state
        if h.dim() == 4:
            return h.mean(dim=(2, 3))
        return h[:, 0]
    raise RuntimeError(f"cannot pool features for {model_type}")
    # end


# 2026-08-29, tianqi, linear or small MLP encoder on frozen feats
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
    # end


class FrozenLinearProbe(nn.Module):
    def __init__(self, backbone_dir: Path, head_kind: str = "linear", interpolate_pos_encoding: bool = False):
        super().__init__()
        top_config = AutoConfig.from_pretrained(backbone_dir, local_files_only=True)
        # 2026-08-29, tianqi, CLIP AutoModel is CLIPModel and needs text ids; image tower only
        if top_config.model_type == "clip":
            self.backbone = CLIPVisionModel.from_pretrained(backbone_dir, local_files_only=True)
        else:
            self.backbone = AutoModel.from_pretrained(backbone_dir, local_files_only=True)
        self.config = self.backbone.config
        # 2026-08-30, tianqi, yun exp4 res336: CLIP interpolates pos-emb for non-224 input
        self._interpolate_pos = bool(interpolate_pos_encoding) and self.config.model_type == "clip_vision_model"
        # end
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        hidden = getattr(self.config, "hidden_size", None)
        if not hidden:
            sizes = getattr(self.config, "hidden_sizes", None)
            hidden = sizes[-1] if sizes else None
        if not hidden:
            raise SystemExit(f"no hidden size in {backbone_dir}")
        # 2026-08-29, tianqi, optional MLP encoder on frozen pooled feats
        self.head = build_head(hidden, head_kind)
        self.head_kind = head_kind
        # end

    def train(self, mode: bool = True):
        super().train(mode)
        # 2026-08-29, tianqi, backbone stays eval even when the probe trains
        self.backbone.eval()
        return self
        # end

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # 2026-08-29, tianqi, freeze-extract once; linear head trains on cached feats
        with torch.no_grad():
            if self._interpolate_pos:
                out = self.backbone(
                    pixel_values=pixel_values, interpolate_pos_encoding=True
                )
            else:
                out = self.backbone(pixel_values=pixel_values)
        return _pool_features(self.config.model_type, out)
        # end

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(pixel_values)).squeeze(-1)
