# 2026-08-30, yun, exp1: unfreeze the last N CLIP vision-tower blocks + differential LR
"""CLIP-B/16 vision tower with only its last N encoder blocks (+ final LN) trainable.

Everything before that stays frozen and in eval() (no dropout/LN-stat drift).
No torch.no_grad() around the backbone call: PyTorch's autograd already skips
building a graph through the frozen prefix (no leaf there requires grad), so
gradients only flow through the unfrozen suffix -- this is the "no feat-cache"
path the spec calls for, but it costs no extra compute on the frozen part.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import CLIPVisionModel

from src.models.sid_linear_probe import build_head

# end


class PartialUnfreezeClipProbe(nn.Module):
    def __init__(
        self,
        backbone_dir: Path,
        n_unfreeze: int = 2,
        head_kind: str = "linear",
        interpolate_pos_encoding: bool = False,
    ):
        super().__init__()
        self.backbone = CLIPVisionModel.from_pretrained(backbone_dir, local_files_only=True)
        self._interpolate_pos = bool(interpolate_pos_encoding)
        layers = self.backbone.encoder.layers
        if not (0 < n_unfreeze < len(layers)):
            raise SystemExit(f"n_unfreeze must be in (0, {len(layers)}), got {n_unfreeze}")
        for p in self.backbone.parameters():
            p.requires_grad = False
        self._unfrozen = list(layers[-n_unfreeze:]) + [self.backbone.post_layernorm]
        for m in self._unfrozen:
            for p in m.parameters():
                p.requires_grad = True
        self.n_unfreeze = n_unfreeze
        self.backbone.eval()
        hidden = self.backbone.config.hidden_size
        self.head = build_head(hidden, head_kind)

    def train(self, mode: bool = True):
        super().train(mode)
        # 2026-08-30, yun, frozen prefix (embeddings + early blocks) always stays eval()
        self.backbone.eval()
        if mode:
            for m in self._unfrozen:
                m.train()
        return self

    def backbone_trainable_parameters(self):
        for m in self._unfrozen:
            yield from m.parameters()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=pixel_values, interpolate_pos_encoding=self._interpolate_pos)
        feat = out.pooler_output
        return self.head(feat).squeeze(-1)
# end
