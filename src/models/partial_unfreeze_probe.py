# 2026-08-30, tianqi, load yun exp1 partial-unfreeze CLIP probe for official_val eval
"""CLIP-B vision tower with last N encoder blocks trainable (eval still uses the saved weights)."""

from __future__ import annotations

from pathlib import Path

import torch.nn as nn
from transformers import CLIPVisionModel

from src.models.linear_probe import build_head

# end


class PartialUnfreezeClipProbe(nn.Module):
    def __init__(
        self,
        backbone_dir: Path,
        n_unfreeze: int = 2,
        head_kind: str = "linear",
        interpolate_pos_encoding: bool = False,
        unfreeze_from: str = "last",
    ):
        super().__init__()
        self.backbone = CLIPVisionModel.from_pretrained(backbone_dir, local_files_only=True)
        self._interpolate_pos = bool(interpolate_pos_encoding)
        layers = self.backbone.encoder.layers
        if not (0 < n_unfreeze < len(layers)):
            raise SystemExit(f"n_unfreeze must be in (0, {len(layers)}), got {n_unfreeze}")
        side = (unfreeze_from or "last").lower()
        if side not in ("last", "first"):
            raise SystemExit(f"unfreeze_from must be last|first, got {unfreeze_from!r}")
        for p in self.backbone.parameters():
            p.requires_grad = False
        # 2026-08-31, tianqi, first-N = encoder blocks 0..N-1 (no post-LN); last-N keeps post-LN
        if side == "first":
            self._unfrozen = list(layers[:n_unfreeze])
        else:
            self._unfrozen = list(layers[-n_unfreeze:]) + [self.backbone.post_layernorm]
        # end
        for m in self._unfrozen:
            for p in m.parameters():
                p.requires_grad = True
        self.n_unfreeze = n_unfreeze
        self.unfreeze_from = side
        self.backbone.eval()
        hidden = self.backbone.config.hidden_size
        self.head = build_head(hidden, head_kind)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        if mode:
            for m in self._unfrozen:
                m.train()
        return self

    def forward(self, pixel_values):
        out = self.backbone(
            pixel_values=pixel_values, interpolate_pos_encoding=self._interpolate_pos
        )
        return self.head(out.pooler_output).squeeze(-1)

    def backbone_trainable_parameters(self):
        # 2026-08-31, tianqi, Adam group for unfrozen CLIP blocks + final LN
        for m in self._unfrozen:
            yield from m.parameters()
        # end
# end
