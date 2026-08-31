# 2026-08-30, tianqi, load yun exp3 dual-branch CLIP probe for official_val eval
"""Frozen CLIP-B RGB branch + shallow CNN on highpass residual."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import CLIPVisionModel

from src.models.linear_probe import build_head

# end


def build_freq_cnn(out_dim: int = 128) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, 3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, out_dim, 3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )


class DualBranchProbe(nn.Module):
    def __init__(self, backbone_dir: Path, freq_dim: int = 128, head_kind: str = "linear"):
        super().__init__()
        self.backbone = CLIPVisionModel.from_pretrained(backbone_dir, local_files_only=True)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.freq_cnn = build_freq_cnn(freq_dim)
        hidden = self.backbone.config.hidden_size
        self.freq_dim = freq_dim
        self.head = build_head(hidden + freq_dim, head_kind)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, rgb: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.backbone(pixel_values=rgb)
        clip_feat = out.pooler_output
        freq_feat = self.freq_cnn(freq)
        combined = torch.cat([clip_feat, freq_feat], dim=1)
        return self.head(combined).squeeze(-1)
# end
