"""
三模态特征投影层: 将原始 T/A/V 特征 (不同维度) 投到统一 d_model.
可作为 MSAClassifier 的可选前端。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FeatureProjection(nn.Module):
    """Linear + LayerNorm + Dropout + GELU"""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D_in) -> (B, L, D_out)
        return self.drop(self.act(self.norm(self.proj(x))))