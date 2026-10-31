"""Small trainable networks: projection heads, MLP encoders, task heads.

A projection head maps a frozen encoder's features into the joint space; an
MLP encoder does the same for a matrix or table trained from scratch. Both are
the same module: Linear -> LayerNorm -> GELU -> Dropout blocks, then a final
Linear. LayerNorm keeps the scale of hidden units stable whatever the input
features' scale, which matters when one side is a pretrained feature and the
other a standardised gene-expression vector.

Residual heads (``residual=True``, input and output of equal size) return the input
plus a learned correction whose last layer starts at zero, so at initialisation the
head is the identity. On a pre-aligned dual encoder this starts training from the
pretrained (zero-shot) space instead of discarding it (the idea of CLIP-Adapter).
"""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512, n_layers: int = 2,
                 dropout: float = 0.1, residual: bool = False):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(max(0, n_layers - 1)):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        self.residual = residual and in_dim == out_dim
        if self.residual:                      # identity at initialisation
            nn.init.zeros_(layers[-1].weight)
            nn.init.zeros_(layers[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return x + y if self.residual else y


def n_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)
