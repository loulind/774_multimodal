"""Small trainable networks: projection heads, MLP encoders, task heads.

A projection head maps a frozen encoder's features into the joint space; an
MLP encoder does the same for a matrix or table trained from scratch. Both are
the same module: Linear -> LayerNorm -> GELU -> Dropout blocks, then a final
Linear. LayerNorm keeps the scale of hidden units stable whatever the input
features' scale, which matters when one side is a pretrained feature and the
other a standardised gene-expression vector.
"""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(max(0, n_layers - 1)):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def n_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)
