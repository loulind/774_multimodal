"""The trainable joint model: one head per modality into a shared space, plus task heads.

    modality input --head--> embedding (embed_dim)      one head per aligned modality
    embeddings     --InfoNCE--  (learned temperature)    contrastive alignment
    embedding      --classifier--> label logits          supervised term (shared across modalities)
    source embedding --regressor--> target features      cross-modal prediction
    embedding      --decoder--> own input                reconstruction / masked denoising

Inputs are frozen-encoder features or preprocessed matrices, so the heads are
the only trainable part in the frozen regime (partial fine-tuning adds the last
backbone blocks; see finetune.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..schema import ConfigSpec
from .heads import MLP
from .losses import normalize


@dataclass
class FitData:
    inputs: dict[str, np.ndarray]                 # modality -> (n, d) model inputs
    labels: np.ndarray | None = None              # int class codes, -1 = unlabelled
    target: np.ndarray | None = None              # regression targets (n, g)
    soft_source: np.ndarray | None = None         # fixed inputs defining soft-InfoNCE targets

    @property
    def n(self) -> int:
        return len(next(iter(self.inputs.values())))

    def subset(self, idx) -> FitData:
        return FitData({k: v[idx] for k, v in self.inputs.items()},
                       None if self.labels is None else self.labels[idx],
                       None if self.target is None else self.target[idx],
                       None if self.soft_source is None else self.soft_source[idx])


class JointModel(nn.Module):
    def __init__(self, cfg: ConfigSpec, in_dims: dict[str, int], n_classes: int = 0,
                 target_dim: int = 0, decode: tuple[str, ...] = ()):
        super().__init__()
        h, t = cfg.head, cfg.training
        self.pair = list(in_dims)
        self.heads = nn.ModuleDict({m: MLP(d, h.embed_dim, h.hidden_dim, h.n_layers, h.dropout)
                                    for m, d in in_dims.items()})
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / t.temperature_init)),
                                        requires_grad=t.learnable_temperature)
        self.classifier = nn.Linear(h.embed_dim, n_classes) if n_classes else None
        self.regressor = MLP(h.embed_dim, target_dim, h.hidden_dim, 2, h.dropout) if target_dim else None
        self.decoders = nn.ModuleDict({m: MLP(h.embed_dim, in_dims[m], h.hidden_dim, 2, h.dropout)
                                       for m in decode})

    def embed(self, name: str, x: torch.Tensor) -> torch.Tensor:
        return self.heads[name](x)

    def scale(self) -> torch.Tensor:
        """Inverse temperature, capped at 100 (temperature >= 0.01) as in CLIP."""
        return self.logit_scale.exp().clamp(max=100.0)

    @property
    def temperature(self) -> float:
        return float(1 / self.scale().detach())

    # --- numpy conveniences for evaluation ------------------------------------------
    def _batched(self, fn, X: np.ndarray, batch: int = 2048) -> np.ndarray:
        dev = next(self.parameters()).device
        was = self.training
        self.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(X), batch):
                out.append(fn(torch.as_tensor(X[s:s + batch], dtype=torch.float32, device=dev)).cpu().numpy())
        self.train(was)
        return np.concatenate(out) if out else np.zeros((0, 0), np.float32)

    def embed_np(self, name: str, X: np.ndarray) -> np.ndarray:
        """L2-normalised embeddings."""
        return self._batched(lambda x: normalize(self.embed(name, x)), X)

    def predict_np(self, source: str, X: np.ndarray) -> np.ndarray:
        return self._batched(lambda x: self.regressor(self.embed(source, x)), X)

    def classify_np(self, name: str, X: np.ndarray) -> np.ndarray:
        return self._batched(lambda x: torch.softmax(self.classifier(normalize(self.embed(name, x))), 1), X)
