"""Geometry of the joint space (computed on L2-normalised embeddings).

  alignment    mean squared distance between partners (Wang & Isola 2020); lower =
               partners closer.
  uniformity   log of the mean Gaussian potential exp(-2 |u - v|^2) over pairs within a
               modality (Wang & Isola 2020); lower (more negative) = embeddings spread
               over the sphere, i.e. more information preserved.
  modality gap distance between the two modalities' mean embeddings (Liang et al. 2022);
               a large gap means the modalities occupy separate regions of the space.
  effective rank  exp(entropy of the normalised singular values) of the embedding matrix
               (Roy & Vetterli 2007); far below the dimension signals collapse onto a few
               directions.
"""

from __future__ import annotations

import numpy as np

from .metrics import l2n

MAX_PAIRS = 2000


def alignment(Za: np.ndarray, Zb: np.ndarray) -> float:
    return float(np.mean(((l2n(Za) - l2n(Zb)) ** 2).sum(1)))


def uniformity(Z: np.ndarray, seed: int = 0) -> float:
    z = l2n(Z)
    if len(z) > MAX_PAIRS:
        z = z[np.random.default_rng(seed).choice(len(z), MAX_PAIRS, replace=False)]
    sq = np.clip(2 - 2 * z @ z.T, 0, None)
    iu = np.triu_indices(len(z), 1)
    return float(np.log(np.mean(np.exp(-2 * sq[iu]))))


def modality_gap(Za: np.ndarray, Zb: np.ndarray) -> float:
    return float(np.linalg.norm(l2n(Za).mean(0) - l2n(Zb).mean(0)))


def effective_rank(Z: np.ndarray) -> float:
    s = np.linalg.svd(l2n(Z) - l2n(Z).mean(0), compute_uv=False)
    p = s / max(s.sum(), 1e-12)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def geometry(Za: np.ndarray, Zb: np.ndarray | None, seed: int = 0) -> dict:
    out = {"uniformity_a": uniformity(Za, seed), "effective_rank_a": effective_rank(Za), "dim": int(Za.shape[1])}
    if Zb is not None:
        out.update(alignment=alignment(Za, Zb), uniformity_b=uniformity(Zb, seed),
                   effective_rank_b=effective_rank(Zb), modality_gap=modality_gap(Za, Zb))
    return out
