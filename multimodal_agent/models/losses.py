"""Training objectives for representation learning.

  infonce               Symmetric InfoNCE / CLIP loss (Oord et al. 2018; Radford et al.
                        2021). In a batch of n pairs, row i of the similarity matrix is
                        a multinomial over the n candidates and the true partner is the
                        target; averaged over both directions. Minimising it maximises a
                        lower bound on the mutual information between the modalities.
                        The temperature (how peaked the multinomial is) is learned.
  soft_infonce          Same, but the target for row i is a distribution over the batch
                        that gives near-identical samples a share of the mass. Targets come
                        from the *input* similarity of the redundant modality (fixed, not
                        learned, so they cannot collapse), sharpened by a small temperature:
                        a sample and its near-twin are not treated as a positive and a
                        negative.
  supervised_ce         Cross-entropy of a shared linear classifier on each modality's
                        normalised embedding; unlabelled samples are skipped.
  cross_modal_regression  Mean squared error of a head predicting the target modality's
                        standardised features from the source modality's embedding.
  reconstruction        An autoencoder term: rebuild a modality's input from its embedding.
  masked_denoising      Self-supervised: hide a random 30% of input features, rebuild them.
  ssl_simclr            Self-supervised on fixed features: two dropout-perturbed views of
                        the same sample are positives (as in SimCSE, Gao et al. 2021).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

MASK_FRACTION = 0.3
VIEW_DROPOUT = 0.2


def normalize(z: torch.Tensor) -> torch.Tensor:
    return F.normalize(z, dim=-1)


def soft_targets(x: torch.Tensor, tau: float) -> torch.Tensor:
    """Row-stochastic targets from the cosine similarity of fixed inputs."""
    xn = normalize(x)
    return F.softmax(xn @ xn.T / tau, dim=1)


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * F.log_softmax(logits, dim=1)).sum(1).mean()


def info_nce(za: torch.Tensor, zb: torch.Tensor, scale: torch.Tensor,
             targets: torch.Tensor | None = None) -> torch.Tensor:
    logits = scale * normalize(za) @ normalize(zb).T
    if targets is None:
        y = torch.arange(len(za), device=za.device)
        return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y))
    # targets come from a symmetric similarity, so the same rows serve both directions
    return 0.5 * (soft_cross_entropy(logits, targets) + soft_cross_entropy(logits.T, targets))


def supervised_ce(zs: list[torch.Tensor], y: torch.Tensor, classifier: torch.nn.Module) -> torch.Tensor:
    m = y >= 0
    if not bool(m.any()):
        return zs[0].sum() * 0.0
    return torch.stack([F.cross_entropy(classifier(normalize(z[m])), y[m]) for z in zs]).mean()


def regression(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def masked_denoising(x: torch.Tensor, embed, decode, generator: torch.Generator | None = None) -> torch.Tensor:
    mask = torch.rand(x.shape, generator=generator, device="cpu").to(x.device) < MASK_FRACTION
    rec = decode(embed(x.masked_fill(mask, 0.0)))
    return F.mse_loss(rec[mask], x[mask]) if bool(mask.any()) else rec.sum() * 0.0


def dropout_views(x: torch.Tensor, embed, scale: torch.Tensor) -> torch.Tensor:
    z1 = embed(F.dropout(x, VIEW_DROPOUT, training=True))
    z2 = embed(F.dropout(x, VIEW_DROPOUT, training=True))
    return info_nce(z1, z2, scale)
