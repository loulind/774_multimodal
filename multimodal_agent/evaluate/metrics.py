"""Small metric functions shared by screening, training and evaluation."""

from __future__ import annotations

import numpy as np


def l2n(Z: np.ndarray) -> np.ndarray:
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)


def recall_at_k(Za: np.ndarray, Zb: np.ndarray, k: int = 10) -> float:
    """Mean of a->b and b->a recall@k for row-aligned pairs (cosine similarity)."""
    a = Za / np.maximum(np.linalg.norm(Za, axis=1, keepdims=True), 1e-9)
    b = Zb / np.maximum(np.linalg.norm(Zb, axis=1, keepdims=True), 1e-9)
    S = a @ b.T
    diag = np.diag(S)
    r_ab = (S > diag[:, None]).sum(1) < k
    r_ba = (S > diag[None, :]).sum(0) < k
    return float((r_ab.mean() + r_ba.mean()) / 2)


def ridge_r2(Xtr, Ytr, Xva, Yva) -> float:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xtr)
    m = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(sc.transform(Xtr), Ytr)
    pred = m.predict(sc.transform(Xva))
    sse = ((Yva - pred) ** 2).sum()
    sst = ((Yva - Ytr.mean(0)) ** 2).sum()
    return float(1 - sse / max(sst, 1e-12))


def columnwise_pearson(P: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Pearson correlation between matching columns of predictions and truth."""
    Pc, Tc = P - P.mean(0), T - T.mean(0)
    den = np.sqrt((Pc ** 2).sum(0) * (Tc ** 2).sum(0))
    return np.divide((Pc * Tc).sum(0), den, out=np.zeros(P.shape[1]), where=den > 0)
