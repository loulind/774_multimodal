"""Linear alignment: ridge-regularised canonical correlation analysis (CCA).

CCA finds pairs of linear projections, one per modality, whose projected values
are maximally correlated (Hotelling 1936). It is the linear, closed-form
counterpart of contrastive alignment, so it serves three roles: screening
candidate encoders, the "linear_alignment" strategy when pairs are too few to
train heads, and the baseline that shows whether a trained alignment adds
anything beyond a linear one. Each view is first reduced by PCA and the
covariances are ridge-regularised, which keeps the estimate stable when
features outnumber samples.
"""

from __future__ import annotations

import numpy as np


def _inv_sqrt(C: np.ndarray, reg: float) -> np.ndarray:
    C = C + reg * np.trace(C) / len(C) * np.eye(len(C))
    w, V = np.linalg.eigh(C)
    return V @ np.diag(1 / np.sqrt(np.clip(w, 1e-12, None))) @ V.T


class CCA:
    """Ridge-regularised CCA on PCA-reduced views (closed form)."""

    def __init__(self, k: int = 32, reg: float = 1e-2, pre: int = 64):
        self.k, self.reg, self.pre = k, reg, pre

    def fit(self, X: np.ndarray, Y: np.ndarray) -> CCA:
        from sklearn.decomposition import PCA

        n = len(X)
        self.px = PCA(min(self.pre, X.shape[1], n - 1), random_state=0).fit(X)
        self.py = PCA(min(self.pre, Y.shape[1], n - 1), random_state=0).fit(Y)
        A, B = self.px.transform(X), self.py.transform(Y)
        Wa, Wb = _inv_sqrt(A.T @ A / n, self.reg), _inv_sqrt(B.T @ B / n, self.reg)
        U, s, Vt = np.linalg.svd(Wa @ (A.T @ B / n) @ Wb)
        k = min(self.k, len(s))
        self.Ua, self.Ub, self.corr = Wa @ U[:, :k], Wb @ Vt.T[:, :k], s[:k]
        return self

    def transform(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.px.transform(X) @ self.Ua, self.py.transform(Y) @ self.Ub


class LinearAligner:
    """CCA as a fitted model with the same embed() interface as a trained JointModel."""

    def __init__(self, pair: list[str], k: int = 64):
        self.pair, self.k = pair, k

    def fit(self, inputs: dict[str, np.ndarray]) -> LinearAligner:
        a, b = self.pair
        n = len(inputs[a])
        self.cca = CCA(k=min(self.k, max(2, n // 10)), pre=min(128, max(2, n // 5))).fit(inputs[a], inputs[b])
        return self

    def embed(self, name: str, X: np.ndarray) -> np.ndarray:
        a, b = self.pair
        if name == a:
            Z = self.cca.px.transform(X) @ self.cca.Ua
        else:
            Z = self.cca.py.transform(X) @ self.cca.Ub
        return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
