"""Matrix preprocessing: library-size normalisation, variable features, PCA.

Everything that learns from data (the normalisation target, which features are
"variable", their means and spreads, the PCA rotation) is fitted on training
samples only and then applied unchanged to validation and test samples, so no
held-out information shapes the inputs.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def lognorm(X: sp.csr_matrix, target: float | None = None) -> tuple[sp.csr_matrix, float]:
    """Library-size normalisation to ``target`` (default: median library size), then log1p."""
    X = sp.csr_matrix(X)
    lib = np.asarray(X.sum(1)).ravel().astype(float)
    target = float(np.median(lib[lib > 0])) if target is None else target
    scale = np.divide(target, lib, out=np.zeros_like(lib), where=lib > 0)
    Y = (sp.diags(scale) @ X.astype(np.float32)).tocsr()
    Y.data = np.log1p(Y.data)
    return Y, target


def top_variable(Y: sp.csr_matrix, k: int) -> np.ndarray:
    """Indices of the k columns with the largest variance."""
    mean = np.asarray(Y.mean(0)).ravel()
    sq = np.asarray(Y.multiply(Y).mean(0)).ravel()
    return np.argsort(-(sq - mean ** 2))[: min(k, Y.shape[1])]


class MatrixPreprocessor:
    """counts -> log-normalised -> top-k variable features -> standardised [-> PCA]."""

    def __init__(self, n_variable: int, mode: str = "log_normalised", pca_components: int | None = None,
                 counts: bool = True):
        self.n_variable, self.mode, self.k_pca, self.counts = n_variable, mode, pca_components, counts

    def _norm(self, X) -> sp.csr_matrix:
        if self.counts:
            Y, _ = lognorm(X, self.target)
            return Y
        return sp.csr_matrix(X, dtype=np.float32)

    def fit(self, X_train) -> MatrixPreprocessor:
        self.target = None
        if self.counts:
            _, self.target = lognorm(X_train)
        Y = self._norm(X_train)
        self.features = np.sort(top_variable(Y, self.n_variable))
        Z = Y[:, self.features].toarray()
        self.mean = Z.mean(0)
        self.std = Z.std(0) + 1e-6
        self.pca = None
        if self.mode == "pca":
            from sklearn.decomposition import PCA

            k = int(min(self.k_pca or 50, Z.shape[0] - 1, Z.shape[1]))
            self.pca = PCA(n_components=k, svd_solver="randomized", random_state=0)
            self.pca.fit((Z - self.mean) / self.std)
        return self

    def standardised(self, X) -> np.ndarray:
        """The variable features, standardised with training statistics (regression targets)."""
        Z = self._norm(X)[:, self.features].toarray()
        return ((Z - self.mean) / self.std).astype(np.float32)

    def transform(self, X) -> np.ndarray:
        Z = self.standardised(X)
        return (self.pca.transform(Z) if self.pca is not None else Z).astype(np.float32)

    @property
    def out_dim(self) -> int:
        return self.pca.n_components_ if self.pca is not None else len(self.features)

    def describe(self) -> str:
        norm = "library-size normalised + log1p" if self.counts else "as stored"
        tail = f", PCA to {self.out_dim}" if self.pca is not None else ""
        return f"{norm}; top {len(self.features)} variable features, standardised{tail}"
