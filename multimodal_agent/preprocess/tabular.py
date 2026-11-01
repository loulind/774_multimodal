"""Tabular preprocessing: a mixed numeric/categorical table -> a standardised numeric matrix.

Every statistic is fitted on training rows only and applied unchanged to validation and
test rows:

  * numeric columns: missing values are replaced by the training median, and a 0/1
    missingness indicator is added when the column has missing values in training (the
    missing-indicator method, which keeps "not measured" as information rather than
    pretending the median was observed);
  * categorical columns: one indicator per level seen in at least ``min_count`` training
    rows (5, or 1% of training rows if more); rarer levels, and levels never seen in
    training, share an "other" indicator; a missing value sets no indicator, plus a
    missingness indicator when missing values are common enough in training;
  * every resulting column is standardised with the training mean and SD, so a 0/1
    indicator and a numeric column enter on the same scale (as in penalised regression);
  * optional PCA of the standardised columns (``mode="pca"``).

Columns that are constant or entirely missing in the training rows carry no information
and are dropped. ``exclude`` names columns that must never be inputs -- the outcome the
plan evaluates -- so a label cannot leak into the representation that is scored on it.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

MIN_COUNT = 5
MIN_SHARE = 0.01


def is_numeric_column(s: pd.Series) -> bool:
    """Numeric or boolean values (booleans enter as 0/1)."""
    return pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)


def describe_columns(df: pd.DataFrame) -> str:
    n_num = sum(is_numeric_column(df[c]) for c in df.columns)
    return f"{n_num} numeric and {df.shape[1] - n_num} categorical columns"


def _categorical(s: pd.Series) -> pd.Series:
    out = s.astype(object).where(s.notna(), None)
    return out.map(lambda v: None if v is None else (str(v).strip() or None))


class TabularPreprocessor:
    def __init__(self, mode: str = "standardised", pca_components: int | None = None,
                 exclude: list[str] | tuple[str, ...] = ()):
        self.mode, self.k_pca = mode, pca_components
        self.exclude = [c for c in exclude if c]

    def fit(self, df: pd.DataFrame) -> TabularPreprocessor:
        n = len(df)
        self.min_count = max(MIN_COUNT, math.ceil(MIN_SHARE * n))
        self.excluded = [c for c in self.exclude if c in df.columns]
        self.numeric: dict[str, tuple[float, bool]] = {}           # column -> (median, add indicator)
        self.categorical: dict[str, tuple[list[str], bool, bool]] = {}  # -> (levels, other, missing ind.)
        self.dropped: list[str] = []
        for col in df.columns:
            if col in self.excluded:
                continue
            s = df[col]
            if is_numeric_column(s):
                x = pd.to_numeric(s, errors="coerce").astype(float)
                if x.notna().sum() == 0 or (x.nunique() < 2 and not x.isna().any()):
                    self.dropped.append(col)
                    continue
                self.numeric[col] = (float(x.median()), bool(x.isna().any()))
            else:
                v = _categorical(s)
                counts = v.dropna().value_counts()
                levels = sorted(counts[counts >= self.min_count].index, key=lambda lv: (-counts[lv], lv))
                other = bool((counts < self.min_count).any())
                missing = int(v.isna().sum()) >= self.min_count
                if len(levels) + other + missing < 2:              # one value (almost) everywhere
                    self.dropped.append(col)
                    continue
                self.categorical[col] = (levels, other, missing)
        Z = self._encode(df)
        self.mean = Z.mean(0) if len(Z) else np.zeros(Z.shape[1])
        self.std = (Z.std(0) if len(Z) else np.ones(Z.shape[1])) + 1e-6
        self.pca = None
        if self.mode == "pca" and Z.shape[1] > 2:
            from sklearn.decomposition import PCA

            k = int(min(self.k_pca or 50, Z.shape[0] - 1, Z.shape[1]))
            self.pca = PCA(n_components=k, random_state=0).fit((Z - self.mean) / self.std)
        return self

    def _encode(self, df: pd.DataFrame) -> np.ndarray:
        cols, names = [], []
        for col, (med, ind) in self.numeric.items():
            x = pd.to_numeric(df[col], errors="coerce").astype(float).to_numpy()
            miss = np.isnan(x)
            cols.append(np.where(miss, med, x))
            names.append(col)
            if ind:
                cols.append(miss.astype(float))
                names.append(f"{col} (missing)")
        for col, (levels, other, missing) in self.categorical.items():
            v = _categorical(df[col]).to_numpy()
            for lv in levels:
                cols.append((v == lv).astype(float))
                names.append(f"{col}={lv}")
            if other:
                known = set(levels)
                cols.append(np.array([x is not None and x not in known for x in v], float))
                names.append(f"{col}=other")
            if missing:
                cols.append(np.array([x is None for x in v], float))
                names.append(f"{col} (missing)")
        self.feature_names = names
        return np.column_stack(cols) if cols else np.zeros((len(df), 0))

    def standardised(self, df: pd.DataFrame) -> np.ndarray:
        return ((self._encode(df) - self.mean) / self.std).astype(np.float32)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        Z = self.standardised(df)
        return (self.pca.transform(Z) if self.pca is not None else Z).astype(np.float32)

    @property
    def features(self) -> np.ndarray:
        return np.arange(len(self.feature_names))

    @property
    def out_dim(self) -> int:
        return self.pca.n_components_ if self.pca is not None else len(self.feature_names)

    def describe(self) -> str:
        n_ind = sum(ind for _, ind in self.numeric.values())
        n_lv = sum(len(lv) + other for lv, other, _ in self.categorical.values())
        parts = []
        if self.numeric:
            parts.append(f"{len(self.numeric)} numeric columns (training-median imputation"
                         + (f", missingness indicators for {n_ind}" if n_ind else "") + ")")
        if self.categorical:
            parts.append(f"{len(self.categorical)} categorical columns one-hot encoded ({n_lv} levels; levels "
                         f"seen in fewer than {self.min_count} training rows pooled as 'other')")
        text = "; ".join(parts) + f"; {len(self.feature_names)} inputs, standardised with training statistics"
        if self.pca is not None:
            text += f", PCA to {self.out_dim}"
        if self.dropped:
            text += f"; dropped as constant or empty in training: {', '.join(self.dropped)}"
        if self.excluded:
            text += f"; excluded as the evaluated outcome: {', '.join(self.excluded)}"
        return text
