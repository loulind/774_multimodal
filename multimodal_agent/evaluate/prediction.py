"""Cross-modal prediction: how well is the target modality predicted from the source?

Per target feature (gene), the Pearson correlation between predicted and observed
standardised values across held-out samples. Reported:

  mean r over the top-50 most variable genes (the primary prediction metric; the
  convention in histology-to-expression studies), median r, mean r over all scored
  genes, and how many genes reach r > 0.3;
  within-group mean r: correlations computed *within* each slide (or other group)
  and averaged, weighted by size. Pooled correlations also reward predicting
  differences *between* slides or patients; within-slide correlations ask whether
  the spatial pattern inside a section is captured, which is the stricter question.

Methods compared, all fitted on training samples only:

  regression head   trained jointly with the alignment (if planned)
  nearest neighbours  the 50 training samples whose *target-modality* embeddings are
                    closest to the query's source embedding; their targets are averaged,
                    weighted by similarity (retrieval through the joint space)
  ridge             ridge regression from the frozen source features (linear baseline)
  mean              the training mean; correlation 0 by construction (reference)
"""

from __future__ import annotations

import numpy as np

from .metrics import columnwise_pearson, l2n

TOP_GENES = 50
GOOD_R = 0.3
KNN_K = 50


def prediction_metrics(P: np.ndarray, T: np.ndarray, variance_order: np.ndarray | None,
                       names: list[str] | None = None, groups: np.ndarray | None = None,
                       top: int = TOP_GENES) -> dict:
    r = columnwise_pearson(P, T)
    order = variance_order if variance_order is not None else np.arange(T.shape[1])
    top_idx = order[: min(top, len(order))]
    out = {"n_samples": len(T), "n_genes": T.shape[1],
           f"mean_r_top{top}": float(np.mean(r[top_idx])), f"median_r_top{top}": float(np.median(r[top_idx])),
           "mean_r_all": float(np.mean(r)), f"n_genes_r_gt_{GOOD_R}": int((r > GOOD_R).sum()),
           "mse": float(np.mean((P - T) ** 2))}
    best = np.argsort(-r)[:10]
    out["best_genes"] = [{"gene": names[j] if names else int(j), "r": round(float(r[j]), 4)} for j in best]
    if groups is not None:
        within, weight = [], []
        for g in np.unique(groups):
            sel = groups == g
            if sel.sum() >= 10:
                within.append(float(np.mean(columnwise_pearson(P[sel][:, top_idx], T[sel][:, top_idx]))))
                weight.append(int(sel.sum()))
        if within:
            out[f"within_group_mean_r_top{top}"] = float(np.average(within, weights=weight))
            out["n_groups"] = len(within)
    out["_r"] = r                                  # per-gene correlations; dropped before writing
    out["_P_top"], out["_T_top"] = P[:, top_idx], T[:, top_idx]
    return out


def knn_predict(Zq: np.ndarray, Zref: np.ndarray, Tref: np.ndarray, k: int = KNN_K,
                batch: int = 1024) -> np.ndarray:
    """Similarity-weighted mean target of the k nearest reference samples (cosine)."""
    q, ref = l2n(Zq), l2n(Zref)
    k = min(k, len(ref))
    out = np.zeros((len(q), Tref.shape[1]), np.float32)
    for s in range(0, len(q), batch):
        S = q[s:s + batch] @ ref.T
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        w = np.take_along_axis(S, idx, 1)
        w = np.exp((w - w.max(1, keepdims=True)) / 0.1)
        w /= w.sum(1, keepdims=True)
        out[s:s + batch] = np.einsum("nk,nkg->ng", w, Tref[idx])
    return out


def ridge_predict(Xtr: np.ndarray, Ttr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xtr)
    m = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(sc.transform(Xtr), Ttr)
    return m.predict(sc.transform(Xte)).astype(np.float32)
