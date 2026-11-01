"""Label-based quality of a representation: linear probe, nearest neighbours, clustering
(and, without labels, agreement between the modalities' clusters).

  linear probe   multinomial logistic regression on the frozen representation,
                 class-balanced weights (so rare classes count, matching macro-F1),
                 ridge penalty chosen by 3-fold cross-validation within training
                 samples; at most 5,000 training samples (stratified), since probe
                 accuracy saturates well before that. Macro-F1 (each class weighs
                 the same) and balanced accuracy.
  kNN            majority vote of the 10 nearest training samples (cosine).
  clustering     k-means with k = number of classes on held-out samples, compared
                 with the labels by ARI (adjusted Rand index; 0 = chance agreement,
                 1 = identical partitions) and NMI; silhouette of the true labels.
  cluster agreement  without labels: k-means on each modality's held-out embeddings
                 separately, with k chosen once on training samples by the largest mean
                 silhouette width; the ARI between the two partitions measures whether
                 both modalities group the samples the same way.
  group silhouette  silhouette of the *grouping* (slide, patient) instead of labels:
                 a high value means the representation is organised by batch.

Fusion baselines for classification: early fusion = probe on the concatenation of
the standardised unimodal features; late fusion = average of the two unimodal
probes' class probabilities.
"""

from __future__ import annotations

import numpy as np

from .metrics import l2n

PROBE_MAX_TRAIN = 5000
KNN_K = 10


def _stratified_cap(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    idx = np.arange(len(y))
    if len(idx) <= n:
        return idx
    rng = np.random.default_rng(seed)
    out = []
    for c in np.unique(y):
        m = idx[y == c]
        out.extend(rng.choice(m, max(1, int(round(n * len(m) / len(idx)))), replace=False))
    return np.sort(np.array(out))


def _fit_probe(Xtr, ytr, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    keep = _stratified_cap(ytr, PROBE_MAX_TRAIN, seed)
    X, y = Xtr[keep], ytr[keep]
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    folds = min(3, int(np.bincount(y).min())) if len(np.unique(y)) > 1 else 0
    if folds >= 2:
        pipe = GridSearchCV(pipe, {"logisticregression__C": [0.01, 0.1, 1.0, 10.0]},
                            cv=StratifiedKFold(folds, shuffle=True, random_state=seed), scoring="f1_macro")
    return pipe.fit(X, y)


def classification_scores(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

    return {"_y_true": y_true, "_y_pred": y_pred,
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "accuracy": float(np.mean(y_true == y_pred)),
            "per_class_f1": [float(v) for v in f1_score(y_true, y_pred, average=None,
                                                        labels=list(range(n_classes)), zero_division=0)],
            "confusion": confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).tolist(),
            "n_test": int(len(y_true))}


def probe(Xtr, ytr, Xte, yte, n_classes: int, seed: int = 0) -> tuple[dict, np.ndarray | None]:
    """Linear probe on labelled samples (-1 = unlabelled). Returns scores and test probabilities."""
    tr, te = ytr >= 0, yte >= 0
    if tr.sum() < 10 or te.sum() < 5 or len(np.unique(ytr[tr])) < 2:
        return {}, None
    model = _fit_probe(Xtr[tr], ytr[tr], seed)
    proba = np.zeros((len(Xte), n_classes))
    proba[:, model.classes_] = model.predict_proba(Xte)
    out = classification_scores(yte[te], proba[te].argmax(1), n_classes)
    if hasattr(model, "best_params_"):
        out["C"] = model.best_params_["logisticregression__C"]
    return out, proba


def late_fusion(probas: list[np.ndarray], yte: np.ndarray, n_classes: int) -> dict:
    te = yte >= 0
    if not probas or te.sum() < 5:
        return {}
    p = np.mean(probas, axis=0)
    return classification_scores(yte[te], p[te].argmax(1), n_classes)


def knn(Xtr, ytr, Xte, yte, n_classes: int, k: int = KNN_K) -> dict:
    from sklearn.neighbors import NearestNeighbors

    tr, te = ytr >= 0, yte >= 0
    if tr.sum() < k or te.sum() < 5:
        return {}
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(Xtr[tr])
    _, idx = nn.kneighbors(Xte[te])
    votes = ytr[tr][idx]
    pred = np.array([np.bincount(v, minlength=n_classes).argmax() for v in votes])
    return classification_scores(yte[te], pred, n_classes)


def clustering(Z: np.ndarray, y: np.ndarray, seed: int = 0) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

    lab = y >= 0
    k = len(np.unique(y[lab]))
    if lab.sum() < 20 or k < 2:
        return {}
    X = l2n(Z[lab])
    pred = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    sub = np.random.default_rng(seed).permutation(lab.sum())[:3000]
    return {"k": k, "ari": float(adjusted_rand_score(y[lab], pred)),
            "nmi": float(normalized_mutual_info_score(y[lab], pred)),
            "label_silhouette": float(silhouette_score(X[sub], y[lab][sub], metric="cosine"))}


CLUSTER_KS = range(2, 13)


def choose_k(views: list[np.ndarray], seed: int = 0, ks=CLUSTER_KS, n_max: int = 2000) -> tuple[int, dict]:
    """Number of clusters when there are no labels: the k whose k-means partition has the largest
    mean silhouette width. ``views`` are training-sample features, one block per modality,
    each L2-normalised so that the modalities count equally."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    X = np.hstack([l2n(v) for v in views])
    X = X[np.random.default_rng(seed).permutation(len(X))[:n_max]]
    sil = {}
    for k in ks:
        if k < len(X):
            pred = KMeans(n_clusters=k, n_init=4, random_state=seed).fit_predict(X)
            sil[k] = float(silhouette_score(X, pred, metric="cosine"))
    return (max(sil, key=sil.get) if sil else 2), sil


def cluster_agreement(Za: np.ndarray, Zb: np.ndarray, k: int, seed: int = 0, fused: np.ndarray | None = None) -> dict:
    """Clustering without labels: k-means on each modality's held-out embeddings separately, and the
    agreement of the two partitions (ARI: 0 = chance, 1 = identical). If the representation captures
    structure the modalities share, both should group the samples the same way. ``fused`` adds the
    silhouette width of the clusters of the fused embedding (how well separated they are)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

    if len(Za) < 20:
        return {}
    ca = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(l2n(Za))
    cb = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(l2n(Zb))
    out = {"k": k, "cross_modal_ari": float(adjusted_rand_score(ca, cb)),
           "cross_modal_nmi": float(normalized_mutual_info_score(ca, cb))}
    if fused is not None:
        X = l2n(fused)
        pred = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        sub = np.random.default_rng(seed).permutation(len(X))[:3000]
        out["silhouette"] = float(silhouette_score(X[sub], pred[sub], metric="cosine"))
    return out


def group_silhouette(Z: np.ndarray, groups: np.ndarray, seed: int = 0) -> float | None:
    from sklearn.metrics import silhouette_score

    g = np.asarray(groups).astype(str)
    if len(np.unique(g)) < 2 or len(g) < 20:
        return None
    sub = np.random.default_rng(seed).permutation(len(g))[:3000]
    return float(silhouette_score(l2n(Z[sub]), g[sub], metric="cosine"))
