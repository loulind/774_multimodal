"""Downstream analyses chosen by the objective (data for figures and tables).

  retrieval_examples      a few held-out queries per direction with their top-5 matches:
                          similarity, whether the match is the true partner, and
                          whether it shares the query's label
  cross_modal_prediction  for one held-out group (slide), observed vs. predicted values
                          of the best-predicted genes at each spot's coordinates
  classification          confusion matrix and per-class F1 of the probe on the image
                          embedding (classifying a new image without its partner), or
                          the fused embedding when there is no image modality
  clustering / spatial_maps  k-means clusters of held-out embeddings on one held-out
                          slide, cross-tabulated with the labels
"""

from __future__ import annotations

import numpy as np

from .evaluate.metrics import columnwise_pearson, l2n


def _pick_queries(n: int, labels: np.ndarray | None, k: int, rng) -> np.ndarray:
    if labels is not None and (labels >= 0).sum() >= k:
        lab = np.flatnonzero(labels >= 0)
        classes = np.unique(labels[lab])
        picks = [rng.choice(lab[labels[lab] == c]) for c in rng.permutation(classes)[:k]]
        rest = rng.permutation(np.setdiff1d(lab, picks))[: max(0, k - len(picks))]
        return np.concatenate([picks, rest]).astype(int)
    return rng.choice(n, min(k, n), replace=False)


def retrieval_examples(Za: np.ndarray, Zb: np.ndarray, ids: np.ndarray, labels: np.ndarray | None,
                       classes: list[str], n_queries: int = 6, k: int = 5, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    a, b = l2n(Za), l2n(Zb)
    out = {}
    for name, Q, G in (("a_to_b", a, b), ("b_to_a", b, a)):
        rows = []
        for q in _pick_queries(len(Q), labels, n_queries, rng):
            sims = G @ Q[q]
            top = np.argsort(-sims)[:k]
            lab = lambda i: (classes[labels[i]] if labels is not None and labels[i] >= 0 else None)  # noqa: E731
            rows.append({"query": str(ids[q]), "query_position": int(q), "query_label": lab(q),
                         "partner_rank": int((sims > sims[q]).sum() + 1),
                         "retrieved": [{"id": str(ids[i]), "position": int(i), "similarity": round(float(sims[i]), 4),
                                        "is_partner": bool(i == q), "label": lab(i),
                                        "same_label": None if lab(q) is None else lab(i) == lab(q)}
                                       for i in top]})
        out[name] = rows
    return out


def prediction_maps(P: np.ndarray, T: np.ndarray, groups: np.ndarray, xy: np.ndarray, names: list[str],
                    order: np.ndarray | None, n_genes: int = 4, top: int = 50) -> dict:
    """Observed vs predicted maps for the best within-slide genes on the largest held-out slide."""
    g = np.asarray(groups).astype(str)
    levels, counts = np.unique(g, return_counts=True)
    slide = levels[np.argmax(counts)]
    sel = g == slide
    cand = (order if order is not None else np.arange(T.shape[1]))[:top]
    r = columnwise_pearson(P[sel][:, cand], T[sel][:, cand])
    best = cand[np.argsort(-r)[:n_genes]]
    return {"group": slide, "n_spots": int(sel.sum()), "x": xy[sel, 0].tolist(), "y": xy[sel, 1].tolist(),
            "genes": [{"gene": names[j] if names else int(j),
                       "r_within": round(float(columnwise_pearson(P[sel][:, [j]], T[sel][:, [j]])[0]), 4),
                       "observed": T[sel, j].round(3).tolist(), "predicted": P[sel, j].round(3).tolist()}
                      for j in best]}


def confusion(probe_result: dict, classes: list[str]) -> dict:
    return {"classes": classes, "matrix": probe_result.get("confusion"),
            "per_class_f1": dict(zip(classes, probe_result.get("per_class_f1", []))),
            "macro_f1": probe_result.get("macro_f1"), "balanced_accuracy": probe_result.get("balanced_accuracy")}


def cluster_map(Z: np.ndarray, labels: np.ndarray | None, classes: list[str], groups: np.ndarray,
                xy: np.ndarray, k: int | None = None, seed: int = 0) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    g = np.asarray(groups).astype(str)
    # the largest held-out slide that carries labels, if any
    lab = labels if labels is not None else np.full(len(g), -1)
    scores = {lv: ((g == lv) & (lab >= 0)).sum() * 10_000 + (g == lv).sum() for lv in np.unique(g)}
    slide = max(scores, key=scores.get)
    sel = g == slide
    k = k or max(2, len(np.unique(lab[sel & (lab >= 0)])) or 5)
    clusters = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(l2n(Z[sel]))
    out = {"group": slide, "k": int(k), "x": xy[sel, 0].tolist(), "y": xy[sel, 1].tolist(),
           "cluster": clusters.tolist()}
    if (lab[sel] >= 0).sum() >= 10:
        m = lab[sel] >= 0
        out["label"] = [classes[v] if v >= 0 else None for v in lab[sel]]
        out["ari"] = float(adjusted_rand_score(lab[sel][m], clusters[m]))
        table = np.zeros((k, len(classes)), int)
        for c, v in zip(clusters[m], lab[sel][m]):
            table[c, v] += 1
        out["contingency"] = {"clusters": list(range(k)), "classes": classes, "counts": table.tolist()}
    return out
