"""Cross-modal retrieval: can a sample's partner be found from the other modality?

For every test query in modality A, all test items of modality B are ranked by
cosine similarity in the joint space; the rank of the true partner is recorded
(and the same from B to A). Reported per direction and averaged:

  recall@k      share of queries whose partner is in the top k; chance = k / n
  median rank   (1 = perfect)
  MRR           mean reciprocal rank
  label match@k share of the top-k retrieved items that carry the query's label
                ("semantic" retrieval); chance = sum of squared class shares.
                Exact-partner retrieval is a very strict test when many samples
                look alike (neighbouring tissue spots, generic captions); label
                match asks the looser, often more relevant, question.
"""

from __future__ import annotations

import numpy as np

from .metrics import l2n


def partner_ranks(Za: np.ndarray, Zb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """0-based rank of each true partner, a->b and b->a (ties counted in the query's favour)."""
    S = l2n(Za) @ l2n(Zb).T
    d = np.diag(S)
    return (S > d[:, None]).sum(1), (S > d[None, :]).sum(0)


def recall_from_ranks(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks < k))


def label_match_at_k(Za: np.ndarray, Zb: np.ndarray, labels: np.ndarray, k: int = 10) -> dict:
    """Precision@k of label agreement among labelled samples (labels: int, -1 = unlabelled)."""
    lab = labels >= 0
    if lab.sum() < k + 1 or len(np.unique(labels[lab])) < 2:
        return {}
    a, b, y = l2n(Za[lab]), l2n(Zb[lab]), labels[lab]
    out = {}
    for name, S in (("a_to_b", a @ b.T), ("b_to_a", b @ a.T)):
        top = np.argsort(-S, axis=1)[:, :k]
        out[name] = float((y[top] == y[:, None]).mean())
    p = np.bincount(y) / len(y)
    out["mean"] = (out["a_to_b"] + out["b_to_a"]) / 2
    out["chance"] = float((p ** 2).sum())
    out["k"] = k
    out["n_labelled"] = int(lab.sum())
    return out


def retrieval_metrics(Za: np.ndarray, Zb: np.ndarray, ks=(1, 5, 10), labels: np.ndarray | None = None) -> dict:
    r_ab, r_ba = partner_ranks(Za, Zb)
    n = len(Za)
    out = {"n": n, "chance": {f"R@{k}": k / n for k in ks}}
    for name, r in (("a_to_b", r_ab), ("b_to_a", r_ba)):
        out[name] = {**{f"R@{k}": recall_from_ranks(r, k) for k in ks},
                     "median_rank": float(np.median(r) + 1), "mrr": float(np.mean(1 / (r + 1)))}
    out["mean"] = {f"R@{k}": (out["a_to_b"][f"R@{k}"] + out["b_to_a"][f"R@{k}"]) / 2 for k in ks}
    out["mean"]["median_rank"] = (out["a_to_b"]["median_rank"] + out["b_to_a"]["median_rank"]) / 2
    if labels is not None:
        lm = label_match_at_k(Za, Zb, labels, max(ks))
        if lm:
            out["label_match"] = lm
    out["_ranks"] = (r_ab, r_ba)                  # for bootstrap; dropped before writing
    return out


def gallery_subsample_recall(Za: np.ndarray, Zb: np.ndarray, size: int, k: int = 10, repeats: int = 5,
                             seed: int = 0) -> float | None:
    """Mean recall@k over random galleries of a fixed size (comparability across test-set sizes)."""
    if len(Za) <= size:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(repeats):
        idx = rng.choice(len(Za), size, replace=False)
        r_ab, r_ba = partner_ranks(Za[idx], Zb[idx])
        vals.append((recall_from_ranks(r_ab, k) + recall_from_ranks(r_ba, k)) / 2)
    return float(np.mean(vals))
