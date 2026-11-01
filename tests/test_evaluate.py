"""Evaluation metrics (known answers), the evaluation suite, and downstream analyses."""

import numpy as np
import pytest

from multimodal_agent import downstream as D
from multimodal_agent.evaluate import geometry as G
from multimodal_agent.evaluate import prediction as P
from multimodal_agent.evaluate import probes as PR
from multimodal_agent.evaluate import retrieval as R
from multimodal_agent.evaluate import suite as S

RNG = np.random.default_rng(0)


def clusters(n=600, k=3, d=16, spread=0.05, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(k, d))
    y = np.arange(n) % k
    return centres[y] + spread * rng.normal(size=(n, d)), y


# --- retrieval ---------------------------------------------------------------------------

def test_retrieval_known_answers():
    Z = RNG.normal(size=(200, 16))
    res = R.retrieval_metrics(Z, Z + 1e-6 * RNG.normal(size=Z.shape))
    assert res["mean"]["R@1"] == 1.0 and res["mean"]["median_rank"] == 1.0
    rnd = R.retrieval_metrics(RNG.normal(size=(500, 16)), RNG.normal(size=(500, 16)))
    assert abs(rnd["mean"]["R@10"] - 10 / 500) < 0.03 and rnd["chance"]["R@10"] == 10 / 500


def test_label_match_and_its_chance_level():
    X, y = clusters()
    lm = R.label_match_at_k(X, X + 0.01 * RNG.normal(size=X.shape), y, 10)
    assert lm["mean"] > 0.99 and lm["chance"] == pytest.approx(1 / 3, abs=0.01)
    y_part = y.copy()
    y_part[:300] = -1                                   # unlabelled samples are excluded
    assert R.label_match_at_k(X, X, y_part, 10)["n_labelled"] == 300


# --- prediction ----------------------------------------------------------------------------

def test_prediction_metrics_known_answers():
    T = RNG.normal(size=(300, 80))
    perfect = P.prediction_metrics(T, T, np.arange(80), groups=np.repeat([0, 1, 2], 100))
    assert perfect["mean_r_top50"] == pytest.approx(1.0) and perfect["n_genes_r_gt_0.3"] == 80
    assert perfect["within_group_mean_r_top50"] == pytest.approx(1.0)
    noise = P.prediction_metrics(RNG.normal(size=T.shape), T, None)
    assert abs(noise["mean_r_top50"]) < 0.05


def test_knn_and_ridge_prediction():
    Z = RNG.normal(size=(400, 8))
    T = Z @ RNG.normal(size=(8, 30))
    Pk = P.knn_predict(Z[300:] + 0.01, Z[:300], T[:300])
    assert np.mean(np.diag(np.corrcoef(Pk.T, T[300:].T)[:30, 30:])) > 0.5
    Pr = P.ridge_predict(Z[:300], T[:300], Z[300:])
    assert np.mean(P.columnwise_pearson(Pr, T[300:])) > 0.95


# --- probes, clustering, geometry -------------------------------------------------------------

def test_probe_knn_clustering_on_separable_classes():
    X, y = clusters(900)
    res, proba = PR.probe(X[:600], y[:600], X[600:], y[600:], 3)
    assert res["macro_f1"] > 0.99 and proba.shape == (300, 3)
    assert PR.knn(X[:600], y[:600], X[600:], y[600:], 3)["macro_f1"] > 0.99
    c = PR.clustering(X[600:], y[600:])
    assert c["ari"] > 0.99 and c["label_silhouette"] > 0.5
    shuffled = RNG.permutation(y)
    res, _ = PR.probe(X[:600], shuffled[:600], X[600:], shuffled[600:], 3)
    assert res["macro_f1"] < 0.5                      # no signal -> near chance (1/3)


def test_geometry_known_answers():
    Z = RNG.normal(size=(300, 16))
    g = G.geometry(Z, Z)
    assert g["alignment"] == pytest.approx(0.0, abs=1e-9) and g["modality_gap"] == pytest.approx(0.0, abs=1e-9)
    collapsed = np.ones((300, 16)) + 1e-6 * RNG.normal(size=(300, 16))
    assert G.uniformity(collapsed) > G.uniformity(Z)          # spread-out is more uniform (lower)
    low_rank = RNG.normal(size=(300, 2)) @ RNG.normal(size=(2, 16))
    assert G.effective_rank(low_rank) < 2.5 < G.effective_rank(Z)


# --- suite ------------------------------------------------------------------------------------

def _fold(seed, zero_shot_quality=1.0):
    rng = np.random.default_rng(seed)
    X, y = clusters(900, seed=seed)
    y = y.copy()
    y[rng.random(900) < 0.2] = -1
    tr, te = np.arange(600), np.arange(600, 900)
    A = X + 0.02 * rng.normal(size=X.shape)
    B = X @ rng.normal(size=(16, 16)) + 0.5 * rng.normal(size=X.shape)
    joint = S.Representation("primary", "joint", {"a": A[tr], "b": A[tr] + 0.05}, {"a": A[te], "b": A[te] + 0.05})
    uni = S.Representation("unimodal", "unimodal", {"a": A[tr], "b": B[tr]}, {"a": A[te], "b": B[te]}, aligned=False)
    noisy = A[te] + (1 - zero_shot_quality) * 10 * rng.normal(size=A[te].shape)
    zs = S.Representation("zero_shot", "zero_shot", {"a": A[tr], "b": A[tr]}, {"a": A[te], "b": noisy})
    fd = S.FoldData("f", ["a", "b"], y[tr], y[te], ["c0", "c1", "c2"], batch_test=np.arange(300) % 2,
                    target_test=B[te], target_order=np.arange(16))
    preds = {"primary:head": B[te] + 0.1 * rng.normal(size=B[te].shape), "ridge": rng.normal(size=B[te].shape)}
    metrics = ["retrieval_recall_at_k", "linear_probe", "knn_accuracy", "clustering_ari_nmi",
               "group_silhouette", "prediction_pearson"]
    return S.evaluate_fold([joint, uni, zs], fd, metrics, predictions=preds)


def test_evaluate_fold_structure_and_primary_values():
    f = _fold(1)
    assert set(f["probe"]) >= {"primary", "unimodal", "early_fusion", "late_fusion"}
    assert "fused" in f["probe"]["primary"] and "fused" not in f["probe"]["unimodal"]
    assert S.metric_value(f, "primary", "retrieval_mean_recall_at_10") > 0.9
    assert S.metric_value(f, "primary", "prediction_mean_pearson_top50") > 0.9
    clean = S.strip_private(f)
    assert "_ranks" not in clean["retrieval"]["primary"]


def test_aggregate_comparisons_bootstrap_and_flags():
    folds = [_fold(1), _fold(2)]
    agg = S.aggregate(folds)
    assert agg["retrieval/primary/mean/R@10"]["n_folds"] == 2
    rows = S.integration_comparisons(folds, "primary", ["prediction_mean_pearson_top50", "probe_macro_f1"])
    ridge = next(r for r in rows if r["alternative"] == "ridge")
    assert ridge["difference_mean"] > 0.5 and ridge["folds_primary_better"] == 2
    one = [_fold(3)]
    boot = S.bootstrap(one[0])
    rows1 = S.integration_comparisons(one, "primary", ["retrieval_mean_recall_at_10",
                                                       "prediction_mean_pearson_top50"], boot)
    assert all("difference_ci95" in r for r in rows1)
    fl = S.flags(one, "primary", rows1, "retrieval_mean_recall_at_10", ["primary"])
    assert all(x["severity"] in ("warning", "info") for x in fl)
    assert any(x["code"] == "zero_shot_near_ceiling" for x in fl)
    fl = S.flags([_fold(4, zero_shot_quality=0.0)], "primary", [], "retrieval_mean_recall_at_10", ["primary"])
    assert not any(x["code"] == "zero_shot_near_ceiling" for x in fl)


# --- downstream --------------------------------------------------------------------------------

def test_downstream_outputs():
    X, y = clusters(300)
    ex = D.retrieval_examples(X, X + 0.01, np.array([f"s{i}" for i in range(300)]), y, ["a", "b", "c"], 4, 5)
    assert len(ex["a_to_b"]) == 4 and ex["a_to_b"][0]["retrieved"][0]["is_partner"]
    groups = np.repeat(["s1", "s2"], [200, 100])
    xy = RNG.uniform(0, 100, (300, 2))
    T = RNG.normal(size=(300, 20))
    maps = D.prediction_maps(T + 0.1 * RNG.normal(size=T.shape), T, groups, xy, [f"g{i}" for i in range(20)],
                             np.arange(20), n_genes=3)
    assert maps["group"] == "s1" and len(maps["genes"]) == 3 and maps["genes"][0]["r_within"] > 0.9
    cm = D.cluster_map(X, y, ["a", "b", "c"], groups, xy)
    assert cm["ari"] > 0.99 and len(cm["cluster"]) == 200


# --- clustering without labels ----------------------------------------------------------------

def test_choose_k_and_cross_modal_cluster_agreement():
    rng = np.random.default_rng(5)
    X, y = clusters(600, k=4, spread=0.1, seed=5)
    A, B = X + 0.05 * rng.normal(size=X.shape), X @ rng.normal(size=(16, 16))   # two views of one structure
    k, sil = PR.choose_k([A[:400], B[:400]], seed=0)
    assert k == 4 and set(sil) == set(PR.CLUSTER_KS)
    shared = PR.cluster_agreement(A[400:], B[400:], 4, fused=A[400:])
    assert shared["cross_modal_ari"] > 0.9 and shared["silhouette"] > 0.5
    unrelated = PR.cluster_agreement(A[400:], rng.normal(size=(200, 16)), 4)
    assert abs(unrelated["cross_modal_ari"]) < 0.1 and "silhouette" not in unrelated


def test_evaluate_fold_without_labels_uses_one_k_for_every_representation():
    rng = np.random.default_rng(6)
    X, _ = clusters(900, k=3, spread=0.1, seed=6)
    tr, te = np.arange(600), np.arange(600, 900)
    A, B = X + 0.02 * rng.normal(size=X.shape), X @ rng.normal(size=(16, 16))
    joint = S.Representation("primary", "joint", {"a": A[tr], "b": A[tr]}, {"a": A[te], "b": A[te] + 0.01})
    uni = S.Representation("unimodal", "unimodal", {"a": A[tr], "b": B[tr]}, {"a": A[te], "b": B[te]}, aligned=False)
    noise = S.Representation("noise", "joint", {"a": A[tr], "b": A[tr]}, {"a": A[te], "b": rng.normal(size=(300, 16))})
    f = S.evaluate_fold([joint, uni, noise], S.FoldData("f", ["a", "b"]), ["retrieval_recall_at_k", "clustering_ari_nmi"])
    assert f["cluster_k"]["k"] == 3 and f["cluster_k"]["chosen_on"] == "unimodal"
    agree = f["cluster_agreement"]
    assert {r["k"] for r in agree.values()} == {3}
    assert agree["primary"]["cross_modal_ari"] > 0.9 > abs(agree["noise"]["cross_modal_ari"]) + 0.8
    assert "silhouette" in agree["primary"] and "silhouette" not in agree["unimodal"]   # no fused view
    assert f["clustering"] == {} and f["probe"] == {}                                   # nothing needs labels


def test_cluster_map_without_labels_or_coordinates_cross_tabulates_the_modalities():
    rng = np.random.default_rng(7)
    X, _ = clusters(300, k=3, spread=0.05, seed=7)
    B = X @ rng.normal(size=(16, 16))
    cm = D.cluster_map(X, None, [], np.array(["all"] * 300), None, 3, views={"rna": X, "peaks": B})
    assert "x" not in cm and "contingency" not in cm
    ag = cm["agreement"]
    counts = np.asarray(ag["counts"])
    assert ag["ari"] > 0.95 and counts.trace() > 0.95 * counts.sum()                   # matched: diagonal
    assert (ag["rows"], ag["cols"]) == ("rna", "peaks")
