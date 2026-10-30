"""Evaluate every representation on one fold, aggregate folds, and compare with baselines.

A *representation* is any way of turning held-out samples into vectors: a
trained configuration's joint embeddings, the unimodal features it started from,
the CCA alignment, or a pretrained dual encoder's own (zero-shot) space. All are
scored on the same held-out samples with the same metrics, so differences are
attributable to the representation.

Uncertainty. With grouped cross-validation every fold is a different set of
held-out groups (patients), so metrics are reported as mean, SD and range over
folds -- the spread between patients, the honest unit of replication. With a
single predefined split, 95% percentile-bootstrap intervals resample held-out
samples (1,000 resamples; 200 for gene correlations), and differences between
two representations use the *same* resamples (paired bootstrap).

"Did integration help?" compares the primary configuration with the best
single-modality alternative, with linear alignment (CCA), with zero-shot
pretrained alignment, with ridge regression, and with early/late fusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geometry as G
from . import prediction as P
from . import probes as PR
from . import retrieval as R
from .metrics import l2n

B_FAST, B_SLOW = 1000, 200
ZERO_SHOT_CEILING = 0.9          # zero-shot recall@10 at or above this suggests pretraining overlap
COLLAPSE_RANK_SHARE = 0.10


@dataclass
class Representation:
    name: str
    kind: str                                    # joint | unimodal | linear | zero_shot
    train: dict[str, np.ndarray]                 # modality -> training vectors
    test: dict[str, np.ndarray]                  # modality -> held-out vectors
    aligned: bool = True                         # modalities share one space


@dataclass
class FoldData:
    name: str
    pair: list[str]
    y_train: np.ndarray | None = None            # label codes, -1 = unlabelled
    y_test: np.ndarray | None = None
    classes: list[str] = field(default_factory=list)
    batch_test: np.ndarray | None = None         # grouping for batch silhouette
    within_test: np.ndarray | None = None        # grouping for within-group correlations
    target_test: np.ndarray | None = None
    target_order: np.ndarray | None = None
    target_names: list[str] = field(default_factory=list)


def _views(rep: Representation, split: str, pair: list[str]) -> dict[str, np.ndarray]:
    d = rep.train if split == "train" else rep.test
    views = {m: d[m] for m in pair if m in d}
    if rep.aligned and all(m in d for m in pair) and len(pair) == 2:
        views["fused"] = l2n(l2n(d[pair[0]]) + l2n(d[pair[1]]))
    return views


def evaluate_fold(reps: list[Representation], fd: FoldData, metrics: list[str], ks=(1, 5, 10),
                  predictions: dict[str, np.ndarray] | None = None, seed: int = 0) -> dict:
    out: dict = {"fold": fd.name, "n_test": int(len(next(iter(reps[0].test.values())))),
                 "retrieval": {}, "geometry": {}, "probe": {}, "knn": {}, "clustering": {},
                 "group_silhouette": {}, "prediction": {}}
    labels = fd.y_test is not None and fd.y_train is not None and (fd.y_test >= 0).any()
    k = len(fd.classes)
    a = fd.pair[0]
    b = fd.pair[1] if len(fd.pair) > 1 else None
    unimodal_probas = []
    for rep in reps:
        te = rep.test
        if rep.aligned and b is not None and a in te and b in te:
            if "retrieval_recall_at_k" in metrics:
                res = R.retrieval_metrics(te[a], te[b], ks, fd.y_test if labels else None)
                g = R.gallery_subsample_recall(te[a], te[b], 1000)
                if g is not None:
                    res["mean_R@10_gallery1000"] = g
                out["retrieval"][rep.name] = res
            out["geometry"][rep.name] = G.geometry(te[a], te[b], seed)
        if labels:
            trv, tev = _views(rep, "train", fd.pair), _views(rep, "test", fd.pair)
            out["probe"][rep.name], out["knn"][rep.name], out["clustering"][rep.name] = {}, {}, {}
            for v in tev:
                if "linear_probe" in metrics:
                    res, proba = PR.probe(trv[v], fd.y_train, tev[v], fd.y_test, k, seed)
                    if res:
                        out["probe"][rep.name][v] = res
                        if rep.kind == "unimodal" and v in fd.pair:
                            unimodal_probas.append(proba)
                if "knn_accuracy" in metrics:
                    res = PR.knn(trv[v], fd.y_train, tev[v], fd.y_test, k)
                    if res:
                        out["knn"][rep.name][v] = res
                if "clustering_ari_nmi" in metrics:
                    res = PR.clustering(tev[v], fd.y_test, seed)
                    if res:
                        out["clustering"][rep.name][v] = res
        if "group_silhouette" in metrics and fd.batch_test is not None:
            tev = _views(rep, "test", fd.pair)
            out["group_silhouette"][rep.name] = {v: PR.group_silhouette(z, fd.batch_test, seed)
                                                 for v, z in tev.items()}
    if labels and "linear_probe" in metrics:
        uni = next((r for r in reps if r.kind == "unimodal"), None)
        if uni is not None and b is not None and a in uni.train and b in uni.train:
            ef = Representation("early_fusion", "fusion",
                                {"x": np.hstack([_std(uni.train[a], uni.train[a]), _std(uni.train[b], uni.train[b])])},
                                {"x": np.hstack([_std(uni.train[a], uni.test[a]), _std(uni.train[b], uni.test[b])])},
                                aligned=False)
            res, _ = PR.probe(ef.train["x"], fd.y_train, ef.test["x"], fd.y_test, k, seed)
            if res:
                out["probe"]["early_fusion"] = {"concat": res}
            if len(unimodal_probas) == 2:
                out["probe"]["late_fusion"] = {"average": PR.late_fusion(unimodal_probas, fd.y_test, k)}
    if fd.target_test is not None:
        for method, Pt in (predictions or {}).items():
            out["prediction"][method] = P.prediction_metrics(Pt, fd.target_test, fd.target_order,
                                                             fd.target_names, fd.within_test)
    return out


def _std(ref: np.ndarray, X: np.ndarray) -> np.ndarray:
    return (X - ref.mean(0)) / (ref.std(0) + 1e-6)


# ---------------------------------------------------------------------------
# primary metric, comparisons, aggregation, bootstrap
# ---------------------------------------------------------------------------

def metric_value(fold: dict, rep: str, metric: str) -> float | None:
    """The primary metric for one representation on one fold (higher is better)."""
    try:
        if metric == "retrieval_mean_recall_at_10":
            return float(fold["retrieval"][rep]["mean"]["R@10"])
        if metric == "prediction_mean_pearson_top50":
            vals = [v["mean_r_top50"] for m, v in fold["prediction"].items() if m.split(":")[0] == rep]
            return float(max(vals)) if vals else None
        if metric == "probe_macro_f1":
            p = fold["probe"][rep]
            return float(p.get("fused", next(iter(p.values())))["macro_f1"])
        if metric == "clustering_ari":
            c = fold["clustering"][rep]
            return float(c.get("fused", next(iter(c.values())))["ari"])
    except (KeyError, StopIteration, TypeError):
        return None
    return None


def _flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            if str(k).startswith("_"):
                continue
            out.update(_flatten(v, f"{prefix}{k}/"))
    elif isinstance(d, (int, float, np.integer, np.floating)) and not isinstance(d, bool):
        out[prefix.rstrip("/")] = float(d)
    return out


def aggregate(folds: list[dict]) -> dict:
    """Mean / SD / min / max over folds for every numeric result present in all folds."""
    flats = [_flatten({k: v for k, v in f.items() if k != "fold"}) for f in folds]
    common = set(flats[0]).intersection(*flats[1:]) if flats else set()
    agg = {}
    for key in sorted(common):
        vals = np.array([f[key] for f in flats])
        agg[key] = {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "min": float(vals.min()), "max": float(vals.max()), "n_folds": len(vals)}
    return agg


def _ci(v: np.ndarray) -> list[float]:
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def bootstrap(fold: dict, seed: int = 0) -> dict:
    """Resampled values of the key metrics on one held-out set (shared resamples per family).

    Returns {family: {key: samples}} for families retrieval (key = representation),
    probe (key = "representation/view") and prediction (key = method).
    """
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, np.ndarray]] = {"retrieval": {}, "probe": {}, "prediction": {}}
    n = fold["n_test"]
    sets = [rng.integers(0, n, n) for _ in range(B_FAST)]
    for rep, res in fold["retrieval"].items():
        if "_ranks" in res:
            r_ab, r_ba = res["_ranks"]
            out["retrieval"][rep] = np.array([(np.mean(r_ab[i] < 10) + np.mean(r_ba[i] < 10)) / 2 for i in sets])
    probes = [(f"{rep}/{v}", r) for rep, views in fold["probe"].items() for v, r in views.items() if "_y_true" in r]
    if probes:
        m = len(probes[0][1]["_y_true"])
        psets = [rng.integers(0, m, m) for _ in range(B_FAST)]
        for key, r in probes:
            yt, yp = r["_y_true"], r["_y_pred"]
            if len(yt) == m:
                out["probe"][key] = np.array([f1_score(yt[i], yp[i], average="macro") for i in psets])
    preds = [(k, r) for k, r in fold["prediction"].items() if "_P_top" in r]
    if preds:
        from .metrics import columnwise_pearson

        m = len(preds[0][1]["_T_top"])
        qsets = [rng.integers(0, m, m) for _ in range(B_SLOW)]
        for key, r in preds:
            Pt, Tt = r["_P_top"], r["_T_top"]
            out["prediction"][key] = np.array([np.mean(columnwise_pearson(Pt[i], Tt[i])) for i in qsets])
    return out


FAMILY = {"retrieval_mean_recall_at_10": "retrieval", "probe_macro_f1": "probe",
          "prediction_mean_pearson_top50": "prediction"}


def _alternative(f: dict, alt: str, metric: str) -> tuple[float | None, str | None]:
    """Value and bootstrap key of an alternative representation for one metric."""
    if alt == "unimodal" and metric in ("probe_macro_f1", "clustering_ari"):
        block = f["probe" if metric == "probe_macro_f1" else "clustering"].get("unimodal", {})
        k = "macro_f1" if metric == "probe_macro_f1" else "ari"
        if not block:
            return None, None
        view = max(block, key=lambda v: block[v][k])
        return block[view][k], f"unimodal/{view}"
    if alt in ("early_fusion", "late_fusion"):
        block = f["probe"].get(alt, {})
        if not block:
            return None, None
        view = next(iter(block))
        return block[view]["macro_f1"], f"{alt}/{view}"
    if alt == "ridge":
        r = f["prediction"].get("ridge")
        return (r["mean_r_top50"], "ridge") if r else (None, None)
    v = metric_value(f, alt, metric)
    key = f"{alt}/fused" if metric == "probe_macro_f1" else alt
    return v, key


def _primary_key(f: dict, rep: str, metric: str) -> str | None:
    if metric == "probe_macro_f1":
        p = f["probe"].get(rep, {})
        return f"{rep}/fused" if "fused" in p else (f"{rep}/{next(iter(p))}" if p else None)
    if metric == "prediction_mean_pearson_top50":
        cands = {m: v["mean_r_top50"] for m, v in f["prediction"].items() if m.split(":")[0] == rep}
        return max(cands, key=cands.get) if cands else None
    return rep


def integration_comparisons(folds: list[dict], primary: str, metric_names: list[str],
                            boot: dict | None = None) -> list[dict]:
    """Primary vs. each alternative: mean difference over folds (SD across folds), or, with a
    single held-out set, the difference with a paired-bootstrap 95% interval."""
    rows = []
    alternatives = {
        "retrieval_mean_recall_at_10": ["cca", "zero_shot"],
        "probe_macro_f1": ["unimodal", "early_fusion", "late_fusion", "cca", "zero_shot"],
        "clustering_ari": ["unimodal", "cca", "zero_shot"],
        "prediction_mean_pearson_top50": ["ridge"],
    }
    for metric in metric_names:
        for alt in alternatives.get(metric, []):
            diffs, prim, other, keys = [], [], [], None
            for f in folds:
                pv = metric_value(f, primary, metric)
                av, akey = _alternative(f, alt, metric)
                if pv is not None and av is not None:
                    diffs.append(pv - av)
                    prim.append(pv)
                    other.append(av)
                    keys = (_primary_key(f, primary, metric), akey)
            if not diffs:
                continue
            d = np.array(diffs)
            row = {"metric": metric, "primary": primary, "alternative": alt,
                   "primary_value": float(np.mean(prim)), "alternative_value": float(np.mean(other)),
                   "difference_mean": float(d.mean()),
                   "difference_sd": float(d.std(ddof=1)) if len(d) > 1 else None,
                   "folds_primary_better": int((d > 0).sum()), "n_folds": len(d)}
            fam = FAMILY.get(metric)
            if boot is not None and fam and keys and len(folds) == 1:
                sa, sb = boot.get(fam, {}).get(keys[0]), boot.get(fam, {}).get(keys[1])
                if sa is not None and sb is not None and len(sa) == len(sb):
                    row["difference_ci95"] = _ci(sa - sb)
                    row["primary_ci95"], row["alternative_ci95"] = _ci(sa), _ci(sb)
            rows.append(row)
    return rows


MIN_EFFECT_ABS = 0.01            # differences smaller than this are treated as noise
MIN_EFFECT_REL = 0.05            # ... or smaller than 5% of the primary's value


def _meaningful(diff: float, base: float) -> bool:
    return abs(diff) > max(MIN_EFFECT_ABS, MIN_EFFECT_REL * abs(base))


def flags(folds: list[dict], primary: str, comparisons: list[dict], primary_metric: str,
          config_names: list[str]) -> list[dict]:
    """Plain-language findings derived from the numbers, for the critic and the report.

    severity "warning": a clear problem (an interval excluding zero, every fold the
    same way, or an effect above the noise threshold); "info": no clear difference.
    Only warnings computed on *validation* data may trigger the critic's revision.
    """
    out = []
    zs = [metric_value(f, "zero_shot", "retrieval_mean_recall_at_10") for f in folds]
    zs = [v for v in zs if v is not None]
    if zs and np.mean(zs) >= ZERO_SHOT_CEILING:
        out.append({"code": "zero_shot_near_ceiling", "severity": "warning", "value": float(np.mean(zs)),
                    "message": f"A pretrained dual encoder already retrieves the true partner in the top 10 for "
                               f"{np.mean(zs):.0%} of held-out samples without any training. Check whether its "
                               "pretraining corpus overlaps the evaluation data; if so, absolute scores are "
                               "optimistic."})
    for c in comparisons:
        d, alt, metric = c["difference_mean"], c["alternative"], c["metric"]
        ci = c.get("difference_ci95")
        if ci is not None:
            worse, unclear = ci[1] < 0, ci[0] <= 0 <= ci[1]
            detail = f"difference {d:+.3f}, 95% CI {ci[0]:+.3f} to {ci[1]:+.3f}"
        else:
            n = c["n_folds"]
            worse = c["folds_primary_better"] == 0 and _meaningful(d, c["primary_value"])
            unclear = not worse and (d <= 0 or c["folds_primary_better"] < n)
            detail = f"mean difference {d:+.3f}; primary better in {c['folds_primary_better']}/{n} folds"
        if worse:
            out.append({"code": f"worse_than_{alt}", "severity": "warning", "metric": metric, "value": d,
                        "message": f"Primary configuration is worse than {alt} on {metric} ({detail})."})
        elif unclear:
            out.append({"code": f"no_clear_gain_over_{alt}", "severity": "info", "metric": metric, "value": d,
                        "message": f"No clear difference between the primary configuration and {alt} on "
                                   f"{metric} ({detail})."})
    for f in folds[:1]:
        gs = f["group_silhouette"].get(primary, {}).get("fused")
        ls = f["clustering"].get(primary, {}).get("fused", {}).get("label_silhouette")
        if gs is not None and ls is not None and gs > ls and gs > 0.1:
            out.append({"code": "batch_confounding", "severity": "warning", "value": gs,
                        "message": f"Held-out embeddings cluster more by group (silhouette {gs:.2f}) than by "
                                   f"label ({ls:.2f})."})
        geo = f["geometry"].get(primary)
        if geo and geo["effective_rank_a"] < COLLAPSE_RANK_SHARE * geo["dim"]:
            out.append({"code": "collapse", "severity": "warning", "value": geo["effective_rank_a"],
                        "message": f"Effective rank {geo['effective_rank_a']:.1f} of {geo['dim']} dimensions: "
                                   "embeddings use few directions."})
    pv = np.mean([v for f in folds if (v := metric_value(f, primary, primary_metric)) is not None] or [np.nan])
    for name in config_names:
        if name == primary:
            continue
        cv = np.mean([v for f in folds if (v := metric_value(f, name, primary_metric)) is not None] or [np.nan])
        if np.isfinite(cv) and np.isfinite(pv) and cv > pv:
            sev = "warning" if _meaningful(cv - pv, pv) else "info"
            out.append({"code": "comparison_better", "severity": sev, "config": name, "value": float(cv - pv),
                        "message": f"Comparison '{name}' scores higher than the primary on {primary_metric} "
                                   f"({cv:.3f} vs {pv:.3f})" + ("." if sev == "warning" else
                                                                 "; within noise.")})
    return out


def strip_private(d):
    """Drop internal arrays (keys starting with '_') before writing JSON."""
    if isinstance(d, dict):
        return {k: strip_private(v) for k, v in d.items() if not str(k).startswith("_")}
    if isinstance(d, list):
        return [strip_private(v) for v in d]
    if isinstance(d, (np.floating, np.integer)):
        return d.item()
    if isinstance(d, np.ndarray):
        return d.tolist()
    return d
