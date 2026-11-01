"""Screen candidate encoders on the data before committing to one.

For each image or text modality of the aligned pair, every candidate encoder
(the registry's candidates for the measured domain, capped by the budget)
encodes the same screening sample. Each is scored on *held-out* screening
samples -- drawn from validation groups, never from the test set -- by how
much its frozen features already relate to the paired modality:

  cca_recall_at_10  canonical correlation analysis between the two feature sets,
                    fitted on screening-train pairs; screening-val pairs are
                    embedded in the canonical space and we ask how often the
                    true partner is among the 10 nearest (both directions).
                    A linear, closed-form version of the alignment the agent
                    will train, so it predicts which features align best.
  ridge_r2          (matrix partner) share of the partner's principal-component
                    variance a ridge regression from the features explains.
  probe_macro_f1    (labels) a logistic-regression probe on the features alone.
  zero_shot_recall_at_10  (pre-aligned dual encoders) recall@10 in the pair's
                    own pretrained space, with no fitting at all.

Candidates are ranked by the score that matches the objective's primary task;
all scores are logged so the choice can be audited.
"""

from __future__ import annotations

import numpy as np

from ..data.dataset import MaterializedSource
from ..evaluate.metrics import recall_at_k, ridge_r2
from ..models.linear import CCA
from ..preprocess.image import ImageTransform, ReinhardNormalizer
from ..preprocess.matrix import MatrixPreprocessor
from ..preprocess.tabular import TabularPreprocessor
from ..preprocess.text import clean
from ..schema import ModalityPlan, ScreeningResult, ScreenScore
from .pretrained import FeatureCache, ids_digest, image_features, load_encoder, release_all, text_features
from .registry import REGISTRY, candidates, is_dual_pair

RANK_METRIC = {"retrieval": "alignment_recall_at_10", "joint_embedding": "alignment_recall_at_10",
               "clustering": "alignment_recall_at_10", "cross_modal_prediction": "ridge_r2",
               "classification": "probe_macro_f1"}


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def probe_f1(Xtr, ytr, Xva, yva) -> float | None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler

    tr, va = ytr != "", yva != ""
    if len(set(ytr[tr])) < 2 or va.sum() < 5:
        return None
    sc = StandardScaler().fit(Xtr[tr])
    clf = LogisticRegression(max_iter=2000, C=0.1).fit(sc.transform(Xtr[tr]), ytr[tr])
    keep = np.isin(yva[va], clf.classes_)
    return float(f1_score(yva[va][keep], clf.predict(sc.transform(Xva[va][keep])), average="macro"))


# ---------------------------------------------------------------------------
# features for one modality
# ---------------------------------------------------------------------------

def fit_normalizer(source, groups: np.ndarray, rng, per_group: int = 30) -> ReinhardNormalizer:
    """Per-group colour statistics from a sample of each group's own images.

    The reference colour is the average over all groups: a fixed template,
    computed once, so image features are extracted once for every fold. It uses
    no labels and nothing from the paired modality; each group is normalised
    with its own statistics, like any per-sample standardisation.
    """
    samples = {}
    for g in np.unique(groups):
        members = np.flatnonzero(groups == g)
        pick = rng.choice(members, min(per_group, len(members)), replace=False)
        samples[str(g)] = [np.asarray(source.load(int(i)).convert("RGB")) for i in source.order(pick)]
    return ReinhardNormalizer().fit(samples)


def image_key_parts(ds, mplan: ModalityPlan, idx: np.ndarray, normalizer=None) -> dict:
    """What determines a cached image feature: data, samples and preprocessing."""
    reinhard = mplan.colour_normalisation == "reinhard" and normalizer is not None
    return {"data": str(ds.path.resolve()), "ids": ids_digest(ds.ids[idx]), "n": len(idx),
            "context": mplan.patch_context, "colour": mplan.colour_normalisation,
            "normaliser": normalizer.describe() if reinhard else None}


def modality_features(ds, mplan: ModalityPlan, encoder: str, idx: np.ndarray, device: str,
                      cache: FeatureCache | None, batch_size: int, normalizer=None,
                      norm_groups: np.ndarray | None = None, progress=None,
                      source_factory=None) -> np.ndarray:
    """Frozen features of one modality for dataset positions ``idx``.

    ``source_factory`` may supply the images already loaded and colour-normalised
    (screening reuses one in-memory sample for every candidate); it is only
    called when the features are not cached.
    """
    mod = ds.modalities[mplan.modality]
    if mod.kind == "image":
        parts = image_key_parts(ds, mplan, idx, normalizer)
        if cache is not None:
            hit = cache.load(cache.key(encoder=encoder, kind="image", **parts))
            if hit is not None and len(hit) == len(idx):
                return hit
        enc = load_encoder(encoder, device)
        source = mod.images
        if mplan.patch_context is not None and hasattr(source, "with_context"):
            source = source.with_context(mplan.patch_context)
        square = getattr(source, "kind", "") == "spot_patches"
        use_norm = normalizer if mplan.colour_normalisation == "reinhard" else None
        if source_factory is not None:
            source, use_norm = source_factory(), None          # already normalised in memory
        tf = ImageTransform(enc.input_size, enc.mean, enc.std, mplan.greyscale_to_rgb, use_norm,
                            square_input=square)
        return image_features(enc, source, idx, tf, norm_groups, batch_size, cache, parts, progress)
    enc = load_encoder(encoder, device)
    parts = {"data": str(ds.path.resolve()), "ids": ids_digest(ds.ids[idx]), "n": len(idx)}
    texts = [clean(mod.data[i], mplan.text_cleaning) for i in idx]
    max_tokens = min(mplan.text_max_tokens or 77, REGISTRY[encoder].max_tokens or 512)
    return text_features(enc, texts, max_tokens, batch_size, cache, {**parts, "cleaning": mplan.text_cleaning})


# ---------------------------------------------------------------------------
# screening
# ---------------------------------------------------------------------------

def _labels(ds, column, idx) -> np.ndarray:
    if column is None:
        return np.full(len(idx), "")
    s = ds.obs[column].iloc[idx]
    return s.astype(object).where(s.notna(), "").astype(str).to_numpy()


def screen(ds, profile, task, prelim_plan, train_idx: np.ndarray, val_idx: np.ndarray, budget, device: str,
           cache: FeatureCache | None, quick: bool, seed: int = 0, progress=None) -> tuple[list[ScreeningResult], dict]:
    """Score every candidate encoder; returns rankings (best first) and a table for the report."""
    rng = np.random.default_rng(seed)
    tr = np.sort(rng.choice(train_idx, min(budget.screening_train, len(train_idx)), replace=False))
    va = np.sort(rng.choice(val_idx, min(budget.screening_val, len(val_idx)), replace=False))
    mplans = {m.modality: m for m in prelim_plan.primary.modalities}
    label_col = prelim_plan.evaluation.label_column
    ytr, yva = _labels(ds, label_col, tr), _labels(ds, label_col, va)
    rank_metric = RANK_METRIC[task.primary_task]
    image_mod = next((m for m in profile.modalities if m.kind == "image"), None)
    domain = image_mod.image.domain_hint if image_mod and image_mod.image else None

    lib = next((c for c in ds.columns_with("library")), None) or profile.cv_group
    norm_groups = ds.obs[lib].astype(str).to_numpy() if lib else None
    normalizer = None
    if any(m.colour_normalisation == "reinhard" for m in mplans.values()) and norm_groups is not None:
        img_name = next(n for n, m in mplans.items() if ds.modalities[n].kind == "image")
        src = ds.modalities[img_name].images
        normalizer = fit_normalizer(src, norm_groups, rng)

    feats: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for name, mp in mplans.items():
        kind = ds.modalities[name].kind
        feats[name] = {}
        if kind in ("image", "text"):
            idx = np.concatenate([tr, va])
            memo: dict = {}

            def factory(mp=mp, idx=idx, memo=memo):
                if "src" not in memo:              # built once, on the first uncached candidate
                    if progress:
                        progress(f"  loading {len(idx)} screening images once for all candidates")
                    src = ds.modalities[mp.modality].images
                    if mp.patch_context is not None and hasattr(src, "with_context"):
                        src = src.with_context(mp.patch_context)
                    norm = normalizer if mp.colour_normalisation == "reinhard" else None
                    memo["src"] = MaterializedSource(src, idx, norm, norm_groups)
                return memo["src"]

            for enc in candidates(kind, domain, budget.max_candidates, quick):
                if progress:
                    progress(f"  screening {name}: {enc}")
                try:
                    both = modality_features(ds, mp, enc, idx, device, cache, budget.feature_batch_size,
                                             normalizer, norm_groups, progress,
                                             factory if kind == "image" else None)
                except Exception as err:          # a candidate that fails to load is reported, not fatal
                    feats[name][enc] = None
                    if progress:
                        progress(f"    {enc} failed: {type(err).__name__}: {str(err)[:200]}")
                    continue
                finally:
                    release_all()
                feats[name][enc] = (both[: len(tr)], both[len(tr):])
        elif kind == "tabular":
            df = ds.modalities[name].data
            pre = TabularPreprocessor(exclude=[label_col]).fit(df.iloc[tr])
            feats[name]["mlp"] = (pre.transform(df.iloc[tr]), pre.transform(df.iloc[va]))
        else:
            X = ds.modalities[name].data
            mx = profile.modality(name).matrix
            pre = MatrixPreprocessor(mp.n_variable_features or 2000, "pca", 50,
                                     counts=bool(mx and mx.integer_valued)).fit(X[tr])
            feats[name]["mlp"] = (pre.transform(X[tr]), pre.transform(X[va]))

    names = list(mplans)
    rows = []
    if len(names) == 2:
        a, b = names
        for ea, fa in feats[a].items():
            for eb, fb in feats[b].items():
                if fa is None or fb is None:
                    continue
                row = {a: ea, b: eb}
                cca = CCA(k=min(32, len(tr) // 10 or 2)).fit(fa[0], fb[0])
                row["cca_recall_at_10"] = recall_at_k(*cca.transform(fa[1], fb[1]))
                row["zero_shot_recall_at_10"] = recall_at_k(fa[1], fb[1]) if is_dual_pair(ea, eb) else None
                row["alignment_recall_at_10"] = max(v for v in (row["cca_recall_at_10"],
                                                                row["zero_shot_recall_at_10"]) if v is not None)
                if task.prediction and task.prediction.target == b:
                    row["ridge_r2"] = ridge_r2(fa[0], fb[0], fa[1], fb[1])
                elif task.prediction and task.prediction.target == a:
                    row["ridge_r2"] = ridge_r2(fb[0], fa[0], fb[1], fa[1])
                rows.append(row)
    probe = {}
    for name in names:
        for enc, f in feats[name].items():
            if f is not None and label_col:
                probe[(name, enc)] = probe_f1(f[0], ytr, f[1], yva)
    for row in rows:
        vals = [probe.get((n, row[n])) for n in names]
        vals = [v for v in vals if v is not None]
        row["probe_macro_f1"] = max(vals) if vals else None

    metric = rank_metric if rows and rows[0].get(rank_metric) is not None else "alignment_recall_at_10"
    results = []
    for name in names:
        if ds.modalities[name].kind not in ("image", "text"):
            continue
        best: dict[str, dict] = {}
        for row in rows:
            if row.get(metric) is None:
                continue
            if row[name] not in best or row[metric] > best[row[name]][metric]:
                best[row[name]] = row
        failed = [e for e, f in feats[name].items() if f is None]
        ranked = sorted(best.items(), key=lambda kv: -kv[1][metric])
        results.append(ScreeningResult(
            modality=name,
            ranked=[ScreenScore(encoder=e, score=float(r[metric]), metric=metric,
                                details={k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                                         if k not in names})
                    for e, r in ranked],
            note=(f"Screened on {len(tr)} training and {len(va)} validation samples from separate groups."
                  + (f" Failed to load: {failed}." if failed else ""))))
    table = {"metric": metric, "n_train": len(tr), "n_val": len(va), "rows": rows,
             "probe_macro_f1": {f"{n}:{e}": v for (n, e), v in probe.items()},
             "normaliser": normalizer.describe() if normalizer else None}
    return results, table
