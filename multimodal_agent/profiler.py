"""Step 1: inspect a multimodal dataset and report measurements the planner can act on.

For every modality: what kind it is, how many samples have it, and the
statistics that matter for representation learning -- image size, colour and
stain signature, text length and duplication, matrix sparsity, count
behaviour, effective dimension and near-twins. Across modalities: how many
samples are paired, which columns are labels and groups, how labels are
distributed over groups, whether a predefined split exists and whether groups
straddle it, and how strongly samples cluster by group (batch effects) versus
by label. With spatial coordinates, how spatially autocorrelated expression is.

Measurement conventions (fixed, recorded in the profile): images are summarised
on <= 240 thumbnails (stratified by group when there is one), matrices on <= 5,000
rows in a log-normalised top-2,000-variable-feature space with 50 principal
components. These are ways of *measuring*, not modelling choices.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .data.dataset import Modality, MultimodalDataset, SpotPatchSource
from .data.roles import SPLIT_TOKENS, coarsest_group
from .preprocess.matrix import lognorm, top_variable
from .preprocess.tabular import is_numeric_column
from .schema import (DataProfile, GroupProfile, ImageProfile, LabelProfile, MatrixProfile,
                     ModalityProfile, PairingProfile, RoleRecord, SpatialProfile, SplitProfile,
                     TabularProfile, TextProfile)

MAX_IMAGES = 240
MAX_HASH = 2000
MAX_ROWS = 5000
N_VARIABLE = 2000
N_PCS = 50
TWIN_COSINE = 0.95
MIN_GROUP_IMAGES = 10            # images a group needs before its mean colour is estimated


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def icc_oneway(values: np.ndarray, groups: np.ndarray) -> float | None:
    """ICC(1) from a one-way ANOVA with unequal group sizes: the between-group
    share of variance (0 = groups look alike, 1 = all variance is between groups)."""
    df = pd.DataFrame({"y": values, "g": groups}).dropna()
    sizes = df.groupby("g")["y"].size()
    sizes = sizes[sizes >= 2]
    df = df[df["g"].isin(sizes.index)]
    g, N = len(sizes), len(df)
    if g < 2 or N <= g:
        return None
    grand = df["y"].mean()
    means = df.groupby("g")["y"].mean()
    msb = float((sizes * (means - grand) ** 2).sum() / (g - 1))
    msw = float(((df["y"] - df["g"].map(means)) ** 2).sum() / (N - g))
    k0 = (N - float((sizes ** 2).sum()) / N) / (g - 1)
    denom = msb + (k0 - 1) * msw
    return float(np.clip((msb - msw) / denom, 0, 1)) if denom > 0 else None


def variance_share(scores: np.ndarray, groups: np.ndarray) -> float | None:
    """Share of the variance of ``scores`` (n x k) explained by a categorical
    grouping, summed over columns (a variance-weighted R^2)."""
    keep = pd.notna(groups)
    if keep.sum() < 10 or pd.Series(groups[keep]).nunique() < 2:
        return None
    S, g = scores[keep], pd.Series(groups[keep]).astype(str).to_numpy()
    S = S - S.mean(0)
    total = float((S ** 2).sum())
    between = 0.0
    for level in np.unique(g):
        rows = S[g == level]
        between += len(rows) * float((rows.mean(0) ** 2).sum())
    return between / total if total > 0 else None


def _sample(idx: np.ndarray, k: int, rng) -> np.ndarray:
    return np.sort(rng.choice(idx, min(k, len(idx)), replace=False)) if len(idx) > k else idx


# ---------------------------------------------------------------------------
# images
# ---------------------------------------------------------------------------

def _thumb(im, side: int = 128) -> np.ndarray:
    im = im.convert("RGB")
    im.thumbnail((side, side))
    return np.asarray(im, dtype=np.uint8)


def _dhash(im) -> int:
    g = np.asarray(im.convert("L").resize((9, 8)), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def _hsv(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = arr.reshape(-1, 3).astype(np.float32) / 255
    mx, mn = a.max(1), a.min(1)
    d = mx - mn
    s = np.where(mx > 0, d / np.maximum(mx, 1e-6), 0)
    r, g, b = a[:, 0], a[:, 1], a[:, 2]
    h = np.zeros_like(mx)
    nz = d > 1e-6
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & ~rm & ~gm
    h[rm] = (60 * ((g[rm] - b[rm]) / d[rm])) % 360
    h[gm] = 60 * ((b[gm] - r[gm]) / d[gm]) + 120
    h[bm] = 60 * ((r[bm] - g[bm]) / d[bm]) + 240
    return h, s


def _image_profile(m: Modality, obs: pd.DataFrame, group_col: str | None, rng) -> ImageProfile:
    from scipy.ndimage import laplace

    src = m.images
    idx = np.flatnonzero(m.present)
    if group_col is not None:
        # Group-stratified sampling (and a between-group colour share) needs groups large enough
        # to estimate their mean colour: it is used when groups of >= 10 images hold most of the
        # images (smaller groups are left out of the sample); otherwise a random sample.
        sizes = obs[group_col].iloc[idx].astype(str).value_counts()
        big = set(sizes.index[sizes >= MIN_GROUP_IMAGES])
        if len(big) < 2 or sizes[sizes >= MIN_GROUP_IMAGES].sum() < 0.5 * len(idx):
            group_col = None
    if group_col is not None:
        levels = obs[group_col].astype(str).to_numpy()
        chosen_levels = rng.permutation([lv for lv in np.unique(levels[idx]) if lv in big])[:12]
        per = max(1, MAX_IMAGES // max(1, len(chosen_levels)))
        pick = np.concatenate([_sample(idx[levels[idx] == lv], per, rng) for lv in chosen_levels])
    else:
        pick = _sample(idx, MAX_IMAGES, rng)
    pick = src.order(pick)
    sizes, grey, means, sats, light, dark, stain_px, tissue_px, sharp = [], [], [], [], [], [], 0, 0, []
    for i in pick:
        im = src.load(int(i))
        sizes.append(im.size)
        arr = _thumb(im)
        chan = arr.astype(np.int16)
        grey.append(float(np.abs(chan[..., 0] - chan[..., 1]).mean() +
                          np.abs(chan[..., 1] - chan[..., 2]).mean()) < 3)
        means.append(arr.reshape(-1, 3).mean(0))
        lt = (arr > 220).all(-1)
        dk = (arr < 25).all(-1)
        light.append(lt.mean())
        dark.append(dk.mean())
        h, s = _hsv(arr)
        t = ~(lt | dk).reshape(-1)
        stain = ((h >= 240) | (h <= 15)) & (s > 0.1)
        stain_px += int((stain & t).sum())
        tissue_px += int(t.sum())
        sats.append(float(s[t].mean()) if t.any() else 0.0)
        gray_img = arr.mean(-1)
        sharp.append(float(laplace(gray_img).var()))
    sizes = np.array(sizes, float)
    means = np.array(means)
    grey_frac = float(np.mean(grey))
    stain_frac = stain_px / max(1, tissue_px)
    share = None
    if group_col is not None:
        g = obs[group_col].astype(str).to_numpy()[pick]
        iccs = [icc_oneway(means[:, c], g) for c in range(3)]
        iccs = [v for v in iccs if v is not None]
        share = float(np.mean(iccs)) if iccs else None
    near_dup = None
    if not isinstance(src, SpotPatchSource):
        hash_idx = _sample(idx, MAX_HASH, rng)
        hashes = [_dhash(src.load(int(i))) for i in hash_idx]
        counts = Counter(hashes)
        near_dup = float(np.mean([counts[h] > 1 for h in hashes]))
    if grey_frac >= 0.5 and stain_frac < 0.3:
        hint, ev = "greyscale_imaging", (f"{grey_frac:.0%} of sampled images are greyscale; "
                                         f"{stain_frac:.0%} of pixels with a stain hue")
    elif grey_frac < 0.2 and stain_frac >= 0.5:
        hint, ev = "stained_tissue", (f"{stain_frac:.0%} of tissue pixels have a pink/purple stain "
                                      f"hue; {grey_frac:.0%} greyscale")
    else:
        hint, ev = "colour_other", (f"{grey_frac:.0%} greyscale; {stain_frac:.0%} of pixels with a "
                                    "stain hue")
    dia = None
    if isinstance(src, SpotPatchSource):
        dia = float(np.median(list(src.diameter.values())))
    return ImageProfile(
        source=src.kind, n_sampled=len(pick), width_median=float(np.median(sizes[:, 0])),
        height_median=float(np.median(sizes[:, 1])),
        width_range=[float(sizes[:, 0].min()), float(sizes[:, 0].max())],
        spot_diameter_px_median=dia, grayscale_fraction=grey_frac,
        mean_rgb=[round(float(v), 1) for v in means.mean(0)], saturation_mean=float(np.mean(sats)),
        light_background_fraction=float(np.mean(light)), dark_background_fraction=float(np.mean(dark)),
        stain_hue_fraction=float(stain_frac), sharpness_median=float(np.median(sharp)),
        near_duplicate_fraction=near_dup, colour_between_group_share=share,
        domain_hint=hint, domain_evidence=ev)


# ---------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------

def _text_profile(m: Modality) -> TextProfile:
    vals = pd.Series([v for v, ok in zip(m.data, m.present) if ok], dtype=object).astype(str)
    words = vals.str.split().str.len()
    norm = vals.str.lower().str.replace(r"[\d\W_]+", " ", regex=True).str.strip()
    vocab = set(t for v in vals.str.lower() for t in re.findall(r"[a-z]+", v))
    return TextProfile(
        n_sampled=len(vals), words_median=float(words.median()), words_p95=float(words.quantile(0.95)),
        chars_median=float(vals.str.len().median()), vocabulary_size=len(vocab),
        duplicate_fraction=float(vals.duplicated(keep=False).mean()),
        near_duplicate_fraction=float(norm.duplicated(keep=False).mean()),
        fraction_over_60_words=float((words > 60).mean()))


# ---------------------------------------------------------------------------
# matrices
# ---------------------------------------------------------------------------

def morans_i(Z: np.ndarray, xy: np.ndarray, k: int = 6) -> np.ndarray:
    """Moran's I for every column of Z on a k-nearest-neighbour graph (row-standardised)."""
    from sklearn.neighbors import NearestNeighbors

    n = len(Z)
    if n <= k + 1:
        return np.full(Z.shape[1], np.nan)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(xy)
    _, ind = nn.kneighbors(xy)
    W = sp.csr_matrix((np.full(n * k, 1.0 / k), (np.repeat(np.arange(n), k), ind[:, 1:].ravel())),
                      shape=(n, n))
    C = Z - Z.mean(0)
    num = (C * (W @ C)).sum(0)
    den = (C ** 2).sum(0)
    return np.divide(num, den, out=np.full(Z.shape[1], np.nan), where=den > 0)


def _matrix_profile(m: Modality, obs: pd.DataFrame, group_col, label_col, spatial: Modality | None,
                    library_col, rng) -> MatrixProfile:
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors

    rows = np.flatnonzero(m.present)
    X = m.data[rows]
    n, p = X.shape
    vals = X.data[: 200_000]
    integer = bool(np.allclose(vals, np.round(vals))) if len(vals) else False
    nonneg = bool(vals.min() >= 0) if len(vals) else True
    lib = np.asarray(X.sum(1)).ravel()
    detected = np.asarray((X > 0).mean(0)).ravel()
    det5 = detected >= 0.05
    over = None
    sub = _sample(np.arange(n), MAX_ROWS, rng)
    Xs = X[sub]
    if integer and nonneg and det5.any():
        mean = np.asarray(Xs[:, det5].mean(0)).ravel()
        var = np.asarray(Xs[:, det5].multiply(Xs[:, det5]).mean(0)).ravel() - mean ** 2
        over = float(np.mean(var > mean))
    if integer and nonneg:
        Ys, _ = lognorm(Xs)
        space = "library-size normalised, log1p"
    else:
        Ys, space = Xs.astype(np.float32).tocsr(), "as stored"
    hv = top_variable(Ys, N_VARIABLE)
    Z = Ys[:, hv].toarray()
    space += f", top {len(hv)} variable features, {N_PCS} PCs"
    k = int(min(N_PCS, Z.shape[0] - 1, Z.shape[1]))
    pca = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(Z)
    total = float(Z.var(0, ddof=1).sum())
    cum = np.cumsum(pca.explained_variance_) / total if total > 0 else np.zeros(k)
    c50 = int(np.searchsorted(cum, 0.5) + 1) if cum[-1] >= 0.5 else None
    c90 = int(np.searchsorted(cum, 0.9) + 1) if cum[-1] >= 0.9 else None
    S = pca.transform(Z)
    tw = _sample(np.arange(len(S)), 3000, rng)
    Sn = S[tw] / np.maximum(np.linalg.norm(S[tw], axis=1, keepdims=True), 1e-9)
    nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(Sn)
    d, _ = nn.kneighbors(Sn)
    twins = float(np.mean(1 - d[:, 1] > TWIN_COSINE))
    top = S[:, : min(10, S.shape[1])]
    sub_obs = obs.iloc[rows[sub]]
    gshare = lshare = gsil = None
    if group_col is not None:
        gvals = sub_obs[group_col].astype(object).to_numpy()
        gshare = variance_share(top, gvals)
        ok = pd.notna(gvals)
        if pd.Series(gvals[ok]).nunique() >= 2:
            sil_idx = _sample(np.flatnonzero(ok), 3000, rng)
            gsil = float(silhouette_score(S[sil_idx, :20], gvals[sil_idx].astype(str), metric="cosine",
                                          random_state=0))
    if label_col is not None:
        lshare = variance_share(top, sub_obs[label_col].astype(object).to_numpy())
    mi_med = mi_frac = None
    if spatial is not None and library_col is not None:
        Y_all, _ = lognorm(X) if integer and nonneg else (X.astype(np.float32).tocsr(), None)
        genes = hv[:100]
        libs = obs.iloc[rows][library_col].astype(str).to_numpy()
        xy = spatial.data[rows]
        per_gene = []
        for lv in np.unique(libs):
            sel = np.flatnonzero(libs == lv)
            if len(sel) >= 20:
                per_gene.append(morans_i(Y_all[sel][:, genes].toarray(), xy[sel]))
        if per_gene:
            mi = np.nanmean(np.vstack(per_gene), axis=0)
            mi_med, mi_frac = float(np.nanmedian(mi)), float(np.nanmean(mi > 0.3))
    kind = "peaks" if m.name == "peaks" else ("genes" if integer and nonneg else "features")
    return MatrixProfile(
        n_features=p, feature_kind=kind, sparsity=float(1 - X.nnz / max(1, n * p)),
        integer_valued=integer, nonnegative=nonneg, library_size_median=float(np.median(lib)),
        library_size_iqr=[float(np.percentile(lib, 25)), float(np.percentile(lib, 75))],
        features_per_sample_median=float(np.median(X.getnnz(1))),
        n_features_detected_5pct=int(det5.sum()), overdispersed_fraction=over, analysis_space=space,
        pca_components_50=c50, pca_components_90=c90,
        pca_top50_variance_fraction=float(cum[-1]) if len(cum) else None, near_twin_fraction=twins,
        group_variance_share=gshare, label_variance_share=lshare, group_silhouette=gsil,
        morans_i_median=mi_med, morans_i_fraction_above_0_3=mi_frac)


# ---------------------------------------------------------------------------
# spatial, tabular
# ---------------------------------------------------------------------------

def _spatial_profile(m: Modality, obs: pd.DataFrame, library_col, image: Modality | None) -> SpatialProfile:
    from sklearn.neighbors import NearestNeighbors

    libs = obs[library_col].astype(str).to_numpy() if library_col else np.zeros(len(obs), str)
    sizes, nn_d = [], []
    for lv in np.unique(libs[m.present]):
        sel = np.flatnonzero((libs == lv) & m.present)
        sizes.append(len(sel))
        if len(sel) > 2:
            d, _ = NearestNeighbors(n_neighbors=2).fit(m.data[sel]).kneighbors(m.data[sel])
            nn_d.append(float(np.median(d[:, 1])))
    dia = None
    if image is not None and isinstance(image.images, SpotPatchSource):
        dia = float(np.median(list(image.images.diameter.values())))
    nn_med = float(np.median(nn_d)) if nn_d else float("nan")
    return SpatialProfile(n_libraries=len(sizes),
                          samples_per_library=[float(min(sizes)), float(np.median(sizes)), float(max(sizes))],
                          nn_distance_px_median=nn_med, spot_diameter_px_median=dia,
                          spacing_over_diameter=nn_med / dia if dia else None)


def _tabular_profile(m: Modality) -> TabularProfile:
    df = m.data[m.present]
    cat = [c for c in df.columns if not is_numeric_column(df[c])]
    return TabularProfile(n_features=df.shape[1], missing_fraction=float(df.isna().mean().mean()),
                          n_numeric=df.shape[1] - len(cat), n_categorical=len(cat),
                          max_levels=max((int(df[c].nunique()) for c in cat), default=0),
                          columns=[str(c) for c in df.columns])


# ---------------------------------------------------------------------------
# labels, groups, splits, pairing
# ---------------------------------------------------------------------------

def _split_names(obs: pd.DataFrame, col: str) -> pd.Series:
    return obs[col].astype(str).str.strip().str.lower().map(SPLIT_TOKENS)


def _label_profile(obs, col, cv_group) -> LabelProfile:
    s = obs[col]
    counts = s.dropna().astype(str).value_counts()
    lab_groups = n_groups = None
    if cv_group:
        n_groups = int(obs[cv_group].nunique())
        lab_groups = int(obs.loc[s.notna(), cv_group].nunique())
    return LabelProfile(column=col, n_classes=len(counts), class_counts=counts.to_dict(),
                        imbalance_ratio=float(counts.max() / max(1, counts.min())),
                        minority_class=str(counts.idxmin()), n_labelled=int(counts.sum()),
                        labelled_fraction=float(counts.sum() / len(s)), labelled_groups=lab_groups,
                        n_groups=n_groups)


def _group_profile(obs, col, role, all_groups, split_col) -> GroupProfile:
    sizes = obs[col].dropna().astype(str).value_counts()
    nested = None
    for other in all_groups:
        if other == col:
            continue
        pairs = obs[[col, other]].dropna().astype(str).drop_duplicates()
        if pairs[col].is_unique and pairs[other].nunique() < pairs[col].nunique():
            nested = other
            break
    overlap = None
    if split_col is not None:
        split = _split_names(obs, split_col)
        levels_in = {s: set(obs.loc[split == s, col].dropna().astype(str)) for s in ("train", "val")}
        overlap = {}
        for s, ref in (("val", "train"), ("test", "train"), ("test", "val")):
            levels = obs.loc[split == s, col].dropna().astype(str)
            if len(levels) and levels_in[ref]:
                overlap[f"{s}_in_{ref}"] = float(levels.isin(levels_in[ref]).mean())
    return GroupProfile(column=col, role=role, n_levels=len(sizes), size_min=int(sizes.min()),
                        size_median=float(sizes.median()), size_max=int(sizes.max()), nested_in=nested,
                        split_overlap=overlap)


def profile(ds: MultimodalDataset, seed: int = 774) -> DataProfile:
    rng = np.random.default_rng(seed)
    obs = ds.obs
    cv_group = coarsest_group(obs, ds.roles)
    libs = ds.columns_with("library")
    library_col = libs[0] if libs else None
    group_cols = ds.columns_with("group") + libs
    label_cols = ds.columns_with("label")
    split_cols = ds.columns_with("split")
    split_col = split_cols[0] if split_cols else None
    group_for_stats = cv_group or library_col
    spatial = next((m for m in ds.modalities.values() if m.kind == "spatial"), None)
    image = next((m for m in ds.modalities.values() if m.kind == "image"), None)

    mods = []
    for m in ds.modalities.values():
        rec = ModalityProfile(name=m.name, kind=m.kind, n_present=m.n_present,
                              missing_fraction=float(1 - m.n_present / max(1, ds.n)), source=m.source)
        if m.kind == "image":
            rec.image = _image_profile(m, obs, group_for_stats, rng)
        elif m.kind == "text":
            rec.text = _text_profile(m)
        elif m.kind == "matrix":
            rec.matrix = _matrix_profile(m, obs, group_for_stats, label_cols[0] if label_cols else None,
                                         spatial, library_col, rng)
        elif m.kind == "tabular":
            rec.tabular = _tabular_profile(m)
        elif m.kind == "spatial":
            rec.spatial = _spatial_profile(m, obs, library_col, image)
        mods.append(rec)

    names = list(ds.modalities)
    pairs = {f"{a}|{b}": int((ds.modalities[a].present & ds.modalities[b].present).sum())
             for i, a in enumerate(names) for b in names[i + 1:]}
    pairing = PairingProfile(present={k: m.n_present for k, m in ds.modalities.items()}, pairs=pairs,
                             n_all_present=int(ds.paired_mask(names).sum()))
    split = None
    if split_col:
        counts = _split_names(obs, split_col).value_counts()
        split = SplitProfile(column=split_col,
                             counts={k: int(counts.get(k, 0)) for k in ("train", "val", "test")
                                     if counts.get(k, 0)})
    return DataProfile(
        name=ds.name, path=str(ds.path), format=ds.format, n_samples=ds.n, modalities=mods,
        pairing=pairing, roles=[RoleRecord(column=r.column, role=r.role, evidence=r.evidence)
                                for r in ds.roles],
        labels=[_label_profile(obs, c, cv_group) for c in label_cols],
        multilabel_columns=ds.columns_with("multilabel"),
        groups=[_group_profile(obs, c, ds.role_of(c), group_cols, split_col) for c in group_cols],
        cv_group=cv_group, split=split, notes=list(ds.notes))
