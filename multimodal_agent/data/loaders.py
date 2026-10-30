"""Read any supported layout into a ``MultimodalDataset``.

The loader reports structure -- which modalities exist, how they are paired,
what each metadata column is -- and nothing else. Deciding how to preprocess,
encode or align them belongs to the planner. There is no lookup by dataset name
anywhere in this module: a dataset is understood only through its format and
contents.

Supported layouts (a file, or a directory holding one of them):

  spatial AnnData   .h5ad with obsm['spatial'] (spot pixel coordinates) and, in
                    uns['spatial'][library], a source image path and spot
                    diameter; gives image patches + expression matrix + spatial.
  AnnData           .h5ad without coordinates; an obs column of image paths
                    adds an image modality.
  table             .csv / .tsv / .parquet with an image-path column, free-text
                    columns and/or numeric columns (image + text, image + table).
  10x HDF5          .h5 with several feature types (e.g. gene expression and
                    chromatin peaks) -> one matrix modality per feature type.
  paired matrices   a directory of several .h5ad files whose rows share ids.

A directory may also hold ``dataset.yaml`` naming the file and overriding
column roles (``label``, ``group``); roles are otherwise inferred.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .dataset import FileImageSource, Modality, MultimodalDataset, SpotPatchSource
from .roles import clean_missing, infer_roles

TABLE_SUFFIXES = (".csv", ".tsv", ".txt", ".parquet")
PEAK_NAME = re.compile(r"^(chr)?[0-9XYMxy]{1,2}[:_-]\d+[-_]\d+$")


def load_path(path: str | Path, label_column: str | None = None,
              group_column: str | None = None) -> MultimodalDataset:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.is_dir():
        return _load_dir(p, label_column, group_column)
    suffix = p.suffix.lower()
    if suffix == ".h5ad":
        return _load_h5ad(p, label_column, group_column)
    if suffix in TABLE_SUFFIXES:
        return _load_table(p, label_column, group_column)
    if suffix == ".h5":
        return _load_10x_h5(p, label_column, group_column)
    raise ValueError(f"unsupported file type: {p.name}")


def _load_dir(p: Path, label_column, group_column) -> MultimodalDataset:
    manifest = p / "dataset.yaml"
    if manifest.exists():
        import yaml

        spec = yaml.safe_load(manifest.read_text()) or {}
        roles = spec.get("roles", {})
        files = spec.get("files") or [spec["file"]]
        label_column = label_column or roles.get("label")
        group_column = group_column or roles.get("group")
        if len(files) > 1:
            return _load_paired_h5ad([p / f for f in files], label_column, group_column)
        return load_path(p / files[0], label_column, group_column)
    h5ads = sorted(p.glob("*.h5ad"))
    tables = sorted(f for f in p.iterdir() if f.suffix.lower() in TABLE_SUFFIXES
                    and not f.name.startswith("."))
    h5s = sorted(p.glob("*.h5"))
    if len(h5ads) == 1 and not tables:
        return _load_h5ad(h5ads[0], label_column, group_column)
    if len(h5ads) > 1:
        return _load_paired_h5ad(h5ads, label_column, group_column)
    if len(tables) == 1:
        return _load_table(tables[0], label_column, group_column)
    if len(h5s) == 1:
        return _load_10x_h5(h5s[0], label_column, group_column)
    raise ValueError(f"cannot tell which file in {p} holds the data; add dataset.yaml "
                     f"(found {[f.name for f in h5ads + tables + h5s]})")


# ---------------------------------------------------------------------------
# matrices
# ---------------------------------------------------------------------------

def _as_csr(X) -> sp.csr_matrix:
    if sp.issparse(X):
        return X.tocsr()
    return sp.csr_matrix(np.asarray(X))


def _matrix_name(feature_names: list[str], default: str) -> str:
    sample = feature_names[: min(200, len(feature_names))]
    if sample and np.mean([bool(PEAK_NAME.match(str(f))) for f in sample]) > 0.8:
        return "peaks"
    return default


def _matrix_modality(name: str, X, features, source: str) -> Modality:
    X = _as_csr(X)
    present = np.asarray(X.getnnz(axis=1) > 0)
    return Modality(name=name, kind="matrix", present=present, data=X,
                    feature_names=[str(f) for f in features], source=source)


def _image_column_modality(obs: pd.DataFrame, col: str, base: Path) -> Modality:
    paths = []
    for v in obs[col]:
        if isinstance(v, str) and v:
            q = Path(v) if Path(v).is_absolute() else base / v
            paths.append(q if q.exists() else None)
        else:
            paths.append(None)
    present = np.array([q is not None for q in paths])
    return Modality(name="image" if col.lower() in ("image", "image_path", "path", "file")
                    else f"image:{col}", kind="image", present=present,
                    images=FileImageSource(paths), source=f"column '{col}'")


def _finish(name: str, path: Path, fmt: str, obs: pd.DataFrame, modalities: dict[str, Modality],
            base: Path, label_column, group_column, library_keys=None,
            notes: list[str] | None = None) -> MultimodalDataset:
    roles = infer_roles(obs, base, library_keys, label_column, group_column)
    for r in roles:
        if r.role in ("label", "group"):
            obs[r.column] = clean_missing(obs[r.column])
        if r.role == "image_path" and not any(m.source == f"column '{r.column}'"
                                               for m in modalities.values()):
            m = _image_column_modality(obs, r.column, base)
            modalities[m.name] = m
        if r.role == "text":
            vals = obs[r.column].astype(object).where(obs[r.column].notna(), None).to_numpy()
            present = np.array([isinstance(v, str) and bool(v.strip()) for v in vals])
            name = "text" if not any(m.kind == "text" for m in modalities.values()) else f"text:{r.column}"
            modalities[name] = Modality(name=name, kind="text", present=present, data=vals,
                                        source=f"column '{r.column}'")
    numeric = [r.column for r in roles if r.role == "numeric"]
    if fmt == "table" and len(numeric) >= 2:
        tab = obs[numeric].apply(pd.to_numeric, errors="coerce")
        modalities["tabular"] = Modality(name="tabular", kind="tabular",
                                         present=tab.notna().any(axis=1).to_numpy(), data=tab,
                                         feature_names=numeric, source=f"{len(numeric)} numeric columns")
    return MultimodalDataset(name=name, path=path, format=fmt, obs=obs, modalities=modalities,
                             roles=roles, notes=notes or [])


# ---------------------------------------------------------------------------
# AnnData
# ---------------------------------------------------------------------------

def _load_h5ad(path: Path, label_column, group_column) -> MultimodalDataset:
    import anndata as ad

    adata = ad.read_h5ad(path)
    obs = adata.obs.copy()
    obs.index = obs.index.astype(str)
    base = path.parent
    X = adata.layers["counts"] if "counts" in adata.layers else adata.X
    mods: dict[str, Modality] = {}
    feats = list(adata.var_names)
    m = _matrix_modality(_matrix_name(feats, "expression"), X, feats, f"{path.name}:X")
    mods[m.name] = m
    notes, library_keys, fmt = [], None, "anndata"
    if "spatial" in adata.obsm:
        xy = np.asarray(adata.obsm["spatial"], float)[:, :2]
        mods["spatial"] = Modality(name="spatial", kind="spatial", present=np.isfinite(xy).all(1),
                                   data=xy, source="obsm['spatial']")
        fmt = "spatial_anndata"
        libs = adata.uns.get("spatial", {}) or {}
        library_keys = {str(k) for k in libs}
        lib_col = next((c for c in obs.columns
                        if not pd.api.types.is_numeric_dtype(obs[c])
                        and set(obs[c].dropna().astype(str)) <= library_keys), None)
        if lib_col is None and len(library_keys) == 1:
            lib_col = "__library__"
            obs[lib_col] = next(iter(library_keys))
        image_path, diameter = {}, {}
        for k, v in libs.items():
            meta = v.get("metadata", {}) or {}
            src = meta.get("source_image_path")
            dia = (v.get("scalefactors", {}) or {}).get("spot_diameter_fullres")
            if src and dia:
                q = Path(src) if Path(src).is_absolute() else base / src
                if q.exists():
                    image_path[str(k)], diameter[str(k)] = q, float(dia)
                else:
                    notes.append(f"source image for '{k}' not found: {q}")
        if lib_col and image_path:
            libs_of = obs[lib_col].astype(str).to_numpy()
            present = np.isin(libs_of, list(image_path)) & mods["spatial"].present
            mods["image"] = Modality(name="image", kind="image", present=present,
                                     images=SpotPatchSource(libs_of, xy, diameter, image_path),
                                     source=f"patches around spot centres ({len(image_path)} "
                                            "source images)")
    return _finish(path.stem, path, fmt, obs, mods, base, label_column, group_column,
                   library_keys, notes)


def _load_paired_h5ad(paths: list[Path], label_column, group_column) -> MultimodalDataset:
    import anndata as ad

    adatas = [ad.read_h5ad(q) for q in paths]
    ids = pd.Index(sorted(set().union(*[set(a.obs_names.astype(str)) for a in adatas])))
    obs = pd.DataFrame(index=ids)
    mods = {}
    for q, a in zip(paths, adatas):
        a.obs_names = a.obs_names.astype(str)
        for c in a.obs.columns:
            if c not in obs.columns:
                obs[c] = a.obs[c].reindex(ids)
        X = _as_csr(a.layers["counts"] if "counts" in a.layers else a.X)
        rows = ids.get_indexer(a.obs_names)
        full = sp.csr_matrix((len(ids), X.shape[1]), dtype=X.dtype).tolil()
        full[rows] = X
        name = _matrix_name(list(a.var_names), q.stem)
        m = _matrix_modality(name if name not in mods else q.stem, full.tocsr(), a.var_names, q.name)
        m.present = np.isin(np.arange(len(ids)), rows)
        mods[m.name] = m
    return _finish(paths[0].parent.name, paths[0].parent, "paired_matrices", obs, mods,
                   paths[0].parent, label_column, group_column)


def _load_10x_h5(path: Path, label_column, group_column) -> MultimodalDataset:
    import scanpy as sc

    adata = sc.read_10x_h5(path, gex_only=False)
    adata.var_names_make_unique()
    obs = pd.DataFrame(index=adata.obs_names.astype(str))
    mods = {}
    types = adata.var.get("feature_types", pd.Series("matrix", index=adata.var_names))
    for t in pd.unique(types):
        keep = (types == t).to_numpy()
        name = re.sub(r"[^a-z0-9]+", "_", str(t).lower()).strip("_") or "matrix"
        mods[name] = _matrix_modality(name, adata.X[:, keep], adata.var_names[keep],
                                      f"{path.name}:{t}")
    return _finish(path.stem, path, "paired_matrices" if len(mods) > 1 else "anndata", obs, mods,
                   path.parent, label_column, group_column)


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    sep = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
    return pd.read_csv(path, sep=sep, low_memory=False)


def _load_table(path: Path, label_column, group_column) -> MultimodalDataset:
    df = _read_table(path)
    first = df.columns[0]
    if df[first].is_unique and not pd.api.types.is_float_dtype(df[first]):
        df = df.set_index(first)
    df.index = df.index.astype(str)
    return _finish(path.stem if path.stem not in ("pairs", "data", "table", "samples")
                   else path.parent.name, path, "table", df, {}, path.parent, label_column,
                   group_column)
