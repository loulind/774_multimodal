"""Infer what each metadata column *is*, from its contents (and generic naming conventions).

Roles:
    id          unique per row
    split       values are train/val/test tokens (a predefined split)
    library     names the source image of each spot (spatial data)
    group       a dependence unit: patient, donor, slide, source article, batch ...
    label       a categorical outcome the objective may name
    multilabel  delimiter-separated codes, several per row
    text        free text (a modality, not metadata)
    image_path  a path to an image file per row (a modality)
    numeric     numeric features (become a tabular modality)
    metadata    anything else

Every assignment carries its evidence, which goes to the profile and the
decision log. The hint lists are naming *conventions* shared across datasets
(what a patient or a label column is usually called), never dataset names.
Callers can override the label and group roles.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import ColumnRole

MISSING_TOKENS = {"", "nan", "na", "n/a", "none", "null", "unknown", "undetermined",
                  "unlabeled", "unlabelled", "unassigned", "not available", "?", "-"}
SPLIT_TOKENS = {"train": "train", "training": "train", "val": "val", "valid": "val",
                "validation": "val", "dev": "val", "test": "test", "testing": "test",
                "holdout": "test"}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
GROUP_HINTS = ("patient", "donor", "subject", "individual", "participant", "case", "slide",
               "section", "sample", "library", "batch", "site", "center", "centre", "study",
               "source", "article", "paper", "plate", "run", "specimen", "mouse")
LABEL_HINTS = ("label", "class", "type", "diagnosis", "status", "grade", "stage", "subtype",
               "modality", "tissue", "celltype", "cell_type", "cluster", "annotation",
               "category", "target", "outcome", "region", "domain", "condition", "phenotype")
COARSE_GROUP_HINTS = ("patient", "donor", "subject", "individual", "participant", "case",
                      "mouse", "specimen")
DELIMITERS = (";", "|")


def clean_missing(s: pd.Series) -> pd.Series:
    """Map conventional 'no value' tokens to NaN (labels and groups only)."""
    out = s.astype(object).where(s.notna(), np.nan)
    mask = out.map(lambda v: isinstance(v, str) and v.strip().lower() in MISSING_TOKENS)
    return out.mask(mask.astype(bool), np.nan)


def _is_text_like(s: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _name_hit(name: str, hints) -> bool:
    tokens = set(re.split(r"[^a-z0-9]+", name.lower())) | {name.lower()}
    return any(h in tokens or name.lower().endswith(h) for h in hints)


def _image_paths(s: pd.Series, base: Path) -> float:
    vals = s.dropna().astype(str)
    if vals.empty:
        return 0.0
    sample = vals.sample(min(50, len(vals)), random_state=0)
    has_ext = sample.str.lower().str.endswith(IMAGE_EXTENSIONS).mean()
    if has_ext < 0.9:
        return 0.0
    exists = np.mean([(Path(v) if Path(v).is_absolute() else base / v).exists() for v in sample])
    return float(exists)


def _multilabel_stats(s: pd.Series) -> tuple[float, float, str | None]:
    vals = s.dropna().astype(str)
    if vals.empty:
        return 0.0, 1.0, None
    best = (0.0, 1.0, None)
    for d in DELIMITERS:
        frac = vals.str.contains(re.escape(d)).mean()
        if frac > best[0]:
            tokens = [t for v in vals for t in v.split(d) if t]
            ratio = len(set(tokens)) / max(1, len(tokens))
            best = (float(frac), float(ratio), d)
    return best


def infer_roles(obs: pd.DataFrame, base: Path, library_keys: set[str] | None = None,
                label_override: str | None = None, group_override: str | None = None,
                index_is_id: bool = True) -> list[ColumnRole]:
    """Assign a role to every column of ``obs`` (the index is the sample id)."""
    roles: list[ColumnRole] = []
    n = len(obs)
    for col in obs.columns:
        s = obs[col]
        name = str(col)
        nun = s.nunique(dropna=True)
        if library_keys and _is_text_like(s) and set(s.dropna().astype(str)) <= library_keys:
            roles.append(ColumnRole(name, "library", f"values match the {len(library_keys)} "
                                                     "source-image keys"))
            continue
        if _is_text_like(s):
            low = set(s.dropna().astype(str).str.strip().str.lower())
            if 2 <= len(low) <= 4 and low <= set(SPLIT_TOKENS) and \
                    {SPLIT_TOKENS[v] for v in low} >= {"train"} and len(low) >= 2:
                roles.append(ColumnRole(name, "split", f"values {sorted(low)} are split names"))
                continue
            if _image_paths(s, base) >= 0.9:
                roles.append(ColumnRole(name, "image_path", "values are paths to existing image files"))
                continue
            frac, ratio, d = _multilabel_stats(s)
            if frac >= 0.2 and ratio < 0.5:
                roles.append(ColumnRole(name, "multilabel",
                                        f"{frac:.0%} of values hold several '{d}'-separated codes; "
                                        f"{ratio:.1%} of codes are distinct"))
                continue
            vals = s.dropna().astype(str)
            words = vals.str.split().str.len()
            uniq = nun / max(1, len(vals))
            if len(vals) and vals.str.len().median() >= 20 and words.median() >= 4 and uniq >= 0.5:
                roles.append(ColumnRole(name, "text",
                                        f"free text: median {int(words.median())} words, "
                                        f"{uniq:.0%} distinct"))
                continue
        if nun == n and n > 1 and (_is_text_like(s) or pd.api.types.is_integer_dtype(s)):
            roles.append(ColumnRole(name, "id", "unique per row"))
            continue
        cleaned = clean_missing(s) if _is_text_like(s) else s
        k = cleaned.nunique(dropna=True)
        multi_member = k >= 2 and cleaned.value_counts().max() >= 2
        if name == group_override or (_name_hit(name, GROUP_HINTS) and multi_member and
                                      (k <= n / 2 or _name_hit(name, ("article", "paper",
                                                                      "source", "study")))):
            roles.append(ColumnRole(name, "group", f"{k} levels; name suggests a dependence unit"))
            continue
        if name == label_override or (2 <= k <= 100 and (_is_text_like(s) or
                                                          (_name_hit(name, LABEL_HINTS) and k <= 20))):
            hint = "named by the user" if name == label_override else \
                ("name suggests an outcome" if _name_hit(name, LABEL_HINTS) else "categorical")
            roles.append(ColumnRole(name, "label", f"{k} classes; {hint}"))
            continue
        if pd.api.types.is_numeric_dtype(s):
            roles.append(ColumnRole(name, "numeric", f"numeric, {nun} distinct values"))
            continue
        roles.append(ColumnRole(name, "metadata", f"{nun} distinct values"))
    _nest_groups(obs, roles)
    return roles


def _partitions(obs: pd.DataFrame, fine: str, coarse: str) -> bool:
    """True when every level of ``fine`` maps to exactly one level of ``coarse``."""
    pairs = obs[[fine, coarse]].dropna().astype(str).drop_duplicates()
    return len(pairs) > 0 and pairs[fine].is_unique and pairs[coarse].nunique() < pairs[fine].nunique()


def _nest_groups(obs: pd.DataFrame, roles: list[ColumnRole]) -> None:
    """A label column that the library (slide) column nests into is really a coarser group.

    Example structure: spots within slides within patients. Labels vary within
    a slide; a patient column does not. So a categorical column that every
    library maps into exactly one level of is a grouping, not an outcome. Only
    libraries are used: with small groups (say, two images per article) a label
    can be constant within every group by chance.
    """
    libs = [r.column for r in roles if r.role == "library"]
    for r in roles:
        if r.role != "label":
            continue
        for lib in libs:
            if _partitions(obs, lib, r.column) and obs[r.column].nunique() >= 2:
                r.role = "group"
                r.evidence = (f"{obs[r.column].nunique()} levels; each '{lib}' level belongs to "
                              "exactly one level (nested grouping)")
                break


def coarsest_group(obs: pd.DataFrame, roles: list[ColumnRole], min_levels: int = 3) -> str | None:
    """The grouping to hold out together: the coarsest one that nests the others.

    Spots on one slide are dependent, and so are slides from one patient, so the
    unit of independent replication is the coarsest nesting level -- provided it
    still has enough levels to split.
    """
    cands = [r.column for r in roles if r.role in ("group", "library")]
    cands = [c for c in cands if obs[c].nunique() >= min_levels]
    if not cands:
        return None
    coarse = [c for c in cands if any(_partitions(obs, o, c) for o in cands if o != c)]
    pool = coarse or cands
    named = [c for c in pool if _name_hit(c, COARSE_GROUP_HINTS)]
    return min(named or pool, key=lambda c: obs[c].nunique())
