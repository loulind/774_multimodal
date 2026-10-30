"""Assemble a configuration's training, validation and test inputs for one fold.

Everything learned from data is fitted on the fold's *training* samples only and
then applied to validation and test samples:

  * matrix / tabular modalities: normalisation, variable features, PCA;
  * the regression target (cross-modal prediction): the target modality's
    variable features, standardised with training means and spreads;
  * label codes: classes are those seen in training; anything else is "unlabelled".

Frozen image/text features do not depend on the fold, so they are computed once
for all samples (and cached) by the caller and only indexed here. Soft-InfoNCE
targets use the model input of the most redundant modality (highest measured
near-twin share), a fixed similarity that the training cannot change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data.splits import Fold
from ..preprocess.matrix import MatrixPreprocessor
from ..schema import ConfigSpec, DataProfile, Plan
from .joint import FitData


@dataclass
class Assembled:
    train: FitData
    val: FitData
    test: FitData
    classes: list[str] = field(default_factory=list)
    target_features: list[str] = field(default_factory=list)
    target_variance_order: np.ndarray | None = None     # target columns, most variable first
    preprocessors: dict = field(default_factory=dict)
    soft_modality: str | None = None
    notes: list[str] = field(default_factory=list)


def label_codes(values, train_idx: np.ndarray) -> tuple[np.ndarray, list[str]]:
    s = np.array([None if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v) for v in values],
                 dtype=object)
    classes = sorted({v for v in s[train_idx] if v is not None})
    lookup = {c: k for k, c in enumerate(classes)}
    return np.array([lookup.get(v, -1) if v is not None else -1 for v in s], dtype=np.int64), classes


def _redundant(profile: DataProfile, names: list[str]) -> str | None:
    best, who = -1.0, None
    for n in names:
        m = profile.modality(n)
        v = (m.matrix.near_twin_fraction if m.matrix else
             m.text.near_duplicate_fraction if m.text else
             (m.image.near_duplicate_fraction or 0.0) if m.image else 0.0)
        if v > best:
            best, who = v, n
    return who


def assemble(ds, profile: DataProfile, plan: Plan, cfg: ConfigSpec, fold: Fold,
             frozen: dict[str, np.ndarray]) -> Assembled:
    """``frozen[m]`` holds pretrained features for *every* dataset row of modality m."""
    tr, va, te = fold.train, fold.val, fold.test
    inputs, pre = {}, {}
    for mp in cfg.modalities:
        mod = ds.modalities[mp.modality]
        if mod.kind in ("matrix", "tabular"):
            X = mod.data if mod.kind == "matrix" else mod.data.to_numpy(np.float32)
            mx = profile.modality(mp.modality).matrix
            p = MatrixPreprocessor(mp.n_variable_features or 2000, mp.matrix_input or "log_normalised",
                                   mp.pca_components, counts=bool(mx and mx.integer_valued and mx.nonnegative))
            pre[mp.modality] = p.fit(X[tr])
            inputs[mp.modality] = p.transform(X)
        else:
            inputs[mp.modality] = frozen[mp.modality]

    labels, classes = (None, [])
    if plan.evaluation.label_column:
        labels, classes = label_codes(ds.obs[plan.evaluation.label_column].to_numpy(), tr)

    target, names, order = None, [], None
    pred = plan.task.prediction
    if pred is not None and ds.modalities[pred.target].kind in ("matrix", "tabular"):
        mod = ds.modalities[pred.target]
        X = mod.data if mod.kind == "matrix" else mod.data.to_numpy(np.float32)
        mp = next(m for m in cfg.modalities if m.modality == pred.target)
        mx = profile.modality(pred.target).matrix
        tp = MatrixPreprocessor(mp.n_variable_features or 2000, "log_normalised",
                                counts=bool(mx and mx.integer_valued and mx.nonnegative)).fit(X[tr])
        target = tp.standardised(X)
        names = [mod.feature_names[j] for j in tp.features] if mod.feature_names else []
        var = np.asarray(tp._norm(X[tr])[:, tp.features].toarray().var(0))
        order = np.argsort(-var)
        pre["target"] = tp

    soft = None
    if any(t.name == "soft_infonce" for t in cfg.losses):
        who = _redundant(profile, [m.modality for m in cfg.modalities])
        soft = inputs[who] if who else None
    else:
        who = None

    def part(idx):
        return FitData({m: x[idx] for m, x in inputs.items()},
                       None if labels is None else labels[idx],
                       None if target is None else target[idx],
                       None if soft is None else soft[idx])

    return Assembled(part(tr), part(va), part(te), classes, names, order, pre, who)
