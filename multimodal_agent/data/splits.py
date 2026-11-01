"""Turn a SplitSpec into train / validation / test folds of sample positions.

Three strategies, chosen by the planner:

  predefined      the dataset's own split column (validation carved from
                  training, by group when groups exist, if none is given).
  grouped_kfold   folds over a grouping column (patient, slide, source...):
                  every group is tested exactly once, and within each fold one
                  training group (more, if one is too small) is held out for
                  validation (early stopping and model selection). With
                  n_folds = 1, one held-out set of groups.
  random          stratified random split (labels as strata when present).

Only samples where both modalities of the aligned pair are present are split;
unpaired samples are reported, not silently dropped. Groups are dealt to folds
so that labelled samples are spread evenly, which keeps label-based metrics
defined in every fold when only some groups carry labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..schema import SplitSpec
from .roles import SPLIT_TOKENS


@dataclass
class Fold:
    name: str
    train: np.ndarray                     # positions into the dataset's rows
    val: np.ndarray
    test: np.ndarray
    test_groups: list[str] = field(default_factory=list)
    val_groups: list[str] = field(default_factory=list)

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def _carve(idx: np.ndarray, frac: float, rng, strata: np.ndarray | None,
           groups: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Split idx into (rest, held) with ~frac held, by group if given, else stratified."""
    if len(idx) < 2:
        return idx, idx[:0]
    if groups is not None and len(np.unique(groups[idx])) >= 3:
        levels = rng.permutation(np.unique(groups[idx]))
        target, held = frac * len(idx), []
        for lv in levels:
            if sum((groups[idx] == h).sum() for h in held) >= target:
                break
            held.append(lv)
        mask = np.isin(groups[idx], held)
        return idx[~mask], idx[mask]
    n_held = max(1, int(round(frac * len(idx))))
    if strata is not None:
        held = []
        s = strata[idx]
        for lv in np.unique(s):
            members = idx[s == lv]
            k = int(round(frac * len(members)))
            held.extend(rng.choice(members, k, replace=False) if k else [])
        held = np.array(sorted(held), dtype=int)
        if len(held) == 0:
            held = rng.choice(idx, n_held, replace=False)
    else:
        held = rng.choice(idx, n_held, replace=False)
    mask = np.isin(idx, held)
    return idx[~mask], idx[mask]


def _strata(obs: pd.DataFrame, column: str | None) -> np.ndarray | None:
    if column is None or column not in obs:
        return None
    return obs[column].astype(object).where(obs[column].notna(), "__unlabelled__").astype(str).to_numpy()


def subsample(obs: pd.DataFrame, idx: np.ndarray, n: int, rng, group_column: str | None) -> np.ndarray:
    """Cap the number of samples (quick mode), keeping every group represented."""
    if n is None or len(idx) <= n:
        return idx
    if group_column is not None:
        g = obs[group_column].astype(str).to_numpy()[idx]
        out = []
        for lv in np.unique(g):
            members = idx[g == lv]
            k = max(1, int(round(n * len(members) / len(idx))))
            out.extend(rng.choice(members, min(k, len(members)), replace=False))
        return np.sort(np.array(out, dtype=int))
    return np.sort(rng.choice(idx, n, replace=False))


def make_folds(obs: pd.DataFrame, paired: np.ndarray, spec: SplitSpec, seed: int,
               split_column: str | None = None, label_column: str | None = None,
               max_samples: int | None = None) -> tuple[list[Fold], list[str]]:
    rng = np.random.default_rng(seed)
    notes = []
    idx = np.flatnonzero(paired)
    if (~paired).any():
        notes.append(f"{int((~paired).sum())} samples lack one of the paired modalities and are "
                     "not used for alignment")
    idx = subsample(obs, idx, max_samples, rng, spec.group_column)
    if max_samples and len(idx) < paired.sum():
        notes.append(f"subsampled to {len(idx)} paired samples (budget cap)")
    strata = _strata(obs, spec.stratify_column or label_column)
    groups = obs[spec.group_column].astype(str).to_numpy() if spec.group_column else None

    if spec.strategy == "predefined":
        split = obs[split_column].astype(str).str.strip().str.lower().map(SPLIT_TOKENS).to_numpy()
        tr, va, te = (idx[split[idx] == s] for s in ("train", "val", "test"))
        if len(va) == 0:
            tr, va = _carve(tr, spec.val_fraction, rng, strata, groups)
            notes.append(f"no validation split given; {len(va)} training samples held out for validation")
        if len(te) == 0:
            tr, te = _carve(tr, spec.test_fraction, rng, strata, groups)
            notes.append(f"no test split given; {len(te)} training samples held out for testing")
        return [Fold("predefined", tr, va, te)], notes

    if spec.strategy == "grouped_kfold":
        levels = np.unique(groups[idx])
        labelled = {lv: int((strata[idx][groups[idx] == lv] != "__unlabelled__").sum())
                    if strata is not None else 0 for lv in levels}
        size = {lv: int((groups[idx] == lv).sum()) for lv in levels}
        order = sorted(rng.permutation(levels), key=lambda lv: (-labelled[lv], -size[lv]))
        k = max(1, spec.n_folds)
        if k == 1:
            n_test = max(1, int(round(spec.test_fraction * len(levels))))
            buckets = [order[:: max(1, len(levels) // n_test)][:n_test]]
        else:
            buckets = [[] for _ in range(k)]
            totals = [(0, 0)] * k
            for lv in order:
                j = min(range(k), key=lambda j: totals[j])
                buckets[j].append(lv)
                totals[j] = (totals[j][0] + labelled[lv], totals[j][1] + size[lv])
        folds = []
        for j, test_levels in enumerate(buckets):
            test = idx[np.isin(groups[idx], test_levels)]
            rest_levels = [lv for lv in order if lv not in test_levels]
            if len(rest_levels) >= 3:
                pool = [lv for lv in rest_levels if labelled[lv] > 0] or rest_levels
                val_levels = [pool[rng.integers(len(pool))]]
                # With groups of unequal size one group can be far too small to select models on:
                # add groups, in random order, until validation holds val_fraction of the remaining
                # samples, keeping at least two training groups.
                target = spec.val_fraction * sum(size[lv] for lv in rest_levels)
                if size[val_levels[0]] < target:
                    more = [lv for lv in rng.permutation(pool) if lv not in val_levels] + \
                           [lv for lv in rng.permutation(rest_levels) if lv not in pool]
                    for lv in more:
                        if sum(size[v] for v in val_levels) >= target or len(val_levels) >= len(rest_levels) - 2:
                            break
                        val_levels.append(lv)
                val = idx[np.isin(groups[idx], val_levels)]
                train = idx[np.isin(groups[idx], [lv for lv in rest_levels if lv not in val_levels])]
            else:
                rest = idx[np.isin(groups[idx], rest_levels)]
                train, val = _carve(rest, spec.val_fraction, rng, strata, None)
                val_levels = []
                notes.append(f"fold {j + 1}: too few groups for a validation group; "
                             "validation carved from training samples")
            folds.append(Fold(f"fold{j + 1}", np.sort(train), np.sort(val), np.sort(test),
                              [str(x) for x in test_levels], [str(x) for x in val_levels]))
        return folds, notes

    # random, stratified
    rest, test = _carve(idx, spec.test_fraction, rng, strata, None)
    train, val = _carve(rest, spec.val_fraction / max(1e-9, 1 - spec.test_fraction), rng, strata, None)
    return [Fold("random", np.sort(train), np.sort(val), np.sort(test))], notes
