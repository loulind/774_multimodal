"""The output folder of a run: tables, figures, embeddings, prose and the report itself.

    report.md / report.html    the analysis report; its prose comes from the LLM writer
                               when available, otherwise from templates that quote the same
                               numbers, and every number in it is checked against the outputs
    methods_summary.md         methods and preprocessing, generated from the executed plan
    plan.md                    the plan in words
    decision_log.json          every decision, its rationale and the alternatives rejected
    evaluation.json            all validation and test results
    tables/*.csv               the numbers behind every table and figure
    figures/*.png              key figures
    downstream/*.json          retrieval examples, prediction maps, confusion, clusters
    embeddings/                held-out joint embeddings (float16, size-capped)
    profile.json, screening.json, plan.json, model_summary.json, run_manifest.json,
    llm_transcript.json        (every LLM prompt and reply; --replay reads it)
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from . import figures as F
from . import writer
from .config import REPO_ROOT
from .encoders.registry import REGISTRY
from .evaluate import suite as S
from .evaluate.metrics import l2n

EXPORT_MAX_BYTES = 5_000_000
COMPARISON_METRICS = ["retrieval_mean_recall_at_10", "prediction_mean_pearson_top50", "probe_macro_f1",
                      "clustering_ari"]
METRIC_LABEL = {"retrieval_mean_recall_at_10": "recall@10 (mean of both directions)",
                "prediction_mean_pearson_top50": "mean Pearson r over the 50 most variable genes",
                "probe_macro_f1": "linear-probe macro-F1", "clustering_ari": "k-means adjusted Rand index"}
SHORT_LABEL = {"retrieval_mean_recall_at_10": "recall@10", "prediction_mean_pearson_top50": "mean gene r",
               "probe_macro_f1": "probe macro-F1", "clustering_ari": "clustering ARI"}
BOOT_FAMILY = {"retrieval_mean_recall_at_10": "retrieval", "probe_macro_f1": "probe",
               "prediction_mean_pearson_top50": "prediction"}
PACKAGES = ["torch", "torchvision", "transformers", "open-clip-torch", "timm", "scikit-learn", "numpy", "scipy",
            "pandas", "anndata", "scanpy", "umap-learn", "pydantic", "claude-agent-sdk"]
DECISION_STEPS = ("objective", "split", "strategy", "encoder", "regime", "preprocessing", "losses",
                  "hyperparameters", "comparison", "evaluation", "downstream", "planner", "critic", "writer")


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=_json_default))


def _round(obj, nd: int = 4):
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else round(float(obj), nd)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, nd) for v in obj]
    return obj


def f3(x) -> str:
    return "–" if x is None else f"{x:.3f}"


def cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def pct(x) -> str:
    return "–" if x is None else f"{100 * x:.0f}%"


INT_COLUMNS = {"n_genes_r_gt_0.3", "best_epoch", "epochs_run", "n_trainable_parameters", "folds_primary_better",
               "n_folds", "embed_dim", "train_seconds", "dim"}


def md_table(rows: list[dict], cols: list[tuple[str, str]], digits: int = 3) -> str:
    if not rows:
        return "_(none)_"

    def cell(v, col=None):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "–"
        if col in INT_COLUMNS and isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.1f}"
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        if isinstance(v, (float, np.floating)):
            return f"{v:.{digits}f}"
        if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
            return f"{v[0]:.{digits}f} to {v[1]:.{digits}f}"
        return str(v).replace("|", "/").replace("\n", " ")

    head = "| " + " | ".join(c[1] for c in cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    return head + "\n".join("| " + " | ".join(cell(r.get(c[0]), c[0]) for c in cols) + " |" for r in rows)


# ---------------------------------------------------------------------------
# Result tables (test data)
# ---------------------------------------------------------------------------

def metric_rows(agent, metric: str) -> list[dict]:
    """One row per representation (and view) for a metric family, over test folds, with the
    across-fold SD (several folds) or a paired-bootstrap interval (one held-out set)."""
    folds, primary = agent.test_folds, agent.primary_name
    values: dict[tuple, list[float]] = {}
    for f in folds:
        if metric == "retrieval_mean_recall_at_10":
            items = [((rep, None), r["mean"]["R@10"]) for rep, r in f["retrieval"].items()]
        elif metric == "prediction_mean_pearson_top50":
            items = [((m, None), r["mean_r_top50"]) for m, r in f["prediction"].items()]
        elif metric == "probe_macro_f1":
            items = [((rep, v), r["macro_f1"]) for rep, views in f["probe"].items() for v, r in views.items()]
        else:
            items = [((rep, v), r["ari"]) for rep, views in f["clustering"].items() for v, r in views.items()]
        for key, val in items:
            values.setdefault(key, []).append(float(val))
    n, fam, rows = len(folds), BOOT_FAMILY.get(metric), []
    for (rep, view), vals in values.items():
        if len(vals) != n:
            continue
        mean = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if n > 1 else None
        lo = hi = None
        interval = ""
        if n > 1:
            lo, hi, interval = mean - sd, mean + sd, f"±1 SD over {n} folds"
        elif agent.test_boot is not None and fam:
            s = agent.test_boot.get(fam, {}).get(rep if view is None else f"{rep}/{view}")
            if s is not None:
                lo, hi = (float(v) for v in np.percentile(s, [2.5, 97.5]))
                interval = "bootstrap 95% interval"
        base = rep.split(":")[0]
        rows.append({"metric": metric, "representation": rep, "view": view,
                     "name": rep if view is None else f"{rep} ({view})",
                     "kind": "configuration" if base in agent.configs else "baseline",
                     "value": mean, "sd": sd, "lo": lo, "hi": hi, "interval": interval,
                     "per_fold": [round(v, 4) for v in vals], "is_primary": base == primary})
    return sorted(rows, key=lambda r: -r["value"])


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def retrieval_table(agent) -> list[dict]:
    acc: dict[str, dict[str, list]] = {}
    for f in agent.test_folds:
        for rep, r in f["retrieval"].items():
            d = acc.setdefault(rep, {})
            for k in ("R@1", "R@5", "R@10", "median_rank"):
                d.setdefault(k, []).append(r["mean"].get(k))
            d.setdefault("a_to_b_R@10", []).append(r["a_to_b"].get("R@10"))
            d.setdefault("b_to_a_R@10", []).append(r["b_to_a"].get("R@10"))
            d.setdefault("chance_R@10", []).append(r["chance"].get("R@10"))
            lm = r.get("label_match") or {}
            d.setdefault("label_match_at_10", []).append(lm.get("mean"))
            d.setdefault("label_match_chance", []).append(lm.get("chance"))
            d.setdefault("gallery1000_R@10", []).append(r.get("mean_R@10_gallery1000"))
    return [{"representation": rep, **{k: _mean(v) for k, v in d.items()}} for rep, d in acc.items()]


def prediction_table(agent) -> list[dict]:
    acc: dict[str, dict[str, list]] = {}
    keys = ("mean_r_top50", "median_r_top50", "mean_r_all", "within_group_mean_r_top50", "n_genes_r_gt_0.3")
    for f in agent.test_folds:
        for meth, r in f["prediction"].items():
            d = acc.setdefault(meth, {})
            for k in keys:
                d.setdefault(k, []).append(r.get(k))
    return sorted([{"method": m, **{k: _mean(v) for k, v in d.items()}} for m, d in acc.items()],
                  key=lambda r: -(r["mean_r_top50"] or -9))


def classification_table(agent) -> list[dict]:
    acc: dict[tuple, dict[str, list]] = {}
    for f in agent.test_folds:
        for rep, views in f["probe"].items():
            for v, r in views.items():
                d = acc.setdefault((rep, v), {})
                d.setdefault("probe_macro_f1", []).append(r.get("macro_f1"))
                d.setdefault("probe_balanced_accuracy", []).append(r.get("balanced_accuracy"))
                d.setdefault("knn_macro_f1", []).append(f["knn"].get(rep, {}).get(v, {}).get("macro_f1"))
                c = f["clustering"].get(rep, {}).get(v, {})
                d.setdefault("cluster_ari", []).append(c.get("ari"))
                d.setdefault("cluster_nmi", []).append(c.get("nmi"))
    return sorted([{"representation": rep, "view": v, **{k: _mean(x) for k, x in d.items()}}
                   for (rep, v), d in acc.items()], key=lambda r: -(r["probe_macro_f1"] or -9))


def geometry_table(agent) -> list[dict]:
    acc: dict[str, dict[str, list]] = {}
    for f in agent.test_folds:
        for rep, g in f["geometry"].items():
            d = acc.setdefault(rep, {})
            for k in ("alignment", "uniformity_a", "uniformity_b", "modality_gap", "effective_rank_a",
                      "effective_rank_b", "dim"):
                d.setdefault(k, []).append(g.get(k))
            gs = f["group_silhouette"].get(rep, {})
            d.setdefault("group_silhouette", []).append(gs.get("fused", next(iter(gs.values()), None)) if gs else None)
    return [{"representation": rep, **{k: _mean(v) for k, v in d.items()}} for rep, d in acc.items()]


def training_table(agent) -> list[dict]:
    rows = []
    for (name, i), rec in agent.fits.items():
        r = rec.record
        rows.append({"configuration": name, "fold": agent.folds[i].name, "best_epoch": r["best_epoch"],
                     "epochs_run": r["epochs_run"], "stopped_because": r["stopped_because"],
                     "best_val_score": r["best_val_score"], "selection_metric": r["selection_metric"],
                     "final_temperature": r.get("final_temperature"),
                     "n_trainable_parameters": r["n_trainable_parameters"], "seconds": r.get("seconds")})
    return rows


def decision_rows(agent) -> list[dict]:
    return [{"step": d.step, "decision": d.decision, "chosen": d.chosen, "by": d.source,
             "rationale": d.rationale} for d in agent.log.entries if d.step in DECISION_STEPS]


def write_tables(agent) -> dict[str, list[dict]]:
    t = {"screening": agent.screening_table.get("rows", []),
         "retrieval": retrieval_table(agent), "prediction": prediction_table(agent),
         "classification": classification_table(agent), "geometry": geometry_table(agent),
         "comparisons_test": agent.test_comparisons, "comparisons_validation": agent.val_comparisons,
         "flags_test": agent.test_flags, "flags_validation": agent.val_flags,
         "training": training_table(agent), "decisions": decision_rows(agent)}
    for m in COMPARISON_METRICS:
        t[f"test_{m}"] = metric_rows(agent, m)
    for name, rows in t.items():
        if rows:
            pd.DataFrame([{k: (json.dumps(v, default=_json_default) if isinstance(v, (list, dict)) else v)
                           for k, v in r.items()} for r in rows]).to_csv(agent.out / "tables" / f"{name}.csv",
                                                                       index=False)
    return t


# ---------------------------------------------------------------------------
# Figures, downstream files, embeddings
# ---------------------------------------------------------------------------

def _content_fn(agent, fold, mplan_by_mod):
    ds = agent.ds
    image_mod = next((m for m in agent.pair if ds.modalities[m].kind == "image"), None)

    def content(position: int, modality: str):
        row = int(fold.test[position])
        kind = ds.modalities[modality].kind
        show = modality if kind in ("image", "text") else image_mod
        if show is None:
            return "text", f"sample {ds.ids[row]}"
        if ds.modalities[show].kind == "text":
            return "text", ds.modalities[show].data[row]
        src = ds.modalities[show].images
        ctx = mplan_by_mod.get(show).patch_context if mplan_by_mod.get(show) else None
        if ctx is not None and hasattr(src, "with_context"):
            src = src.with_context(ctx)
        return "image", src.load(row).convert("RGB")
    return content


def make_figures(agent, tables: dict) -> dict[str, str]:
    figs, fdir = {}, agent.out / "figures"
    d = agent.downstream_index
    fold = agent.folds[d]
    pm = agent.plan.evaluation.primary_metric
    panels = []
    for m in [pm] + [x for x in COMPARISON_METRICS if x != pm]:
        rows = tables.get(f"test_{m}") or []
        if m in ("probe_macro_f1", "clustering_ari"):      # whole representations; per-view rows are in the table
            rows = [r for r in rows if r["view"] in (None, "fused", "concat", "average")
                    or r["representation"] == "unimodal"]
        if not rows:
            continue
        chance = None
        if m == "retrieval_mean_recall_at_10":
            chance = _mean([r["chance"]["R@10"] for f in agent.test_folds for r in list(f["retrieval"].values())[:1]])
        panels.append({"title": f"Test {METRIC_LABEL[m]}", "rows": rows, "chance": chance,
                       "interval": rows[0]["interval"]})
    figs["comparison"] = F.comparison(panels, fdir / "comparison.png")
    names = agent.trained()
    colours = F.config_colours(names)
    curves = {n: agent.fits[(n, d)].record for n in names if (n, d) in agent.fits}
    figs["training_curves"] = F.training_curves(curves, colours, curves[agent.primary_name]["selection_metric"],
                                                fold.name, fdir / "training_curves.png")
    rec = agent.fits[(agent.primary_name, d)]
    asm = agent.baselines(d, agent.config(agent.primary_name))["asm"]
    a, b = agent.pair[0], agent.pair[-1]
    if len(agent.pair) == 2:
        figs["embedding_map"] = F.embedding_map(rec.emb["test"][a], rec.emb["test"][b], (a, b), asm.test.labels,
                                                asm.classes, fdir / "embedding_map.png", agent.cfg.seed)
    ds_res = agent.downstream_results
    mplans = {m.modality: m for m in agent.config(agent.primary_name).modalities}
    if ds_res.get("retrieval_examples"):
        figs["retrieval_examples"] = F.retrieval_examples(ds_res["retrieval_examples"],
                                                          _content_fn(agent, fold, mplans), (a, b),
                                                          fdir / "retrieval_examples.jpg")
    if ds_res.get("prediction_maps"):
        figs["prediction_maps"] = F.prediction_maps(ds_res["prediction_maps"], fdir / "prediction_maps.png")
    if ds_res.get("confusion"):
        figs["confusion"] = F.confusion(ds_res["confusion"], fdir / "confusion.png")
    if ds_res.get("cluster_map"):
        figs["cluster_map"] = F.cluster_map(ds_res["cluster_map"], fdir / "cluster_map.png")
    return {k: v for k, v in figs.items() if v}


def write_downstream(agent) -> list[str]:
    files = []
    for key, value in agent.downstream_results.items():
        if isinstance(value, dict):
            dump(value, agent.out / "downstream" / f"{key}.json")
            files.append(f"downstream/{key}.json")
    return files


def export_embeddings(agent) -> dict:
    """Held-out (test) joint embeddings of the final configuration, one row per test sample of
    each fold, embedded by the model that never saw it. Rows from different folds come from
    different models: compare rows within a fold. Subsampled (stratified by fold) to the cap."""
    ds, parts = agent.ds, []
    for i, fold in enumerate(agent.folds):
        rec = agent.fits.get((agent.primary_name, i))
        if rec is None:
            continue
        asm = agent.baselines(i, agent.config(agent.primary_name))["asm"]
        lab = asm.test.labels
        parts.append({"ids": ds.ids[fold.test].astype(str), "fold": np.array([fold.name] * len(fold.test)),
                      "label": np.array([asm.classes[v] if lab is not None and v >= 0 else "" for v in
                                         (lab if lab is not None else np.full(len(fold.test), -1))]),
                      **{m: rec.emb["test"][m] for m in agent.pair}})
    if not parts:
        return {}
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    n = len(cat["ids"])
    per_row = sum(cat[m].shape[1] for m in agent.pair) * 2
    cap = max(1, EXPORT_MAX_BYTES // per_row)
    keep = np.arange(n)
    if n > cap:
        rng = np.random.default_rng(agent.cfg.seed)
        keep = np.sort(np.concatenate([rng.permutation(np.flatnonzero(cat["fold"] == f))[:int(round(
            cap * (cat["fold"] == f).mean()))] for f in np.unique(cat["fold"])]))
    arrays = {k: (v[keep].astype(np.float16) if k in agent.pair else v[keep]) for k, v in cat.items()}
    path = agent.out / "embeddings" / "held_out_embeddings.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    about = {"file": "held_out_embeddings.npz", "configuration": agent.primary_name, "modalities": agent.pair,
             "dim": {m: int(cat[m].shape[1]) for m in agent.pair}, "n_rows": int(len(keep)), "n_held_out": int(n),
             "subsampled": bool(n > cap), "dtype": "float16", "l2_normalised": True,
             "arrays": ["ids", "fold", "label"] + agent.pair,
             "note": "Each row is a test sample embedded by its fold's model (never trained on it). Rows from "
                     "different folds come from different models, so compare rows within one fold."}
    dump(about, agent.out / "embeddings" / "about.json")
    return about


# ---------------------------------------------------------------------------
# The record the writer sees (and the number check uses)
# ---------------------------------------------------------------------------

def _modality_summary(m) -> dict:
    d = {"name": m.name, "kind": m.kind, "n_present": m.n_present}
    if m.image:
        d.update(image_source=m.image.source, domain_hint=m.image.domain_hint,
                 greyscale_fraction=m.image.grayscale_fraction, near_duplicate_fraction=m.image.near_duplicate_fraction,
                 colour_between_group_share=m.image.colour_between_group_share,
                 spot_diameter_px=m.image.spot_diameter_px_median)
    if m.text:
        d.update(words_median=m.text.words_median, vocabulary_size=m.text.vocabulary_size,
                 near_duplicate_fraction=m.text.near_duplicate_fraction)
    if m.matrix:
        x = m.matrix
        d.update(n_features=x.n_features, sparsity=x.sparsity, integer_counts=x.integer_valued,
                 library_size_median=x.library_size_median, near_twin_fraction=x.near_twin_fraction,
                 group_variance_share=x.group_variance_share, label_variance_share=x.label_variance_share,
                 group_silhouette=x.group_silhouette, morans_i_median=x.morans_i_median)
    if m.spatial:
        d.update(n_libraries=m.spatial.n_libraries, spacing_over_diameter=m.spatial.spacing_over_diameter)
    return d


def _config_summary(c) -> dict:
    return {"name": c.name, "role": c.role, "strategy": c.strategy, "fusion": c.fusion,
            "encoders": {m.modality: m.encoder for m in c.modalities},
            "regimes": {m.modality: m.regime + (f" (last {m.trainable_blocks} blocks)" if m.trainable_blocks else "")
                        for m in c.modalities},
            "losses": {t.name: t.weight for t in c.losses}, "embed_dim": c.head.embed_dim,
            "hidden_dim": c.head.hidden_dim, "dropout": c.head.dropout, "lr": c.training.lr,
            "batch_size": c.training.batch_size, "max_epochs": c.training.max_epochs,
            "train_seconds": c.training.train_seconds, "rationale": c.rationale[:300]}


def build_record(agent) -> dict:
    prof, plan, pm = agent.prof, agent.plan, agent.plan.evaluation.primary_metric
    t = plan.task
    res = {m: [{k: r[k] for k in ("name", "kind", "value", "sd", "lo", "hi", "interval", "per_fold")}
               for r in metric_rows(agent, m)] for m in COMPARISON_METRICS}
    chance = _mean([r["chance"]["R@10"] for f in agent.test_folds for r in list(f["retrieval"].values())[:1]])
    crit = agent.critique
    ds_res = agent.downstream_results
    downstream = {"fold": ds_res.get("fold")}
    if ds_res.get("prediction"):
        downstream["prediction"] = {"method": ds_res["prediction"]["method"],
                                    "best_genes": ds_res["prediction"]["best_genes"][:6]}
    if ds_res.get("confusion"):
        c = ds_res["confusion"]
        downstream["classification"] = {"view": c["view"], "macro_f1": c["macro_f1"],
                                        "balanced_accuracy": c["balanced_accuracy"], "per_class_f1": c["per_class_f1"]}
    if ds_res.get("cluster_map"):
        downstream["clustering"] = {"k": ds_res["cluster_map"]["k"], "ari": ds_res["cluster_map"].get("ari"),
                                    "slide": ds_res["cluster_map"].get("group")}
    if ds_res.get("retrieval_examples"):
        ex = ds_res["retrieval_examples"]
        downstream["retrieval_examples"] = {
            k: [{"query_label": q["query_label"], "partner_rank": q["partner_rank"],
                 "top5_same_label": sum(bool(x["same_label"]) for x in q["retrieved"])} for q in v]
            for k, v in ex.items()}
    llm = agent.client.summary() if agent.client is not None else {}
    rec = {
        "objective": agent.objective, "quick_mode": agent.cfg.quick,
        "data": {"summary": prof.brief(), "n_samples": prof.n_samples,
                 "modalities": [_modality_summary(m) for m in prof.modalities], "pairs": prof.pairing.pairs,
                 "labels": [{"column": lab.column, "n_classes": lab.n_classes, "n_labelled": lab.n_labelled,
                             "labelled_fraction": lab.labelled_fraction, "class_counts": lab.class_counts,
                             "imbalance_ratio": lab.imbalance_ratio} for lab in prof.labels],
                 "groups": [{"column": g.column, "role": g.role, "n_levels": g.n_levels} for g in prof.groups],
                 "cv_group": prof.cv_group, "predefined_split": prof.split.counts if prof.split else None},
        "task": {"tasks": t.tasks, "primary_task": t.primary_task, "pair": t.pair,
                 "prediction": t.prediction.model_dump() if t.prediction else None, "label_column": t.label_column,
                 "label_evidence": t.label_evidence},
        "plan": {"source": plan.source, "split": plan.split.model_dump(),
                 "folds": [{"name": f.name, **f.sizes(), "test_groups": f.test_groups, "val_groups": f.val_groups}
                           for f in agent.folds],
                 "configurations": [_config_summary(agent.config(n)) for n in agent.trained()],
                 "primary_metric": pm, "baselines": plan.evaluation.baselines, "downstream": plan.downstream},
        "decisions": [{k: (v[:400] if isinstance(v, str) else v) for k, v in r.items()} for r in decision_rows(agent)
                      if r["step"] != "writer"],
        "screening": {"metric": agent.screening_table.get("metric"), "rows": agent.screening_table.get("rows", [])},
        "training": {n: [{k: fr.record[k] for k in ("best_epoch", "epochs_run", "stopped_because", "best_val_score")}
                         for (name, _), fr in agent.fits.items() if name == n] for n in agent.trained()},
        "test": {"primary": agent.primary_name, "primary_metric": pm, "n_folds": len(agent.test_folds),
                 "chance_recall_at_10": chance, "results": res, "retrieval": retrieval_table(agent),
                 "prediction": prediction_table(agent)[:8], "comparisons": agent.test_comparisons,
                 "flags": agent.test_flags},
        "validation": {"comparisons": agent.val_comparisons, "flags": agent.val_flags},
        "critic": None if crit is None else {
            "action": crit.action, "target": crit.target_config, "changes": {c.path: c.value for c in crit.changes},
            "assessment": crit.assessment, "rationale": crit.rationale, "source": crit.source,
            "triggers": [{"code": x.code, "severity": x.severity, "message": x.message} for x in crit.triggers]},
        "revision": agent.revision, "downstream": downstream,
        "llm": {k: llm.get(k) for k in ("model_requested", "models_used", "n_calls", "total_cost_usd")},
    }
    return _round(rec, 6)


# ---------------------------------------------------------------------------
# Template prose (used when no LLM is available; quotes only record numbers)
# ---------------------------------------------------------------------------

def _primary_row(record: dict, metric: str) -> dict | None:
    rows = record["test"]["results"].get(metric) or []
    p = record["test"]["primary"]
    own = [r for r in rows if r["name"].split(":")[0].split(" (")[0] == p]
    fused = [r for r in own if r["name"].endswith("(fused)")]
    return (fused or own or [None])[0]


def _iv(r: dict) -> str:
    if r.get("lo") is None:
        return ""
    return f" ({r['interval']}: {f3(r['lo'])} to {f3(r['hi'])})"


def template_sections(record: dict) -> dict[str, str]:
    t, plan, test = record["task"], record["plan"], record["test"]
    pm, primary = test["primary_metric"], test["primary"]
    prim = next((c for c in plan["configurations"] if c["name"] == primary), plan["configurations"][0])
    head = _primary_row(record, pm)
    encs = ", ".join(f"{e} for {m}" for m, e in prim["encoders"].items())
    losses = " + ".join(f"{n} ({w})" for n, w in prim["losses"].items())
    split = plan["split"]
    split_txt = (f"{split['strategy'].replace('_', ' ')} split"
                 + (f" by '{split['group_column']}' with {len(plan['folds'])} fold(s)" if split["group_column"] else ""))
    comps = test["comparisons"]
    main = [c for c in comps if c["metric"] == pm]
    crit, rev = record.get("critic"), record.get("revision") or {}

    def comp_sentence(c):
        ci = c.get("difference_ci95")
        spread = (f"95% interval {f3(ci[0])} to {f3(ci[1])}" if ci else
                  f"better in {c['folds_primary_better']} of {c['n_folds']} folds")
        return (f"{SHORT_LABEL.get(c['metric'], c['metric'])} {f3(c['primary_value'])} vs {f3(c['alternative_value'])} "
                f"for {c['alternative']} (difference {f3(c['difference_mean'])}; {spread})")

    s = {}
    headline = (f"'{primary}' reached a test {METRIC_LABEL[pm]} of {f3(head['value'])}{_iv(head)}" if head
                else f"'{primary}' was evaluated on held-out data")
    if pm == "retrieval_mean_recall_at_10" and test.get("chance_recall_at_10") is not None:
        headline += f", against {f3(test['chance_recall_at_10'])} expected by chance"
    s["executive_summary"] = (
        f"The objective was: \"{record['objective']}\" The agent read it as {', '.join(t['tasks'])} on the "
        f"modalities {' and '.join(t['pair'])}"
        + (f", predicting {t['prediction']['target']} from {t['prediction']['source']}" if t["prediction"] else "")
        + f". It chose {prim['strategy'].replace('_', ' ')} with {encs}, trained with {losses}, and a {split_txt}. "
        + f"{headline}. "
        + (" ".join(cap(comp_sentence(c)) + "." for c in main[:2]) + " " if main else "")
        + (f"The critic chose '{crit['action']}'" + (f"; the revision was {'kept' if rev.get('kept') else 'discarded'}"
                                                      if rev.get("ran") else "") + "." if crit else ""))
    mods = []
    for m in record["data"]["modalities"]:
        bits = [f"{m['n_present']:,} samples"]
        if m.get("domain_hint"):
            bits.append(f"image domain '{m['domain_hint']}'")
        if m.get("colour_between_group_share") is not None:
            bits.append(f"between-group colour share {f3(m['colour_between_group_share'])}")
        if m.get("words_median") is not None:
            bits.append(f"median {m['words_median']:.0f} words")
        if m.get("n_features") is not None:
            bits.append(f"{m['n_features']:,} features")
        if m.get("group_variance_share") is not None:
            bits.append(f"group explains {pct(m['group_variance_share'])} of top-PC variance")
        if m.get("near_twin_fraction") is not None:
            bits.append(f"near-twin fraction {f3(m['near_twin_fraction'])}")
        mods.append(f"- **{m['name']}** ({m['kind']}): " + "; ".join(bits))
    labs = [f"- label '{lab['column']}': {lab['n_classes']} classes on {lab['n_labelled']:,} samples "
            f"({pct(lab['labelled_fraction'])})" for lab in record["data"]["labels"]]
    s["data_and_objective"] = "\n".join([record["data"]["summary"] + ".", ""] + mods + labs + [
        "", f"Parsed objective: tasks {', '.join(t['tasks'])}; primary task {t['primary_task']}"
        + (f"; label column '{t['label_column']}' ({t['label_evidence']})" if t["label_column"] else "") + "."])
    picks = [d for d in record["decisions"] if d["step"] in ("split", "strategy", "encoder", "regime", "losses",
                                                              "preprocessing", "comparison", "evaluation", "planner")]
    s["decisions_explained"] = "\n".join(f"- **{d['step']}**: {d['chosen']}. {d['rationale']}" for d in picks[:14])
    lines = []
    for m in COMPARISON_METRICS:
        rows = test["results"].get(m) or []
        if not rows:
            continue
        p = _primary_row(record, m)
        others = [r for r in rows if r is not p][:3]
        lines.append(f"- **{METRIC_LABEL[m]}**: " + (f"{p['name']} {f3(p['value'])}{_iv(p)}; " if p else "")
                     + "; ".join(f"{r['name']} {f3(r['value'])}" for r in others) + ".")
    ret = next((r for r in test["retrieval"] if r["representation"] == primary), None)
    if ret and ret.get("label_match_at_10") is not None:
        lines.append(f"- Retrieved matches share the query's label {pct(ret['label_match_at_10'])} of the time "
                     f"(chance {pct(ret['label_match_chance'])}).")
    s["results_interpretation"] = "\n".join(lines) or "No held-out metrics were computed."
    flags = [f for f in test["flags"] if f["severity"] == "warning"]
    s["integration_benefit"] = "\n".join(
        [f"- {cap(comp_sentence(c))}." for c in comps] or ["- No comparison was possible."]
    ) + ("\n\nWarnings:\n" + "\n".join(f"- {f['message']}" for f in flags) if flags else "")
    if crit:
        trig = "; ".join(x["message"] for x in crit["triggers"] if x["severity"] == "warning") or "no warning-level trigger"
        s["critic_and_revision"] = (
            f"Triggers on validation data: {trig}. Decision ({crit['source']}): {crit['action']}"
            + (f" with {', '.join(f'{k} = {v}' for k, v in crit['changes'].items())}" if crit["changes"] else "")
            + (f" -> '{crit['target']}'" if crit.get("target") else "") + ". "
            + (f"Validation {SHORT_LABEL[pm]}: original {f3(rev.get('original_val_mean'))}, revised "
               f"{f3(rev.get('revised_val_mean'))}; the revision was {'kept' if rev.get('kept') else 'discarded'}."
               if rev.get("ran") else "No revision was run."))
    else:
        s["critic_and_revision"] = "The critic was disabled for this run."
    lim = []
    if test["n_folds"] == 1:
        lim.append("- One held-out set: intervals are bootstrap intervals over its samples, not over replicate datasets.")
    else:
        lim.append(f"- {test['n_folds']} folds: the across-fold SD reflects few held-out groups.")
    if record.get("quick_mode"):
        lim.append("- Quick mode: small subsample, small encoders and seconds of training per fit; the numbers check "
                   "the pipeline, not the method.")
    lim += [f"- {f['message']}" for f in test["flags"] if f["code"] in ("zero_shot_near_ceiling", "batch_confounding")]
    lim.append("- Fits are capped by time as well as epochs; a fit that stops on its time cap can run a different "
               "number of epochs on a busier machine.")
    lim.append("- LLM decisions cannot be re-sampled identically; they are cached, and --replay repeats them exactly.")
    s["limitations"] = "\n".join(lim)
    return s


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

def plan_markdown(agent) -> str:
    plan = agent.plan
    out = [f"# Plan ({plan.source})", "", f"Objective: \"{plan.task.raw_objective}\"", "",
           f"- Tasks: {', '.join(plan.task.tasks)} (primary: {plan.task.primary_task})",
           f"- Modalities aligned: {' + '.join(plan.task.pair)}"
           + (f"; prediction {plan.task.prediction.source} -> {plan.task.prediction.target}" if plan.task.prediction
              else ""),
           f"- Split: {plan.split.strategy}" + (f" by '{plan.split.group_column}', {plan.split.n_folds} fold(s)"
                                                 if plan.split.group_column else ""),
           f"- Primary metric: {plan.evaluation.primary_metric}; metrics: {', '.join(plan.evaluation.metrics)}",
           f"- Baselines: {', '.join(plan.evaluation.baselines)}", f"- Downstream: {', '.join(plan.downstream)}", ""]
    for c in [agent.config(n) for n in agent.configs]:
        out += [f"## {c.name} ({c.role})", "", c.rationale, "",
                f"- Strategy: {c.strategy}, fusion {c.fusion}",
                f"- Losses: {', '.join(f'{t.name} x {t.weight}' for t in c.losses) or 'none (linear)'}",
                f"- Head: embedding {c.head.embed_dim}, hidden {c.head.hidden_dim}, {c.head.n_layers} layers, "
                f"dropout {c.head.dropout}",
                f"- Training: batch {c.training.batch_size}, lr {c.training.lr}, weight decay {c.training.weight_decay}, "
                f"up to {c.training.max_epochs} epochs, patience {c.training.patience}, "
                f"cap {c.training.train_seconds:.0f}s per fit, temperature {c.training.temperature_init}"
                + (" (learned)" if c.training.learnable_temperature else " (fixed)")]
        for m in c.modalities:
            out.append(f"- {m.modality}: {m.encoder}, {m.regime}" + "".join(
                f", {k} {v}" for k, v in m.model_dump(exclude={"modality", "encoder", "regime"}).items()
                if v not in (None, False, 0, "none")))
        out.append("")
    return "\n".join(out)


def methods_summary(agent) -> str:
    plan, prof = agent.plan, agent.prof
    prim = agent.config(agent.primary_name)
    asm = agent.baselines(0, prim)["asm"]
    lines = ["# Methods and preprocessing", "",
             f"Generated from the executed plan (source: {plan.source}). Final configuration: '{agent.primary_name}'.",
             "", "## Data and split", "", f"- {prof.brief()}",
             f"- Split: {plan.split.strategy}" + (f", grouped by '{plan.split.group_column}'" if plan.split.group_column
                                                   else "") + f"; {len(agent.folds)} fold(s). Every preprocessing "
             "statistic (feature selection, normalisation, PCA, label codes, regression targets) is fitted on the "
             "fold's training samples only. Model selection uses validation samples; test samples are scored once.",
             "", "## Preprocessing and encoders", ""]
    for m in prim.modalities:
        kind = agent.ds.modalities[m.modality].kind
        info = REGISTRY.get(m.encoder)
        if kind in ("matrix", "tabular") and m.modality in asm.preprocessors:
            pre = asm.preprocessors[m.modality].describe()
            lines.append(f"- **{m.modality}** ({kind}): {pre}; encoded by an MLP trained from scratch.")
        else:
            bits = []
            if m.patch_context is not None:
                bits.append(f"patch = {m.patch_context} x the spot diameter")
            if m.colour_normalisation != "none":
                bits.append(f"{m.colour_normalisation} colour normalisation per slide/group")
            if m.greyscale_to_rgb:
                bits.append("greyscale replicated to RGB")
            if m.text_max_tokens:
                bits.append(f"text truncated at {m.text_max_tokens} tokens ({m.text_cleaning} cleaning)")
            lines.append(f"- **{m.modality}** ({kind}): {m.encoder} ({info.pretraining if info else ''}; "
                         f"licence {info.license if info else ''}), {m.regime}"
                         + (f" with the last {m.trainable_blocks} blocks fine-tuned" if m.trainable_blocks else "")
                         + (f"; {', '.join(bits)}" if bits else "") + ".")
    if "target" in asm.preprocessors:
        lines.append(f"- Regression target: {asm.preprocessors['target'].describe()}.")
    t = prim.training
    lines += ["", "## Representation learning", "",
              f"- Strategy: {prim.strategy} ({prim.fusion} fusion); projection heads of {prim.head.n_layers} layers "
              f"(hidden {prim.head.hidden_dim}, dropout {prim.head.dropout}) into a {prim.head.embed_dim}-dimensional "
              "L2-normalised joint space.",
              f"- Objective: {' + '.join(f'{x.weight} x {x.name}' for x in prim.losses)}.",
              f"- Optimiser AdamW, lr {t.lr}, weight decay {t.weight_decay}, batch {t.batch_size}, up to {t.max_epochs} "
              f"epochs, early stopping after {t.patience} epochs without validation improvement, time cap "
              f"{t.train_seconds:.0f}s per fit; best validation epoch kept. Seed {agent.cfg.seed}.",
              "", "## Evaluation", "",
              f"- Primary metric: {METRIC_LABEL[plan.evaluation.primary_metric]}.",
              f"- Metrics: {', '.join(plan.evaluation.metrics)}. Baselines: {', '.join(plan.evaluation.baselines)}.",
              "- Uncertainty: mean and SD across folds when there are several; otherwise paired-bootstrap 95% intervals "
              "over held-out samples.",
              f"- Downstream analyses: {', '.join(plan.downstream)}.",
              "", "## Comparison configurations", ""]
    lines += [f"- **{c.name}**: {c.rationale}" for c in plan.configs if c.role != "primary"] or ["- none"]
    if agent.revision.get("ran"):
        lines.append(f"- Critic revision '{agent.revision.get('target')}': "
                     f"{'kept' if agent.revision.get('kept') else 'discarded'}.")
    return "\n".join(lines) + "\n"


CSS = """
:root { color-scheme: light; --bg: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e;
  --muted: #898781; --rule: #e1e0d9; --accent: #2a78d6; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; --bg: #0d0d0d;
  --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781; --rule: #2c2c2a; --accent: #3987e5; } }
:root[data-theme="dark"] { color-scheme: dark; --bg: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7;
  --muted: #898781; --rule: #2c2c2a; --accent: #3987e5; }
body { background: var(--bg); color: var(--ink); font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
  margin: 0; padding: 24px 16px; }
main { max-width: 980px; margin: 0 auto; background: var(--surface); padding: 24px 28px; border-radius: 8px; }
h1 { font-size: 1.6em; line-height: 1.25; } h2 { margin-top: 1.8em; border-bottom: 1px solid var(--rule);
  padding-bottom: 4px; } h3 { margin-top: 1.4em; }
a { color: var(--accent); } code { font-size: 0.92em; }
table { border-collapse: collapse; font-size: 13px; margin: 10px 0 18px; display: block; overflow-x: auto; }
th, td { border-bottom: 1px solid var(--rule); padding: 4px 10px; text-align: left; vertical-align: top;
  font-variant-numeric: tabular-nums; } th { color: var(--ink2); font-weight: 600; }
img { max-width: 100%; height: auto; background: #fcfcfb; border-radius: 6px; }
.meta { color: var(--ink2); font-size: 13px; }
"""


def to_html(md_text: str, title: str) -> str:
    import markdown

    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
    return (f"<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>{title}</title>"
            f"<style>{CSS}</style></head><body><main>\n{body}\n</main></body></html>\n")


def report_markdown(agent, record: dict, prose: dict, figs: dict, tables: dict, manifest: dict,
                    exported: dict) -> str:
    sec, pm = prose["sections"], record["test"]["primary_metric"]
    check = prose["number_check"]

    def fig(key, caption):
        return f"![{caption}]({figs[key]})\n\n*{caption}*\n" if key in figs else ""

    rows_for = {m: tables.get(f"test_{m}") or [] for m in COMPARISON_METRICS}
    llm = record["llm"]
    out = ["# Multimodal representation learning report", "",
           f"<p class=\"meta\">Objective: “{agent.objective}”<br>Data: <code>{agent.data_path}</code> · "
           f"generated {manifest['finished'][:16].replace('T', ' ')} UTC · plan: {agent.plan.source} · "
           f"prose: {prose['source']}" + (" · quick mode" if agent.cfg.quick else "") + "</p>", "",
           "## Summary", "", sec["executive_summary"], "",
           "## Data and objective", "", sec["data_and_objective"], "",
           "## What the agent decided and why", "", sec["decisions_explained"], "",
           "### Configurations trained", "",
           md_table([{**c, "encoders": ", ".join(f"{k}: {v}" for k, v in c["encoders"].items()),
                      "regimes": ", ".join(c["regimes"].values()), "lr": f"{c['lr']:g}",
                      "losses": ", ".join(f"{k} x {v}" for k, v in c["losses"].items()) or "none (closed form)"}
                     for c in record["plan"]["configurations"]],
                    [("name", "configuration"), ("role", "role"), ("strategy", "strategy"), ("encoders", "encoders"),
                     ("regimes", "regime"), ("losses", "losses"), ("embed_dim", "dim"), ("lr", "lr"),
                     ("train_seconds", "cap (s)")]), "",
           "### Encoder screening (frozen features, training vs validation samples of fold 1)", "",
           md_table(tables["screening"], [(k, k) for k in (tables["screening"][0].keys() if tables["screening"] else [])]),
           "", "## Results on held-out test data", "", sec["results_interpretation"], "",
           fig("comparison", "Test metrics for every configuration (blue: final configuration) and baseline (grey)."),
           f"### {METRIC_LABEL[pm]} (primary metric)", "",
           md_table(rows_for[pm], [("name", "representation"), ("kind", "kind"), ("value", "mean"), ("lo", "lower"),
                                   ("hi", "upper"), ("interval", "interval")]), ""]
    if tables["retrieval"]:
        out += ["### Retrieval", "", md_table(tables["retrieval"], [
            ("representation", "representation"), ("R@1", "R@1"), ("R@5", "R@5"), ("R@10", "R@10"),
            ("a_to_b_R@10", f"{agent.pair[0]}→{agent.pair[-1]} R@10"), ("b_to_a_R@10", f"{agent.pair[-1]}→{agent.pair[0]} R@10"),
            ("median_rank", "median rank"), ("chance_R@10", "chance R@10"), ("label_match_at_10", "label match@10"),
            ("label_match_chance", "label chance")]), ""]
    if tables["prediction"]:
        out += ["### Cross-modal prediction", "", md_table(tables["prediction"], [
            ("method", "method"), ("mean_r_top50", "mean r (top 50)"), ("median_r_top50", "median r"),
            ("within_group_mean_r_top50", "within-slide r"), ("mean_r_all", "mean r (all)"),
            ("n_genes_r_gt_0.3", "genes r > 0.3")]), ""]
    if tables["classification"]:
        out += ["### Classification and clustering", "", md_table(tables["classification"], [
            ("representation", "representation"), ("view", "view"), ("probe_macro_f1", "probe macro-F1"),
            ("probe_balanced_accuracy", "balanced acc."), ("knn_macro_f1", "kNN macro-F1"),
            ("cluster_ari", "ARI"), ("cluster_nmi", "NMI")]), ""]
    if tables["geometry"]:
        out += ["### Embedding geometry", "", md_table(tables["geometry"], [
            ("representation", "representation"), ("alignment", "alignment"), ("uniformity_a", "uniformity a"),
            ("uniformity_b", "uniformity b"), ("modality_gap", "modality gap"), ("effective_rank_a", "eff. rank a"),
            ("effective_rank_b", "eff. rank b"), ("group_silhouette", "group silhouette")]), ""]
    out += ["## Did integration help?", "", sec["integration_benefit"], "",
            md_table(record["test"]["comparisons"], [
                ("metric", "metric"), ("alternative", "versus"), ("primary_value", "final config"),
                ("alternative_value", "alternative"), ("difference_mean", "difference"), ("difference_sd", "SD"),
                ("difference_ci95", "95% interval"), ("folds_primary_better", "folds better"), ("n_folds", "folds")]),
            "", "## Downstream analyses", "",
            fig("retrieval_examples", "Held-out queries and their five nearest matches in the joint space."),
            fig("prediction_maps", "Observed and predicted expression of the best-predicted genes on a held-out slide."),
            fig("confusion", "Confusion matrix of the linear probe (rows: true class)."),
            fig("cluster_map", "k-means clusters of held-out joint embeddings and their label composition."),
            fig("embedding_map", "Held-out joint embeddings of both modalities."),
            fig("training_curves", "Validation score and training loss by epoch.")]
    pred = record["downstream"].get("prediction")
    if pred:
        out += ["Best-predicted genes (" + pred["method"] + "):", "",
                md_table(pred["best_genes"], [("gene", "gene"), ("r", "test r")]), ""]
    out += ["## Critic review", "", sec["critic_and_revision"], "",
            md_table([{"code": x["code"], "severity": x["severity"], "message": x["message"]}
                      for x in (record["critic"] or {}).get("triggers", [])],
                     [("code", "trigger"), ("severity", "severity"), ("message", "finding")]), "",
            "## Limitations", "", sec["limitations"], "",
            "## Number check", "",
            (f"All {check['n_checked']} numbers in the prose above were found in the computed outputs."
             if not check["n_unmatched"] else
             f"{check['n_unmatched']} of {check['n_checked']} numbers in the prose were **not found** in the computed "
             "outputs; treat them with caution:\n\n" + "\n".join(f"- {u['number']} ({u['section']}): {u['context']}"
                                                              for u in check["unmatched"])), "",
            "## Reproducibility", "",
            f"- Command: `{manifest['command']}`",
            f"- Seed {agent.cfg.seed}; device {agent.hw.device} ({agent.hw.device_name}); projection heads on "
            f"{agent.heads_device}.",
            f"- LLM: " + (f"{llm.get('model_requested')} via the Claude CLI, {llm.get('n_calls')} call(s), "
                          f"${llm.get('total_cost_usd') or 0:.2f}; prompts and replies in `llm_transcript.json`; "
                          f"`--replay {agent.out}` repeats them without calling a model."
                          if llm.get("n_calls") else "not used (rule-based planner, critic and writer)."),
            f"- Embeddings: `embeddings/{exported.get('file', '')}` ({exported.get('n_rows', 0):,} held-out rows"
            + (", subsampled" if exported.get("subsampled") else "") + ", float16).",
            "- Files: decision_log.json, plan.json/plan.md, profile.json, screening.json, evaluation.json, tables/, "
            "figures/, downstream/, methods_summary.md, model_summary.json, run_manifest.json.", "",
            "## Appendix: training summaries", "",
            md_table(tables["training"], [("configuration", "configuration"), ("fold", "fold"),
                                          ("best_epoch", "best epoch"), ("epochs_run", "epochs"),
                                          ("stopped_because", "stopped because"), ("best_val_score", "best val score"),
                                          ("n_trainable_parameters", "trainable params")]), ""]
    return "\n".join(out)


def build_manifest(agent) -> dict:
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    git = {}
    try:
        git["commit"] = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True,
                                       text=True, timeout=10).stdout.strip() or None
        git["uncommitted_changes"] = bool(subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
                                                         capture_output=True, text=True, timeout=10).stdout.strip())
    except Exception:
        pass
    llm = agent.client.summary() if agent.client is not None else {}
    argv = (["mmagent"] + sys.argv[1:]) if Path(sys.argv[0]).name == "mmagent" else sys.argv
    return {"command": shlex.join(argv), "data": str(agent.data_path), "objective": agent.objective,
            "out_dir": str(agent.out), "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": agent.cfg.model_dump(), "hardware": agent.hw.to_record(), "heads_device": agent.heads_device,
            "python": sys.version.split()[0], "versions": versions, "git": git, "timings_seconds": agent.timings,
            "train_seconds": {"planned_caps": round(agent.planned_seconds, 1),
                              "measured": round(agent.measured_seconds, 1)},
            "llm": {k: v for k, v in llm.items() if k != "calls"}, "llm_calls": llm.get("calls", [])}


def model_summary(agent) -> dict:
    out = {}
    for name in agent.trained():
        c = agent.config(name)
        out[name] = {"role": "final" if name == agent.primary_name else c.role, "strategy": c.strategy,
                     "fusion": c.fusion,
                     "modalities": [{"modality": m.modality, "encoder": m.encoder, "regime": m.regime,
                                     "trainable_blocks": m.trainable_blocks,
                                     "encoder_dim": REGISTRY[m.encoder].dim if m.encoder in REGISTRY else None}
                                    for m in c.modalities],
                     "losses": {t.name: t.weight for t in c.losses}, "head": c.head.model_dump(),
                     "training": c.training.model_dump(),
                     "folds": [{"fold": agent.folds[i].name, **{k: rec.record[k] for k in (
                         "n_trainable_parameters", "best_epoch", "epochs_run", "stopped_because", "best_val_score")}}
                               for (n, i), rec in agent.fits.items() if n == name]}
    return out


def write_all(agent) -> dict:
    out = agent.out
    for sub in ("tables", "figures", "downstream", "embeddings"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    tables = write_tables(agent)
    figs = make_figures(agent, tables)
    write_downstream(agent)
    exported = export_embeddings(agent)
    dump({"primary": agent.primary_name, "primary_metric": agent.plan.evaluation.primary_metric,
          "test": {"folds": S.strip_private(agent.test_folds), "aggregate": agent.test_aggregate,
                   "comparisons": agent.test_comparisons, "flags": agent.test_flags},
          "validation": {"folds": S.strip_private(agent.val_folds), "comparisons": agent.val_comparisons,
                         "flags": agent.val_flags},
          "revision": agent.revision}, out / "evaluation.json")
    record = build_record(agent)
    prose = writer.write_sections(record, agent.client, agent.log, template_sections)
    manifest = build_manifest(agent)
    md = report_markdown(agent, record, prose, figs, tables, manifest, exported)
    (out / "report.md").write_text(md)
    (out / "report.html").write_text(to_html(md, "Multimodal representation learning report"))
    (out / "methods_summary.md").write_text(methods_summary(agent))
    (out / "plan.md").write_text(plan_markdown(agent))
    dump(model_summary(agent), out / "model_summary.json")
    dump(manifest, out / "run_manifest.json")
    if agent.client is not None and agent.client.records:
        dump({"model_requested": agent.client.model, "calls": agent.client.records}, out / "llm_transcript.json")
    pm = agent.plan.evaluation.primary_metric
    head = _primary_row(record, pm)
    return {"out_dir": str(out), "primary": agent.primary_name, "primary_metric": pm,
            "test_value": head["value"] if head else None, "plan_source": agent.plan.source,
            "critic_action": agent.critique.action if agent.critique else None,
            "revision_kept": agent.revision.get("kept"), "number_check": prose["number_check"]["n_unmatched"],
            "prose_source": prose["source"], "figures": sorted(figs), "llm": record["llm"]}
