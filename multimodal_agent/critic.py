"""Critic: read the validation results and decide whether one revision is worth running.

This is what makes the agent more than a fixed pipeline: some choices can only be judged
once a model has been trained. The critic sees *validation* results and training curves
only; test data are scored once, after the critic has finished.

Two layers (as in Projects 1-2):
  1. Deterministic triggers, each with a concrete remedy:
       comparison_better   a comparison configuration beat the primary on the primary
                           metric by more than the noise threshold -> promote it
       worse_than_<b>      a baseline (CCA, ridge, zero-shot, unimodal, fusion) beat the
                           primary on the primary metric -> train longer if the fits stopped
                           while still improving, otherwise regularise (against ridge on a
                           prediction metric: weight the regression term more)
       collapse            embeddings use few directions, or the temperature sits at its
                           bound -> a fixed, warmer temperature and a lower learning rate
       underfitting        most folds stopped on the time or epoch cap while still improving
                           -> more time or epochs if the budget allows, else a higher learning rate
       overfitting         the validation score peaked early and fell by >= 10% while the
                           training loss kept falling -> more dropout and weight decay, a
                           narrower head
       batch_confounding   embeddings group by slide/patient more than by label -> colour
                           normalisation for an image modality that lacks it
     Findings on secondary metrics, and "no clear difference" results, are recorded for the
     report but do not trigger a revision: the revision is judged on the primary metric.
  2. A judgement: the LLM's when available (validated: allowed actions, configuration
     names, editable fields and ranges; the revised configuration must also pass the
     planner's guardrails), otherwise the first actionable trigger by priority.

Limits: at most one revision per run, and it is kept only if the validation primary metric
improves (``keep_revision``); otherwise the original stands and the log records both.
"""

from __future__ import annotations

import json
import typing

import numpy as np

from .config import Budget
from .decision_log import DecisionLog
from .encoders.registry import REGISTRY
from .evaluate import suite as S
from .llm import LLMClient, LLMUnavailable
from .planner import validator as V
from .planner.llm import load_prompt
from .schema import (Alternative, Change, ConfigSpec, CriticReply, Critique, DataProfile, LossName,
                     ModalityPlan, Plan, Trigger)

PRIORITY = ["comparison_better", "worse_than_", "collapse", "underfitting", "overfitting", "batch_confounding"]
TEMPERATURE_FLOOR = 0.0105        # the logit scale is clamped at 100, i.e. temperature 0.01
EARLY_PEAK = 0.3                  # overfitting: best epoch within the first 30% of epochs run ...
PEAK_DROP = 0.10                  # ... and a final validation score 10% below that peak
VOLATILE = {"seconds"}            # left out of prompts, so a replayed run sends the same text
CURVE_POINTS = 25                 # points per training curve shown to the LLM
LOSS_NAMES = set(typing.get_args(LossName))
MODALITY_FIELDS = set(ModalityPlan.model_fields) - {"modality"}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def training_record(fit) -> dict:
    """What the critic needs from one models.train.FitResult: its summary and its curves."""
    h = fit.history
    return {**fit.summary(), "val_score": list(h.get("val_score", [])),
            "train_loss": list(h.get("train_loss", []))}


def _majority(xs: list[bool]) -> bool:
    return bool(xs) and sum(xs) > len(xs) / 2


def _underfit(r: dict) -> bool:
    capped = str(r.get("stopped_because", "")).startswith(("time cap", "epoch cap"))
    return capped and r.get("best_epoch", 0) >= r.get("epochs_run", 0) - 1


def _overfit(r: dict) -> bool:
    vs, tl, best = r.get("val_score") or [], r.get("train_loss") or [], r.get("best_epoch", 0)
    if not vs or not tl or best < 1 or not str(r.get("stopped_because", "")).startswith("no validation"):
        return False
    peak = vs[best - 1]
    return (best <= EARLY_PEAK * len(vs) and peak > 0 and vs[-1] <= (1 - PEAK_DROP) * peak
            and tl[-1] < tl[best - 1])


def _capped_by(records: list[dict]) -> str:
    reasons = [str(r.get("stopped_because", "")) for r in records]
    return "time" if sum(x.startswith("time cap") for x in reasons) >= sum(x.startswith("epoch cap")
                                                                          for x in reasons) else "epochs"


# ---------------------------------------------------------------------------
# Remedies (each a list of edits to the primary configuration)
# ---------------------------------------------------------------------------

def _regularise(cfg: ConfigSpec) -> list[Change]:
    return [Change(path="head.dropout", value=round(min(0.5, cfg.head.dropout + 0.2), 2)),
            Change(path="training.weight_decay", value=float(min(0.1, max(1e-3, cfg.training.weight_decay * 10)))),
            Change(path="head.hidden_dim", value=max(64, cfg.head.hidden_dim // 2))]


def _fit_cap(cfg: ConfigSpec, budget: Budget) -> float:
    ft = any(m.regime == "partial_finetune" for m in cfg.modalities)
    return budget.finetune_seconds_per_fit if ft else budget.train_seconds_per_fit


def _longer(cfg: ConfigSpec, records: list[dict], budget: Budget, n_folds: int,
            remaining: float | None) -> list[Change]:
    """More training where the cap that stopped it can still be raised; otherwise larger steps.
    Uses planned caps only (never measured times), so the same run always gets the same advice."""
    t = cfg.training
    room = _fit_cap(cfg, budget) if remaining is None else min(_fit_cap(cfg, budget), remaining / max(1, n_folds))
    lr_up = [Change(path="training.lr", value=float(min(V.RANGES[("training", "lr")][1], 2 * t.lr)))]
    if _capped_by(records) == "epochs":
        if t.max_epochs >= budget.max_epochs:
            return lr_up                                   # no more epochs allowed: take larger steps
        epochs = min(budget.max_epochs, 2 * t.max_epochs)
        changes = [Change(path="training.max_epochs", value=int(epochs))]
        secs = min(room, t.train_seconds * epochs / t.max_epochs)
        if secs > t.train_seconds:
            changes.append(Change(path="training.train_seconds", value=float(round(secs))))
        return changes
    secs = min(room, 2 * t.train_seconds)                  # stopped on the time cap
    if secs >= 1.2 * t.train_seconds:
        return [Change(path="training.train_seconds", value=float(round(secs)))]
    return lr_up


def _against(cfg: ConfigSpec, plan: Plan, alt: str, underfit: bool, records, budget, n_folds,
             remaining) -> list[Change]:
    if alt == "zero_shot" and not cfg.head.residual:
        return [Change(path="head.residual", value=True)]   # start from the pretrained space
    if alt == "ridge" and plan.task.prediction is not None:
        w = next((t.weight for t in cfg.losses if t.name == "cross_modal_regression"), None)
        return [Change(path="loss.cross_modal_regression.weight",
                       value=1.0 if w is None else float(min(V.LOSS_WEIGHT[1], 2 * w)))]
    if underfit:
        return _longer(cfg, records, budget, n_folds, remaining)
    return _regularise(cfg)


def _warmer(cfg: ConfigSpec) -> list[Change]:
    return [Change(path="training.learnable_temperature", value=False),
            Change(path="training.temperature_init", value=float(max(0.1, cfg.training.temperature_init))),
            Change(path="training.lr", value=float(max(V.RANGES[("training", "lr")][0], cfg.training.lr / 2)))]


# ---------------------------------------------------------------------------
# Deterministic triggers
# ---------------------------------------------------------------------------

def find_triggers(plan: Plan, val_folds: list[dict], flags: list[dict], training: dict[str, list[dict]],
                  budget: Budget, remaining_seconds: float | None = None) -> list[Trigger]:
    prim, pm = plan.primary, plan.evaluation.primary_metric
    n_folds = max(1, len(val_folds))
    records = training.get(prim.name, [])
    under = _majority([_underfit(r) for r in records])
    over = _majority([_overfit(r) for r in records])
    out: list[Trigger] = []
    closed_form = prim.strategy == "linear_alignment"      # CCA: only preprocessing changes can matter
    for fl in flags:
        code, msg = fl["code"], fl["message"]
        if code == "comparison_better":
            act = fl["severity"] == "warning" and any(c.name == fl.get("config") for c in plan.configs)
            out.append(Trigger(code=code, message=msg, severity=fl["severity"], config=fl.get("config"),
                               suggested_action="switch_primary" if act else "none",
                               suggested_target=fl.get("config") if act else None))
        elif code.startswith("worse_than_"):
            on_primary = fl.get("metric") == pm
            changes = _against(prim, plan, code[len("worse_than_"):], under, records, budget, n_folds,
                               remaining_seconds) if on_primary and not closed_form else []
            out.append(Trigger(code=code, severity="warning" if on_primary else "info", config=prim.name,
                               message=msg + ("" if on_primary else " (secondary metric: reported, not revised)"),
                               suggested_action="modify_primary" if changes else "none",
                               suggested_changes=changes))
        elif code == "collapse":
            out.append(Trigger(code=code, message=msg, config=prim.name,
                               suggested_action="none" if closed_form else "modify_primary",
                               suggested_changes=[] if closed_form else _warmer(prim)))
        elif code == "batch_confounding":
            img = next((m for m in prim.modalities if m.colour_normalisation == "none"
                        and m.encoder in REGISTRY and "image" in REGISTRY[m.encoder].kinds), None)
            changes = [Change(path=f"modalities.{img.modality}.colour_normalisation", value="reinhard")] if img else []
            out.append(Trigger(code=code, message=msg, config=prim.name,
                               severity="warning" if changes else "info",
                               suggested_action="modify_primary" if changes else "none",
                               suggested_changes=changes))
        else:
            out.append(Trigger(code=code, message=msg, severity="info", config=fl.get("config")))
    temps = [r.get("final_temperature") for r in records if r.get("final_temperature") is not None]
    if temps and _majority([t <= TEMPERATURE_FLOOR for t in temps]) and not any(t.code == "collapse" for t in out):
        out.append(Trigger(code="collapse", config=prim.name, suggested_action="modify_primary",
                           message=f"The learned temperature reached its lower bound ({min(temps):.4f}) in most "
                                   "folds: the contrastive objective is pushing towards over-confident matches.",
                           suggested_changes=_warmer(prim)))
    if under:
        n = sum(_underfit(r) for r in records)
        out.append(Trigger(code="underfitting", config=prim.name, suggested_action="modify_primary",
                           message=f"Training stopped on its {_capped_by(records)} cap while the validation score "
                                   f"was still improving in {n} of {len(records)} folds.",
                           suggested_changes=_longer(prim, records, budget, n_folds, remaining_seconds)))
    elif over:
        n = sum(_overfit(r) for r in records)
        out.append(Trigger(code="overfitting", config=prim.name, suggested_action="modify_primary",
                           message=f"The validation score peaked in the first {EARLY_PEAK:.0%} of epochs and fell by "
                                   f">= {PEAK_DROP:.0%} while the training loss kept falling, in {n} of "
                                   f"{len(records)} folds (the early-stopped model is used either way).",
                           suggested_changes=_regularise(prim)))
    return out


# ---------------------------------------------------------------------------
# Applying a revision
# ---------------------------------------------------------------------------

def apply_changes(cfg: ConfigSpec, changes: list[Change]) -> ConfigSpec:
    """Edit a copy of ``cfg``; raises ValueError for a field that is not editable or a bad value."""
    d = cfg.model_dump()
    for ch in changes:
        parts = ch.path.split(".")
        if len(parts) == 2 and parts[0] in ("training", "head") and parts[1] in d[parts[0]]:
            d[parts[0]][parts[1]] = ch.value
        elif len(parts) == 3 and parts[0] == "loss" and parts[1] in LOSS_NAMES and parts[2] == "weight":
            w = float(ch.value)
            terms = d["losses"]
            hit = next((t for t in terms if t["name"] == parts[1]), None)
            if w <= 0:
                d["losses"] = [t for t in terms if t["name"] != parts[1]]
            elif hit:
                hit["weight"] = w
            else:
                terms.append({"name": parts[1], "weight": w})
        elif len(parts) == 3 and parts[0] == "modalities" and parts[2] in MODALITY_FIELDS:
            m = next((m for m in d["modalities"] if m["modality"] == parts[1]), None)
            if m is None:
                raise ValueError(f"no modality '{parts[1]}' in configuration {cfg.name}")
            m[parts[2]] = ch.value
        else:
            raise ValueError(f"not an editable field: {ch.path}")
    try:
        return ConfigSpec(**d)
    except Exception as err:                               # pydantic: wrong type or unknown choice
        raise ValueError(f"invalid value: {err}") from err


def revised_config(plan: Plan, critique: Critique, profile: DataProfile, budget: Budget, quick: bool,
                   n_folds: int, remaining_seconds: float | None) -> tuple[ConfigSpec | None, list[str], str | None]:
    """The primary with the critique's edits, checked by the planner's guardrails.

    Returns (config, repairs, error). Its time cap is clipped so the revision fits in what is
    left of the run budget on every fold."""
    try:
        cfg = apply_changes(plan.primary, critique.changes)
    except ValueError as err:
        return None, [], str(err)
    cfg.name, cfg.role = f"{plan.primary.name}_revised", "primary"
    cfg.rationale = f"Critic revision of {plan.primary.name}: {critique.rationale}"[:500]
    trial = plan.model_copy(deep=True)
    trial.configs = [cfg]
    res = V.validate(trial, profile, budget, quick)
    if not res.ok:
        return None, res.repairs, "; ".join(res.errors)
    out, repairs = res.plan.configs[0], list(res.repairs)
    if remaining_seconds is not None:
        room = remaining_seconds / max(1, n_folds)
        if room < V.MIN_FIT_SECONDS:
            return None, repairs, f"no compute left for a revision ({remaining_seconds:.0f}s for {n_folds} folds)"
        if out.training.train_seconds > room:
            repairs.append(f"revision time cap {out.training.train_seconds:.0f}s -> {room:.0f}s (remaining budget)")
            out.training.train_seconds = room
    return out, repairs, None


# ---------------------------------------------------------------------------
# Judgement
# ---------------------------------------------------------------------------

def _rule_critique(triggers: list[Trigger]) -> Critique:
    for code in PRIORITY:
        for t in triggers:
            if t.code.startswith(code) and t.severity == "warning" and t.suggested_action != "none":
                return Critique(
                    assessment=f"Rule-based critic: trigger '{t.code}': {t.message}",
                    action=t.suggested_action, target_config=t.suggested_target,
                    changes=t.suggested_changes,
                    rationale=f"Highest-priority actionable trigger (order: {', '.join(c.rstrip('_') for c in PRIORITY)}).",
                    alternatives_rejected=[Alternative(option="no revision", reason=t.message)],
                    expected_effect="the validation primary metric improves", triggers=triggers)
    msg = "; ".join(t.message for t in triggers if t.severity == "warning") or "no warning-level trigger fired"
    return Critique(assessment=f"Rule-based critic: no actionable problem ({msg}).", action="none",
                    rationale="No warning-level trigger with a concrete remedy on the primary metric.",
                    triggers=triggers)


def _round(obj, nd: int = 4):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, nd) for v in obj]
    return obj


def _thin(xs: list, n: int = CURVE_POINTS) -> list:
    if len(xs) <= n:
        return xs
    idx = np.unique(np.linspace(0, len(xs) - 1, n).round().astype(int))
    return [xs[i] for i in idx]


def editable_fields(plan: Plan, profile: DataProfile, budget: Budget) -> dict:
    prim = plan.primary
    out = {f"{part}.{attr}": list(r) for (part, attr), r in V.RANGES.items()}
    out["training.max_epochs"] = [1, budget.max_epochs]
    out["training.train_seconds"] = [V.MIN_FIT_SECONDS, round(_fit_cap(prim, budget), 1)]
    out["training.learnable_temperature"] = [True, False]
    out["head.residual"] = [True, False]
    out.update({f"loss.{n}.weight": list(V.LOSS_WEIGHT) for n in sorted(LOSS_NAMES)})
    for m in prim.modalities:
        kind = profile.modality(m.modality).kind
        out[f"modalities.{m.modality}.encoder"] = [e for e, i in REGISTRY.items() if kind in i.kinds]
        if kind == "image":
            out[f"modalities.{m.modality}.colour_normalisation"] = ["none", "reinhard"]
            if m.patch_context is not None:
                out[f"modalities.{m.modality}.patch_context"] = list(V.PATCH_CONTEXT)
        if kind == "text":
            out[f"modalities.{m.modality}.text_max_tokens"] = [8, REGISTRY[m.encoder].max_tokens or 512]
        if kind == "matrix":
            out[f"modalities.{m.modality}.n_variable_features"] = [50, "number of features"]
            out[f"modalities.{m.modality}.matrix_input"] = ["log_normalised", "pca", "standardised"]
            out[f"modalities.{m.modality}.pca_components"] = [2, 256]
        if kind == "tabular":
            out[f"modalities.{m.modality}.matrix_input"] = ["standardised", "pca"]
            out[f"modalities.{m.modality}.pca_components"] = [2, 256]
    return out


def validation_summary(plan: Plan, val_folds: list[dict], comparisons: list[dict]) -> dict:
    pm = plan.evaluation.primary_metric
    per_fold = {}
    for name in [c.name for c in plan.configs] + ["cca", "zero_shot"]:
        vals = [S.metric_value(f, name, pm) for f in val_folds]
        if any(v is not None for v in vals):
            per_fold[name] = vals
    out = {"primary_metric": pm, "n_folds": len(val_folds), "per_fold": per_fold, "comparisons": comparisons}
    if pm == "retrieval_mean_recall_at_10" and val_folds:
        out["retrieval_chance_R@10"] = [f["retrieval"][plan.primary.name]["chance"]["R@10"]
                                        for f in val_folds if plan.primary.name in f.get("retrieval", {})]
    return _round(out)


def _validated(reply: dict, plan: Plan, profile: DataProfile, budget: Budget, quick: bool,
               triggers: list[Trigger], n_folds: int, remaining: float | None
               ) -> tuple[Critique | None, ConfigSpec | None, list[str], str]:
    try:
        r = CriticReply(**reply)
    except Exception as err:
        return None, None, [], f"reply does not match the critic schema: {str(err)[:300]}"
    c = Critique(assessment=r.assessment, action=r.action, target_config=r.target_config, changes=r.changes,
                 rationale=r.rationale, alternatives_rejected=r.alternatives_rejected,
                 expected_effect=r.expected_effect, triggers=triggers, source="llm")
    if c.action == "none":
        c.target_config, c.changes = None, []
        return c, None, [], ""
    if c.action == "switch_primary":
        names = [x.name for x in plan.configs if x.role != "primary"]
        if c.target_config not in names:
            return None, None, [], f"switch target '{c.target_config}' is not a comparison configuration {names}"
        c.changes = []
        return c, None, [], ""
    if not c.changes:
        return None, None, [], "modify_primary without any changes"
    if plan.primary.strategy == "linear_alignment" and any(not x.path.startswith("modalities.") for x in c.changes):
        return None, None, [], ("the primary is a closed-form linear alignment (CCA); only modality preprocessing "
                                "changes can alter it")
    cfg, repairs, err = revised_config(plan, c, profile, budget, quick, n_folds, remaining)
    if cfg is None:
        return None, None, repairs, err or "revision invalid"
    c.target_config = None
    return c, cfg, repairs, ""


def critique(plan: Plan, profile: DataProfile, val_folds: list[dict], comparisons: list[dict], flags: list[dict],
             training: dict[str, list[dict]], client: LLMClient | None, log: DecisionLog, budget: Budget,
             quick: bool = False, remaining_seconds: float | None = None) -> tuple[Critique, ConfigSpec | None]:
    """Judge the validation results. Returns the critique and, for ``modify_primary``, the
    validated configuration to train on every fold (None otherwise)."""
    n_folds = max(1, len(val_folds))
    triggers = find_triggers(plan, val_folds, flags, training, budget, remaining_seconds)
    for t in triggers:
        log.add("critic", f"trigger: {t.code}", t.suggested_action, t.message, [],
                {"severity": t.severity, "suggested_changes": {c.path: c.value for c in t.suggested_changes}},
                source="critic")
    out, cfg, repairs = None, None, []
    if client is not None and client.available:
        payload = {
            "objective": plan.task.raw_objective, "profile_summary": profile.brief(plan.evaluation.label_column),
            "plan": plan.model_dump(exclude={"source"}),
            "validation": validation_summary(plan, val_folds, comparisons), "flags": _round(flags),
            "training": {name: [_round({**{k: v for k, v in r.items() if k not in VOLATILE},
                                        "val_score": _thin(r.get("val_score", [])),
                                        "train_loss": _thin(r.get("train_loss", []))}) for r in recs]
                         for name, recs in training.items()},
            "triggers": [t.model_dump() for t in triggers],
            "compute": {"n_folds": n_folds, "per_fit_seconds_cap": round(_fit_cap(plan.primary, budget), 1),
                        "remaining_train_seconds": None if remaining_seconds is None else round(remaining_seconds)},
            "actions": ["none", "modify_primary", "switch_primary"],
            "editable_fields": editable_fields(plan, profile, budget),
        }
        try:
            resp = client.complete("critic", load_prompt("critic"),
                                   "Review this run.\n\n" + json.dumps(payload, indent=1, default=str),
                                   CriticReply.model_json_schema())
            out, cfg, repairs, why = _validated(resp.json(), plan, profile, budget, quick, triggers, n_folds,
                                                remaining_seconds)
            if out is None:
                log.add("critic", "LLM critique", "rejected", f"{why}; the rule-based critic is used.",
                        source="validator")
        except (LLMUnavailable, ValueError) as err:
            log.add("critic", "LLM critique", "unavailable", str(err), source="system")
    if out is None:
        out = _rule_critique(triggers)
        if out.action == "modify_primary":
            cfg, repairs, err = revised_config(plan, out, profile, budget, quick, n_folds, remaining_seconds)
            if cfg is None:
                log.add("critic", "rule-based revision", "not run", f"The suggested edits failed the guardrails: "
                        f"{err}", source="validator")
                out = out.model_copy(update={"action": "none", "changes": [],
                                             "rationale": out.rationale + f" Not run: {err}."})
    for r in repairs:
        log.add("validator", "revision repair", r, "Critic revision adjusted to satisfy a guardrail.",
                source="validator")
    chosen = {"none": "no revision", "switch_primary": f"promote '{out.target_config}' to primary",
              "modify_primary": "modify the primary: " + ", ".join(f"{c.path}={c.value}" for c in out.changes)}
    log.add("critic", "revise the plan?", chosen[out.action], f"{out.assessment} {out.rationale}".strip(),
            [(a.option, a.reason) for a in out.alternatives_rejected],
            {"expected_effect": out.expected_effect, "source": out.source}, source="critic")
    return out, cfg


def keep_revision(original: list[float | None], revised: list[float | None], metric: str,
                  log: DecisionLog) -> bool:
    """Keep the revision only if the mean validation primary metric improves."""
    pairs = [(a, b) for a, b in zip(original, revised) if a is not None and b is not None]
    if not pairs:
        log.add("critic", "keep the revision?", "discarded", f"No validation {metric} values to compare.",
                source="critic")
        return False
    a, b = np.array(pairs).T
    keep = bool(b.mean() > a.mean())
    log.add("critic", "keep the revision?", "kept" if keep else "discarded",
            f"Validation {metric}: revised {b.mean():.4f} vs original {a.mean():.4f} (difference "
            f"{b.mean() - a.mean():+.4f}; revised better in {(b > a).sum()} of {len(a)} folds). "
            + ("Kept: the primary metric improved." if keep else
               "Discarded: the primary metric did not improve, so the original result stands."),
            [], {"original": a.round(4).tolist(), "revised": b.round(4).tolist()}, source="critic")
    return keep
