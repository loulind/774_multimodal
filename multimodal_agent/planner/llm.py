"""LLM planner: profile + objective + screening + rule-based proposal -> validated Plan.

Flow (from Project 2; every branch is logged):
  1. The rule-based planner writes a complete, validated proposal; its decisions are
     already in the log as the baseline reasoning.
  2. The LLM may adopt, modify or reject any element, recording a decision for each
     change. It sees the profile without the dataset's name or path, so its choices
     must come from measurements. The user's objective text cannot be changed.
  3. The result is validated against the same guardrails as any plan. Errors are sent
     back once for repair; if the repaired plan still fails, the rule-based proposal
     is used and the log says why.
"""

from __future__ import annotations

import json
import typing
from pathlib import Path

from ..config import Budget
from ..decision_log import DecisionLog
from ..encoders import registry
from ..llm import LLMClient, LLMUnavailable
from ..schema import (ConfigSpec, DataProfile, DownstreamName, LossName, ModalityPlan, Plan,
                      PlannerReply, PrimaryMetric, TaskSpec)
from . import validator as V

PROMPTS = Path(__file__).resolve().parents[2] / "prompts"
HIDDEN_PROFILE_FIELDS = {"name", "path"}
TASK_FIELDS = ("tasks", "primary_task", "pair", "prediction", "label_column")
MAX_REPLY_ECHO = 30_000                      # characters of a failed reply sent back for repair
DIFF_SKIP = {"rationale", "source", "evidence", "label_evidence", "modality_mentions"}


def load_prompt(name: str) -> str:
    return (PROMPTS / f"{name}.md").read_text()


def _choices(model, field: str) -> list:
    return list(typing.get_args(model.model_fields[field].annotation))


def allowed_values() -> dict:
    """Everything the validator accepts, stated up front so the LLM can stay inside it."""
    return {
        "encoders": [e["name"] for e in registry.summary()],
        "strategies": _choices(ConfigSpec, "strategy"), "fusion": _choices(ConfigSpec, "fusion"),
        "regimes": _choices(ModalityPlan, "regime"), "losses": list(typing.get_args(LossName)),
        "primary_metrics": list(typing.get_args(PrimaryMetric)), "metrics": sorted(V.KNOWN_METRICS),
        "baselines": sorted(V.KNOWN_BASELINES), "downstream": list(typing.get_args(DownstreamName)),
        "ranges": {f"{part}.{attr}": list(r) for (part, attr), r in V.RANGES.items()},
        "loss_weight": list(V.LOSS_WEIGHT), "patch_context": list(V.PATCH_CONTEXT),
        "max_configs": V.MAX_CONFIGS, "finetune_batch_size_max": V.FT_BATCH_MAX,
    }


def budget_summary(budget: Budget, quick: bool) -> dict:
    return {"device": budget.device, "quick_mode": quick,
            "train_seconds_per_fit": round(budget.train_seconds_per_fit, 1),
            "finetune_seconds_per_fit": round(budget.finetune_seconds_per_fit, 1),
            "total_train_seconds": round(budget.total_train_seconds, 1), "max_epochs": budget.max_epochs,
            "max_folds": budget.max_folds, "max_samples": budget.max_samples,
            "total_rule": "n_folds x sum(train_seconds over configs) + primary refit + n_folds x primary "
                          "(critic revision) <= total_train_seconds"}


def build_user_message(profile: DataProfile, task: TaskSpec, budget: Budget, proposal: Plan,
                       proposal_log: DecisionLog, screening_table: dict | None, quick: bool) -> str:
    payload = {
        "objective": task.raw_objective,
        "parsed_task": task.model_dump(exclude={"source"}),
        "data_profile": profile.model_dump(exclude=HIDDEN_PROFILE_FIELDS),
        "encoder_screening": screening_table or {},
        "compute_budget": budget_summary(budget, quick),
        "encoder_library": registry.summary(),
        "allowed": allowed_values(),
        "proposal": {"plan": proposal.model_dump(exclude={"source"}),
                     "reasoning": [{"step": d.step, "decision": d.decision, "chosen": d.chosen,
                                    "rationale": d.rationale,
                                    "alternatives_rejected": [a.model_dump() for a in d.alternatives_rejected]}
                                   for d in proposal_log.entries]},
    }
    return ("Plan the multimodal representation-learning workflow for this dataset and objective.\n\n"
            + json.dumps(payload, indent=1, default=str))


def _flatten(obj, prefix: str = "") -> dict:
    """Dotted paths for a plan; lists of named items are keyed by name, so order is irrelevant."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k not in DIFF_SKIP:
                out.update(_flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj) and \
            all("name" in x or "modality" in x for x in obj):
        for x in obj:
            name = x.get("name", x.get("modality"))
            out.update(_flatten({k: v for k, v in x.items() if k not in ("name", "modality")},
                                f"{prefix}[{name}]"))
    else:
        out[prefix] = obj
    return out


def plan_diff(a: Plan, b: Plan) -> list[str]:
    fa, fb = _flatten(a.model_dump()), _flatten(b.model_dump())
    keys = list(fa) + [k for k in fb if k not in fa]
    miss = "(absent)"
    return [f"{k}: {fa.get(k, miss)} -> {fb.get(k, miss)}" for k in keys if fa.get(k, miss) != fb.get(k, miss)]


def _candidate(raw: dict, task: TaskSpec) -> tuple[dict, bool]:
    plan_dict = dict(raw.get("plan", raw)) if isinstance(raw, dict) else {}
    plan_dict.pop("source", None)                 # bookkeeping, set by us
    t = plan_dict.get("task")
    t = dict(t) if isinstance(t, dict) else task.model_dump()
    tampered = t.get("raw_objective") != task.raw_objective
    t["raw_objective"] = task.raw_objective
    plan_dict["task"] = t
    return plan_dict, tampered


def _log_llm_decisions(log: DecisionLog, decisions) -> None:
    for d in decisions or []:
        if not isinstance(d, dict):
            continue
        log.add(step=str(d.get("step", "llm")), decision=str(d.get("decision", "")),
                chosen=str(d.get("chosen", "")), rationale=str(d.get("rationale", "")),
                alternatives=[(str(a.get("option", "")), str(a.get("reason", "")))
                              for a in d.get("alternatives_rejected", []) or [] if isinstance(a, dict)],
                source="llm")


def plan(proposal: Plan, proposal_log: DecisionLog, profile: DataProfile, task: TaskSpec, budget: Budget,
         log: DecisionLog, client: LLMClient | None, screening_table: dict | None = None,
         quick: bool = False) -> Plan:
    """Return the plan to run. ``proposal`` must already be validated (planner.flow does this)."""
    attempted = False

    def fallback(reason: str) -> Plan:
        log.add("planner", "planner used", "rule-based (fallback)" if attempted else "rule-based", reason,
                [], {}, source="system")
        out = proposal.model_copy(deep=True)
        out.source = "heuristic_fallback" if attempted else "heuristic"
        return out

    if client is None or not client.available:
        why = client.disabled if client is not None and client.disabled else "--no-llm or no LLM backend"
        return fallback(f"No LLM available ({why}); the rule-based plan is used.")

    system = load_prompt("planner")
    user = build_user_message(profile, task, budget, proposal, proposal_log, screening_table, quick)
    schema = PlannerReply.model_json_schema()
    attempted = True
    try:
        resp = client.complete("planner", system, user, schema)
        raw = resp.json()
    except (LLMUnavailable, ValueError) as err:
        return fallback(f"LLM planning failed: {err}")
    candidate, tampered = _candidate(raw, task)
    result = V.validate(candidate, profile, budget, quick)
    source = "llm"
    if not result.ok:
        log.add("validator", "LLM plan validation", "failed; one repair attempt", "; ".join(result.errors),
                source="validator")
        repair = (user + "\n\nYour previous answer:\n" + resp.text[:MAX_REPLY_ECHO] +
                  "\n\nIt failed validation with these errors:\n- " + "\n- ".join(result.errors) +
                  "\n\nReturn the corrected JSON object.")
        try:
            resp = client.complete("planner_repair", system, repair, schema)
            raw = resp.json()
        except (LLMUnavailable, ValueError) as err:
            return fallback(f"LLM plan invalid ({'; '.join(result.errors)}) and the repair failed: {err}")
        candidate, tampered = _candidate(raw, task)
        result = V.validate(candidate, profile, budget, quick)
        source = "llm_repaired"
        if not result.ok:
            return fallback(f"LLM plan still invalid after one repair: {'; '.join(result.errors)}")

    final = result.plan
    final.source = source
    _log_llm_decisions(log, raw.get("decisions") if isinstance(raw, dict) else None)
    if tampered:
        log.add("validator", "objective text", "restored", "The LLM altered the user's objective text; "
                "the original was restored.", source="validator")
    for r in result.repairs:
        log.add("validator", "plan repair", r, "LLM plan adjusted to satisfy a guardrail.", source="validator")
    before = proposal.task
    if any(getattr(final.task, f) != getattr(before, f) for f in TASK_FIELDS):
        final.task.source = "llm"
        log.add("objective", "parsed task", "corrected by the LLM",
                "; ".join(f"{f}: {getattr(before, f)} -> {getattr(final.task, f)}" for f in TASK_FIELDS
                          if getattr(final.task, f) != getattr(before, f)), source="llm")
    changes = plan_diff(proposal, final)
    log.add("planner", "planner used", source,
            ("LLM revised the rule-based proposal: " + "; ".join(changes[:25]) +
             (f"; ... ({len(changes)} changes)" if len(changes) > 25 else "")) if changes
            else "LLM adopted the rule-based proposal unchanged.",
            [], {"llm_model": resp.model_used, "n_changes": len(changes), "changes": changes}, source="llm")
    return final
