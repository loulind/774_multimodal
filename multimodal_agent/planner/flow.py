"""Planning as the agent runs it: rules -> split -> encoder screening -> plan -> validation.

  1. A preliminary rule-based plan fixes everything screening needs: the split
     strategy and each modality's preprocessing (patch size, colour
     normalisation, text length, matrix features). Its log is discarded.
  2. Folds are drawn; candidate encoders are screened on the first fold's
     training and validation samples (never its test samples).
  3. The final plan is made with the screening scores, so the encoder decision
     in the log cites them.
  4. The plan is validated against the guardrails; repairs are logged.
  5. ``agent_plan`` only: the LLM planner adopts or modifies the validated proposal
     (planner/llm.py). If it changes the split, the pair or the label column, the
     folds are drawn again from the final plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..data.splits import Fold, make_folds
from ..decision_log import DecisionLog
from ..encoders.pretrained import FeatureCache
from ..encoders.screening import screen
from ..llm import LLMClient
from ..schema import Plan, ScreeningResult
from . import heuristic
from . import llm as llm_planner
from .validator import validate


@dataclass
class PlanOutcome:
    plan: Plan
    folds: list[Fold]
    screening: list[ScreeningResult] = field(default_factory=list)
    screening_table: dict = field(default_factory=dict)
    split_notes: list[str] = field(default_factory=list)


def folds_for(ds, plan: Plan, seed: int, max_samples: int | None) -> tuple[list[Fold], list[str]]:
    split_cols = ds.columns_with("split")
    return make_folds(ds.obs, ds.paired_mask(plan.task.pair), plan.split, seed,
                      split_column=split_cols[0] if split_cols else None,
                      label_column=plan.evaluation.label_column, max_samples=max_samples)


def rule_based(ds, profile, task, budget, device: str, cache: FeatureCache | None, log: DecisionLog,
               quick: bool, seed: int, do_screen: bool = True, progress=None) -> PlanOutcome:
    prelim = heuristic.plan(profile, task, budget, DecisionLog(), quick=quick)
    folds, notes = folds_for(ds, prelim, seed, budget.max_samples)
    screening, table = [], {}
    if do_screen and any(ds.modalities[n].kind in ("image", "text") for n in task.pair):
        screening, table = screen(ds, profile, task, prelim, folds[0].train, folds[0].val, budget, device,
                                  cache, quick, seed, progress)
    plan = heuristic.plan(profile, task, budget, log, screening=screening, quick=quick)
    checked = validate(plan, profile, budget, quick)
    for r in checked.repairs:
        log.add("validator", "plan repair", r, "guardrail", source="validator")
    if not checked.ok:
        raise ValueError(f"rule-based plan failed validation: {checked.errors}")
    for n in notes:
        log.add("split", "sample selection", n, "Recorded while drawing the folds.", source="system")
    return PlanOutcome(checked.plan, folds, screening, table, notes)


def agent_plan(ds, profile, task, budget, device: str, cache: FeatureCache | None, log: DecisionLog,
               quick: bool, seed: int, client: LLMClient | None = None, do_screen: bool = True,
               progress=None) -> PlanOutcome:
    """The planning step of a run: the rule-based proposal, then the LLM planner if one is available."""
    proposal_log = DecisionLog()
    out = rule_based(ds, profile, task, budget, device, cache, proposal_log, quick, seed, do_screen, progress)
    log.extend(proposal_log.entries)
    final = llm_planner.plan(out.plan, proposal_log, profile, task, budget, log, client,
                             out.screening_table, quick)
    redraw = (final.split != out.plan.split or final.task.pair != out.plan.task.pair
              or final.evaluation.label_column != out.plan.evaluation.label_column)
    if redraw:
        folds, notes = folds_for(ds, final, seed, budget.max_samples)
        log.add("split", "folds", "redrawn from the final plan",
                "The LLM changed the split, the modality pair or the label column, so the folds were "
                "drawn again (screening used the proposal's first fold).",
                [], {"n_folds": len(folds)}, source="system")
        return PlanOutcome(final, folds, out.screening, out.screening_table, notes)
    out.plan = final
    return out
