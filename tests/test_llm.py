"""LLM client, LLM planner, critic and writer -- every fallback path, with a scripted fake CLI."""

import json

import numpy as np
import pandas as pd
import pytest

from factories import image_text_profile, spatial_profile
from multimodal_agent import critic as C
from multimodal_agent import writer as W
from multimodal_agent.config import LLMConfig, load_config
from multimodal_agent.data.loaders import load_path
from multimodal_agent.decision_log import DecisionLog
from multimodal_agent.llm import LLMClient, LLMResponse, LLMUnavailable, OfflineBackend, extract_json, make_client
from multimodal_agent.objective import parse_objective
from multimodal_agent.planner import flow, heuristic
from multimodal_agent.planner import llm as P
from multimodal_agent.planner.validator import validate
from multimodal_agent.profiler import profile as profile_data
from multimodal_agent.schema import Change, Critique

R1 = ("Learn joint embeddings of tissue image patches and spot-level gene expression to enable "
      "cross-modal retrieval and prediction of gene expression from histology.")
R2 = "Learn a joint image–caption embedding for image–text retrieval and imaging-modality classification."
MPS = load_config().budget("mps")


class Scripted:
    """A fake CLI backend: answers in order (dicts are JSON replies), records every call."""
    name, model, max_output_tokens = "fake", "fake-model", 100

    def __init__(self, replies, cost=0.1, used="fake-model"):
        self.replies, self.cost, self.used, self.calls = list(replies), cost, used, []

    def complete(self, system, user, schema=None, effort=None, budget_usd=None):
        self.calls.append({"system": system, "user": user, "schema": schema, "effort": effort,
                           "budget": budget_usd})
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        structured = isinstance(r, dict) and schema is not None
        return LLMResponse(text=r if isinstance(r, str) else json.dumps(r), backend=self.name,
                           model_requested=self.model, model_used=self.used, seconds=0.01, cost_usd=self.cost,
                           data=r if structured else None, structured=structured)


def client_for(tmp_path, replies, **kw):
    backend = Scripted(replies, **{k: kw.pop(k) for k in ("cost", "used") if k in kw})
    return LLMClient(backend, tmp_path / "llm_cache", effort={"planner": "high", "critic": "high",
                                                              "writer": "medium"}, **kw), backend


# --- client -----------------------------------------------------------------------------------

def test_cache_replay_and_role_position_matching(tmp_path):
    client, backend = client_for(tmp_path, [{"a": 1}, {"b": 2}])
    assert client.complete("planner", "sys", "user 1", {"type": "object"}).json() == {"a": 1}
    assert client.complete("critic", "sys", "user 2").json() == {"b": 2}
    assert backend.calls[0]["effort"] == "high" and backend.calls[1]["schema"] is None
    rec = json.loads(next((tmp_path / "llm_cache").glob("planner_*.json")).read_text())
    assert rec["sequence"] == 0 and rec["model_used"] == "fake-model" and rec["user"] == "user 1"

    replay = LLMClient(OfflineBackend("fake-model"), tmp_path / "run2", replay_from=tmp_path / "llm_cache")
    assert replay.available
    assert replay.complete("planner", "sys", "user 1", {"type": "object"}).json() == {"a": 1}
    # a prompt that differs in incidental digits is matched by role and position
    resp = replay.complete("critic", "sys", "user 2 with other digits 0.1234")
    assert resp.json() == {"b": 2} and "role and position" in resp.note
    with pytest.raises(LLMUnavailable):
        replay.complete("writer", "sys", "never cached")
    assert len(list((tmp_path / "run2").glob("*.json"))) == 2       # the new run's cache is complete


def test_spend_cap_passes_remaining_budget_then_stops(tmp_path):
    client, backend = client_for(tmp_path, [{"x": 1}, {"x": 2}, {"x": 3}], cost=0.1, max_budget_usd=0.15)
    client.complete("planner", "s", "u1")
    client.complete("critic", "s", "u2")
    assert backend.calls[0]["budget"] == 0.15 and backend.calls[1]["budget"] == pytest.approx(0.05)
    with pytest.raises(LLMUnavailable, match="spend cap"):
        client.complete("writer", "s", "u3")
    assert len(backend.calls) == 2


def test_fatal_error_stops_further_calls(tmp_path):
    client, backend = client_for(tmp_path, [LLMUnavailable("claude CLI failed: Failed to authenticate"), {"x": 1}])
    with pytest.raises(LLMUnavailable):
        client.complete("planner", "s", "u")
    assert not client.available and "authenticate" in client.disabled
    with pytest.raises(LLMUnavailable, match="disabled"):
        client.complete("critic", "s", "u")
    assert len(backend.calls) == 1 and client.summary()["disabled"]


def test_model_substitution_is_noted(tmp_path):
    client, _ = client_for(tmp_path, [{"x": 1}], used="other-model")
    client.complete("planner", "s", "u")
    assert any("answered by other-model" in n for n in client.notes)
    assert client.summary()["models_used"] == ["other-model"]


def test_make_client_modes(tmp_path):
    cfg = LLMConfig()
    assert not make_client(cfg, tmp_path, enabled=False).available
    assert make_client(cfg, tmp_path, replay_from=tmp_path).available


def test_extract_json_variants():
    assert extract_json('Here:\n```json\n{"a": 1}\n```\nDone.') == {"a": 1}
    assert extract_json('prefix {"a": {"b": 2}} suffix') == {"a": {"b": 2}}
    assert extract_json('```json\n{"a": 1}') == {"a": 1}                      # unterminated fence
    assert extract_json('```\n{"a": 1}\n```\n```\n{"b": 2}\n```') == {"a": 1, "b": 2}
    with pytest.raises(ValueError):
        extract_json("no json here")


# --- LLM planner ------------------------------------------------------------------------------

def proposal_for(prof, objective):
    task = parse_objective(objective, prof)
    log = DecisionLog()
    checked = validate(heuristic.plan(prof, task, MPS, log), prof, MPS)
    assert checked.ok
    return checked.plan, log, task


@pytest.fixture()
def r2():
    prof = image_text_profile().model_copy(update={"name": "secretname", "path": "/data/secretname"})
    return (prof, *proposal_for(prof, R2))


def reply_with(plan, decisions=()):
    return {"plan": plan.model_dump(), "decisions": list(decisions)}


def test_offline_uses_rule_based_plan(r2):
    prof, proposal, plog, task = r2
    log = DecisionLog()
    out = P.plan(proposal, plog, prof, task, MPS, log, None)
    assert out.source == "heuristic" and out.configs == proposal.configs
    assert log.by_step("planner")[-1].chosen == "rule-based"


def test_llm_adopts_proposal_and_never_sees_dataset_name(r2, tmp_path):
    prof, proposal, plog, task = r2
    client, backend = client_for(tmp_path, [reply_with(proposal)])
    log = DecisionLog()
    out = P.plan(proposal, plog, prof, task, MPS, log, client)
    assert out.source == "llm" and "unchanged" in log.by_step("planner")[-1].rationale
    assert "secretname" not in backend.calls[0]["user"]
    assert "plan" in backend.calls[0]["schema"]["properties"]
    assert P.plan_diff(proposal, out) == []


def test_llm_modification_is_logged_with_its_reason(r2, tmp_path):
    prof, proposal, plog, task = r2
    mod = proposal.model_copy(deep=True)
    mod.primary.training.lr = 5e-4
    mod.primary.head.dropout = 1.5                                   # out of range -> clipped
    dec = {"step": "hyperparameters", "decision": "learning rate", "chosen": "5e-4",
           "rationale": "20,000 pairs; a smaller step is safer", "alternatives_rejected": []}
    client, _ = client_for(tmp_path, [reply_with(mod, [dec])])
    log = DecisionLog()
    out = P.plan(proposal, plog, prof, task, MPS, log, client)
    assert out.primary.training.lr == 5e-4 and out.primary.head.dropout == 0.5
    assert any(d.source == "llm" and d.decision == "learning rate" for d in log.entries)
    assert any("dropout" in d.chosen for d in log.by_step("validator"))
    assert "training.lr" in log.by_step("planner")[-1].rationale


def test_invalid_llm_plan_is_repaired_once(r2, tmp_path):
    prof, proposal, plog, task = r2
    bad = proposal.model_copy(deep=True)
    bad.primary.modalities[0].encoder = "no_such_encoder"
    client, backend = client_for(tmp_path, [reply_with(bad), reply_with(proposal)])
    out = P.plan(proposal, plog, prof, task, MPS, DecisionLog(), client)
    assert out.source == "llm_repaired" and len(backend.calls) == 2
    assert "unknown encoder" in backend.calls[1]["user"]


def test_still_invalid_or_unparseable_falls_back(r2, tmp_path):
    prof, proposal, plog, task = r2
    bad = proposal.model_copy(deep=True)
    bad.primary.modalities[0].encoder = "no_such_encoder"
    client, _ = client_for(tmp_path / "a", [reply_with(bad), reply_with(bad)])
    log = DecisionLog()
    out = P.plan(proposal, plog, prof, task, MPS, log, client)
    assert out.source == "heuristic_fallback" and "after one repair" in log.by_step("planner")[-1].rationale
    client, _ = client_for(tmp_path / "b", ["I would rather not answer in JSON."])
    out = P.plan(proposal, plog, prof, task, MPS, DecisionLog(), client)
    assert out.source == "heuristic_fallback"
    client, _ = client_for(tmp_path / "c", [LLMUnavailable("claude CLI error: overloaded")])
    assert P.plan(proposal, plog, prof, task, MPS, DecisionLog(), client).source == "heuristic_fallback"


def test_objective_text_cannot_be_changed(r2, tmp_path):
    prof, proposal, plog, task = r2
    mod = proposal.model_copy(deep=True)
    mod.task.raw_objective = "something else"
    client, _ = client_for(tmp_path, [reply_with(mod)])
    log = DecisionLog()
    out = P.plan(proposal, plog, prof, task, MPS, log, client)
    assert out.task.raw_objective == R2 and any(d.decision == "objective text" for d in log.entries)


def test_agent_plan_end_to_end_on_a_small_table(tmp_path):
    from PIL import Image

    (tmp_path / "images").mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for i in range(80):
        Image.fromarray(rng.integers(0, 255, (24, 32), dtype=np.uint8)).save(tmp_path / "images" / f"{i}.png")
        mod = ["CT scan", "MR image"][i % 2]
        rows.append({"id": f"s{i}", "image_path": f"images/{i}.png", "caption": f"{mod} number {i} of a lesion",
                     "split": ["train", "train", "valid", "test"][i % 4], "modality": mod})
    pd.DataFrame(rows).to_csv(tmp_path / "pairs.csv", index=False)
    ds = load_path(tmp_path)
    prof = profile_data(ds)
    task = parse_objective(R2, prof)
    out = flow.agent_plan(ds, prof, task, MPS, "cpu", None, DecisionLog(), False, 0, None, do_screen=False)
    assert out.plan.source == "heuristic" and len(out.folds) == 1
    client, _ = client_for(tmp_path, [reply_with(out.plan)])
    out2 = flow.agent_plan(ds, prof, task, MPS, "cpu", None, DecisionLog(), False, 0, client, do_screen=False)
    assert out2.plan.source == "llm" and np.array_equal(out2.folds[0].test, out.folds[0].test)


# --- critic -----------------------------------------------------------------------------------

@pytest.fixture()
def r1_plan():
    prof = spatial_profile()
    plan, _, _ = proposal_for(prof, R1)
    return prof, plan


def val_folds(plan, values, chance=0.02):
    return [{"retrieval": {name: {"mean": {"R@10": v}, "chance": {"R@10": chance}} for name, v in fold.items()}}
            for fold in values]


def record(best, run, why, vs=None, tl=None, temp=0.07, seconds=60.0):
    return {"best_epoch": best, "epochs_run": run, "stopped_because": why, "final_temperature": temp,
            "seconds": seconds, "val_score": vs or [0.1] * run, "train_loss": tl or [1.0] * run}


def test_comparison_better_promotes_the_comparison(r1_plan):
    prof, plan = r1_plan
    comp = plan.configs[1].name
    flags = [{"code": "comparison_better", "severity": "warning", "config": comp, "value": 0.05, "message": "m"}]
    log = DecisionLog()
    crit, cfg = C.critique(plan, prof, val_folds(plan, [{"primary": 0.1, comp: 0.15}]), [], flags, {}, None,
                           log, MPS)
    assert crit.action == "switch_primary" and crit.target_config == comp and cfg is None
    assert log.by_step("critic")[-1].decision == "revise the plan?"


def test_worse_than_baseline_regularises_or_trains_longer(r1_plan):
    prof, plan = r1_plan
    flags = [{"code": "worse_than_cca", "severity": "warning", "metric": plan.evaluation.primary_metric,
              "value": -0.02, "message": "worse than cca"}]
    folds = val_folds(plan, [{"primary": 0.05}] * 4)
    early = {"primary": [record(10, 25, "no validation improvement for 15 epochs")] * 4}
    crit, cfg = C.critique(plan, prof, folds, [], flags, early, None, DecisionLog(), MPS)
    paths = {c.path for c in crit.changes}
    assert crit.action == "modify_primary" and {"head.dropout", "training.weight_decay"} <= paths
    assert cfg.name == "primary_revised" and cfg.head.dropout > plan.primary.head.dropout
    capped = {"primary": [record(40, 40, "time cap (60s)", seconds=60.0)] * 4}
    short = plan.model_copy(deep=True)
    short.primary.training.train_seconds = 60.0
    crit, cfg = C.critique(short, prof, folds, [], flags, capped, None, DecisionLog(), MPS,
                           remaining_seconds=5000)
    assert [c.path for c in crit.changes] == ["training.train_seconds"] and cfg.training.train_seconds == 120
    at_cap = plan.model_copy(deep=True)
    at_cap.primary.training.train_seconds = MPS.train_seconds_per_fit     # no more time per fit: raise lr
    crit, cfg = C.critique(at_cap, prof, folds, [], flags, capped, None, DecisionLog(), MPS,
                           remaining_seconds=5000)
    assert [c.path for c in crit.changes] == ["training.lr"] and cfg.training.lr == 2 * plan.primary.training.lr


def test_secondary_metric_findings_are_not_revised(r1_plan):
    prof, plan = r1_plan
    flags = [{"code": "worse_than_unimodal", "severity": "warning", "metric": "probe_macro_f1", "value": -0.1,
              "message": "worse than unimodal"},
             {"code": "zero_shot_near_ceiling", "severity": "warning", "value": 0.95, "message": "ceiling"}]
    crit, cfg = C.critique(plan, prof, val_folds(plan, [{"primary": 0.1}]), [], flags, {}, None, DecisionLog(), MPS)
    assert crit.action == "none" and cfg is None
    assert all(t.severity == "info" for t in crit.triggers)


def test_overfitting_and_collapse_triggers(r1_plan):
    prof, plan = r1_plan
    vs = [0.1, 0.3, 0.32, 0.31] + [0.2] * 16                   # peak at epoch 3, then a 37% fall
    tl = list(np.linspace(2.0, 0.5, 20))
    over = {"primary": [record(3, 20, "no validation improvement for 15 epochs", vs, tl)] * 4}
    trig = C.find_triggers(plan, [{}] * 4, [], over, MPS)
    assert [t.code for t in trig] == ["overfitting"]
    cold = {"primary": [record(30, 60, "no validation improvement for 15 epochs", temp=0.01)] * 4}
    trig = C.find_triggers(plan, [{}] * 4, [], cold, MPS)
    assert trig[0].code == "collapse" and {c.path for c in trig[0].suggested_changes} >= {
        "training.learnable_temperature", "training.temperature_init"}


def test_batch_confounding_turns_on_colour_normalisation():
    prof = spatial_profile(icc=0.02)                          # low colour ICC: rules do not normalise
    plan, _, _ = proposal_for(prof, R1)
    img = next(m for m in plan.primary.modalities if m.modality == "image")
    assert img.colour_normalisation == "none"
    flags = [{"code": "batch_confounding", "severity": "warning", "value": 0.4, "message": "groups"}]
    crit, cfg = C.critique(plan, prof, [{}], [], flags, {}, None, DecisionLog(), MPS)
    assert crit.action == "modify_primary"
    assert next(m for m in cfg.modalities if m.modality == "image").colour_normalisation == "reinhard"


def test_llm_critique_is_validated(r1_plan, tmp_path):
    prof, plan = r1_plan
    good = {"assessment": "a", "action": "modify_primary", "changes": [{"path": "training.lr", "value": 0.0005}],
            "rationale": "r", "alternatives_rejected": [], "expected_effect": "R@10 up"}
    client, backend = client_for(tmp_path / "a", [good])
    crit, cfg = C.critique(plan, prof, val_folds(plan, [{"primary": 0.1}]), [], [],
                           {"primary": [record(5, 20, "no validation improvement for 15 epochs")]}, client,
                           DecisionLog(), MPS)
    assert crit.source == "llm" and cfg.training.lr == 0.0005
    assert '"seconds"' not in backend.calls[0]["user"]          # volatile timings stay out of the prompt
    for bad in ({**good, "changes": [{"path": "training.no_such_field", "value": 1}]},
                {**good, "action": "switch_primary", "target_config": "no_such_config"},
                {**good, "changes": []}):
        client, _ = client_for(tmp_path / "b", [bad])
        log = DecisionLog()
        crit, _ = C.critique(plan, prof, val_folds(plan, [{"primary": 0.1}]), [], [], {}, client, log, MPS)
        assert crit.source == "rules" and any(d.chosen == "rejected" for d in log.entries)


def test_apply_changes_and_keep_revision(r1_plan):
    prof, plan = r1_plan
    cfg = C.apply_changes(plan.primary, [Change(path="loss.cross_modal_regression.weight", value=0),
                                         Change(path="loss.supcon.weight", value=0.3),
                                         Change(path="modalities.image.patch_context", value=2.0)])
    names = [t.name for t in cfg.losses]
    assert "cross_modal_regression" not in names and "supcon" in names
    assert next(m for m in cfg.modalities if m.modality == "image").patch_context == 2.0
    with pytest.raises(ValueError):
        C.apply_changes(plan.primary, [Change(path="strategy", value="linear_alignment")])
    with pytest.raises(ValueError):
        C.apply_changes(plan.primary, [Change(path="training.lr", value="fast")])
    log = DecisionLog()
    assert C.keep_revision([0.10, 0.20], [0.15, 0.22], "m", log)
    assert not C.keep_revision([0.10, 0.20], [0.10, 0.19], "m", log)
    assert [d.chosen for d in log.entries] == ["kept", "discarded"]
    crit = Critique(assessment="a", action="modify_primary", changes=[Change(path="training.lr", value=1e-4)])
    cfg, _, err = C.revised_config(plan, crit, prof, MPS, False, 4, remaining_seconds=20)
    assert cfg is None and "no compute left" in err


# --- writer -----------------------------------------------------------------------------------

RECORD = {"retrieval": {"R@10": 0.33412, "chance": 0.002}, "n_spots": 1762,
          "flags": [{"message": "difference +0.094, 95% CI +0.041 to +0.151"}]}


def test_number_check_allows_rounding_and_percentages():
    text = ("Recall@10 was 0.334 (Fig. 2; Table 1), i.e. 33.4% of queries, against chance 0.002 on "
            "1,762 spots over 4 folds of the top-50 genes; the gain was +0.094 (95% CI 0.041 to 0.151). "
            "An invented value: 0.777. A model name such as ResNet-50 is not a result.")
    check = W.check_numbers({"results_interpretation": text}, RECORD)
    assert [u["number"] for u in check["unmatched"]] == ["0.777"]
    assert check["n_checked"] == 8          # top-50, Recall@10, Fig. 2, 4 folds and 95% are not checked


def test_writer_llm_template_and_partial(tmp_path):
    def template(rec):
        return {k: f"Template {k}: R@10 {rec['retrieval']['R@10']:.3f}." for k in W.SECTIONS}

    full = {k: f"LLM {k}: recall 0.334." for k in W.SECTIONS}
    client, _ = client_for(tmp_path / "a", [full])
    out = W.write_sections(RECORD, client, DecisionLog(), template)
    assert out["source"].startswith("LLM") and out["number_check"]["n_unmatched"] == 0
    partial = {**full, "limitations": ""}
    client, _ = client_for(tmp_path / "b", [partial])
    out = W.write_sections(RECORD, client, DecisionLog(), template)
    assert out["sections"]["limitations"].startswith("Template") and "template for limitations" in out["source"]
    client, _ = client_for(tmp_path / "c", [LLMUnavailable("claude CLI error: overloaded")])
    log = DecisionLog()
    out = W.write_sections(RECORD, client, log, template)
    assert out["source"] == "template" and log.by_step("writer")[0].chosen == "template"
    assert W.write_sections(RECORD, None, DecisionLog(), template)["source"] == "template"


def test_below_zero_shot_suggests_a_residual_head(r2):
    prof, proposal, _, _ = r2
    plan = proposal.model_copy(deep=True)
    plan.primary.head.residual = False
    flags = [{"code": "worse_than_zero_shot", "severity": "warning", "metric": plan.evaluation.primary_metric,
              "value": -0.04, "message": "below zero-shot"}]
    crit, cfg = C.critique(plan, prof, [{}], [], flags, {}, None, DecisionLog(), MPS)
    assert [c.path for c in crit.changes] == ["head.residual"] and cfg.head.residual
