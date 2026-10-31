"""Objective parsing, rule-based planning, guardrails and splits (synthetic profiles only)."""

import numpy as np
import pandas as pd
import pytest

from factories import image_text_profile, label, matrix, profile, spatial_profile
from multimodal_agent.config import load_config
from multimodal_agent.data.splits import make_folds
from multimodal_agent.decision_log import DecisionLog
from multimodal_agent.objective import parse_objective
from multimodal_agent.planner import heuristic
from multimodal_agent.planner.validator import validate
from multimodal_agent.schema import SplitSpec

R1 = ("Learn joint embeddings of tissue image patches and spot-level gene expression to enable "
      "cross-modal retrieval and prediction of gene expression from histology.")
R2 = "Learn a joint image–caption embedding for image–text retrieval and imaging-modality classification."
MPS, CPU = load_config().budget("mps"), load_config().budget("cpu")
QUICK = load_config(quick=True).budget("mps")


def plan_for(prof, objective, budget=MPS, quick=False):
    log = DecisionLog()
    p = heuristic.plan(prof, parse_objective(objective, prof), budget, log, quick=quick)
    return p, log


# --- objective --------------------------------------------------------------------------

def test_parse_spatial_objective():
    t = parse_objective(R1, spatial_profile())
    assert t.primary_task == "retrieval" and "cross_modal_prediction" in t.tasks
    assert t.pair == ["image", "expression"]
    assert (t.prediction.source, t.prediction.target) == ("image", "expression")
    assert t.label_column == "label" and "evaluation only" in t.label_evidence


def test_parse_image_text_objective_names_label():
    t = parse_objective(R2, image_text_profile())
    assert t.tasks == ["joint_embedding", "retrieval", "classification"]
    assert t.pair == ["image", "text"] and t.label_column == "modality"
    assert "names column" in t.label_evidence


def test_parse_matrix_families_and_clustering():
    prof = profile([matrix(5000, "rna"), matrix(5000, "peaks")], 5000,
                   labels=[label("cell_type", {"a": 3000, "b": 2000}, 5000)])
    t = parse_objective("Integrate chromatin accessibility and RNA to cluster cells into types", prof)
    assert t.primary_task == "clustering" and t.pair == ["peaks", "rna"]


def test_predicting_a_label_is_classification_not_cross_modal():
    prof = image_text_profile()
    t = parse_objective("Predict the scan type from the images", prof)
    assert "classification" in t.tasks and t.prediction is None


def test_no_measurable_task_assumes_retrieval():
    t = parse_objective("Learn a joint representation of both modalities", image_text_profile())
    assert t.primary_task == "retrieval" and "notes" in t.evidence


# --- rule-based planner -------------------------------------------------------------------

def test_spatial_plan_follows_profile():
    p, log = plan_for(spatial_profile(), R1)
    assert p.split.strategy == "grouped_kfold" and p.split.n_folds == 4
    assert p.split.group_column == "patient"
    prim = p.primary
    assert prim.strategy == "contrastive_alignment"
    assert [t.name for t in prim.losses] == ["soft_infonce", "cross_modal_regression"]
    img, expr = prim.modalities
    assert img.encoder == "phikon" and img.colour_normalisation == "reinhard" and img.patch_context == 1.0
    assert expr.encoder == "mlp" and expr.regime == "from_scratch" and expr.matrix_input == "pca"
    assert [c.name for c in p.configs[1:]] == ["hard_infonce", "patch_context_2x"]
    assert p.evaluation.primary_metric == "retrieval_mean_recall_at_10"
    assert {"ridge", "cca", "unimodal"} <= set(p.evaluation.baselines)
    assert "spatial_maps" in p.downstream
    assert all(d.rationale for d in log.entries)


def test_low_redundancy_and_colour_keep_defaults():
    p, _ = plan_for(spatial_profile(twins=0.02, icc=0.05), R1)
    assert p.primary.losses[0].name == "infonce"
    assert p.primary.modalities[0].colour_normalisation == "none"


def test_image_text_plan_finetunes_as_comparison_on_gpu():
    p, _ = plan_for(image_text_profile(), R2)
    assert p.split.strategy == "predefined"
    assert [m.encoder for m in p.primary.modalities] == ["biomedclip_image", "biomedclip_text"]
    assert {t.name for t in p.primary.losses} == {"infonce", "supervised_ce"}
    names = [c.name for c in p.configs]
    assert names == ["primary", "partial_finetune", "contrastive_only"]
    ft = p.configs[1]
    assert all(m.regime == "partial_finetune" for m in ft.modalities)
    assert "zero_shot" in p.evaluation.baselines


def test_no_finetuning_on_cpu_or_with_few_pairs():
    p, log = plan_for(image_text_profile(), R2, budget=CPU)
    assert "partial_finetune" not in [c.name for c in p.configs]
    assert any("no GPU" in d.rationale for d in log.by_step("regime"))
    p, _ = plan_for(image_text_profile(n_train=5000, n_val=500, n_test=1000), R2)
    assert "partial_finetune" not in [c.name for c in p.configs]


def test_finetuning_becomes_primary_with_many_pairs():
    big = load_config().budget("mps").model_copy(update={"finetune_seconds_per_fit": 3000})
    p, _ = plan_for(image_text_profile(n_train=60_000), R2, budget=big)
    assert p.primary.name == "partial_finetune"


def test_few_pairs_use_linear_alignment():
    p, _ = plan_for(image_text_profile(n_train=400, n_val=100, n_test=100), R2)
    assert p.primary.strategy == "linear_alignment" and len(p.configs) == 1


def test_quick_mode_uses_one_fold_and_small_encoders():
    p, _ = plan_for(spatial_profile(), R1, budget=QUICK, quick=True)
    assert p.split.n_folds == 1 and p.primary.modalities[0].encoder == "dinov2_small"


@pytest.mark.parametrize("prof, obj", [(spatial_profile(), R1), (image_text_profile(), R2),
                                       (image_text_profile(labelled=False), R2)])
def test_heuristic_plans_pass_validation(prof, obj):
    p, _ = plan_for(prof, obj)
    res = validate(p, prof, MPS)
    assert res.ok, res.errors


# --- guardrails ---------------------------------------------------------------------------

def _dict(prof=None, obj=R2):
    p, _ = plan_for(prof or image_text_profile(), obj)
    return p.model_dump()


def test_validator_rejects_unknown_primary_encoder():
    d = _dict()
    d["configs"][0]["modalities"][0]["encoder"] = "gated_foundation_model"
    res = validate(d, image_text_profile(), MPS)
    assert not res.ok and "unknown encoder" in res.errors[0]


def test_validator_repairs():
    d = _dict()
    d["configs"][0]["training"]["lr"] = 5.0
    d["configs"][0]["head"]["dropout"] = 0.9
    d["configs"][0]["modalities"][1]["text_max_tokens"] = 4096
    d["evaluation"]["baselines"].append("magic")
    res = validate(d, image_text_profile(), MPS)
    assert res.ok
    c = res.plan.configs[0]
    assert c.training.lr == 1e-2 and c.head.dropout == 0.5
    assert c.modalities[1].text_max_tokens == 256
    assert "magic" not in res.plan.evaluation.baselines


def test_validator_drops_supervised_loss_without_labels_and_finetuning_on_cpu():
    prof = image_text_profile(labelled=False)
    d = _dict()
    res = validate(d, prof, CPU)
    assert res.ok
    for c in res.plan.configs:
        assert "supervised_ce" not in [t.name for t in c.losses]
        assert all(m.regime == "frozen" for m in c.modalities)


def test_validator_quick_mode_swaps_large_encoders_and_fixes_split():
    d = _dict(spatial_profile(), R1)
    d["split"] = {"strategy": "predefined"}
    res = validate(d, spatial_profile(), QUICK, quick=True)
    assert res.ok
    assert res.plan.split.strategy == "grouped_kfold" and res.plan.split.n_folds == 1
    assert res.plan.configs[0].modalities[0].encoder == "dinov2_small"


def test_validator_enforces_run_budget():
    d = _dict(spatial_profile(), R1)
    tight = MPS.model_copy(update={"total_train_seconds": 600})
    res = validate(d, spatial_profile(), tight)
    assert res.ok and any("budget" in r for r in res.repairs)
    folds = res.plan.split.n_folds
    p = res.plan.configs[0].training.train_seconds
    total = folds * sum(c.training.train_seconds for c in res.plan.configs) + p * (1 + folds)
    assert total <= 600 + 1e-6


def test_schema_garbage_is_rejected():
    assert not validate({"configs": "lots"}, image_text_profile(), MPS).ok


# --- splits ---------------------------------------------------------------------------------

def _obs(n_groups=8, per=50, labelled_groups=8):
    rows = []
    for g in range(n_groups):
        for i in range(per):
            lab = (["a", "b"][i % 2] if g < labelled_groups and i < per // 2 else np.nan)
            rows.append({"patient": f"P{g}", "label": lab})
    return pd.DataFrame(rows)


def test_grouped_kfold_tests_every_group_once():
    obs = _obs()
    folds, _ = make_folds(obs, np.ones(len(obs), bool),
                          SplitSpec(strategy="grouped_kfold", group_column="patient", n_folds=4), 0,
                          label_column="label")
    assert len(folds) == 4
    tested = [g for f in folds for g in f.test_groups]
    assert sorted(tested) == sorted(obs["patient"].unique())
    for f in folds:
        parts = [set(obs["patient"].iloc[x]) for x in (f.train, f.val, f.test)]
        assert not (parts[0] & parts[1]) and not (parts[0] & parts[2]) and not (parts[1] & parts[2])
        assert obs["label"].iloc[f.test].notna().any()


def test_predefined_split_without_validation_is_carved():
    obs = pd.DataFrame({"split": ["train"] * 80 + ["test"] * 20, "label": ["a", "b"] * 50})
    folds, notes = make_folds(obs, np.ones(100, bool), SplitSpec(strategy="predefined"), 0,
                              split_column="split", label_column="label")
    f = folds[0]
    assert len(f.test) == 20 and len(f.val) > 0 and len(f.train) + len(f.val) == 80
    assert any("validation" in n for n in notes)


def test_random_split_and_subsample_keep_groups():
    obs = _obs()
    folds, notes = make_folds(obs, np.ones(len(obs), bool),
                              SplitSpec(strategy="grouped_kfold", group_column="patient", n_folds=1), 0,
                              max_samples=120)
    f = folds[0]
    used = np.concatenate([f.train, f.val, f.test])
    assert len(used) <= 130 and obs["patient"].iloc[used].nunique() == 8
    folds, _ = make_folds(obs, np.ones(len(obs), bool), SplitSpec(strategy="random"), 0)
    assert abs(len(folds[0].test) - 80) <= 2


def test_pre_aligned_pair_gets_residual_heads_and_validator_guards_them():
    p, log = plan_for(image_text_profile(), R2)
    encs = [m.encoder for m in p.primary.modalities]
    assert set(encs) == {"biomedclip_image", "biomedclip_text"}
    assert p.primary.head.residual and p.primary.head.embed_dim == 512
    assert all(c.head.residual for c in p.configs)                       # comparisons inherit it
    assert log.by_step("hyperparameters")[-1].decision == "head initialisation"
    spatial, _ = plan_for(spatial_profile(), R1)
    assert not spatial.primary.head.residual                            # phikon + MLP: no aligned pair
    bad = spatial.model_copy(deep=True)
    bad.primary.head.residual = True
    checked = validate(bad, spatial_profile(), MPS)
    assert not checked.plan.primary.head.residual and any("residual" in r for r in checked.repairs)
