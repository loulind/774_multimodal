"""End-to-end agent runs on a small synthetic paired-matrix dataset (no downloads, CPU only)."""

import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from multimodal_agent import figures
from multimodal_agent.llm import LLMClient, LLMResponse
from multimodal_agent.orchestrator import run

OBJ = "Integrate RNA and chromatin accessibility to learn a joint embedding for cross-modal retrieval " \
      "and cell-type classification."


def write_paired(d, n, n_donors, seed=0, labelled=True):
    """Two count matrices (RNA, chromatin peaks) driven by a shared cell-type signal; ``labelled=False``
    leaves out the cell-type and donor columns."""
    import anndata as ad

    rng = np.random.default_rng(seed)
    k = 3
    y = np.arange(n) % k
    z = rng.normal(size=(k, 6))[y] + 0.6 * rng.normal(size=(n, 6))
    rna = rng.poisson(np.exp(0.4 + 0.5 * z @ rng.normal(size=(6, 80)))).astype(np.float32)
    peaks = rng.poisson(np.exp(-0.6 + 0.5 * z @ rng.normal(size=(6, 60)))).astype(np.float32)
    ids = [f"c{i}" for i in range(n)]
    obs = pd.DataFrame({"cell_type": [f"type{v}" for v in y], "donor": [f"d{i % n_donors}" for i in range(n)]},
                       index=ids) if labelled else pd.DataFrame(index=ids)
    ad.AnnData(sp.csr_matrix(rna), obs=obs, var=pd.DataFrame(index=[f"G{i}" for i in range(80)])
               ).write_h5ad(d / "rna.h5ad")
    ad.AnnData(sp.csr_matrix(peaks), obs=pd.DataFrame(index=ids),
               var=pd.DataFrame(index=[f"chr1:{1000 * i}-{1000 * i + 500}" for i in range(60)])).write_h5ad(d / "atac.h5ad")
    return d


@pytest.fixture(scope="module")
def paired_dir(tmp_path_factory):
    return write_paired(tmp_path_factory.mktemp("paired"), 1000, 8)   # 6 training donors: ~750 pairs


class FakeCLI:
    """Answers like the planner, critic and writer would: adopts the proposal, asks for one
    revision, and writes prose containing one invented number."""
    name, model, max_output_tokens = "fake", "fake-model", 100

    def __init__(self):
        self.roles = []

    def complete(self, system, user, schema=None, effort=None, budget_usd=None):
        payload = json.loads(user.split("\n\n", 1)[1])
        if "planning component" in system:
            self.roles.append("planner")
            reply = {"plan": payload["proposal"]["plan"],
                     "decisions": [{"step": "split", "decision": "split", "chosen": "keep", "rationale": "groups"}]}
        elif system.startswith("You are the critic"):
            self.roles.append("critic")
            reply = {"assessment": "Validation scores are modest.", "action": "modify_primary",
                     "changes": [{"path": "head.dropout", "value": 0.3}], "rationale": "regularise",
                     "alternatives_rejected": [], "expected_effect": "recall up"}
        else:
            self.roles.append("writer")
            reply = {k: f"Section {k}." for k in ("executive_summary", "data_and_objective", "decisions_explained",
                                                  "results_interpretation", "integration_benefit",
                                                  "critic_and_revision", "limitations")}
            reply["executive_summary"] = "The invented value 0.987654 is not in the record."
        return LLMResponse(text=json.dumps(reply), backend=self.name, model_requested=self.model,
                           model_used=self.model, seconds=0.0, cost_usd=0.01, data=reply, structured=True)


def test_rule_based_run_writes_every_output(paired_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "USE_UMAP", False)
    out = tmp_path / "run"
    res = run(paired_dir, OBJ, out, quick=True, no_llm=True, device="cpu")
    for f in ("report.md", "report.html", "methods_summary.md", "plan.md", "plan.json", "decision_log.json",
              "evaluation.json", "profile.json", "model_summary.json", "run_manifest.json",
              "embeddings/held_out_embeddings.npz", "embeddings/about.json", "tables/training.csv",
              "figures/comparison.png", "figures/training_curves.png"):
        assert (out / f).exists(), f
    assert res["plan_source"] == "heuristic" and res["prose_source"] == "template"
    assert res["number_check"] == 0                       # template prose quotes only computed numbers
    steps = {d["step"] for d in json.loads((out / "decision_log.json").read_text())}
    assert {"hardware", "profile", "objective", "split", "planner", "training", "critic", "evaluation",
            "writer"} <= steps
    emb = np.load(out / "embeddings" / "held_out_embeddings.npz")
    assert emb["rna"].dtype == np.float16 and len(emb["ids"]) == len(emb["peaks"]) > 0
    ev = json.loads((out / "evaluation.json").read_text())
    assert ev["primary"] == res["primary"] and ev["test"]["comparisons"]


def test_llm_run_then_replay_reproduces_the_decisions(paired_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "USE_UMAP", False)
    fake = FakeCLI()
    out1, out2 = tmp_path / "llm", tmp_path / "replay"
    res = run(paired_dir, OBJ, out1, quick=True, device="cpu", client=LLMClient(fake, out1 / "llm_cache"))
    assert fake.roles == ["planner", "critic", "writer"] and res["plan_source"] == "llm"
    assert res["critic_action"] == "modify_primary" and res["revision_kept"] in (True, False)
    assert res["number_check"] == 1                       # the invented number is flagged in the report
    assert "0.987654" in (out1 / "report.md").read_text().split("## Number check")[1]
    assert len(json.loads((out1 / "llm_transcript.json").read_text())["calls"]) == 3
    res2 = run(paired_dir, OBJ, out2, quick=True, device="cpu", replay=str(out1))
    key = lambda d: (d["step"], d["decision"], d["chosen"])  # noqa: E731
    log1 = [key(d) for d in json.loads((out1 / "decision_log.json").read_text())]
    log2 = [key(d) for d in json.loads((out2 / "decision_log.json").read_text())]
    assert log1 == log2 and res2["test_value"] == res["test_value"]


def test_few_pairs_give_closed_form_linear_alignment(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "USE_UMAP", False)
    (tmp_path / "data").mkdir()
    data = write_paired(tmp_path / "data", 720, 4)                 # 2 training donors: 360 pairs (< 500)
    out = tmp_path / "run"
    res = run(data, OBJ, out, quick=True, no_llm=True, device="cpu")
    plan = json.loads((out / "plan.json").read_text())
    assert plan["configs"][0]["strategy"] == "linear_alignment" and res["number_check"] == 0
    fits = [d for d in json.loads((out / "decision_log.json").read_text()) if d["step"] == "training"]
    assert fits and fits[0]["chosen"].startswith("closed form")
    assert not (out / "figures" / "training_curves.png").exists()  # no epochs to plot


def test_unlabelled_clustering_is_measured_by_cross_modal_agreement(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "USE_UMAP", False)
    (tmp_path / "data").mkdir()
    data = write_paired(tmp_path / "data", 900, 1, labelled=False)
    out = tmp_path / "run"
    res = run(data, "Align RNA with chromatin peaks in a shared space; evaluate retrieval and clustering.", out,
              quick=True, no_llm=True, device="cpu")
    ev = json.loads((out / "evaluation.json").read_text())
    fold = ev["test"]["folds"][0]
    assert {res["primary"], "unimodal", "cca"} <= set(fold["cluster_agreement"])
    assert len({r["k"] for r in fold["cluster_agreement"].values()}) == 1
    for f in ("tables/cluster_agreement.csv", "figures/cluster_map.png", "downstream/cluster_map.json"):
        assert (out / f).exists(), f
    cm = json.loads((out / "downstream" / "cluster_map.json").read_text())
    assert "x" not in cm and "agreement" in cm                    # no coordinates to map; no labels
    report = (out / "report.md").read_text()
    assert "Clustering without labels" in report and res["number_check"] == 0
    log = json.loads((out / "decision_log.json").read_text())
    split = next(d for d in log if d["step"] == "split")
    assert split["chosen"].startswith("random")                   # nothing to stratify by
    regime = next(d for d in log if d["step"] == "regime")
    assert regime["chosen"] == "trained from scratch"             # no pretrained backbone to freeze
