"""Tables of clinical variables: mixed numeric/categorical preprocessing fitted on training rows,
the outcome kept out of the inputs, and an end-to-end run with a (fake) image encoder."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from multimodal_agent import figures
from multimodal_agent.encoders import screening
from multimodal_agent.encoders.registry import REGISTRY, EncoderInfo
from multimodal_agent.data.loaders import load_path
from multimodal_agent.objective import task_from_objective
from multimodal_agent.orchestrator import run
from multimodal_agent.preprocess.tabular import TabularPreprocessor
from multimodal_agent.profiler import profile

OBJ = "Learn a shared representation of scan images and patient covariates to classify stage and find clusters."


def test_tabular_preprocessor_is_fitted_on_training_rows_only():
    rng = np.random.default_rng(0)
    n, tr = 120, np.arange(100)
    df = pd.DataFrame({"age": rng.normal(60, 10, n),
                       "sex": rng.choice(["f", "m"], n),
                       "grade": ["g1"] * 50 + ["g2"] * 47 + ["g3"] * 3 + ["g1", "g2"] * 10,
                       "constant": "same", "empty": np.nan,
                       "stage": rng.choice(["I", "II"], n)})
    df.loc[::7, "age"] = np.nan
    df.loc[105, "grade"] = "g9"                                       # a level never seen in training
    p = TabularPreprocessor(exclude=["stage"]).fit(df.iloc[tr])
    names = p.feature_names
    assert not any(c.startswith("stage") for c in names) and p.excluded == ["stage"]
    assert {"age", "age (missing)", "sex=f", "sex=m", "grade=g1", "grade=g2", "grade=other"} == set(names)
    assert set(p.dropped) == {"constant", "empty"}                     # carry no information
    raw = p._encode(df)
    assert raw[0, names.index("age")] == pytest.approx(df["age"].iloc[tr].median())   # training median
    assert raw[105, names.index("grade=other")] == 1                  # unseen level -> 'other'
    Z = p.transform(df)
    assert np.isfinite(Z).all() and Z.shape == (n, len(names))
    np.testing.assert_allclose(Z[tr].mean(0), 0, atol=1e-5)
    changed = df.copy()
    changed.loc[100:, "age"] = 1e6                                    # test rows cannot move the statistics
    np.testing.assert_allclose(TabularPreprocessor(exclude=["stage"]).fit(changed.iloc[tr]).mean, p.mean)
    assert "one-hot" in p.describe() and "excluded as the evaluated outcome: stage" in p.describe()


def write_clinical(d, n=240, seed=3):
    """Greyscale images whose brightness tracks a latent severity; clinical columns that track it too."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    (d / "images").mkdir()
    z = rng.uniform(0, 1, n)
    for i, v in enumerate(z):
        Image.fromarray(np.full((16, 16), int(30 + 200 * v), np.uint8)).save(d / "images" / f"p{i}.png")
    age = 45 + 30 * z + rng.normal(0, 3, n)
    age[rng.random(n) < 0.1] = np.nan
    stage = np.array(["Stage I", "Stage II", "Stage III"])[np.minimum(2, (3 * z).astype(int))].astype(object)
    stage[rng.random(n) < 0.05] = "unknown"
    pd.DataFrame({"patient": [f"p{i}" for i in range(n)], "image_path": [f"images/p{i}.png" for i in range(n)],
                  "centre": [f"c{i % 6}" for i in range(n)], "age": age,
                  "sex": rng.choice(["female", "male"], n),
                  "grade": np.where(z + rng.normal(0, 0.15, n) > 0.5, "high", "low"),
                  "never_recorded": np.nan, "stage": stage}).to_csv(d / "pairs.csv", index=False)
    return d


def test_clinical_table_inputs_and_the_outcome_removed(tmp_path):
    ds = load_path(write_clinical(tmp_path))
    assert set(ds.modalities) == {"image", "tabular"}
    assert set(ds.modalities["tabular"].feature_names) == {"age", "sex", "grade", "stage"}
    assert any("never_recorded" in note for note in ds.notes)
    task, prof, removed = task_from_objective(ds, profile(ds), OBJ, 0)
    assert task.pair == ["image", "tabular"] and task.label_column == "stage" and removed == ["stage"]
    assert list(ds.modalities["tabular"].data.columns) == ["age", "sex", "grade"]
    tab = prof.modality("tabular").tabular
    assert (tab.n_features, tab.n_numeric, tab.n_categorical) == (3, 1, 2)


class FakeImageEncoder:
    kind, input_size, mean, std, device, dim = "image", 16, (0.0,) * 3, (1.0,) * 3, "cpu", 6
    info = EncoderInfo("fake_image", ("image",), "fake:x", ("generic",), 6, 16, quick=True)

    def encode(self, x):
        m = x.mean(dim=(1, 2, 3))
        return torch.stack([m, m ** 2, m.sin(), m.cos(), -m, 2 * m], 1)


def test_image_and_clinical_table_run_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "USE_UMAP", False)
    monkeypatch.setitem(REGISTRY, "fake_image", FakeImageEncoder.info)
    monkeypatch.setattr(screening, "candidates", lambda *a, **k: ["fake_image"])
    monkeypatch.setattr(screening, "load_encoder", lambda name, device: FakeImageEncoder())
    monkeypatch.setattr(screening, "release_all", lambda: None)
    (tmp_path / "data").mkdir()
    data = write_clinical(tmp_path / "data")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"cache_dir: {tmp_path / 'cache'}\n")
    out = tmp_path / "run"
    res = run(data, OBJ, out, quick=True, no_llm=True, device="cpu", config=str(cfg))
    plan = json.loads((out / "plan.json").read_text())
    assert plan["task"]["pair"] == ["image", "tabular"] and plan["evaluation"]["label_column"] == "stage"
    log = json.loads((out / "decision_log.json").read_text())
    assert any(d["decision"] == "outcome kept out of the inputs" for d in log)
    assert any(d["step"] == "preprocessing" and d["chosen"] == "impute, one-hot encode, standardise" for d in log)
    methods = (out / "methods_summary.md").read_text()
    assert "one-hot encoded" in methods and "stage" not in methods.split("**tabular**")[1].split("\n")[0]
    assert res["number_check"] == 0 and np.isfinite(res["test_value"])
