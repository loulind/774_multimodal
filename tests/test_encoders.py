"""Preprocessing, encoder screening and heads -- offline, with fake encoders."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from multimodal_agent.encoders import pretrained, screening
from multimodal_agent.encoders.pretrained import FeatureCache
from multimodal_agent.encoders.registry import REGISTRY, EncoderInfo, candidates
from multimodal_agent.encoders.screening import CCA, recall_at_k, ridge_r2
from multimodal_agent.models.heads import MLP, n_parameters
from multimodal_agent.preprocess.image import ImageTransform, ReinhardNormalizer, lab_to_rgb, rgb_to_lab
from multimodal_agent.preprocess.matrix import MatrixPreprocessor


# --- image preprocessing -----------------------------------------------------------------

def test_lab_round_trip():
    rgb = np.random.default_rng(0).integers(0, 256, (50, 50, 3), dtype=np.uint8)
    assert np.abs(lab_to_rgb(rgb_to_lab(rgb)).astype(int) - rgb).max() <= 1


def test_reinhard_removes_group_colour_shift_but_keeps_contrast():
    rng = np.random.default_rng(1)
    base = [rng.integers(60, 200, (32, 32, 3)).astype(np.uint8) for _ in range(10)]
    shifted = {"g1": base[:5], "g2": [np.clip(b.astype(int) + [40, -20, 10], 0, 255).astype(np.uint8)
                                      for b in base[5:]]}
    norm = ReinhardNormalizer().fit(shifted)
    before = abs(rgb_to_lab(np.stack(shifted["g1"]))[..., 1].mean() -
                 rgb_to_lab(np.stack(shifted["g2"]))[..., 1].mean())
    after_g1 = np.stack([norm(x, "g1") for x in shifted["g1"]])
    after_g2 = np.stack([norm(x, "g2") for x in shifted["g2"]])
    after = abs(rgb_to_lab(after_g1)[..., 1].mean() - rgb_to_lab(after_g2)[..., 1].mean())
    assert after < 0.25 * before
    L0, L1 = rgb_to_lab(shifted["g2"][0])[..., 0].ravel(), rgb_to_lab(after_g2[0])[..., 0].ravel()
    assert np.corrcoef(L0, L1)[0, 1] > 0.95                     # within-image structure kept


def test_image_transform_shapes():
    from PIL import Image

    tf = ImageTransform(224, (0.5,) * 3, (0.5,) * 3)
    grey = Image.fromarray(np.zeros((100, 180), np.uint8))
    assert tf(grey).shape == (3, 224, 224)
    aug = ImageTransform(64, (0.5,) * 3, (0.5,) * 3, augment=True, square_input=True)
    assert aug(Image.new("RGB", (40, 40)), rng=np.random.default_rng(0)).shape == (3, 64, 64)


def test_matrix_preprocessor_fits_on_training_rows_only():
    rng = np.random.default_rng(2)
    D = rng.poisson(1.0, (300, 400)).astype(np.float32)
    D[:100, :10] = rng.poisson(30.0, (100, 10))       # genes 0-9 vary across training rows
    D[200:, 10:20] = rng.poisson(30.0, (100, 10))     # genes 10-19 vary only in held-out rows
    X = sp.csr_matrix(D)
    pre = MatrixPreprocessor(10, "pca", 5).fit(X[:200])
    assert set(pre.features) == set(range(10))
    Z = pre.transform(X)
    assert Z.shape == (300, 5) and pre.out_dim == 5
    S = pre.standardised(X[:200])
    assert np.allclose(S.mean(0), 0, atol=1e-4)


# --- screening statistics -----------------------------------------------------------------

def test_cca_retrieval_known_answers():
    rng = np.random.default_rng(3)
    Z = rng.normal(size=(600, 8))
    X = Z @ rng.normal(size=(8, 40)) + 0.05 * rng.normal(size=(600, 40))
    Y = Z @ rng.normal(size=(8, 30)) + 0.05 * rng.normal(size=(600, 30))
    cca = CCA(k=8).fit(X[:400], Y[:400])
    assert recall_at_k(*cca.transform(X[400:], Y[400:])) > 0.95
    noise = rng.normal(size=(600, 30))
    cca = CCA(k=8).fit(X[:400], noise[:400])
    assert recall_at_k(*cca.transform(X[400:], noise[400:])) < 0.15       # chance is 10/200
    assert ridge_r2(X[:400], Y[:400], X[400:], Y[400:]) > 0.9
    assert ridge_r2(X[:400], noise[:400], X[400:], noise[400:]) < 0.05


# --- screening end-to-end with fake encoders ---------------------------------------------------

class FakeEncoder:
    kind, input_size, mean, std, device = "image", 16, (0.0,) * 3, (1.0,) * 3, "cpu"

    def __init__(self, name, good):
        self.info = EncoderInfo(name, ("image",), "fake:x", ("generic",), 6, 16)
        self.dim, self.good = 6, good

    def encode(self, x):
        m = x.mean(dim=(1, 2, 3))
        if self.good:
            return torch.stack([m, m ** 2, m.sin(), m.cos(), -m, m * 2], 1)
        g = torch.Generator().manual_seed(int(1000 * float(m.sum())) % 10_000)
        return torch.randn(len(x), 6, generator=g)


@pytest.fixture()
def image_matrix_dir(tmp_path):
    import anndata as ad
    from PIL import Image

    rng = np.random.default_rng(4)
    (tmp_path / "im").mkdir()
    z = rng.uniform(0, 1, 300)
    for i, v in enumerate(z):
        Image.fromarray(np.full((16, 16), int(40 + 200 * v), np.uint8)).save(tmp_path / "im" / f"{i}.png")
    lam = np.outer(np.exp(2 * z), rng.uniform(0.5, 2, 60))
    obs = pd.DataFrame({"image_path": [f"im/{i}.png" for i in range(300)],
                        "patient": [f"P{i % 6}" for i in range(300)]}, index=[f"s{i}" for i in range(300)])
    ad.AnnData(sp.csr_matrix(rng.poisson(lam).astype(np.float32)), obs=obs,
               var=pd.DataFrame(index=[f"g{j}" for j in range(60)])).write_h5ad(tmp_path / "d.h5ad")
    return tmp_path


def test_screening_ranks_informative_encoder_first(image_matrix_dir, monkeypatch, tmp_path):
    from multimodal_agent.config import load_config
    from multimodal_agent.data.loaders import load_path
    from multimodal_agent.decision_log import DecisionLog
    from multimodal_agent.objective import parse_objective
    from multimodal_agent.planner import heuristic
    from multimodal_agent.planner.flow import folds_for
    from multimodal_agent.profiler import profile

    fakes = {"fake_good": FakeEncoder("fake_good", True), "fake_bad": FakeEncoder("fake_bad", False)}
    for name, enc in fakes.items():
        monkeypatch.setitem(REGISTRY, name, enc.info)
    monkeypatch.setattr(screening, "candidates", lambda *a, **k: list(fakes))
    monkeypatch.setattr(screening, "load_encoder", lambda name, device: fakes[name])
    monkeypatch.setattr(screening, "release_all", lambda: None)
    ds = load_path(image_matrix_dir)
    prof = profile(ds)
    task = parse_objective("Align images with gene expression for retrieval", prof)
    budget = load_config(quick=True).budget("cpu").model_copy(update={"screening_train": 150,
                                                                       "screening_val": 60})
    prelim = heuristic.plan(prof, task, budget, DecisionLog(), quick=True)
    folds, _ = folds_for(ds, prelim, 0, None)
    cache = FeatureCache(tmp_path / "cache")
    results, table = screening.screen(ds, prof, task, prelim, folds[0].train, folds[0].val, budget, "cpu",
                                      cache, quick=True)
    img = next(r for r in results if r.modality == "image")
    assert [s.encoder for s in img.ranked] == ["fake_good", "fake_bad"]
    assert img.ranked[0].score > img.ranked[1].score + 0.2
    assert len(list((tmp_path / "cache").glob("*.npy"))) == 2              # features were cached
    log = DecisionLog()
    final = heuristic.plan(prof, task, budget, log, screening=results, quick=True)
    assert final.primary.modalities[0].encoder == "fake_good"
    assert "screening score" in log.by_step("encoder")[0].rationale


# --- registry and heads ---------------------------------------------------------------------

def test_candidates_always_include_a_generic_model():
    for domain in ("stained_tissue", "greyscale_imaging", "colour_other", None):
        c = candidates("image", domain, 4)
        assert any("generic" in REGISTRY[x].domains for x in c) and len(c) <= 4
    assert all(REGISTRY[x].quick for x in candidates("image", "stained_tissue", 2, quick=True))


def test_registry_lists_open_weights_only():
    gated = ("uni", "conch", "virchow", "gigapath", "h-optimus", "hibou")
    assert not any(g in e.source.lower() for e in REGISTRY.values() for g in gated)


def test_mlp_head():
    head = MLP(384, 128, hidden=256, n_layers=2, dropout=0.1)
    assert head(torch.zeros(5, 384)).shape == (5, 128)
    assert n_parameters(head) == 384 * 256 + 256 + 2 * 256 + 256 * 128 + 128


def test_find_blocks_handles_library_layouts():
    from torch import nn

    class A(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(2, 2)] * 3)

    class B(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList([nn.Linear(2, 2)] * 4)

    assert len(pretrained.find_blocks(A())) == 3 and len(pretrained.find_blocks(B())) == 4
