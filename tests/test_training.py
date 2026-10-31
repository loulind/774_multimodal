"""Losses (known answers), the training loop, input assembly and partial fine-tuning."""

import math

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from multimodal_agent.encoders.pretrained import Encoder
from multimodal_agent.encoders.registry import EncoderInfo
from multimodal_agent.models import losses as L
from multimodal_agent.models.finetune import finetune, unfreeze_last
from multimodal_agent.models.inputs import label_codes
from multimodal_agent.models.joint import FitData
from multimodal_agent.models.train import fit
from multimodal_agent.schema import ConfigSpec, HeadSpec, LossTerm, ModalityPlan, TrainingSpec


# --- losses -----------------------------------------------------------------------------

def test_infonce_known_answers():
    n = 32
    z = torch.eye(n)
    assert float(L.info_nce(z, z, torch.tensor(100.0))) < 1e-3                 # perfectly aligned
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(n, 16, generator=g), torch.randn(n, 16, generator=g)
    assert abs(float(L.info_nce(a, b, torch.tensor(1e-4))) - math.log(n)) < 1e-3   # flat logits -> log n


def test_soft_targets_share_mass_between_twins():
    x = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    T = L.soft_targets(x, 0.1)
    assert torch.allclose(T.sum(1), torch.ones(3))
    assert abs(float(T[0, 0] - T[0, 1])) < 1e-6 and float(T[0, 2]) < 1e-3
    z = torch.randn(3, 4, requires_grad=True)
    L.info_nce(z, z.detach() + 0.1, torch.tensor(10.0), T).backward()
    assert torch.isfinite(z.grad).all()


def test_supervised_ce_skips_unlabelled():
    clf = nn.Linear(4, 3)
    z = torch.randn(5, 4)
    assert float(L.supervised_ce([z], torch.full((5,), -1), clf)) == 0.0
    assert float(L.supervised_ce([z], torch.tensor([0, 1, -1, 2, -1]), clf).detach()) > 0


# --- training on synthetic paired data ---------------------------------------------------------

def paired(n=1200, seed=0, n_classes=3):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 4))
    A = (z @ rng.normal(size=(4, 32)) + 0.3 * rng.normal(size=(n, 32))).astype(np.float32)
    B = (np.tanh(z @ rng.normal(size=(4, 20))) + 0.3 * rng.normal(size=(n, 20))).astype(np.float32)
    y = np.digitize(z[:, 0], np.quantile(z[:, 0], np.linspace(0, 1, n_classes + 1)[1:-1]))
    y[rng.random(n) < 0.3] = -1
    return A, B, y.astype(np.int64)


def cfg(losses, strategy="contrastive_alignment", **train):
    t = dict(batch_size=128, lr=3e-3, max_epochs=60, patience=8, train_seconds=60)
    t.update(train)
    mods = [ModalityPlan(modality="a", encoder="mlp", regime="from_scratch"),
            ModalityPlan(modality="b", encoder="mlp", regime="from_scratch")]
    if strategy == "self_supervised":
        mods = mods[:1]
    return ConfigSpec(name="t", strategy=strategy, modalities=mods,
                      losses=[LossTerm(name=n, weight=w) for n, w in losses],
                      head=HeadSpec(embed_dim=16, hidden_dim=64, n_layers=2, dropout=0.0),
                      training=TrainingSpec(**t))


def split(A, B, y, target=None):
    tr, va = np.arange(900), np.arange(900, 1200)
    mk = lambda i: FitData({"a": A[i], "b": B[i]}, y[i], None if target is None else target[i], B[i])  # noqa: E731
    return mk(tr), mk(va)


def test_contrastive_training_beats_chance_and_is_deterministic():
    A, B, y = paired()
    tr, va = split(A, B, y)
    r1 = fit(cfg([("infonce", 1.0)]), tr, va, selection="retrieval", seed=1)
    r2 = fit(cfg([("infonce", 1.0)]), tr, va, selection="retrieval", seed=1)
    assert r1.best_score > 0.5                      # chance is 10/300
    assert r1.best_score == r2.best_score           # CPU training is exactly reproducible
    assert len(r1.history["val_score"]) == r1.epochs_run >= r1.best_epoch >= 1
    assert r1.stopped_because.startswith(("no validation", "epoch cap"))


def test_regression_head_learns_prediction():
    A, B, y = paired()
    tr, va = split(A, B, y, target=B)
    r = fit(cfg([("infonce", 1.0), ("cross_modal_regression", 1.0)]), tr, va, selection="prediction",
            pred_source="a", seed=0)
    pred = r.model.predict_np("a", va.inputs["a"])
    r2 = 1 - ((pred - va.target) ** 2).sum() / ((va.target - va.target.mean(0)) ** 2).sum()
    assert r.best_score > 0.6 and r2 > 0.4


def test_supervised_soft_and_time_cap():
    A, B, y = paired()
    tr, va = split(A, B, y)
    r = fit(cfg([("soft_infonce", 1.0), ("supervised_ce", 0.5)]), tr, va, selection="classification",
            n_classes=3, seed=0)
    assert r.model.classifier is not None and r.best_score > 0.5
    quick = fit(cfg([("infonce", 1.0)], train_seconds=1e-6), tr, va, selection="retrieval", seed=0)
    assert quick.epochs_run == 1 and "time cap" in quick.stopped_because


def test_self_supervised_single_modality():
    A, B, y = paired()
    tr = FitData({"a": A[:900]})
    va = FitData({"a": A[900:]})
    r = fit(cfg([("masked_denoising", 1.0)], strategy="self_supervised", max_epochs=3), tr, va,
            selection="loss", decode=("a",), seed=0)
    assert r.epochs_run == 3 and r.model.embed_np("a", A[:5]).shape == (5, 16)


def test_label_codes_come_from_training_classes():
    codes, classes = label_codes(np.array(["x", "y", None, "z", "x"], dtype=object), np.array([0, 1, 4]))
    assert classes == ["x", "y"] and codes.tolist() == [0, 1, -1, -1, 0]


# --- partial fine-tuning -----------------------------------------------------------------------

class FakeImageTower(Encoder):
    kind, input_size, mean, std = "image", 16, (0.5,) * 3, (0.5,) * 3

    def __init__(self):
        self.info = EncoderInfo("fake_img", ("image",), "fake:x", ("generic",), 8, 16,
                                partner="fake_txt", finetunable=True, n_blocks=3)
        self.device = "cpu"
        self.stem = nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.layers = nn.ModuleList([nn.Linear(4, 8), nn.Linear(8, 8), nn.Linear(8, 8)])
        self.proj = nn.Linear(8, 8)

    def modules(self):
        return [self.stem, self.layers, self.proj]

    def blocks(self):
        return list(self.layers)

    def encode(self, x):
        h = self.stem(x)
        for b in self.layers:
            h = torch.relu(b(h))
        return self.proj(h)


class FakeTextTower(Encoder):
    kind = "text"

    def __init__(self):
        self.info = EncoderInfo("fake_txt", ("text",), "fake:x", ("generic",), 8, max_tokens=16,
                                partner="fake_img", finetunable=True, n_blocks=2)
        self.device = "cpu"
        self.emb = nn.EmbeddingBag(97, 8)
        self.layers = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])

    def modules(self):
        return [self.emb, self.layers]

    def blocks(self):
        return list(self.layers)

    def tokenize(self, texts, max_tokens):
        ids = [[sum(map(ord, w)) % 97 for w in t.split()[:max_tokens]] or [0] for t in texts]
        offsets = torch.tensor([0] + [len(x) for x in ids[:-1]]).cumsum(0)
        return {"ids": torch.tensor([i for x in ids for i in x]), "offsets": offsets}

    def encode(self, batch):
        h = self.emb(batch["ids"], batch["offsets"])
        for b in self.layers:
            h = torch.relu(b(h))
        return h


def test_unfreeze_last_blocks_only():
    enc = FakeImageTower()
    trainable = unfreeze_last(enc, 2)
    names = {id(p) for p in trainable}
    assert all(id(p) not in names for p in enc.stem.parameters())
    assert all(id(p) not in names for p in enc.layers[0].parameters())
    assert all(p.requires_grad for p in list(enc.layers[1].parameters()) + list(enc.proj.parameters()))


@pytest.fixture()
def image_text_ds(tmp_path):
    from PIL import Image

    from multimodal_agent.data.loaders import load_path

    (tmp_path / "images").mkdir()
    words = ["alpha", "beta", "gamma", "delta"]
    rows = []
    for i in range(160):
        k = i % 4
        Image.fromarray(np.full((20, 20), 30 + 60 * k, np.uint8)).save(tmp_path / "images" / f"{i}.png")
        rows.append({"id": f"s{i}", "image_path": f"images/{i}.png",
                     "caption": f"an image of {words[k]} type number {i} in the series",
                     "kind": words[k]})
    pd.DataFrame(rows).to_csv(tmp_path / "pairs.csv", index=False)
    return load_path(tmp_path)


def test_finetune_runs_and_embeds(image_text_ds):
    c = ConfigSpec(name="ft", strategy="contrastive_alignment",
                   modalities=[ModalityPlan(modality="image", encoder="fake_img", regime="partial_finetune",
                                            trainable_blocks=2, augment=True),
                               ModalityPlan(modality="text", encoder="fake_txt", regime="partial_finetune",
                                            trainable_blocks=1, text_max_tokens=16)],
                   losses=[LossTerm(name="infonce"), LossTerm(name="supervised_ce", weight=0.5)],
                   head=HeadSpec(embed_dim=8, hidden_dim=16, n_layers=1, dropout=0.0),
                   training=TrainingSpec(batch_size=16, lr=1e-2, finetune_lr=1e-3, max_epochs=2, patience=5,
                                         train_seconds=60))
    labels, _ = label_codes(image_text_ds.obs["kind"].to_numpy(), np.arange(120))
    encs = {"image": FakeImageTower(), "text": FakeTextTower()}
    ft, res = finetune(c, image_text_ds, encs, np.arange(120), np.arange(120, 160), labels, 4, "cpu", 0)
    assert res.epochs_run == 2 and res.n_parameters > 0
    z = ft.embed_indices("image", np.arange(150, 160))
    assert z.shape == (10, 8) and np.allclose(np.linalg.norm(z, axis=1), 1, atol=1e-5)
    assert not any(p.requires_grad for p in encs["image"].stem.parameters())


def test_residual_head_starts_from_the_pretrained_space():
    from multimodal_agent.models.heads import MLP

    x = torch.randn(5, 16)
    assert torch.allclose(MLP(16, 16, 32, 2, 0.0, residual=True)(x), x)          # identity at the start
    assert not MLP(16, 8, residual=True).residual                               # sizes differ: plain head
    # pre-aligned inputs: the untrained start already retrieves perfectly, so it is kept (epoch 0)
    X = np.random.default_rng(0).normal(size=(400, 16)).astype(np.float32)
    c = cfg([("infonce", 1.0)], max_epochs=3)
    c.head = HeadSpec(embed_dim=16, hidden_dim=32, n_layers=2, dropout=0.0, residual=True)
    res = fit(c, FitData({"a": X[:300], "b": X[:300]}), FitData({"a": X[300:], "b": X[300:]}),
              selection="retrieval", seed=0)
    assert res.best_epoch == 0 and res.best_score == pytest.approx(1.0)
    assert res.history["initial_val_score"] == pytest.approx(1.0) and res.summary()["train_loss_at_best"] is None
    Z = X[300:] / np.linalg.norm(X[300:], axis=1, keepdims=True)
    assert np.allclose(res.model.embed_np("a", X[300:]), Z, atol=1e-5)
