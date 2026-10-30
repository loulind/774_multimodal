"""Train a joint model on fixed inputs (frozen features / preprocessed matrices).

Each epoch shuffles the training pairs into mini-batches, minimises the weighted
sum of the configuration's losses with AdamW, and clips gradients. After every
epoch the validation samples are scored with the *selection metric* -- the
validation version of the plan's primary metric -- and the best epoch's weights
are kept. Training stops when the score has not improved for ``patience`` epochs,
at the epoch cap, or when the wall-clock cap runs out, whichever comes first.
Test data are never seen here. The history (loss per term on training and
validation data, validation score, temperature, time) feeds the critic and report.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ..evaluate.metrics import columnwise_pearson, recall_at_k
from ..schema import ConfigSpec
from . import losses as L
from .heads import n_parameters
from .joint import FitData, JointModel

SELECTION = {"retrieval_mean_recall_at_10": "retrieval", "prediction_mean_pearson_top50": "prediction",
             "probe_macro_f1": "classification", "clustering_ari": "retrieval"}
VAL_CAP = 2000          # validation samples scored per epoch (fixed subsample)


@dataclass
class FitResult:
    model: JointModel
    selection: str
    best_epoch: int
    best_score: float
    epochs_run: int
    seconds: float
    stopped_because: str
    n_parameters: int
    history: dict = field(default_factory=dict)

    def summary(self) -> dict:
        h = self.history
        return {"selection_metric": self.selection, "best_epoch": self.best_epoch,
                "best_val_score": round(self.best_score, 4), "epochs_run": self.epochs_run,
                "seconds": round(self.seconds, 1), "stopped_because": self.stopped_because,
                "n_trainable_parameters": self.n_parameters,
                "final_temperature": h["temperature"][-1] if h.get("temperature") else None,
                "train_loss_at_best": h["train_loss"][self.best_epoch - 1] if h.get("train_loss") else None,
                "val_loss_at_best": h["val_loss"][self.best_epoch - 1] if h.get("val_loss") else None}


def _to(data: FitData, device) -> dict:
    t = lambda a, dt=torch.float32: None if a is None else torch.as_tensor(a, dtype=dt, device=device)  # noqa: E731
    return {"inputs": {k: t(v) for k, v in data.inputs.items()},
            "labels": t(data.labels, torch.long), "target": t(data.target), "soft": t(data.soft_source)}


def _slice(b: dict, idx) -> dict:
    s = lambda v: None if v is None else v[idx]  # noqa: E731
    return {"inputs": {k: v[idx] for k, v in b["inputs"].items()}, "labels": s(b["labels"]),
            "target": s(b["target"]), "soft": s(b["soft"])}


def compute_losses(model: JointModel, cfg: ConfigSpec, batch: dict, pred_source: str | None,
                   gen: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    pair = model.pair
    z = {m: model.embed(m, batch["inputs"][m]) for m in pair}
    out: dict[str, torch.Tensor] = {}
    for term in cfg.losses:
        n = term.name
        if n in ("infonce", "soft_infonce") and len(pair) == 2:
            targets = None
            if n == "soft_infonce" and batch["soft"] is not None:
                targets = L.soft_targets(batch["soft"], cfg.training.soft_target_temperature)
            out[n] = L.info_nce(z[pair[0]], z[pair[1]], model.scale(), targets)
        elif n in ("supervised_ce", "supcon") and model.classifier is not None and batch["labels"] is not None:
            if cfg.strategy == "supervised_fusion":
                zs = [L.normalize(torch.stack([L.normalize(z[m]) for m in pair]).mean(0))]
            else:
                zs = [z[m] for m in pair]
            out[n] = L.supervised_ce(zs, batch["labels"], model.classifier)
        elif n == "cross_modal_regression" and model.regressor is not None and batch["target"] is not None:
            out[n] = L.regression(model.regressor(z[pred_source]), batch["target"])
        elif n == "reconstruction" and len(model.decoders):
            out[n] = torch.stack([torch.nn.functional.mse_loss(model.decoders[m](z[m]), batch["inputs"][m])
                                  for m in model.decoders]).mean()
        elif n == "masked_denoising" and len(model.decoders):
            out[n] = torch.stack([L.masked_denoising(batch["inputs"][m], (lambda x, m=m: model.embed(m, x)),
                                                     model.decoders[m], gen) for m in model.decoders]).mean()
        elif n == "ssl_simclr":
            m = pair[0]
            out[n] = L.dropout_views(batch["inputs"][m], (lambda x: model.embed(m, x)), model.scale())
    return out


def _total(cfg: ConfigSpec, terms: dict[str, torch.Tensor]) -> torch.Tensor:
    w = {t.name: t.weight for t in cfg.losses}
    return sum(w[k] * v for k, v in terms.items())


@torch.no_grad()
def score(model: JointModel, data: FitData, selection: str, pred_source: str | None) -> float:
    """Validation score, higher is better (on at most VAL_CAP samples)."""
    keep = np.random.default_rng(0).permutation(data.n)[:VAL_CAP]
    d = data.subset(np.sort(keep))
    pair = model.pair
    if selection == "retrieval" and len(pair) == 2:
        return recall_at_k(model.embed_np(pair[0], d.inputs[pair[0]]), model.embed_np(pair[1], d.inputs[pair[1]]))
    if selection == "prediction" and model.regressor is not None and d.target is not None:
        return float(np.mean(columnwise_pearson(model.predict_np(pred_source, d.inputs[pred_source]), d.target)))
    if selection == "classification" and model.classifier is not None and d.labels is not None:
        from sklearn.metrics import f1_score

        lab = d.labels >= 0
        if lab.sum() >= 2:
            probs = np.mean([model.classify_np(m, d.inputs[m][lab]) for m in pair], axis=0)
            return float(f1_score(d.labels[lab], probs.argmax(1), average="macro"))
    if len(pair) == 2:
        return recall_at_k(model.embed_np(pair[0], d.inputs[pair[0]]), model.embed_np(pair[1], d.inputs[pair[1]]))
    return float("nan")


def fit(cfg: ConfigSpec, train: FitData, val: FitData, *, selection: str, n_classes: int = 0,
        pred_source: str | None = None, decode: tuple[str, ...] = (), device: str = "cpu",
        seed: int = 0, progress=None) -> FitResult:
    torch.manual_seed(seed)
    t = cfg.training
    in_dims = {m: train.inputs[m].shape[1] for m in [mp.modality for mp in cfg.modalities]}
    needs_decoder = any(x.name in ("reconstruction", "masked_denoising") for x in cfg.losses)
    model = JointModel(cfg, in_dims, n_classes if any(x.name in ("supervised_ce", "supcon") for x in cfg.losses)
                       else 0,
                       train.target.shape[1] if train.target is not None and any(
                           x.name == "cross_modal_regression" for x in cfg.losses) else 0,
                       decode if needs_decoder else ()).to(device)
    decay = [p for n, p in model.named_parameters() if p.requires_grad and n != "logit_scale"]
    groups = [{"params": decay, "weight_decay": t.weight_decay}]
    if model.logit_scale.requires_grad:
        groups.append({"params": [model.logit_scale], "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, lr=t.lr)
    tr, va = _to(train, device), _to(val, device)
    n, bs = train.n, min(t.batch_size, train.n)
    drop_last = n >= 2 * bs
    gen = torch.Generator().manual_seed(seed)
    hist = {"train_loss": [], "val_loss": [], "val_score": [], "temperature": [], "seconds": [],
            "terms": {x.name: [] for x in cfg.losses}}
    best, best_epoch, best_state, stale = -np.inf, 0, None, 0
    start, why = time.perf_counter(), "epoch cap"
    for epoch in range(1, t.max_epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        stops = range(0, n - bs + 1 if drop_last else n, bs)
        sums: dict[str, float] = {}
        total, steps = 0.0, 0
        for s in stops:
            batch = _slice(tr, perm[s:s + bs].to(device))
            terms = compute_losses(model, cfg, batch, pred_source, gen)
            loss = _total(cfg, terms)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            steps += 1
            for k, v in terms.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
        model.eval()
        with torch.no_grad():
            vt = _total(cfg, compute_losses(model, cfg, _slice(va, torch.arange(min(val.n, VAL_CAP),
                                                                                 device=device)), pred_source))
        sc = score(model, val, selection, pred_source)
        hist["train_loss"].append(round(total / max(1, steps), 5))
        hist["val_loss"].append(round(float(vt), 5))
        hist["val_score"].append(round(sc, 5))
        hist["temperature"].append(round(model.temperature, 5))
        hist["seconds"].append(round(time.perf_counter() - start, 2))
        for k in hist["terms"]:
            hist["terms"][k].append(round(sums.get(k, float("nan")) / max(1, steps), 5))
        if sc > best + 1e-6:
            best, best_epoch, stale = sc, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
        if progress and (epoch == 1 or epoch % 10 == 0):
            progress(f"    {cfg.name} epoch {epoch}: train loss {hist['train_loss'][-1]:.4f}, "
                     f"val {selection} {sc:.4f}, temperature {model.temperature:.4f}")
        if stale >= t.patience:
            why = f"no validation improvement for {t.patience} epochs"
            break
        if time.perf_counter() - start > t.train_seconds:
            why = f"time cap ({t.train_seconds:.0f}s)"
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return FitResult(model, selection, best_epoch, float(best), len(hist["train_loss"]),
                     time.perf_counter() - start, why, n_parameters(model), hist)
