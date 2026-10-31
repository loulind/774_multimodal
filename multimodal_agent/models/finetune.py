"""Partial fine-tuning of pre-aligned towers: only the last blocks move.

With a few thousand to tens of thousands of pairs, updating all ~86M parameters
of each tower would overfit and is slow; freezing everything except the last
N transformer blocks (and the layers after them: final norms, projections)
lets the representation adapt to the domain while most pretrained weights stay
fixed. Backbone weights get a learning rate about 100x smaller than the new
heads, so the pretrained alignment is adjusted, not overwritten.

The loop mirrors models/train.py: the same losses, validation selection metric
on held-out samples after every epoch, the best epoch kept, early stopping, and a
wall-clock cap. Images are augmented only here (see preprocess/image.py).
"""

from __future__ import annotations

import copy
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from ..encoders.pretrained import Encoder
from ..evaluate.metrics import recall_at_k
from ..preprocess.image import ImageTransform
from ..preprocess.text import clean
from ..schema import ConfigSpec, ModalityPlan
from .heads import n_parameters
from .joint import JointModel
from .losses import normalize
from .train import FitResult, _total, compute_losses


def unfreeze_last(enc: Encoder, n_blocks: int) -> list[torch.nn.Parameter]:
    """Freeze the tower, then unfreeze its last ``n_blocks`` blocks and everything after them."""
    params = list(enc.parameters())
    for p in params:
        p.requires_grad = False
    blocks = enc.blocks()
    if not blocks or n_blocks <= 0:
        return []
    first = {id(p) for p in blocks[-min(n_blocks, len(blocks))].parameters()}
    start = next(i for i, p in enumerate(params) if id(p) in first)
    for p in params[start:]:
        p.requires_grad = True
    return params[start:]


class TowerInput:
    """Turns dataset positions into a batch for one tower."""

    def __init__(self, ds, mplan: ModalityPlan, enc: Encoder, normalizer=None, groups=None):
        self.enc, self.kind = enc, ds.modalities[mplan.modality].kind
        mod = ds.modalities[mplan.modality]
        if self.kind == "image":
            src = mod.images
            if mplan.patch_context is not None and hasattr(src, "with_context"):
                src = src.with_context(mplan.patch_context)
            square = getattr(src, "kind", "") == "spot_patches"
            norm = normalizer if mplan.colour_normalisation == "reinhard" else None
            self.source, self.groups = src, groups
            self.tf_eval = ImageTransform(enc.input_size, enc.mean, enc.std, True, norm, False, square)
            self.tf_train = ImageTransform(enc.input_size, enc.mean, enc.std, True, norm, mplan.augment, square)
            self.pool = ThreadPoolExecutor(max_workers=8)
        else:
            self.texts = mod.data
            self.max_tokens = min(mplan.text_max_tokens or 77, enc.info.max_tokens or 512)
            self.cleaning = mplan.text_cleaning

    def batch(self, idx: np.ndarray, train: bool, rng: np.random.Generator):
        if self.kind == "image":
            tf = self.tf_train if train else self.tf_eval
            seeds = rng.integers(0, 2 ** 31, len(idx))

            def one(k):
                i = int(idx[k])
                g = None if self.groups is None else str(self.groups[i])
                return tf(self.source.load(i), g, np.random.default_rng(int(seeds[k])))

            return torch.stack(list(self.pool.map(one, range(len(idx))))).to(self.enc.device)
        return self.enc.tokenize([clean(self.texts[int(i)], self.cleaning) for i in idx], self.max_tokens)


class FineTuned:
    """Fine-tuned towers + heads, embedding samples by dataset position."""

    def __init__(self, encoders: dict[str, Encoder], inputs: dict[str, TowerInput], model: JointModel,
                 batch_size: int):
        self.encoders, self.inputs, self.model, self.batch_size = encoders, inputs, model, batch_size

    @torch.no_grad()
    def embed_indices(self, name: str, idx: np.ndarray) -> np.ndarray:
        self.model.eval()
        self.encoders[name].train(False)
        rng = np.random.default_rng(0)
        out = []
        for s in range(0, len(idx), self.batch_size):
            x = self.encoders[name].encode(self.inputs[name].batch(idx[s:s + self.batch_size], False, rng))
            out.append(normalize(self.model.embed(name, x.float())).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.model.heads[name].net[-1].out_features))


def finetune(cfg: ConfigSpec, ds, encoders: dict[str, Encoder], train_idx: np.ndarray, val_idx: np.ndarray,
             labels: np.ndarray | None, n_classes: int, device: str, seed: int, normalizer=None,
             norm_groups=None, progress=None) -> tuple[FineTuned, FitResult]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t = cfg.training
    pair = [mp.modality for mp in cfg.modalities]
    inputs = {mp.modality: TowerInput(ds, mp, encoders[mp.modality], normalizer, norm_groups)
              for mp in cfg.modalities}
    backbone = []
    for mp in cfg.modalities:
        backbone += unfreeze_last(encoders[mp.modality], mp.trainable_blocks)
    model = JointModel(cfg, {m: encoders[m].dim for m in pair},
                       n_classes if any(x.name == "supervised_ce" for x in cfg.losses) else 0).to(device)
    heads = [p for n, p in model.named_parameters() if p.requires_grad and n != "logit_scale"]
    groups = [{"params": backbone, "lr": t.finetune_lr, "weight_decay": t.weight_decay},
              {"params": heads, "lr": t.lr, "weight_decay": t.weight_decay}]
    if model.logit_scale.requires_grad:
        groups.append({"params": [model.logit_scale], "lr": t.lr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    y_all = None if labels is None else torch.as_tensor(labels, dtype=torch.long)
    ft = FineTuned(encoders, inputs, model, max(t.batch_size, 64))
    hist = {"train_loss": [], "val_loss": [], "val_score": [], "temperature": [], "seconds": [],
            "terms": {x.name: [] for x in cfg.losses}}
    best, best_epoch, best_state, stale = -np.inf, 0, None, 0
    start, why, steps_done = time.perf_counter(), "epoch cap", 0
    val_cap = np.sort(rng.permutation(val_idx)[:2000])
    # epoch 0: the pretrained towers with the starting heads (identity when residual) are a candidate
    init = recall_at_k(ft.embed_indices(pair[0], val_cap), ft.embed_indices(pair[1], val_cap))
    hist["initial_val_score"] = round(init, 5)
    best = init
    best_state = {"model": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}),
                  "backbone": [p.detach().cpu().clone() for p in backbone]}
    if progress:
        progress(f"    {cfg.name} before training: val recall@10 {init:.4f}")
    for epoch in range(1, t.max_epochs + 1):
        model.train()
        for m in pair:
            # Backbones stay in inference mode (dropout off) while their last blocks
            # train: the Apple-GPU attention kernel has no dropout, and frozen blocks
            # then behave exactly as at inference. Regularisation comes from the small
            # backbone learning rate, weight decay, augmentation and early stopping.
            encoders[m].train(False)
        perm = rng.permutation(train_idx)
        total, steps, sums = 0.0, 0, {}
        for s in range(0, len(perm) - t.batch_size + 1, t.batch_size):
            b = perm[s:s + t.batch_size]
            feats = {m: encoders[m].encode(inputs[m].batch(b, True, rng)).float() for m in pair}
            batch = {"inputs": feats, "target": None, "soft": None,
                     "labels": None if y_all is None else y_all[b].to(device)}
            terms = compute_losses(model, cfg, batch, None)
            loss = _total(cfg, terms)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            opt.step()
            total += float(loss.detach())
            steps += 1
            steps_done += 1
            for k, v in terms.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
            if progress and steps_done % 50 == 0:
                rate = steps_done * t.batch_size / (time.perf_counter() - start)
                progress(f"    {cfg.name} epoch {epoch} step {steps}: loss {total / steps:.4f} ({rate:.0f} pairs/s)")
            if time.perf_counter() - start > t.train_seconds:
                break
        za = ft.embed_indices(pair[0], val_cap)
        zb = ft.embed_indices(pair[1], val_cap)
        sc = recall_at_k(za, zb)
        hist["train_loss"].append(round(total / max(1, steps), 5))
        hist["val_loss"].append(None)
        hist["val_score"].append(round(sc, 5))
        hist["temperature"].append(round(model.temperature, 5))
        hist["seconds"].append(round(time.perf_counter() - start, 2))
        for k in hist["terms"]:
            hist["terms"][k].append(round(sums.get(k, float("nan")) / max(1, steps), 5))
        if progress:
            progress(f"    {cfg.name} epoch {epoch}: train loss {hist['train_loss'][-1]:.4f}, "
                     f"val recall@10 {sc:.4f}")
        if sc > best + 1e-6:
            best, best_epoch, stale = sc, epoch, 0
            best_state = {"model": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}),
                          "backbone": [p.detach().cpu().clone() for p in backbone]}
        else:
            stale += 1
        if stale >= t.patience:
            why = f"no validation improvement for {t.patience} epochs"
            break
        if time.perf_counter() - start > t.train_seconds:
            why = f"time cap ({t.train_seconds:.0f}s)"
            break
    if best_state is not None:
        model.load_state_dict(best_state["model"])
        with torch.no_grad():
            for p, v in zip(backbone, best_state["backbone"]):
                p.copy_(v.to(p.device))
    model.eval()
    for m in pair:
        encoders[m].train(False)
    n_train_params = n_parameters(model) + sum(p.numel() for p in backbone)
    result = FitResult(model, "retrieval", best_epoch, float(best), len(hist["train_loss"]),
                       time.perf_counter() - start, why, n_train_params, hist)
    return ft, result
