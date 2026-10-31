"""Load registry encoders and turn images or texts into feature vectors.

Every pretrained model, whatever library it comes from, is wrapped in the same
small interface:

    enc.encode(batch)      differentiable forward pass of one tower
    enc.blocks()           its transformer blocks, last ones unfrozen for fine-tuning
    enc.mean / enc.std / enc.input_size     image normalisation (image towers)
    enc.tokenize(texts, max_tokens)          text towers

Features from frozen encoders are computed once and cached on disk
(``runs/cache/features``), keyed by encoder, preprocessing and the exact sample
ids, so screening, comparisons and the critic's revision never recompute them.

Supported sources (see registry.py):
    hf_vit           transformers ViT-type backbone; CLS token of the last layer
    hf_clip_image    transformers CLIP vision tower + projection
    hf_clip_text     transformers CLIP text tower + projection
    open_clip_image  OpenCLIP (incl. Hugging Face hub models) image tower
    open_clip_text   OpenCLIP text tower
    timm             timm CNN, global-pooled features
    sentence / hf_text  transformers text encoder, attention-masked mean pooling
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .registry import REGISTRY, EncoderInfo

_BACKBONES: dict[str, object] = {}       # shared between the towers of one dual encoder


def _backbone(key: str, build):
    if key not in _BACKBONES:
        _BACKBONES[key] = build()
    return _BACKBONES[key]


def release_all() -> None:
    """Free every loaded backbone (between screening candidates)."""
    _BACKBONES.clear()
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def find_blocks(module) -> list:
    """The transformer blocks of a backbone, whichever library laid it out."""
    for path in ("layers", "encoder.layer", "encoder.layers", "blocks", "trunk.blocks",
                 "transformer.resblocks", "transformer.encoder.layer"):
        obj = module
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__len__") and len(obj) > 0:
            return list(obj)
    return []


class Encoder:
    """One tower of a pretrained model."""

    kind = "image"
    input_size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(self, info: EncoderInfo, device: str):
        self.info, self.device = info, device
        self.loader, self.identifier = info.source.split(":", 1)

    @property
    def dim(self) -> int:
        return self.info.dim

    def modules(self) -> list[torch.nn.Module]:
        raise NotImplementedError

    def blocks(self) -> list[torch.nn.Module]:
        return []

    def encode(self, batch) -> torch.Tensor:
        raise NotImplementedError

    def tokenize(self, texts: list[str], max_tokens: int):
        raise NotImplementedError

    def train(self, mode: bool = True) -> None:
        for m in self.modules():
            m.train(mode)

    def parameters(self):
        for m in self.modules():
            yield from m.parameters()


class HFViT(Encoder):
    def __init__(self, info, device):
        super().__init__(info, device)
        from transformers import AutoImageProcessor, AutoModel

        self.model = _backbone(info.source, lambda: AutoModel.from_pretrained(self.identifier)).to(device).eval()
        proc = AutoImageProcessor.from_pretrained(self.identifier)
        self.mean, self.std = tuple(proc.image_mean), tuple(proc.image_std)

    def modules(self):
        return [self.model]

    def blocks(self):
        return find_blocks(self.model)

    def encode(self, x):
        return self.model(pixel_values=x).last_hidden_state[:, 0]


def _clip(identifier: str):
    from transformers import CLIPModel, CLIPProcessor

    return CLIPModel.from_pretrained(identifier), CLIPProcessor.from_pretrained(identifier)


class HFClipImage(Encoder):
    def __init__(self, info, device):
        super().__init__(info, device)
        self.model, proc = _backbone(f"clip:{self.identifier}", lambda: _clip(self.identifier))
        self.model.to(device).eval()
        self.mean, self.std = tuple(proc.image_processor.image_mean), tuple(proc.image_processor.image_std)

    def modules(self):
        return [self.model.vision_model, self.model.visual_projection]

    def blocks(self):
        return find_blocks(self.model.vision_model)

    def encode(self, x):
        return self.model.visual_projection(self.model.vision_model(pixel_values=x).pooler_output)


class HFClipText(Encoder):
    kind = "text"

    def __init__(self, info, device):
        super().__init__(info, device)
        self.model, proc = _backbone(f"clip:{self.identifier}", lambda: _clip(self.identifier))
        self.model.to(device).eval()
        self.tokenizer = proc.tokenizer

    def modules(self):
        return [self.model.text_model, self.model.text_projection]

    def blocks(self):
        return find_blocks(self.model.text_model)

    def tokenize(self, texts, max_tokens):
        t = self.tokenizer(texts, padding=True, truncation=True, max_length=min(max_tokens, 77),
                           return_tensors="pt")
        return {k: v for k, v in t.items() if k in ("input_ids", "attention_mask")}

    def encode(self, batch):
        out = self.model.text_model(input_ids=batch["input_ids"].to(self.device),
                                    attention_mask=batch["attention_mask"].to(self.device))
        return self.model.text_projection(out.pooler_output)


def _open_clip(identifier: str):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(identifier)
    norm = next(t for t in preprocess.transforms if type(t).__name__ == "Normalize")
    return model, open_clip.get_tokenizer(identifier), (tuple(norm.mean), tuple(norm.std))


class OpenClipImage(Encoder):
    def __init__(self, info, device):
        super().__init__(info, device)
        self.model, _, (self.mean, self.std) = _backbone(f"oc:{self.identifier}",
                                                         lambda: _open_clip(self.identifier))
        self.model.to(device).eval()

    def modules(self):
        return [self.model.visual]

    def blocks(self):
        return find_blocks(self.model.visual)

    def encode(self, x):
        return self.model.encode_image(x)


class OpenClipText(Encoder):
    kind = "text"

    def __init__(self, info, device):
        super().__init__(info, device)
        self.model, self.tokenizer, _ = _backbone(f"oc:{self.identifier}", lambda: _open_clip(self.identifier))
        self.model.to(device).eval()

    def modules(self):
        return [self.model.text]

    def blocks(self):
        return find_blocks(self.model.text)

    def tokenize(self, texts, max_tokens):
        return {"tokens": self.tokenizer(texts, context_length=max_tokens)}

    def encode(self, batch):
        return self.model.encode_text(batch["tokens"].to(self.device))


class Timm(Encoder):
    def __init__(self, info, device):
        super().__init__(info, device)
        import timm

        self.model = _backbone(info.source, lambda: timm.create_model(self.identifier, pretrained=True,
                                                                      num_classes=0)).to(device).eval()
        cfg = timm.data.resolve_model_data_config(self.model)
        self.mean, self.std = tuple(cfg["mean"]), tuple(cfg["std"])

    def modules(self):
        return [self.model]

    def encode(self, x):
        return self.model(x)


class MeanPooledText(Encoder):
    kind = "text"

    def __init__(self, info, device):
        super().__init__(info, device)
        from transformers import AutoModel, AutoTokenizer

        self.model = _backbone(info.source, lambda: AutoModel.from_pretrained(self.identifier)).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.identifier)

    def modules(self):
        return [self.model]

    def blocks(self):
        return find_blocks(self.model)

    def tokenize(self, texts, max_tokens):
        t = self.tokenizer(texts, padding=True, truncation=True, max_length=max_tokens, return_tensors="pt")
        return {k: v for k, v in t.items() if k in ("input_ids", "attention_mask")}

    def encode(self, batch):
        mask = batch["attention_mask"].to(self.device)
        out = self.model(input_ids=batch["input_ids"].to(self.device), attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(out.dtype)
        return (out * m).sum(1) / m.sum(1).clamp_min(1.0)


LOADERS = {"hf_vit": HFViT, "hf_clip_image": HFClipImage, "hf_clip_text": HFClipText,
           "open_clip_image": OpenClipImage, "open_clip_text": OpenClipText, "timm": Timm,
           "sentence": MeanPooledText, "hf_text": MeanPooledText}


def load_encoder(name: str, device: str) -> Encoder:
    info = REGISTRY[name]
    loader = info.source.split(":", 1)[0]
    if loader not in LOADERS:
        raise ValueError(f"{name} is not a pretrained encoder (source {info.source})")
    return LOADERS[loader](info, device)


# ---------------------------------------------------------------------------
# feature extraction with a disk cache
# ---------------------------------------------------------------------------

class FeatureCache:
    def __init__(self, root: str | Path | None):
        self.root = Path(root) if root else None

    @staticmethod
    def key(**parts) -> str:
        return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:20]

    def load(self, key: str) -> np.ndarray | None:
        if self.root is None or not (self.root / f"{key}.npy").exists():
            return None
        return np.load(self.root / f"{key}.npy").astype(np.float32)

    def save(self, key: str, arr: np.ndarray, meta: dict) -> None:
        if self.root is None:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        np.save(self.root / f"{key}.npy", arr.astype(np.float16))
        (self.root / f"{key}.json").write_text(json.dumps(meta, indent=1, default=str))


def as_cached(arr: np.ndarray) -> np.ndarray:
    """Features exactly as a later cache hit returns them (stored as float16), so a first run and
    its reruns compute on identical inputs."""
    return arr.astype(np.float16).astype(np.float32)


def ids_digest(ids) -> str:
    return hashlib.sha1("\n".join(map(str, ids)).encode()).hexdigest()[:16]


@torch.inference_mode()
def image_features(enc: Encoder, source, indices: np.ndarray, transform, groups: np.ndarray | None,
                   batch_size: int, cache: FeatureCache | None = None, key_parts: dict | None = None,
                   progress=None) -> np.ndarray:
    """Encode the images of ``indices`` (positions in the dataset), in that order."""
    indices = np.asarray(indices)
    key = None
    if cache is not None and key_parts is not None:
        key = cache.key(encoder=enc.info.name, kind="image", **key_parts)
        hit = cache.load(key)
        if hit is not None and len(hit) == len(indices):
            return hit
    order = source.order(indices)
    pos = {int(i): k for k, i in enumerate(indices)}
    out = np.zeros((len(indices), enc.dim), np.float32)
    batches = [order[s:s + batch_size] for s in range(0, len(order), batch_size)]
    q: queue.Queue = queue.Queue(maxsize=3)

    def produce():
        # Images are *loaded* by this one thread, in the source's preferred order
        # (spot patches grouped by slide), so each slide is decoded once per pass;
        # only the per-image transforms run in parallel.
        try:
            with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 2)) as pool:
                for b in batches:
                    raw = [source.load(int(i)) for i in b]
                    grp = [None if groups is None else str(groups[int(i)]) for i in b]
                    q.put((b, torch.stack(list(pool.map(transform, raw, grp)))))
        except Exception as err:          # surfaced in the consumer
            q.put(err)
        q.put(None)

    worker = threading.Thread(target=produce, daemon=True)
    worker.start()
    b_idx = 0
    while (item := q.get()) is not None:
        if isinstance(item, Exception):
            raise item
        b, x = item
        out[[pos[int(i)] for i in b]] = enc.encode(x.to(enc.device)).float().cpu().numpy()
        b_idx += 1
        if progress and b_idx % 20 == 0:
            progress(f"    {enc.info.name}: {min(b_idx * batch_size, len(order))}/{len(order)} images")
    worker.join()
    if key is not None:
        cache.save(key, out, {"encoder": enc.info.name, "n": len(indices), **key_parts})
        out = as_cached(out)
    return out


@torch.inference_mode()
def text_features(enc: Encoder, texts: list[str], max_tokens: int, batch_size: int,
                  cache: FeatureCache | None = None, key_parts: dict | None = None) -> np.ndarray:
    key = None
    if cache is not None and key_parts is not None:
        key = cache.key(encoder=enc.info.name, kind="text", max_tokens=max_tokens, **key_parts)
        hit = cache.load(key)
        if hit is not None and len(hit) == len(texts):
            return hit
    out = []
    for s in range(0, len(texts), batch_size):
        out.append(enc.encode(enc.tokenize(texts[s:s + batch_size], max_tokens)).float().cpu().numpy())
    arr = np.concatenate(out) if out else np.zeros((0, enc.dim), np.float32)
    if key is not None:
        cache.save(key, arr, {"encoder": enc.info.name, "n": len(texts), "max_tokens": max_tokens, **key_parts})
        arr = as_cached(arr)
    return arr
