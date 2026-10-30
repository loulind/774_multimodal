"""Image preprocessing: stain colour normalisation, channels, resizing, augmentation.

Colour normalisation (Reinhard et al. 2001) works in CIE L*a*b* space, where
the three channels are roughly decorrelated: each image's channel means and
spreads are matched to a reference. Statistics are taken per *group of images
that share a staining/scanning batch* -- a slide, for spot patches -- not per
image: a single patch's colour also reflects what tissue it shows (fat is pale,
tumour dark), and normalising every patch to the same colour would erase that
biology. Matching slide-level statistics removes slide-to-slide stain shifts
and keeps within-slide contrasts. The reference is the average over groups, a
fixed target in the role of a template slide; it is built from images only,
never from labels or the other modality.
"""

from __future__ import annotations

import numpy as np
import torch

# --- sRGB <-> CIE L*a*b* (D65), vectorised ---------------------------------------------
_M = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]])
_M_INV = np.linalg.inv(_M)
_WHITE = np.array([0.95047, 1.0, 1.08883])


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """uint8 or [0, 1] RGB array (..., 3) -> L*a*b*."""
    c = rgb.astype(np.float64)
    if c.max() > 1.0:
        c = c / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ _M.T / _WHITE
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], -1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """L*a*b* -> uint8 RGB."""
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], -1)
    xyz = np.where(f ** 3 > 216 / 24389, f ** 3, (116 * f - 16) / (24389 / 27)) * _WHITE
    c = xyz @ _M_INV.T
    c = np.where(c > 0.0031308, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055, 12.92 * c)
    return (np.clip(c, 0, 1) * 255 + 0.5).astype(np.uint8)


def _tissue_stats(images: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Mean and s.d. of L*a*b* over non-background pixels of a set of RGB arrays."""
    labs = [rgb_to_lab(im).reshape(-1, 3) for im in images]
    lab = np.concatenate(labs)
    tissue = lab[:, 0] < 85                      # bright, washed-out pixels are slide background
    if tissue.mean() < 0.05:
        tissue = np.ones(len(lab), bool)
    sel = lab[tissue]
    return sel.mean(0), sel.std(0) + 1e-6


class ReinhardNormalizer:
    """Per-group Reinhard colour normalisation towards the average group."""

    def __init__(self):
        self.stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.target: tuple[np.ndarray, np.ndarray] | None = None

    def fit(self, samples_by_group: dict[str, list[np.ndarray]],
            target_groups: list[str] | None = None) -> ReinhardNormalizer:
        """Each group's own statistics; the target averages ``target_groups``
        (training groups), so held-out images never shape the reference."""
        self.stats = {g: _tissue_stats(ims) for g, ims in samples_by_group.items() if ims}
        ref = [g for g in (target_groups or self.stats) if g in self.stats] or list(self.stats)
        self.target = (np.mean([self.stats[g][0] for g in ref], 0),
                       np.mean([self.stats[g][1] for g in ref], 0))
        return self

    def __call__(self, rgb: np.ndarray, group: str) -> np.ndarray:
        if group not in self.stats or self.target is None:
            return rgb
        mu, sd = self.stats[group]
        tmu, tsd = self.target
        lab = rgb_to_lab(rgb)
        return lab_to_rgb((lab - mu) / sd * tsd + tmu)

    def describe(self) -> dict:
        return {"n_groups": len(self.stats),
                "target_lab_mean": [round(float(v), 2) for v in self.target[0]] if self.target else None,
                "target_lab_std": [round(float(v), 2) for v in self.target[1]] if self.target else None}


class ImageTransform:
    """PIL image -> normalised tensor for one encoder."""

    def __init__(self, input_size: int, mean, std, greyscale_to_rgb: bool = True,
                 normalizer: ReinhardNormalizer | None = None, augment: bool = False,
                 square_input: bool = False):
        self.size, self.mean, self.std = input_size, np.asarray(mean, np.float32), np.asarray(std, np.float32)
        self.greyscale_to_rgb, self.normalizer, self.augment = greyscale_to_rgb, normalizer, augment
        self.square_input = square_input

    def __call__(self, im, group: str | None = None, rng: np.random.Generator | None = None) -> torch.Tensor:
        """Greyscale is copied to three channels (pretrained encoders expect RGB).

        Augmentation (only when backbones are fine-tuned): square tissue patches
        have no natural orientation, so they get flips and 90-degree rotations;
        other images (e.g. radiographs with captions saying "left"/"right") only
        a mild random crop, since a flip could contradict the paired text.
        """
        from PIL import Image

        im = im.convert("RGB")
        if self.normalizer is not None and group is not None:
            im = Image.fromarray(self.normalizer(np.asarray(im), group))
        rng = rng or np.random.default_rng()
        w, h = im.size
        if self.augment and not self.square_input:
            f = np.sqrt(rng.uniform(0.85, 1.0))
            cw, ch = max(1, int(w * f)), max(1, int(h * f))
            left, top = int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1))
            im = im.crop((left, top, left + cw, top + ch))
            w, h = im.size
        if self.square_input or w == h:
            im = im.resize((self.size, self.size), Image.Resampling.BICUBIC)
        else:                                    # shorter side to size, then centre crop
            s = self.size / min(w, h)
            im = im.resize((max(self.size, round(w * s)), max(self.size, round(h * s))),
                           Image.Resampling.BICUBIC)
            w, h = im.size
            left, top = (w - self.size) // 2, (h - self.size) // 2
            im = im.crop((left, top, left + self.size, top + self.size))
        arr = np.asarray(im, np.float32) / 255.0
        if self.augment and self.square_input:
            if rng.random() < 0.5:
                arr = arr[:, ::-1]
            arr = np.rot90(arr, int(rng.integers(4)))
        arr = (arr - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
