"""The one container every later step reads: samples x modalities.

A dataset is a table of samples (``obs``, indexed by sample id) plus a set of
modalities aligned to its rows. Each modality has a *kind* -- image, text,
matrix, tabular or spatial -- and a presence mask, so unpaired and partially
missing data are represented explicitly rather than dropped. Nothing here knows
what a dataset is called; kinds are inferred from file structure and content.

Images are never decoded up front. An ``ImageSource`` yields one PIL image per
sample on demand, either from a file per sample or by cutting a patch around a
spot's coordinates from a larger source image (spatial transcriptomics), with
the amount of surrounding context left to the planner.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

ModalityKind = Literal["image", "text", "matrix", "tabular", "spatial"]


class ImageSource:
    """One image per sample, loaded lazily. Subclasses implement ``load``."""

    kind = "files"

    def __len__(self) -> int:
        raise NotImplementedError

    def load(self, i: int):
        raise NotImplementedError

    def order(self, indices) -> np.ndarray:
        """An access order that is cheap for this source (identity by default)."""
        return np.asarray(indices)

    def describe(self) -> dict:
        return {"kind": self.kind}


class FileImageSource(ImageSource):
    """A file path per sample (``None`` where the image is missing)."""

    kind = "files"

    def __init__(self, paths: list[Path | None]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def load(self, i: int):
        from PIL import Image

        with Image.open(self.paths[i]) as im:
            im.load()
            return im.copy()

    def describe(self) -> dict:
        exts = pd.Series([p.suffix.lower() for p in self.paths if p is not None]).value_counts()
        return {"kind": self.kind, "extensions": exts.to_dict()}


class SpotPatchSource(ImageSource):
    """Square patches centred on spot coordinates, cut from one image per group.

    ``context`` scales the patch side relative to the spot diameter: 1.0 is the
    spot's own footprint (where its RNA came from); 2.0 adds a ring of
    neighbouring tissue. Regions outside the source image are padded white,
    like slide background. Source images are decoded on first use and the most
    recent few are kept in memory.
    """

    kind = "spot_patches"

    def __init__(self, library: np.ndarray, xy: np.ndarray, diameter: dict[str, float],
                 image_path: dict[str, Path], context: float = 1.0, max_cached: int = 2):
        self.library, self.xy = np.asarray(library).astype(str), np.asarray(xy, float)
        self.diameter, self.image_path = diameter, image_path
        self.context, self.max_cached = context, max_cached
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()          # patches may be cut from several threads

    def __len__(self) -> int:
        return len(self.library)

    def with_context(self, context: float) -> SpotPatchSource:
        return SpotPatchSource(self.library, self.xy, self.diameter, self.image_path,
                               context, self.max_cached)

    def _image(self, lib: str):
        from PIL import Image

        with self._lock:
            if lib not in self._cache:
                Image.MAX_IMAGE_PIXELS = None      # whole-slide images exceed PIL's default
                with Image.open(self.image_path[lib]) as im:
                    self._cache[lib] = im.convert("RGB")
                while len(self._cache) > self.max_cached:
                    self._cache.popitem(last=False)
            self._cache.move_to_end(lib)
            return self._cache[lib]

    def patch_side(self, i: int) -> int:
        return max(4, int(round(self.context * self.diameter[self.library[i]])))

    def load(self, i: int):
        from PIL import Image

        img = self._image(self.library[i])
        side = self.patch_side(i)
        x, y = self.xy[i]
        left, top = int(round(x - side / 2)), int(round(y - side / 2))
        box = (max(left, 0), max(top, 0), min(left + side, img.width), min(top + side, img.height))
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        if box[2] > box[0] and box[3] > box[1]:
            canvas.paste(img.crop(box), (box[0] - left, box[1] - top))
        return canvas

    def order(self, indices) -> np.ndarray:
        """Group by source image so each is decoded once per pass."""
        indices = np.asarray(indices)
        return indices[np.argsort(self.library[indices], kind="stable")]

    def describe(self) -> dict:
        return {"kind": self.kind, "n_source_images": len(self.image_path), "context": self.context,
                "spot_diameter_px": {k: round(v, 1) for k, v in self.diameter.items()}}


class MaterializedSource(ImageSource):
    """A fixed set of images held in memory, optionally colour-normalised once.

    Encoder screening scores several candidates on the same sample; holding the
    sample's images in memory means slides are decoded (and stain-normalised)
    once rather than once per candidate.
    """

    def __init__(self, source: ImageSource, indices, transform_rgb=None, groups=None):
        from PIL import Image

        self.kind = getattr(source, "kind", "files")
        self.images: dict[int, Any] = {}
        for i in source.order(np.asarray(indices)):
            im = source.load(int(i)).convert("RGB")
            if transform_rgb is not None and groups is not None:
                im = Image.fromarray(transform_rgb(np.asarray(im), str(groups[int(i)])))
            self.images[int(i)] = im

    def __len__(self) -> int:
        return len(self.images)

    def load(self, i: int):
        return self.images[int(i)]


@dataclass
class Modality:
    name: str
    kind: ModalityKind
    present: np.ndarray                    # bool, one per sample
    data: Any = None                       # matrix: (n, p) sparse/dense; text: object array;
                                           # tabular: DataFrame; spatial: (n, 2) array
    images: ImageSource | None = None      # image modalities only
    feature_names: list[str] | None = None
    source: str = ""                       # file / column it came from

    @property
    def n_present(self) -> int:
        return int(self.present.sum())


@dataclass
class ColumnRole:
    column: str
    role: str        # id | split | group | library | label | multilabel | text | image_path |
                     # numeric | metadata
    evidence: str


@dataclass
class MultimodalDataset:
    name: str
    path: Path
    format: str                            # spatial_anndata | anndata | table | paired_matrices
    obs: pd.DataFrame                      # per-sample metadata, index = sample id
    modalities: dict[str, Modality]
    roles: list[ColumnRole] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.obs)

    @property
    def ids(self) -> np.ndarray:
        return self.obs.index.to_numpy()

    def columns_with(self, role: str) -> list[str]:
        return [r.column for r in self.roles if r.role == role]

    def role_of(self, column: str) -> str | None:
        return next((r.role for r in self.roles if r.column == column), None)

    def by_kind(self, kind: str) -> list[Modality]:
        return [m for m in self.modalities.values() if m.kind == kind]

    def paired_mask(self, names: list[str]) -> np.ndarray:
        mask = np.ones(self.n, bool)
        for n in names:
            mask &= self.modalities[n].present
        return mask

    def exclude_inputs(self, columns) -> list[str]:
        """Remove outcome columns from the tabular modalities, so that a label the run evaluates is
        never also an input. Returns the columns removed."""
        from ..preprocess.tabular import describe_columns

        removed = []
        for name, m in list(self.modalities.items()):
            drop = [c for c in columns if m.kind == "tabular" and c and c in m.data.columns]
            if not drop:
                continue
            m.data = m.data.drop(columns=drop)
            removed += drop
            if m.data.shape[1] == 0:
                del self.modalities[name]
                continue
            m.feature_names = list(m.data.columns)
            m.present = m.data.notna().any(axis=1).to_numpy()
            m.source = describe_columns(m.data)
        return removed
