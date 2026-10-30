"""Typed records passed between the agent's steps.

Every object here is JSON-serialisable, because each one is shown to the LLM,
written to the run directory and quoted in the decision log. Profile fields are
measurements, not judgements: the profile says "between-group share of colour
variance 0.41", never "strong batch effect". Turning numbers into decisions is
the planner's job.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Data profile
# ---------------------------------------------------------------------------


class ImageProfile(BaseModel):
    source: str                                   # files | spot_patches
    n_sampled: int
    width_median: float
    height_median: float
    width_range: list[float]
    spot_diameter_px_median: float | None = None  # spot patches: the spot's footprint
    grayscale_fraction: float                     # images whose colour channels are (near) equal
    mean_rgb: list[float]
    saturation_mean: float
    light_background_fraction: float              # near-white pixels (slide background)
    dark_background_fraction: float               # near-black pixels (radiograph surround)
    stain_hue_fraction: float                     # tissue pixels with pink/purple hue (H&E-like)
    sharpness_median: float                       # variance of the Laplacian, 128-px greyscale
    near_duplicate_fraction: float | None = None  # perceptual-hash near-twins among sampled images
    colour_between_group_share: float | None = None   # ICC of per-image mean colour by group
    domain_hint: str                              # stained_tissue | greyscale_imaging | colour_other
    domain_evidence: str


class TextProfile(BaseModel):
    n_sampled: int
    words_median: float
    words_p95: float
    chars_median: float
    vocabulary_size: int
    duplicate_fraction: float                     # exact duplicates
    near_duplicate_fraction: float                # duplicates after lowercasing and stripping digits
    fraction_over_60_words: float                 # roughly beyond a 77-token text encoder window


class MatrixProfile(BaseModel):
    n_features: int
    feature_kind: str                             # genes | peaks | features
    sparsity: float                               # fraction of zero entries
    integer_valued: bool
    nonnegative: bool
    library_size_median: float
    library_size_iqr: list[float]
    features_per_sample_median: float
    n_features_detected_5pct: int                 # nonzero in >= 5% of samples
    overdispersed_fraction: float | None = None   # var > mean, among detected features (counts)
    analysis_space: str                           # how the measurements below were taken
    pca_components_50: int | None = None          # components for 50% / 90% of variance
    pca_components_90: int | None = None          # (None: not reached within the top 50)
    pca_top50_variance_fraction: float | None = None
    near_twin_fraction: float                     # NN cosine similarity > 0.95 in PCA space
    group_variance_share: float | None = None     # share of top-PC variance explained by group
    label_variance_share: float | None = None     # ... by label (biology), for comparison
    group_silhouette: float | None = None         # silhouette of groups in PCA space
    morans_i_median: float | None = None          # spatial autocorrelation of variable features
    morans_i_fraction_above_0_3: float | None = None


class TabularProfile(BaseModel):
    n_features: int
    missing_fraction: float


class SpatialProfile(BaseModel):
    n_libraries: int
    samples_per_library: list[float]              # min, median, max
    nn_distance_px_median: float
    spot_diameter_px_median: float | None = None
    spacing_over_diameter: float | None = None


class ModalityProfile(BaseModel):
    name: str
    kind: Literal["image", "text", "matrix", "tabular", "spatial"]
    n_present: int
    missing_fraction: float
    source: str
    image: ImageProfile | None = None
    text: TextProfile | None = None
    matrix: MatrixProfile | None = None
    tabular: TabularProfile | None = None
    spatial: SpatialProfile | None = None


class RoleRecord(BaseModel):
    column: str
    role: str
    evidence: str


class LabelProfile(BaseModel):
    column: str
    n_classes: int
    class_counts: dict[str, int]
    imbalance_ratio: float                        # largest / smallest class
    minority_class: str
    n_labelled: int
    labelled_fraction: float
    labelled_groups: int | None = None            # CV-unit levels with any labelled sample
    n_groups: int | None = None


class GroupProfile(BaseModel):
    column: str
    role: str                                     # group | library
    n_levels: int
    size_min: int
    size_median: float
    size_max: int
    nested_in: str | None = None                  # a coarser grouping it partitions into
    split_overlap: dict[str, float] | None = None  # e.g. test_in_train: share of test samples
                                                   # whose level also occurs in train


class SplitProfile(BaseModel):
    column: str
    counts: dict[str, int]                        # normalised names: train / val / test


class PairingProfile(BaseModel):
    present: dict[str, int]                       # modality -> samples where present
    pairs: dict[str, int]                         # "a|b" -> samples where both present
    n_all_present: int


class DataProfile(BaseModel):
    name: str
    path: str
    format: str
    n_samples: int
    modalities: list[ModalityProfile]
    pairing: PairingProfile
    roles: list[RoleRecord]
    labels: list[LabelProfile] = Field(default_factory=list)
    multilabel_columns: list[str] = Field(default_factory=list)
    groups: list[GroupProfile] = Field(default_factory=list)
    cv_group: str | None = None                   # unit of independent replication
    split: SplitProfile | None = None
    notes: list[str] = Field(default_factory=list)

    def modality(self, name: str) -> ModalityProfile:
        return next(m for m in self.modalities if m.name == name)

    def brief(self) -> str:
        """One line for logs."""
        mods = ", ".join(f"{m.name} ({m.kind}, {m.n_present:,})" for m in self.modalities)
        parts = [f"{self.format}", f"n={self.n_samples:,}", mods]
        if self.labels:
            lab = self.labels[0]
            parts.append(f"label '{lab.column}' {lab.n_classes} classes on "
                         f"{lab.labelled_fraction:.0%} of samples")
        if self.cv_group:
            g = next(g for g in self.groups if g.column == self.cv_group)
            parts.append(f"{g.n_levels} '{g.column}' groups")
        if self.split:
            parts.append("predefined split " + "/".join(f"{v:,}" for v in self.split.counts.values()))
        return "; ".join(parts)
