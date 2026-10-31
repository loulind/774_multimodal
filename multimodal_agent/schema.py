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

    def label(self, column: str | None) -> LabelProfile | None:
        return next((lab for lab in self.labels if lab.column == column), None)

    def group(self, column: str | None) -> GroupProfile | None:
        return next((g for g in self.groups if g.column == column), None)


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

TaskType = Literal["retrieval", "cross_modal_prediction", "classification", "clustering",
                   "joint_embedding"]
# Tasks with a metric of their own; "joint_embedding" is a means, not an end.
MEASURABLE_TASKS = ("retrieval", "cross_modal_prediction", "classification", "clustering")


class PredictionSpec(BaseModel):
    source: str                                   # modality predicted from
    target: str                                   # modality predicted


class TaskSpec(BaseModel):
    """The user's objective, made explicit. Produced by objective.py (+ LLM)."""
    raw_objective: str
    tasks: list[TaskType]                         # in order of first mention
    primary_task: TaskType                        # first measurable task
    pair: list[str]                               # the two modalities to integrate
    prediction: PredictionSpec | None = None
    label_column: str | None = None               # for classification / clustering / probes
    label_evidence: str = ""
    modality_mentions: dict[str, list[str]] = Field(default_factory=dict)
    evidence: dict[str, list[str]] = Field(default_factory=dict)   # task -> matched phrases
    source: Literal["heuristic", "llm"] = "heuristic"


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

class SplitSpec(BaseModel):
    strategy: Literal["predefined", "grouped_kfold", "random"]
    group_column: str | None = None
    n_folds: int = 1                              # grouped_kfold: folds over groups (1 = one held-out set)
    val_fraction: float = 0.1                     # random / carved from train when no val split
    test_fraction: float = 0.2                    # random, and grouped with one fold
    stratify_column: str | None = None


class ModalityPlan(BaseModel):
    modality: str                                 # name in the dataset
    encoder: str                                  # encoders.registry name
    regime: Literal["frozen", "partial_finetune", "from_scratch"] = "frozen"
    trainable_blocks: int = 0                     # partial_finetune: last N transformer blocks
    # preprocessing; fields that do not apply to the modality's kind are ignored
    patch_context: float | None = None            # spot patches: patch side / spot diameter
    colour_normalisation: Literal["none", "reinhard"] = "none"
    greyscale_to_rgb: bool = False
    augment: bool = False                         # flips/crops; only when backbones train
    text_max_tokens: int | None = None
    text_cleaning: Literal["none", "basic"] = "basic"
    n_variable_features: int | None = None        # matrix: top variable features kept
    matrix_input: Literal["log_normalised", "pca", "standardised"] | None = None
    pca_components: int | None = None


LossName = Literal["infonce", "soft_infonce", "supervised_ce", "supcon", "cross_modal_regression",
                   "reconstruction", "ssl_simclr", "masked_denoising"]


class LossTerm(BaseModel):
    name: LossName
    weight: float = 1.0


class HeadSpec(BaseModel):
    embed_dim: int = 256                          # dimension of the joint space
    hidden_dim: int = 512
    n_layers: int = 2
    dropout: float = 0.1
    residual: bool = False                        # identity at initialisation (pre-aligned encoders)


class TrainingSpec(BaseModel):
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 200
    patience: int = 15
    temperature_init: float = 0.07
    learnable_temperature: bool = True
    soft_target_temperature: float = 0.1          # soft_infonce: sharpness of similarity targets
    finetune_lr: float = 1e-5                     # backbone blocks under partial fine-tuning
    train_seconds: float = 120.0                  # wall-clock cap per fit


class ConfigSpec(BaseModel):
    """One trained configuration: the primary, or a comparison that tests a decision."""
    name: str
    role: Literal["primary", "comparison"] = "primary"
    strategy: Literal["contrastive_alignment", "supervised_fusion", "self_supervised",
                      "linear_alignment"]
    fusion: Literal["joint", "early", "late"] = "joint"
    modalities: list[ModalityPlan]
    losses: list[LossTerm]
    head: HeadSpec = Field(default_factory=HeadSpec)
    training: TrainingSpec = Field(default_factory=TrainingSpec)
    rationale: str = ""


PrimaryMetric = Literal["retrieval_mean_recall_at_10", "prediction_mean_pearson_top50",
                        "probe_macro_f1", "clustering_ari"]


class EvalSpec(BaseModel):
    primary_metric: PrimaryMetric
    metrics: list[str]
    baselines: list[str]
    label_column: str | None = None
    group_column: str | None = None               # batch / group silhouette
    retrieval_ks: list[int] = Field(default_factory=lambda: [1, 5, 10])
    prediction_genes: int = 50                    # most variable target features scored


DownstreamName = Literal["retrieval_examples", "cross_modal_prediction", "classification",
                         "clustering", "spatial_maps"]


class Plan(BaseModel):
    task: TaskSpec
    split: SplitSpec
    configs: list[ConfigSpec]
    evaluation: EvalSpec
    downstream: list[DownstreamName] = Field(default_factory=list)
    rationale: str = ""
    source: Literal["heuristic", "llm", "llm_repaired", "heuristic_fallback"] = "heuristic"

    @property
    def primary(self) -> ConfigSpec:
        return next(c for c in self.configs if c.role == "primary")


# ---------------------------------------------------------------------------
# Decision log
# ---------------------------------------------------------------------------

class Alternative(BaseModel):
    option: str
    reason: str


class Decision(BaseModel):
    step: str                     # profile | objective | split | strategy | encoder | ...
    decision: str                 # what was being decided
    chosen: str
    rationale: str
    alternatives_rejected: list[Alternative] = Field(default_factory=list)
    evidence: dict[str, float | int | str | bool | None | list | dict] = Field(default_factory=dict)
    source: Literal["heuristic", "llm", "validator", "critic", "system"] = "heuristic"


# ---------------------------------------------------------------------------
# Encoder screening (filled in by encoders/screening.py)
# ---------------------------------------------------------------------------

class ScreenScore(BaseModel):
    encoder: str
    score: float                          # higher is better
    metric: str
    details: dict[str, float | int | str | None] = Field(default_factory=dict)


class ScreeningResult(BaseModel):
    modality: str
    ranked: list[ScreenScore]             # best first
    note: str = ""


# ---------------------------------------------------------------------------
# LLM replies (shape enforced by the CLI's structured output, content by the validator)
# ---------------------------------------------------------------------------

class LLMDecision(BaseModel):
    step: str
    decision: str
    chosen: str
    rationale: str
    alternatives_rejected: list[Alternative] = Field(default_factory=list)


class PlannerReply(BaseModel):
    plan: Plan
    decisions: list[LLMDecision] = Field(default_factory=list)


class Change(BaseModel):
    path: str                     # e.g. "training.lr", "head.dropout", "loss.supervised_ce.weight"
    value: float | int | str | bool


class CriticReply(BaseModel):
    assessment: str
    action: Literal["none", "modify_primary", "switch_primary"]
    target_config: str | None = None
    changes: list[Change] = Field(default_factory=list)
    rationale: str
    alternatives_rejected: list[Alternative] = Field(default_factory=list)
    expected_effect: str = ""


class WriterReply(BaseModel):
    executive_summary: str
    data_and_objective: str
    decisions_explained: str
    results_interpretation: str
    integration_benefit: str
    critic_and_revision: str
    limitations: str


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

CriticAction = Literal["none", "modify_primary", "switch_primary"]


class Trigger(BaseModel):
    code: str                     # comparison_better, worse_than_<baseline>, overfitting, ...
    message: str
    severity: Literal["warning", "info"] = "warning"
    config: str | None = None
    suggested_action: CriticAction = "none"
    suggested_target: str | None = None
    suggested_changes: list[Change] = Field(default_factory=list)


class Critique(BaseModel):
    assessment: str
    action: CriticAction = "none"
    target_config: str | None = None          # switch_primary: the comparison to promote
    changes: list[Change] = Field(default_factory=list)   # modify_primary: edits to the primary
    rationale: str = ""
    alternatives_rejected: list[Alternative] = Field(default_factory=list)
    expected_effect: str = ""
    triggers: list[Trigger] = Field(default_factory=list)
    source: Literal["rules", "llm"] = "rules"
