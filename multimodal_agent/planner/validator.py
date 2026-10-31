"""Check a plan against the guardrails; repair what can be repaired.

Two layers, as in Projects 1-2. Pydantic rejects anything malformed (wrong
types, unknown loss or metric names). This module then enforces what a type
system cannot:

  * the aligned modalities exist, are paired often enough, and each config
    encodes exactly those modalities;
  * encoders exist in the registry (open weights only), suit the modality, and
    are small enough in quick mode;
  * a regime is allowed for its encoder (fine-tuning needs a GPU and a
    fine-tunable backbone; MLPs train from scratch);
  * losses are usable with the data (no supervised loss without labels, no
    regression without a matrix target, InfoNCE needs two modalities);
  * hyperparameters and preprocessing lie in fixed ranges (clipped, with a note);
  * the split matches the data (a predefined split must exist; grouped
    cross-validation needs >= 3 groups and at most the budget's folds);
  * compute: per-fit caps and a total-run cap that reserves room for the final
    refit and one critic revision;
  * metrics, baselines and downstream analyses are known and feasible.

Repairs are recorded and logged. A plan whose *primary* configuration is
unusable, or that fails schema validation outright, is not repaired: the caller
falls back to the rule-based planner and the decision log says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import ValidationError

from ..config import Budget
from ..encoders.registry import REGISTRY, candidates, is_dual_pair
from ..schema import ConfigSpec, DataProfile, LossTerm, Plan

KNOWN_METRICS = {"retrieval_recall_at_k", "retrieval_label_match_at_k", "prediction_pearson",
                 "linear_probe", "knn_accuracy", "clustering_ari_nmi", "alignment_uniformity",
                 "modality_gap", "effective_rank", "group_silhouette"}
KNOWN_BASELINES = {"unimodal", "chance", "cca", "ridge", "early_fusion", "late_fusion", "zero_shot"}
RANGES = {
    ("head", "embed_dim"): (32, 1024), ("head", "hidden_dim"): (64, 2048),
    ("head", "n_layers"): (1, 4), ("head", "dropout"): (0.0, 0.5),
    ("training", "batch_size"): (16, 2048), ("training", "lr"): (1e-6, 1e-2),
    ("training", "weight_decay"): (0.0, 0.1), ("training", "patience"): (1, 50),
    ("training", "temperature_init"): (0.01, 1.0), ("training", "soft_target_temperature"): (0.01, 1.0),
    ("training", "finetune_lr"): (1e-7, 1e-4),
}
FT_BATCH_MAX = 256
LOSS_WEIGHT = (0.0, 10.0)
PATCH_CONTEXT = (0.5, 3.0)
MAX_CONFIGS = 3
MIN_PAIRS = 50
MIN_FIT_SECONDS = 10.0


@dataclass
class ValidationResult:
    plan: Plan | None
    errors: list[str] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.plan is not None and not self.errors


def _clip(obj, attr: str, lo, hi, label: str, repairs: list[str]) -> None:
    v = getattr(obj, attr)
    if v is None:
        return
    new = v
    if lo is not None and v < lo:
        new = lo
    if hi is not None and v > hi:
        new = hi
    if new != v:
        setattr(obj, attr, type(v)(new))
        repairs.append(f"{label}: {attr} {v} -> {new} (allowed {lo}..{hi})")


def _check_config(c: ConfigSpec, plan: Plan, profile: DataProfile, budget: Budget, quick: bool,
                  repairs: list[str]) -> str | None:
    """Repair one configuration in place; return a problem string if it is unusable."""
    pair = plan.task.pair
    names = [m.modality for m in c.modalities]
    if sorted(names) != sorted(pair):
        return f"encodes {names}, but the aligned modalities are {pair}"
    for m in c.modalities:
        kind = profile.modality(m.modality).kind
        info = REGISTRY.get(m.encoder)
        if info is None:
            return f"unknown encoder '{m.encoder}'; allowed: {sorted(REGISTRY)}"
        if kind not in info.kinds:
            return f"encoder '{m.encoder}' does not encode {kind} data"
        if quick and not info.trainable and not info.quick:
            img = next((x for x in profile.modalities if x.kind == "image"), None)
            new = candidates(kind, img.image.domain_hint if img and img.image else None, 1, True)[0]
            repairs.append(f"{c.name}: {m.modality} encoder {m.encoder} -> {new} (quick mode)")
            m.encoder, info = new, REGISTRY[new]
        if info.trainable and m.regime != "from_scratch":
            repairs.append(f"{c.name}: {m.modality} regime {m.regime} -> from_scratch (trainable encoder)")
            m.regime = "from_scratch"
        if not info.trainable and m.regime == "from_scratch":
            repairs.append(f"{c.name}: {m.modality} regime from_scratch -> frozen (pretrained encoder)")
            m.regime = "frozen"
        if m.regime == "partial_finetune":
            if not info.finetunable or budget.device == "cpu":
                why = "no GPU" if budget.device == "cpu" else f"{m.encoder} not fine-tunable here"
                repairs.append(f"{c.name}: {m.modality} partial_finetune -> frozen ({why})")
                m.regime, m.trainable_blocks = "frozen", 0
            else:
                _clip(m, "trainable_blocks", 1, min(4, info.n_blocks or 4), f"{c.name}/{m.modality}", repairs)
        elif m.trainable_blocks:
            m.trainable_blocks = 0
        prof = profile.modality(m.modality)
        if kind == "image":
            spot = prof.image is not None and prof.image.source == "spot_patches"
            if spot and m.patch_context is None:
                m.patch_context = 1.0
                repairs.append(f"{c.name}: {m.modality} patch_context set to 1.0")
            if not spot:
                m.patch_context = None
            _clip(m, "patch_context", *PATCH_CONTEXT, f"{c.name}/{m.modality}", repairs)
            if m.augment and m.regime == "frozen":
                m.augment = False
        if kind == "text":
            cap = info.max_tokens or 512
            if m.text_max_tokens is None:
                m.text_max_tokens = min(cap, 77)
            _clip(m, "text_max_tokens", 8, cap, f"{c.name}/{m.modality}", repairs)
        if kind in ("matrix", "tabular"):
            p = prof.matrix.n_features if prof.matrix else (prof.tabular.n_features if prof.tabular else 0)
            if m.matrix_input is None:
                m.matrix_input = "log_normalised" if prof.matrix and prof.matrix.integer_valued \
                    else "standardised"
                repairs.append(f"{c.name}: {m.modality} matrix_input set to {m.matrix_input}")
            if m.n_variable_features is None:
                m.n_variable_features = min(2000, p)
            _clip(m, "n_variable_features", min(50, p), p, f"{c.name}/{m.modality}", repairs)
            if m.matrix_input == "pca":
                if m.pca_components is None:
                    m.pca_components = 50
                _clip(m, "pca_components", 2, min(256, m.n_variable_features),
                      f"{c.name}/{m.modality}", repairs)
            else:
                m.pca_components = None

    # losses
    label = profile.label(plan.evaluation.label_column)
    pred = plan.task.prediction
    kept = []
    for t in c.losses:
        problem = None
        if t.name in ("supervised_ce", "supcon") and (label is None or label.n_labelled == 0):
            problem = "no labels"
        elif t.name == "cross_modal_regression" and (
                pred is None or profile.modality(pred.target).kind not in ("matrix", "tabular")):
            problem = "no matrix/tabular prediction target"
        elif t.name in ("infonce", "soft_infonce") and len(pair) < 2:
            problem = "needs two modalities"
        elif t.name == "masked_denoising" and not any(
                profile.modality(n).kind in ("matrix", "tabular") for n in pair):
            problem = "needs a matrix or tabular modality"
        elif t.name == "ssl_simclr" and not any(profile.modality(n).kind == "image" for n in pair):
            problem = "needs an image modality"
        if problem:
            repairs.append(f"{c.name}: dropped loss {t.name} ({problem})")
            continue
        _clip(t, "weight", *LOSS_WEIGHT, f"{c.name}/{t.name}", repairs)
        kept.append(t)
    c.losses = kept
    if c.strategy == "linear_alignment":
        c.losses = []
    elif c.strategy == "contrastive_alignment" and not any(
            t.name in ("infonce", "soft_infonce") for t in c.losses):
        if len(pair) < 2:
            return "contrastive alignment needs two modalities"
        c.losses.insert(0, LossTerm(name="infonce"))
        repairs.append(f"{c.name}: added infonce (contrastive alignment without a contrastive loss)")
    elif not c.losses:
        return f"strategy {c.strategy} has no usable loss"

    # hyperparameters
    for (part, attr), (lo, hi) in RANGES.items():
        _clip(getattr(c, part), attr, lo, hi, c.name, repairs)
    if c.head.residual:
        infos = [REGISTRY.get(m.encoder) for m in c.modalities]
        dims = {i.dim for i in infos if i is not None}
        if any(i is None or i.trainable for i in infos) or len(dims) != 1:
            c.head.residual = False
            repairs.append(f"{c.name}: residual head needs pretrained encoders of one dimension; "
                           "random initialisation used")
        elif c.head.embed_dim != next(iter(dims)):
            repairs.append(f"{c.name}: head embed_dim {c.head.embed_dim} -> {next(iter(dims))} "
                           "(a residual head keeps the pretrained dimension)")
            c.head.embed_dim = next(iter(dims))
    _clip(c.training, "max_epochs", 1, budget.max_epochs, c.name, repairs)
    finetunes = any(m.regime == "partial_finetune" for m in c.modalities)
    if finetunes:
        _clip(c.training, "batch_size", 8, FT_BATCH_MAX, c.name, repairs)
    cap = budget.finetune_seconds_per_fit if finetunes else budget.train_seconds_per_fit
    _clip(c.training, "train_seconds", MIN_FIT_SECONDS, cap, c.name, repairs)
    return None


def validate(candidate: dict | Plan, profile: DataProfile, budget: Budget,
             quick: bool = False) -> ValidationResult:
    try:
        plan = candidate if isinstance(candidate, Plan) else Plan(**candidate)
    except ValidationError as err:
        return ValidationResult(None, errors=[f"schema: {e['loc']}: {e['msg']}" for e in err.errors()][:10])
    plan = plan.model_copy(deep=True)
    errors, repairs = [], []
    t = plan.task
    names = {m.name: m for m in profile.modalities}

    # task
    if not t.pair or any(n not in names or names[n].kind == "spatial" for n in t.pair) or len(t.pair) > 2:
        return ValidationResult(None, [f"pair {t.pair} must name 1-2 non-spatial modalities of "
                                       f"{[n for n, m in names.items() if m.kind != 'spatial']}"])
    if len(t.pair) == 2:
        a, b = t.pair
        n_pair = profile.pairing.pairs.get(f"{a}|{b}", profile.pairing.pairs.get(f"{b}|{a}", 0))
        if n_pair < MIN_PAIRS:
            return ValidationResult(None, [f"only {n_pair} samples have both {a} and {b} (< {MIN_PAIRS})"])
    if t.prediction and (t.prediction.source not in t.pair or t.prediction.target not in t.pair):
        repairs.append(f"prediction {t.prediction.source}->{t.prediction.target} outside the pair; removed")
        t.prediction = None
        t.tasks = [x for x in t.tasks if x != "cross_modal_prediction"]
    if t.label_column and profile.label(t.label_column) is None:
        repairs.append(f"label column '{t.label_column}' does not exist; removed")
        t.label_column = None
    if plan.evaluation.label_column and profile.label(plan.evaluation.label_column) is None:
        repairs.append(f"evaluation label column '{plan.evaluation.label_column}' does not exist; removed")
        plan.evaluation.label_column = None
    if plan.evaluation.label_column is None and t.label_column:
        plan.evaluation.label_column = t.label_column

    # split
    s = plan.split
    groups = {g.column: g for g in profile.groups}
    if s.strategy == "predefined" and profile.split is None:
        s.strategy = "grouped_kfold" if profile.cv_group else "random"
        s.group_column = profile.cv_group
        repairs.append(f"no predefined split in the data; split -> {s.strategy}")
    if s.strategy == "grouped_kfold":
        g = groups.get(s.group_column or "")
        if g is None or g.n_levels < 3:
            repairs.append(f"group column '{s.group_column}' unusable for grouped CV; split -> random")
            s.strategy, s.group_column, s.n_folds = "random", None, 1
        else:
            _clip(s, "n_folds", 1, min(g.n_levels, budget.max_folds), "split", repairs)
    if s.strategy != "grouped_kfold":
        s.n_folds = 1
    if s.group_column and s.group_column not in groups:
        repairs.append(f"group column '{s.group_column}' does not exist; removed")
        s.group_column = None
    _clip(s, "val_fraction", 0.05, 0.4, "split", repairs)
    _clip(s, "test_fraction", 0.05, 0.4, "split", repairs)
    if s.stratify_column and profile.label(s.stratify_column) is None:
        s.stratify_column = None

    # configurations
    primaries = [c for c in plan.configs if c.role == "primary"]
    if not primaries:
        if not plan.configs:
            return ValidationResult(None, ["plan has no configurations"])
        plan.configs[0].role = "primary"
        repairs.append(f"promoted {plan.configs[0].name} to primary")
    for extra in primaries[1:]:
        extra.role = "comparison"
        repairs.append(f"demoted {extra.name} to comparison (one primary allowed)")
    plan.configs.sort(key=lambda c: c.role != "primary")
    kept, seen = [], set()
    for c in plan.configs:
        if c.name in seen:
            c.name = f"{c.name}_{len(seen)}"
            repairs.append(f"renamed duplicate configuration to {c.name}")
        seen.add(c.name)
        problem = _check_config(c, plan, profile, budget, quick, repairs)
        if problem:
            if c.role == "primary":
                errors.append(f"primary configuration invalid: {problem}")
            else:
                repairs.append(f"dropped comparison {c.name}: {problem}")
            continue
        kept.append(c)
    if errors:
        return ValidationResult(None, errors, repairs)
    if len(kept) > MAX_CONFIGS:
        repairs.append(f"kept {MAX_CONFIGS} of {len(kept)} configurations")
        kept = kept[:MAX_CONFIGS]
    plan.configs = kept

    # compute: every config on every fold, plus a final refit and one revision of the primary
    folds = max(1, s.n_folds)
    def total() -> float:
        p = plan.configs[0].training.train_seconds
        return folds * sum(c.training.train_seconds for c in plan.configs) + p + folds * p
    if total() > budget.total_train_seconds:
        comps = plan.configs[1:]
        room = budget.total_train_seconds - plan.configs[0].training.train_seconds * (2 * folds + 1)
        if comps and room > 0:
            scale = room / (folds * sum(c.training.train_seconds for c in comps))
            for c in comps:
                c.training.train_seconds = max(MIN_FIT_SECONDS, c.training.train_seconds * scale)
            repairs.append(f"scaled comparison time caps by {scale:.2f} to fit the run budget")
        while total() > budget.total_train_seconds and len(plan.configs) > 1:
            dropped = plan.configs.pop()
            repairs.append(f"dropped comparison {dropped.name} (run budget)")
        if total() > budget.total_train_seconds:
            p = plan.configs[0].training
            new = max(MIN_FIT_SECONDS, budget.total_train_seconds / (2 * folds + 1))
            repairs.append(f"primary time cap {p.train_seconds:.0f}s -> {new:.0f}s (run budget)")
            p.train_seconds = new

    # evaluation
    ev = plan.evaluation
    bad = [m for m in ev.metrics if m not in KNOWN_METRICS]
    if bad:
        ev.metrics = [m for m in ev.metrics if m in KNOWN_METRICS]
        repairs.append(f"dropped unknown metrics {bad}")
    has_label = ev.label_column is not None
    encs = [m.encoder for m in plan.configs[0].modalities]
    feasible = {"zero_shot": len(encs) == 2 and is_dual_pair(*encs), "ridge": t.prediction is not None,
                "early_fusion": has_label, "late_fusion": has_label, "cca": len(t.pair) == 2}
    drop = [b for b in ev.baselines if b not in KNOWN_BASELINES or not feasible.get(b, True)]
    if drop:
        ev.baselines = [b for b in ev.baselines if b not in drop]
        repairs.append(f"dropped unknown or infeasible baselines {drop}")
    for b in ("unimodal", "chance"):
        if b not in ev.baselines:
            ev.baselines.append(b)
    if not has_label:
        ev.metrics = [m for m in ev.metrics if m not in ("linear_probe", "knn_accuracy",
                                                         "retrieval_label_match_at_k")]
    if ev.group_column and ev.group_column not in groups:
        ev.group_column = None
    need = {"prediction_mean_pearson_top50": t.prediction is not None,
            "probe_macro_f1": has_label, "clustering_ari": has_label,
            "retrieval_mean_recall_at_10": len(t.pair) == 2}
    if not need[ev.primary_metric]:
        new = "retrieval_mean_recall_at_10" if len(t.pair) == 2 else "clustering_ari"
        repairs.append(f"primary metric {ev.primary_metric} infeasible -> {new}")
        ev.primary_metric = new
    required = {"retrieval_mean_recall_at_10": "retrieval_recall_at_k",
                "prediction_mean_pearson_top50": "prediction_pearson",
                "probe_macro_f1": "linear_probe", "clustering_ari": "clustering_ari_nmi"}[ev.primary_metric]
    if required not in ev.metrics:
        ev.metrics.append(required)

    # downstream
    spatial = any(m.kind == "spatial" for m in profile.modalities)
    ok_ds = {"retrieval_examples": len(t.pair) == 2, "cross_modal_prediction": t.prediction is not None,
             "classification": has_label, "clustering": True, "spatial_maps": spatial}
    drop = [d for d in plan.downstream if not ok_ds[d]]
    if drop:
        plan.downstream = [d for d in plan.downstream if d not in drop]
        repairs.append(f"dropped infeasible downstream analyses {drop}")
    return ValidationResult(plan, errors, repairs)
