"""Deterministic planner: profile + objective + budget (+ encoder screening) -> Plan.

This is both the ``--no-llm`` planner and the fallback when an LLM plan cannot
be repaired. Every rule reads only the profile, the TaskSpec, the compute
budget and the screening scores -- never a dataset name -- and every decision is
written to the decision log with the numbers that triggered it and the options
it rejected.

Thresholds are named constants so the report can quote them; each carries the
reasoning for its value.
"""

from __future__ import annotations

import math

from ..config import Budget
from ..decision_log import DecisionLog
from ..encoders.registry import REGISTRY, candidates, is_dual_pair
from ..schema import (ConfigSpec, DataProfile, EvalSpec, HeadSpec, LossTerm, ModalityPlan, Plan,
                      ScreeningResult, SplitSpec, TaskSpec, TrainingSpec)

# --- strategy -----------------------------------------------------------------------
# Contrastive alignment uses the other pairs in a batch as negatives. Below this many
# training pairs, batches are small, each positive meets few negatives and a learned
# head overfits; a linear method (CCA) is the lower-variance estimator.
MIN_PAIRS_CONTRASTIVE = 500
# Self-supervised pretraining of one modality pays off only when it has many more
# samples than the paired set that alignment can learn from.
UNPAIRED_SSL_RATIO = 2.0
# --- fine-tuning ----------------------------------------------------------------------
# Updating the last blocks of an ~86M-parameter backbone needs many pairs to beat a
# frozen backbone plus a ~0.5M-parameter head (bias-variance). Below FT_MIN_PAIRS we
# only freeze; between the two thresholds fine-tuning is tried as a comparison.
FT_MIN_PAIRS = 10_000
FT_PRIMARY_PAIRS = 50_000
FT_BLOCKS = 2
FT_MIN_EPOCHS = 2
# Partial-fine-tuning throughput (pairs/second) of a ViT-B/16 image tower with a
# BERT-base text tower, used only to check that FT_MIN_EPOCHS fit the time budget.
# MPS value measured on an M2 Pro (~45 pairs/s, last 2 blocks of each tower trainable);
# the others are conservative guesses.
FT_THROUGHPUT = {"cuda": 250.0, "mps": 45.0, "cpu": 4.0}
# --- losses -----------------------------------------------------------------------------
# Share of samples with a near-identical twin above which hard InfoNCE would push apart
# many pairs that are effectively the same sample (false negatives).
REDUNDANCY_SOFT = 0.10
# Minimum labelled share of training pairs for an auxiliary supervised loss term.
SUPERVISED_MIN_COVERAGE = 0.10
SUPERVISED_WEIGHT = 0.5
REGRESSION_WEIGHT = 1.0
# --- preprocessing ------------------------------------------------------------------------
# Between-group share of per-image colour variance (an ICC) above which stain colour is
# normalised: a frozen encoder would otherwise encode slide/patient identity as signal.
COLOUR_ICC_NORMALISE = 0.20
MAX_VARIABLE_FEATURES = 2000
# Fewer training samples than this per input feature -> compress the matrix with PCA
# before the MLP (fewer parameters, less noise).
SAMPLES_PER_FEATURE = 5
TEXT_TOKENS_PER_WORD = 1.5
# --- splits --------------------------------------------------------------------------------
MAX_FOLDS = 4
MIN_GROUPS_FOR_CV = 3
MAX_COMPARISONS = 2
MIN_GROUP_SIZE_BATCH = 5

PRIMARY_METRIC = {"retrieval": "retrieval_mean_recall_at_10",
                  "cross_modal_prediction": "prediction_mean_pearson_top50",
                  "classification": "probe_macro_f1", "clustering": "clustering_ari"}


def _pow2(x: float, lo: int, hi: int) -> int:
    return int(min(hi, max(lo, 2 ** math.floor(math.log2(max(x, 1))))))


class Context:
    """Derived quantities the rules share."""

    def __init__(self, profile: DataProfile, task: TaskSpec, budget: Budget,
                 screening: list[ScreeningResult] | None, quick: bool):
        self.profile, self.task, self.budget, self.quick = profile, task, budget, quick
        self.screening = {s.modality: s for s in screening or []}
        self.pair = task.pair
        self.mod = {n: profile.modality(n) for n in self.pair}
        if len(self.pair) == 2:
            a, b = self.pair
            self.n_paired = profile.pairing.pairs.get(f"{a}|{b}", profile.pairing.pairs.get(f"{b}|{a}", 0))
        else:
            self.n_paired = profile.pairing.present.get(self.pair[0], 0)
        self.n_used = min(self.n_paired, budget.max_samples or self.n_paired)
        self.label = profile.label(task.label_column)
        self.group = profile.group(profile.cv_group)
        image = next((m for m in profile.modalities if m.kind == "image"), None)
        self.image_domain = image.image.domain_hint if image and image.image else None
        self.spot_patches = bool(image and image.image and image.image.source == "spot_patches")
        self.spatial = any(m.kind == "spatial" for m in profile.modalities)
        self.train_fraction = 0.7
        self.n_train = int(self.n_used * self.train_fraction)

    @property
    def device(self) -> str:
        return self.budget.device


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------

def choose_split(ctx: Context, log: DecisionLog) -> SplitSpec:
    pr, g = ctx.profile, ctx.group
    label = ctx.label.column if ctx.label else None
    morans = next((m.matrix.morans_i_median for m in pr.modalities
                   if m.matrix and m.matrix.morans_i_median is not None), None)
    gshare = next((m.matrix.group_variance_share for m in pr.modalities
                   if m.matrix and m.matrix.group_variance_share is not None), None)
    if pr.split is not None:
        c = pr.split.counts
        ctx.train_fraction = c.get("train", 0) / max(1, sum(c.values()))
        overlap = g.split_overlap if g and g.split_overlap else {}
        caveat = ""
        if overlap.get("test_in_val", 0) > 0:
            caveat = (f" Note: {overlap['test_in_val']:.0%} of test samples share a '{g.column}' "
                      "with a validation sample, so choices made on validation may flatter test "
                      "results slightly.")
        if overlap.get("test_in_train", 0) > 0:
            caveat += (f" {overlap['test_in_train']:.0%} of test samples share a '{g.column}' with "
                       "training samples (possible leakage).")
        spec = SplitSpec(strategy="predefined", group_column=g.column if g else None,
                         stratify_column=label)
        log.add("split", "how to split the data", "predefined split",
                f"The data come with their own split ({', '.join(f'{k} {v:,}' for k, v in c.items())}); "
                "using it keeps results comparable with other work and keeps the test set unseen. "
                "Model selection uses validation only; test is scored once at the end." + caveat,
                [("re-split by group", "would break comparability with the published split"),
                 ("random re-split", "same, and ignores any grouping")],
                {"counts": c, "group_overlap": overlap})
    elif g is not None and g.n_levels >= MIN_GROUPS_FOR_CV:
        k = max(1, min(g.n_levels, MAX_FOLDS, ctx.budget.max_folds))
        if k >= 2:
            ctx.train_fraction = 1 - 1 / k - 1 / g.n_levels
        else:
            ctx.train_fraction = 1 - 0.2 - 1 / g.n_levels
        why = [f"Samples come in {g.n_levels} '{g.column}' groups"]
        if gshare is not None:
            why.append(f"'{g.column}' explains {gshare:.0%} of the variance of the top expression "
                       "components")
        if morans is not None:
            why.append(f"neighbouring samples are spatially correlated (median Moran's I {morans:.2f})")
        rationale = ("; ".join(why) + ". Samples within a group are not independent, so whole "
                     f"groups are held out: {'grouped ' + str(k) + '-fold cross-validation' if k > 1 else 'one held-out set of groups'}"
                     f"{', each group tested exactly once' if k > 1 else ''}, with one further group "
                     "per fold held out for validation (early stopping and model selection).")
        alts = [("random sample-level split", "leaks group-level and spatial dependence into the "
                 "test set and overstates performance")]
        if k < g.n_levels:
            alts.append((f"leave-one-{g.column}-out ({g.n_levels} folds)",
                         f"{g.n_levels} fits per configuration; {k} folds still test every group once"
                         if k > 1 else "compute cap for this mode allows one fold"))
        spec = SplitSpec(strategy="grouped_kfold", group_column=g.column, n_folds=k, stratify_column=label)
        log.add("split", "how to split the data", f"grouped by '{g.column}', {k} fold(s)", rationale, alts,
                {"n_groups": g.n_levels, "group_variance_share": gshare, "morans_i_median": morans})
    else:
        ctx.train_fraction = 0.7
        spec = SplitSpec(strategy="random", stratify_column=label)
        log.add("split", "how to split the data", "stratified random 70/10/20",
                "No predefined split and no grouping with enough levels; samples are treated as "
                "exchangeable" + (f", stratified by '{label}'" if label else "") + ".",
                [("grouped split", "no grouping column with >= 3 levels")], {})
    ctx.n_train = int(ctx.n_used * ctx.train_fraction)
    return spec


# ---------------------------------------------------------------------------
# strategy
# ---------------------------------------------------------------------------

def choose_strategy(ctx: Context, log: DecisionLog) -> str:
    tasks, n = ctx.task.tasks, ctx.n_train
    ev = {"n_paired": ctx.n_paired, "n_train_est": n, "tasks": tasks}
    for name, m in ctx.mod.items():
        unpaired = m.n_present - ctx.n_paired
        if ctx.n_paired and unpaired >= UNPAIRED_SSL_RATIO * ctx.n_paired:
            log.add("strategy", f"self-supervised pretraining for '{name}'", "noted, not run",
                    f"'{name}' has {unpaired:,} unpaired samples ({unpaired / ctx.n_paired:.1f}x the "
                    "paired set); a self-supervised stage on them could improve its encoder.", [], ev)
    if len(ctx.pair) < 2:
        log.add("strategy", "integration strategy", "self_supervised",
                "Only one usable modality: learn its representation by self-supervision.", [], ev)
        return "self_supervised"
    shared = any(t in tasks for t in ("retrieval", "cross_modal_prediction", "joint_embedding"))
    if n < MIN_PAIRS_CONTRASTIVE:
        log.add("strategy", "integration strategy", "linear_alignment (CCA)",
                f"About {n:,} training pairs (< {MIN_PAIRS_CONTRASTIVE}): too few negatives per batch "
                "for contrastive training to beat a linear alignment; frozen features + CCA.",
                [("CLIP-style contrastive alignment", f"needs >= {MIN_PAIRS_CONTRASTIVE} pairs")], ev)
        return "linear_alignment"
    if shared or "clustering" in tasks:
        need = [t for t in tasks if t in ("retrieval", "cross_modal_prediction", "joint_embedding",
                                          "clustering")]
        log.add("strategy", "integration strategy", "contrastive_alignment (CLIP-style)",
                f"The objective needs a shared space ({', '.join(need)}) and there are about {n:,} "
                "training pairs. Contrastive alignment learns that space directly from the pairing: "
                "each sample's partner is its positive and the other samples in the batch its "
                "negatives (InfoNCE, a lower bound on the mutual information between modalities).",
                [("early fusion (concatenate features)", "gives no cross-modal space: cannot retrieve "
                  "or predict one modality from the other"),
                 ("late fusion (combine per-modality predictions)", "same; used as a baseline only"),
                 ("supervised representation learning only", "labels are not the objective's target; "
                  "they are kept for evaluation" if "classification" not in tasks else
                  "would ignore the pairing; used as an auxiliary loss instead")], ev)
        return "contrastive_alignment"
    log.add("strategy", "integration strategy", "supervised_fusion",
            "Classification only: learn the fused representation with the labels.",
            [("contrastive alignment", "no retrieval or prediction requested")], ev)
    return "supervised_fusion"


# ---------------------------------------------------------------------------
# per-modality encoder and preprocessing
# ---------------------------------------------------------------------------

def _encoder(ctx: Context, name: str, log: DecisionLog) -> str:
    m = ctx.mod[name]
    if m.kind not in ("image", "text"):
        log.add("encoder", f"encoder for '{name}'", "mlp (trained from scratch)",
                f"'{name}' is a {m.kind} modality; no pretrained encoder applies, so a small MLP is "
                "trained on the preprocessed features.",
                [("PCA projection only", "linear; kept as the unimodal baseline")], {})
        return "mlp"
    cands = candidates(m.kind, ctx.image_domain, ctx.budget.max_candidates, ctx.quick)
    scr = ctx.screening.get(name)
    if scr and scr.ranked:
        best = scr.ranked[0]
        alts = [(s.encoder, f"{s.metric} {s.score:.3f} < {best.score:.3f}") for s in scr.ranked[1:]]
        log.add("encoder", f"encoder for '{name}'", best.encoder,
                f"Highest screening score among {len(scr.ranked)} candidates ({best.metric} "
                f"{best.score:.3f} on held-out screening samples). {scr.note}".strip(), alts,
                {"scores": {s.encoder: round(s.score, 4) for s in scr.ranked}})
        return best.encoder
    chosen = cands[0]
    info = REGISTRY[chosen]
    log.add("encoder", f"encoder for '{name}'", chosen,
            f"No screening scores; first candidate for {m.kind} data with measured domain "
            f"'{ctx.image_domain}' (pretrained on: {info.pretraining}).",
            [(c, "lower prior; not screened") for c in cands[1:]], {"candidates": cands})
    return chosen


def _preprocess(ctx: Context, name: str, encoder: str, log: DecisionLog) -> ModalityPlan:
    m, plan = ctx.mod[name], ModalityPlan(modality=name, encoder=encoder,
                                          regime="from_scratch" if REGISTRY[encoder].trainable else "frozen")
    if m.kind == "image" and m.image:
        im = m.image
        if im.source == "spot_patches":
            plan.patch_context = 1.0
            log.add("preprocessing", f"'{name}' patch size", "1.0 x spot diameter",
                    f"Each patch covers the spot's own footprint ({im.spot_diameter_px_median:.0f} px), "
                    "the tissue its RNA was captured from; resized to the encoder's input size.",
                    [("2.0 x spot diameter", "adds tissue context but also cells outside the spot; "
                      "tested as a comparison when the budget allows")],
                    {"spot_diameter_px": im.spot_diameter_px_median})
        icc = im.colour_between_group_share
        if icc is not None and icc >= COLOUR_ICC_NORMALISE:
            plan.colour_normalisation = "reinhard"
            log.add("preprocessing", f"'{name}' colour normalisation", "Reinhard",
                    f"{icc:.0%} of the variance in mean image colour lies between groups (ICC >= "
                    f"{COLOUR_ICC_NORMALISE:.0%}); a frozen encoder would encode stain/scanner "
                    "differences as signal. Colour statistics are matched to the training data.",
                    [("none", "group-level colour shifts would act as a batch effect")], {"icc": icc})
        elif icc is not None:
            log.add("preprocessing", f"'{name}' colour normalisation", "none",
                    f"Between-group colour share {icc:.0%} < {COLOUR_ICC_NORMALISE:.0%}.", [], {"icc": icc})
        if im.domain_hint == "greyscale_imaging":
            plan.greyscale_to_rgb = True
            log.add("preprocessing", f"'{name}' channels", "greyscale replicated to RGB",
                    f"{im.grayscale_fraction:.0%} of images are greyscale; pretrained encoders expect "
                    "three channels, so luminance is copied to all three (no information added).",
                    [("convert colour images to greyscale", "would discard colour in the minority "
                      "of colour images (overlays, Doppler)")], {})
    elif m.kind == "text" and m.text:
        cap = REGISTRY[encoder].max_tokens or 512
        want = int(math.ceil(TEXT_TOKENS_PER_WORD * m.text.words_p95 / 8) * 8)
        plan.text_max_tokens = min(cap, max(16, want))
        log.add("preprocessing", f"'{name}' text length", f"{plan.text_max_tokens} tokens",
                f"95% of texts have <= {m.text.words_p95:.0f} words (~{want} tokens); capped at the "
                f"encoder's context ({cap}). Whitespace and Unicode are normalised.",
                [("encoder maximum", "pads most texts for no gain")] if want < cap else
                [("longer context", f"encoder limit is {cap} tokens; "
                  f"{m.text.fraction_over_60_words:.1%} of texts exceed ~60 words")],
                {"words_p95": m.text.words_p95})
    elif m.kind in ("matrix", "tabular") and m.matrix:
        mx = m.matrix
        k = int(min(MAX_VARIABLE_FEATURES, max(50, mx.n_features_detected_5pct), mx.n_features))
        plan.n_variable_features = k
        if ctx.n_train / max(1, k) < SAMPLES_PER_FEATURE:
            plan.matrix_input = "pca"
            plan.pca_components = 100 if ctx.n_train >= 2000 else 50
            how = (f"PCA to {plan.pca_components} components (fitted on training data): about "
                   f"{ctx.n_train:,} training samples for {k:,} features is < {SAMPLES_PER_FEATURE} "
                   "per feature, so compressing first reduces parameters and count noise")
            alt = [("all variable features as MLP input", f"{k:,} inputs x hidden units parameters "
                    "for few samples per feature")]
        else:
            plan.matrix_input = "log_normalised" if mx.integer_valued else "standardised"
            how = "the variable features directly"
            alt = [("PCA first", "enough samples per feature to use them directly")]
        norm = ("library-size normalisation to the median library size, then log1p"
                if mx.integer_valued and mx.nonnegative else "standardisation")
        log.add("preprocessing", f"'{name}' features", f"{norm}; top {k:,} variable; {plan.matrix_input}",
                f"{mx.feature_kind} matrix, {mx.sparsity:.0%} zeros, median library {mx.library_size_median:,.0f}"
                f"{' (integer counts)' if mx.integer_valued else ''}: {norm}; keep the {k:,} most variable "
                f"features (fitted on training folds only); then {how}.", alt,
                {"sparsity": mx.sparsity, "n_detected_5pct": mx.n_features_detected_5pct})
    elif m.kind == "tabular":
        plan.matrix_input = "standardised"
    return plan


# ---------------------------------------------------------------------------
# losses and hyperparameters
# ---------------------------------------------------------------------------

def _redundancy(ctx: Context) -> tuple[float, str]:
    best, where = 0.0, ""
    for name, m in ctx.mod.items():
        vals = []
        if m.matrix:
            vals.append(m.matrix.near_twin_fraction)
        if m.text:
            vals.append(m.text.near_duplicate_fraction)
        if m.image and m.image.near_duplicate_fraction is not None:
            vals.append(m.image.near_duplicate_fraction)
        for v in vals:
            if v > best:
                best, where = v, name
    return best, where


def choose_losses(ctx: Context, strategy: str, log: DecisionLog) -> list[LossTerm]:
    tasks, losses = ctx.task.tasks, []
    if strategy == "contrastive_alignment":
        red, where = _redundancy(ctx)
        if red > REDUNDANCY_SOFT:
            losses.append(LossTerm(name="soft_infonce"))
            log.add("loss", "contrastive loss", "soft-target InfoNCE",
                    f"{red:.0%} of '{where}' samples have a near-identical twin (> {REDUNDANCY_SOFT:.0%}). "
                    "Hard InfoNCE treats every other sample in the batch as a negative, so look-alike "
                    "samples would be pushed apart (false negatives); soft targets weight each candidate "
                    "by its within-modality similarity instead.",
                    [("hard InfoNCE (CLIP)", "penalises near-identical negatives; tested as a "
                      "comparison when the budget allows")], {"near_twin_fraction": red})
        else:
            losses.append(LossTerm(name="infonce"))
            log.add("loss", "contrastive loss", "symmetric InfoNCE (CLIP)",
                    f"Near-duplicates are rare (max {red:.1%} <= {REDUNDANCY_SOFT:.0%}), so treating the "
                    "other batch samples as negatives is sound.",
                    [("soft-target InfoNCE", "adds a similarity model without a redundancy problem")],
                    {"near_twin_fraction": red})
    elif strategy == "self_supervised":
        kind = ctx.mod[ctx.pair[0]].kind
        losses.append(LossTerm(name="masked_denoising" if kind in ("matrix", "tabular") else "ssl_simclr"))
    elif strategy == "supervised_fusion":
        losses.append(LossTerm(name="supervised_ce"))
    lab = ctx.label
    if "classification" in tasks and lab and strategy in ("contrastive_alignment", "self_supervised"):
        if lab.labelled_fraction >= SUPERVISED_MIN_COVERAGE:
            losses.append(LossTerm(name="supervised_ce", weight=SUPERVISED_WEIGHT))
            log.add("loss", "supervised term", f"cross-entropy on '{lab.column}', weight {SUPERVISED_WEIGHT}",
                    f"The objective asks for classification and {lab.labelled_fraction:.0%} of samples "
                    f"carry '{lab.column}' ({lab.n_classes} classes): a supervised head on the joint "
                    "embedding shapes it for that task while InfoNCE keeps the pairing.",
                    [("contrastive only", "tested as a comparison"),
                     ("supervised only", "would ignore the pairing the retrieval task needs")],
                    {"labelled_fraction": lab.labelled_fraction})
        else:
            log.add("loss", "supervised term", "none",
                    f"Only {lab.labelled_fraction:.0%} of samples are labelled (< "
                    f"{SUPERVISED_MIN_COVERAGE:.0%}); labels are used for evaluation only.", [], {})
    pred = ctx.task.prediction
    if pred and ctx.profile.modality(pred.target).kind in ("matrix", "tabular") and \
            strategy != "linear_alignment":
        losses.append(LossTerm(name="cross_modal_regression", weight=REGRESSION_WEIGHT))
        log.add("loss", "prediction head", f"regress '{pred.target}' from '{pred.source}'",
                f"The objective asks to predict {pred.target} from {pred.source}: a regression head on "
                f"the {pred.source} embedding is trained jointly (mean squared error on the "
                "preprocessed target features). Nearest-neighbour prediction through the joint space "
                "and ridge regression on frozen features are evaluated alongside it.",
                [("retrieval-only prediction", "kept as a comparison method at evaluation")], {})
    return losses


def choose_hyperparameters(ctx: Context, log: DecisionLog) -> tuple[HeadSpec, TrainingSpec]:
    n, b = ctx.n_train, ctx.budget
    head = HeadSpec(embed_dim=128 if n < 5000 else 256, hidden_dim=512, n_layers=2,
                    dropout=0.2 if n < 5000 else 0.1)
    train = TrainingSpec(batch_size=_pow2(n / 16, 32, 512), lr=1e-3, weight_decay=1e-4,
                         max_epochs=b.max_epochs, patience=15 if b.max_epochs > 20 else 2,
                         train_seconds=b.train_seconds_per_fit)
    log.add("hyperparameters", "head and training", f"embed {head.embed_dim}, batch {train.batch_size}, "
            f"lr {train.lr:g}, dropout {head.dropout}",
            f"About {n:,} training pairs: a {head.embed_dim}-d joint space with 2-layer heads "
            f"({head.hidden_dim} hidden) is small relative to n; batch {train.batch_size} gives "
            f"{train.batch_size - 1} negatives per positive; early stopping on validation (patience "
            f"{train.patience}) and a {train.train_seconds:.0f}s cap per fit; temperature learned from "
            f"{train.temperature_init}.", [("larger heads", "more parameters than the data support")],
            {"n_train_est": n})
    return head, train


# ---------------------------------------------------------------------------
# comparisons, evaluation, downstream
# ---------------------------------------------------------------------------

def choose_head_initialisation(mods: list[ModalityPlan], strategy: str, head: HeadSpec,
                               log: DecisionLog) -> HeadSpec:
    """Identity-initialised (residual) heads when the two encoders were pretrained together into one
    aligned space, so training starts from the zero-shot solution instead of discarding it."""
    encs = [m.encoder for m in mods]
    if strategy == "linear_alignment" or len(encs) != 2 or not (is_dual_pair(*encs) or is_dual_pair(*encs[::-1])):
        return head
    dim = REGISTRY[encs[0]].dim
    log.add("hyperparameters", "head initialisation", f"residual: identity at the start, {dim}-d",
            f"{encs[0]} and {encs[1]} were pretrained together into one aligned space. Each head adds a learned "
            "correction (initialised at zero) to that space, so training starts from the pretrained (zero-shot) "
            "solution, and the starting point itself is a candidate when the best epoch is chosen on validation.",
            [("randomly initialised heads", "they discard the pretrained alignment and must relearn it from far "
              "fewer pairs than pretraining used")], {"embed_dim": dim})
    return head.model_copy(update={"residual": True, "embed_dim": dim})


def _finetune_ok(ctx: Context, cfg: ConfigSpec) -> tuple[bool, str]:
    encs = [m.encoder for m in cfg.modalities]
    if len(encs) != 2 or not is_dual_pair(encs[0], encs[1]):
        return False, "no pre-aligned dual encoder among the chosen encoders"
    if not all(REGISTRY[e].finetunable for e in encs):
        return False, "encoders do not support partial fine-tuning here"
    if ctx.device == "cpu":
        return False, "no GPU; fine-tuning on CPU would not finish within budget"
    if ctx.n_train < FT_MIN_PAIRS:
        return False, f"about {ctx.n_train:,} training pairs (< {FT_MIN_PAIRS:,})"
    secs = FT_MIN_EPOCHS * ctx.n_train / FT_THROUGHPUT.get(ctx.device, 6.0)
    if secs > ctx.budget.finetune_seconds_per_fit:
        return False, (f"{FT_MIN_EPOCHS} epochs would take ~{secs:.0f}s > "
                       f"{ctx.budget.finetune_seconds_per_fit:.0f}s budget")
    return True, (f"pre-aligned dual encoder ({encs[0]} + {encs[1]}), about {ctx.n_train:,} training "
                  f"pairs and ~{secs:.0f}s for {FT_MIN_EPOCHS} epochs on {ctx.device}")


def _variant(base: ConfigSpec, name: str, why: str, **changes) -> ConfigSpec:
    c = base.model_copy(deep=True)
    c.name, c.role, c.rationale = name, "comparison", why
    for k, v in changes.items():
        setattr(c, k, v)
    return c


def choose_configs(ctx: Context, primary: ConfigSpec, log: DecisionLog) -> list[ConfigSpec]:
    options: list[ConfigSpec] = []
    ok, why = _finetune_ok(ctx, primary)
    if ok:
        mods = [m.model_copy(update={"regime": "partial_finetune", "trainable_blocks": FT_BLOCKS,
                                     "augment": ctx.mod[m.modality].kind == "image"})
                for m in primary.modalities]
        tr = primary.training.model_copy(update={"batch_size": 64, "max_epochs": max(FT_MIN_EPOCHS, 5),
                                                 "patience": 2,
                                                 "train_seconds": ctx.budget.finetune_seconds_per_fit})
        ft = _variant(primary, "partial_finetune", f"Tests the frozen-backbone decision: {why}.",
                      modalities=mods, training=tr)
        if ctx.n_train >= FT_PRIMARY_PAIRS:
            ft.role, primary.role = "primary", "comparison"
        options.append(ft)
    log.add("regime", "frozen backbones or fine-tuning", "fine-tune as comparison" if ok else "frozen",
            ("Frozen pretrained backbones with small trainable projection heads: few trainable "
             "parameters for the number of pairs (low variance), features computed once. ") +
            (f"Partial fine-tuning of the last {FT_BLOCKS} blocks is run as a comparison: {why}."
             if ok else f"Fine-tuning not tried: {why}."),
            [("full fine-tuning", "~86M parameters per tower for the available pairs; high variance "
              "and compute")], {"n_train_est": ctx.n_train})
    names = [t.name for t in primary.losses]
    if "supervised_ce" in names and primary.strategy == "contrastive_alignment":
        options.append(_variant(primary, "contrastive_only", "Tests whether the supervised term helps.",
                                losses=[t for t in primary.losses if t.name != "supervised_ce"]))
    if "soft_infonce" in names:
        options.append(_variant(primary, "hard_infonce", "Tests the soft-target decision.",
                                losses=[LossTerm(name="infonce") if t.name == "soft_infonce" else t
                                        for t in primary.losses]))
    if ctx.spot_patches:
        mods = [m.model_copy(update={"patch_context": 2.0}) if m.patch_context else m
                for m in primary.modalities]
        options.append(_variant(primary, "patch_context_2x", "Tests the patch-size decision "
                                "(2 x spot diameter: more tissue context).", modalities=mods))
    keep = options[:MAX_COMPARISONS]
    for c in keep:
        log.add("comparison", f"comparison configuration '{c.name}'", "run", c.rationale, [], {})
    for c in options[MAX_COMPARISONS:]:
        log.add("comparison", f"comparison configuration '{c.name}'", "not run", c.rationale,
                [(c.name, f"at most {MAX_COMPARISONS} comparisons per run (compute)")], {})
    configs = [primary] + keep
    configs.sort(key=lambda c: c.role != "primary")
    return configs


def choose_evaluation(ctx: Context, strategy: str, configs: list[ConfigSpec], log: DecisionLog) -> EvalSpec:
    t = ctx.task
    metrics = []
    if "retrieval" in t.tasks or len(ctx.pair) == 2:
        metrics.append("retrieval_recall_at_k")
    if t.prediction:
        metrics.append("prediction_pearson")
    if ctx.label:
        metrics += ["linear_probe", "knn_accuracy", "clustering_ari_nmi"]
        if "retrieval_recall_at_k" in metrics:
            metrics.append("retrieval_label_match_at_k")
    elif "clustering" in t.tasks:
        metrics.append("clustering_ari_nmi")
    metrics += ["alignment_uniformity", "modality_gap", "effective_rank"]
    # Clustering by group only means something when groups hold many samples each.
    batch_group = ctx.group if ctx.group and ctx.group.size_median >= MIN_GROUP_SIZE_BATCH else None
    if batch_group:
        metrics.append("group_silhouette")
    baselines = ["unimodal", "chance"]
    if len(ctx.pair) == 2:
        baselines.append("cca")
    if t.prediction:
        baselines.append("ridge")
    if ctx.label:
        baselines += ["early_fusion", "late_fusion"]
    encs = [m.encoder for m in configs[0].modalities]
    if len(encs) == 2 and is_dual_pair(encs[0], encs[1]):
        baselines.append("zero_shot")
    primary = PRIMARY_METRIC[t.primary_task]
    if primary == "probe_macro_f1" and not ctx.label:
        primary = "retrieval_mean_recall_at_10"
    spec = EvalSpec(primary_metric=primary, metrics=metrics, baselines=baselines,
                    label_column=ctx.label.column if ctx.label else None,
                    group_column=batch_group.column if batch_group else None)
    log.add("evaluation", "primary metric", primary,
            f"First measurable task in the objective is '{t.primary_task}'. Every metric is computed on "
            "held-out data and compared with unimodal features, linear alignment (CCA)"
            f"{', ridge regression' if t.prediction else ''}"
            f"{', zero-shot pretrained alignment' if 'zero_shot' in baselines else ''} and chance, to show "
            "whether integration helped.", [], {"metrics": metrics, "baselines": baselines})
    return spec


def choose_downstream(ctx: Context) -> list[str]:
    t, out = ctx.task, []
    if "retrieval" in t.tasks:
        out.append("retrieval_examples")
    if t.prediction:
        out.append("cross_modal_prediction")
    if "classification" in t.tasks and ctx.label:
        out.append("classification")
    if "clustering" in t.tasks:
        out.append("clustering")
    if ctx.spatial and (t.prediction or "clustering" in t.tasks):
        out.append("spatial_maps")
    return out


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def plan(profile: DataProfile, task: TaskSpec, budget: Budget, log: DecisionLog,
         screening: list[ScreeningResult] | None = None, quick: bool = False) -> Plan:
    ctx = Context(profile, task, budget, screening, quick)
    split = choose_split(ctx, log)
    strategy = choose_strategy(ctx, log)
    mods = [_preprocess(ctx, n, _encoder(ctx, n, log), log) for n in ctx.pair]
    losses = choose_losses(ctx, strategy, log)
    head, training = choose_hyperparameters(ctx, log)
    head = choose_head_initialisation(mods, strategy, head, log)
    primary = ConfigSpec(name="primary", role="primary", strategy=strategy, fusion="joint",
                         modalities=mods, losses=losses, head=head, training=training,
                         rationale="Rule-based choice; see the decision log.")
    configs = choose_configs(ctx, primary, log) if strategy != "linear_alignment" else [primary]
    evaluation = choose_evaluation(ctx, strategy, configs, log)
    return Plan(task=task, split=split, configs=configs, evaluation=evaluation,
                downstream=choose_downstream(ctx),
                rationale=f"{strategy} on {' + '.join(ctx.pair)} for {', '.join(task.tasks)}",
                source="heuristic")
