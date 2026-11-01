"""The agent loop: from a dataset and an objective to a finished report, in one call.

    hardware -> load -> profile -> parse the objective -> plan (rules + encoder screening
    + LLM) -> per fold: features, preprocessing, training of every configuration
    -> validation evaluation -> critic (at most one revision, kept only if the validation
    primary metric improves) -> test evaluation of every configuration and baseline,
    scored once -> downstream analyses -> figures, tables, embeddings -> report

Nothing here decides *what* to do: the planner and the critic do. This module executes
their decisions within the compute budget and records what happened. Every decision is
written to decision_log.json as it is made, so a run that stops part-way still shows how
far it got and why.

Compute is accounted by each fit's planned time cap, not its measured time, so budget
decisions (and the prompts the LLM sees) are the same when a run is repeated.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import critic as critic_mod
from . import downstream as D
from . import report
from .config import REPO_ROOT, RunConfig, load_config
from .data.loaders import load_path
from .decision_log import DecisionLog
from .encoders.pretrained import FeatureCache, load_encoder, release_all
from .encoders.registry import REGISTRY, is_dual_pair
from .encoders.screening import fit_normalizer, modality_features
from .evaluate import prediction as P
from .evaluate import suite as S
from .evaluate.metrics import l2n, recall_at_k
from .hardware import detect, seed_everything
from .llm import LLMClient, make_client
from .models.finetune import finetune
from .models.inputs import assemble, label_codes
from .models.linear import LinearAligner
from .models.train import SELECTION, fit
from .objective import task_from_objective
from .planner.flow import agent_plan
from .profiler import profile as profile_data
from .schema import ConfigSpec

EVAL_TRAIN_CAP = 5000            # training rows embedded per fold for probes and kNN (probes use <= 5000)
COMPARISON_METRICS = ["retrieval_mean_recall_at_10", "prediction_mean_pearson_top50", "probe_macro_f1",
                      "clustering_ari"]
VALIDATION_METRICS = {"retrieval_recall_at_k", "clustering_ari_nmi", "group_silhouette", "prediction_pearson"}
METRIC_FAMILY = {"retrieval_mean_recall_at_10": "retrieval_recall_at_k",
                 "prediction_mean_pearson_top50": "prediction_pearson", "probe_macro_f1": "linear_probe",
                 "clustering_ari": "clustering_ari_nmi"}


@dataclass
class FitRecord:
    config: str
    fold: int
    record: dict                                            # critic.training_record: summary + curves
    emb: dict[str, dict[str, np.ndarray]]                   # train (evaluation rows) / val / test -> modality
    preds: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)    # split -> method -> values
    finetuned: bool = False
    target_features: list[str] = field(default_factory=list)


class Agent:
    def __init__(self, data_path, objective: str, out_dir, cfg: RunConfig, client: LLMClient | None = None):
        self.data_path, self.objective, self.out = Path(data_path), objective, Path(out_dir)
        self.cfg, self.client = cfg, client
        self.t0 = time.perf_counter()
        self.timings: dict[str, float] = {}
        self.fits: dict[tuple[str, int], FitRecord] = {}
        self.configs: dict[str, ConfigSpec] = {}
        self.planned_seconds = 0.0
        self.measured_seconds = 0.0
        self._frozen: dict[tuple, np.ndarray] = {}
        self._base: dict[tuple, dict] = {}
        self._normalizer = None
        self.critique = None
        self.revision: dict = {"action": "none", "ran": False, "kept": False}
        self.downstream_results: dict = {}

    # -- small helpers -------------------------------------------------------------------

    def say(self, msg: str) -> None:
        print(f"[{time.perf_counter() - self.t0:7.1f}s] {msg}", flush=True)

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        self.say(f"== {name}")
        yield
        self.timings[name] = round(self.timings.get(name, 0.0) + time.perf_counter() - start, 1)

    def config(self, name: str) -> ConfigSpec:
        return self.configs[name]

    def trained(self) -> list[str]:
        return [n for n in self.configs if any((n, i) in self.fits for i in range(len(self.folds)))]

    # -- steps ---------------------------------------------------------------------------

    def execute(self) -> dict:
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = DecisionLog(self.out / "decision_log.json")
        self.setup()
        self.load()
        self.make_plan()
        self.train_all()
        self.review()
        self.final_evaluation()
        self.run_downstream()
        return self.write_outputs()

    def setup(self) -> None:
        cfg = self.cfg
        self.hw = detect(cfg.device, self.out)
        seed_everything(cfg.seed)
        self.budget = cfg.budget(self.hw.device)
        self.heads_device = self.hw.device if cfg.heads_device == "auto" else cfg.heads_device
        self.cache = FeatureCache(REPO_ROOT / cfg.cache_dir / "features")
        hw = self.hw
        self.log.add("hardware", "compute device", f"{hw.device} ({hw.device_name})",
                     f"Detected {hw.n_cpu} CPU cores and {hw.memory_gb} GB memory. Feature extraction and "
                     f"fine-tuning run on {hw.device}; projection heads train on {self.heads_device}"
                     + (" so reruns are bit-for-bit reproducible." if self.heads_device == "cpu" else "."),
                     [], {"device": hw.device, "heads_device": self.heads_device, "quick": cfg.quick,
                          "budget": self.budget.model_dump()}, source="system")

    def load(self) -> None:
        cfg = self.cfg
        with self.stage("load and profile"):
            self.ds = load_path(self.data_path, cfg.label_column, cfg.group_column)
            t, self.prof, removed = task_from_objective(self.ds, profile_data(self.ds, cfg.seed),
                                                        self.objective, cfg.seed)
        (self.out / "profile.json").write_text(self.prof.model_dump_json(indent=1))
        self.log.add("profile", "data profile", self.prof.brief(t.label_column),
                     "Measured from the data before any modeling decision (profile.json).", [],
                     {"n_samples": self.prof.n_samples,
                      "modalities": {m.name: m.kind for m in self.prof.modalities}}, source="system")
        self.task = t
        self.log.add("objective", "parsed objective",
                     f"tasks {t.tasks}; primary {t.primary_task}; modalities {t.pair}"
                     + (f"; predict {t.prediction.target} from {t.prediction.source}" if t.prediction else ""),
                     "; ".join(f"{k}: {', '.join(v)}" for k, v in t.evidence.items()) or "no task words found",
                     [], {"label_column": t.label_column, "label_evidence": t.label_evidence}, source="heuristic")
        if removed:
            self.log.add("preprocessing", "outcome kept out of the inputs", f"removed {', '.join(removed)}",
                         f"'{t.label_column}' is the label this run evaluates ({t.label_evidence}), so it is not "
                         "one of the tabular inputs; the data were profiled again without it.", [],
                         {"removed": removed}, source="system")

    def make_plan(self) -> None:
        cfg = self.cfg
        if self.client is None:
            self.client = make_client(cfg.llm, self.out / "llm_cache", cfg.replay)
        for note in self.client.notes:
            self.log.add("llm", "LLM availability", "rule-based", note, source="system")
        with self.stage("plan (screening + planner)"):
            out = agent_plan(self.ds, self.prof, self.task, self.budget, self.hw.device, self.cache, self.log,
                             cfg.quick, cfg.seed, self.client, progress=self.say)
        self.plan, self.folds, self.screening_table = out.plan, out.folds, out.screening_table
        self.configs = {c.name: c for c in self.plan.configs}
        self.pair = list(self.plan.task.pair)
        (self.out / "plan.json").write_text(self.plan.model_dump_json(indent=1))
        (self.out / "screening.json").write_text(json.dumps(self.screening_table, indent=1, default=str))
        self.used = np.unique(np.concatenate([np.concatenate([f.train, f.val, f.test]) for f in self.folds]))
        lib = next(iter(self.ds.columns_with("library")), None)
        grp = lib or self.prof.cv_group
        self.norm_groups = self.ds.obs[grp].astype(str).to_numpy() if grp else None
        self.within_col = lib
        rng = np.random.default_rng(cfg.seed)
        self.train_pos = [np.sort(rng.permutation(len(f.train))[:EVAL_TRAIN_CAP]) for f in self.folds]
        for f in self.folds:
            n = f.sizes()
            self.log.add("split", f"fold {f.name}", f"{n['train']:,} train / {n['val']:,} validation / "
                         f"{n['test']:,} test samples",
                         f"Test groups: {', '.join(map(str, f.test_groups or [])) or 'none (not grouped)'}; validation "
                         f"groups: {', '.join(map(str, f.val_groups or [])) or 'none (not grouped)'}.", [],
                         {"sizes": n}, source="system")

    # -- features --------------------------------------------------------------------------

    def normalizer(self):
        if self._normalizer is None and self.norm_groups is not None:
            img = next((n for n in self.pair if self.ds.modalities[n].kind == "image"), None)
            if img is not None:
                self._normalizer = fit_normalizer(self.ds.modalities[img].images, self.norm_groups,
                                                  np.random.default_rng(self.cfg.seed))
        return self._normalizer

    def frozen_for(self, cfg: ConfigSpec) -> dict[str, np.ndarray]:
        """Frozen pretrained features of every image/text modality, for all rows any fold uses."""
        out = {}
        for mp in cfg.modalities:
            if self.ds.modalities[mp.modality].kind not in ("image", "text"):
                continue
            key = (mp.modality, mp.encoder, mp.patch_context, mp.colour_normalisation, mp.greyscale_to_rgb,
                   mp.text_max_tokens, mp.text_cleaning)
            if key not in self._frozen:
                norm = self.normalizer() if mp.colour_normalisation == "reinhard" else None
                full = np.zeros((self.ds.n, REGISTRY[mp.encoder].dim), np.float32)
                full[self.used] = modality_features(self.ds, mp, mp.encoder, self.used, self.hw.device, self.cache,
                                                    self.budget.feature_batch_size, norm, self.norm_groups,
                                                    progress=self.say)
                self._frozen[key] = full
                release_all()
            out[mp.modality] = self._frozen[key]
        return out

    def baselines(self, i: int, cfg: ConfigSpec) -> dict:
        """Inputs, CCA, zero-shot availability and ridge predictions for one fold, from the frozen
        version of ``cfg`` (its encoders and preprocessing without any fine-tuning)."""
        frozen = cfg.model_copy(deep=True)
        for m in frozen.modalities:
            if m.regime == "partial_finetune":
                m.regime, m.trainable_blocks, m.augment = "frozen", 0, False
        sig = json.dumps([m.model_dump(exclude={"regime", "trainable_blocks", "augment"}) for m in frozen.modalities],
                         sort_keys=True)
        if (i, sig) not in self._base:
            asm = assemble(self.ds, self.prof, self.plan, frozen, self.folds[i], self.frozen_for(frozen))
            encs = [m.encoder for m in frozen.modalities]
            ridge = {}
            pred = self.plan.task.prediction
            if pred is not None and asm.train.target is not None:
                for split in ("val", "test"):
                    ridge[split] = P.ridge_predict(asm.train.inputs[pred.source], asm.train.target,
                                                   getattr(asm, split).inputs[pred.source])
            self._base[(i, sig)] = {
                "asm": asm, "ridge": ridge,
                "cca": LinearAligner(self.pair).fit(asm.train.inputs) if len(self.pair) == 2 else None,
                "zero_shot": len(encs) == 2 and (is_dual_pair(*encs) or is_dual_pair(*encs[::-1]))}
        return self._base[(i, sig)]

    # -- training ----------------------------------------------------------------------------

    def train_all(self) -> None:
        reserve = len(self.folds) * self.plan.primary.training.train_seconds if self.cfg.max_critic_revisions else 0
        with self.stage("training"):
            for cfg in self.plan.configs:
                for i in range(len(self.folds)):
                    cap = cfg.training.train_seconds
                    if cfg.role != "primary" and self.planned_seconds + cap > self.budget.total_train_seconds - reserve:
                        self.log.add("training", f"comparison '{cfg.name}'", "stopped",
                                     f"Run budget: {self.planned_seconds:.0f}s of {self.budget.total_train_seconds:.0f}s "
                                     f"planned, {reserve:.0f}s reserved for a critic revision.", source="system")
                        break
                    self.fit_config(cfg, i)

    def fit_config(self, cfg: ConfigSpec, i: int) -> FitRecord:
        fold, ds, plan = self.folds[i], self.ds, self.plan
        self.say(f"training '{cfg.name}' on {fold.name} ({len(fold.train):,} train / {len(fold.val):,} val)")
        if any(m.regime == "partial_finetune" for m in cfg.modalities):
            rec = self._fit_finetuned(cfg, i)
        elif cfg.strategy == "linear_alignment":
            rec = self._fit_linear(cfg, i)
        else:
            asm = assemble(ds, self.prof, plan, cfg, fold, self.frozen_for(cfg))
            pred = plan.task.prediction
            decode = tuple(m for m in self.pair if ds.modalities[m].kind in ("matrix", "tabular"))
            res = fit(cfg, asm.train, asm.val, selection=SELECTION[plan.evaluation.primary_metric],
                      n_classes=len(asm.classes), pred_source=pred.source if pred else None, decode=decode,
                      device=self.heads_device, seed=self.cfg.seed, progress=self.say)
            m = res.model
            train_all = {k: m.embed_np(k, asm.train.inputs[k]) for k in self.pair}
            emb = {"train": {k: v[self.train_pos[i]] for k, v in train_all.items()},
                   "val": {k: m.embed_np(k, asm.val.inputs[k]) for k in self.pair},
                   "test": {k: m.embed_np(k, asm.test.inputs[k]) for k in self.pair}}
            preds = {}
            if pred is not None and asm.train.target is not None:
                for split in ("val", "test"):
                    x = getattr(asm, split).inputs[pred.source]
                    p = {f"{cfg.name}:knn": P.knn_predict(m.embed_np(pred.source, x), train_all[pred.target],
                                                          asm.train.target)}
                    if m.regressor is not None:
                        p[f"{cfg.name}:head"] = m.predict_np(pred.source, x)
                    preds[split] = p
            rec = FitRecord(cfg.name, i, {**critic_mod.training_record(res)}, emb, preds, False, asm.target_features)
        self.configs.setdefault(cfg.name, cfg)
        self.planned_seconds += cfg.training.train_seconds
        self.measured_seconds += rec.record.get("seconds") or 0.0
        self.fits[(cfg.name, i)] = rec
        r = rec.record
        iterative = r["epochs_run"] > 0
        self.log.add("training", f"fit '{cfg.name}' on {fold.name}",
                     f"best epoch {r['best_epoch']} of {r['epochs_run']}" if iterative else r["stopped_because"],
                     (f"Stopped: {r['stopped_because']}. " if iterative else "") +
                     f"Validation {r['selection_metric']} {r['best_val_score']:.4f}"
                     + ("." if not iterative else
                        "; no epoch beat the untrained starting point, which is kept." if r["best_epoch"] == 0 else
                        "; the best epoch's weights are kept."), [],
                     {"n_trainable_parameters": r["n_trainable_parameters"]}, source="system")
        return rec

    def _fit_linear(self, cfg: ConfigSpec, i: int) -> FitRecord:
        """The linear_alignment strategy: CCA in closed form, no iterative training."""
        fold, a, b = self.folds[i], self.pair[0], self.pair[-1]
        asm = assemble(self.ds, self.prof, self.plan, cfg, fold, self.frozen_for(cfg))
        start = time.perf_counter()
        lin = LinearAligner(self.pair).fit(asm.train.inputs)
        train_all = {k: lin.embed(k, asm.train.inputs[k]) for k in self.pair}
        emb = {"train": {k: v[self.train_pos[i]] for k, v in train_all.items()},
               "val": {k: lin.embed(k, asm.val.inputs[k]) for k in self.pair},
               "test": {k: lin.embed(k, asm.test.inputs[k]) for k in self.pair}}
        preds = {}
        pred = self.plan.task.prediction
        if pred is not None and asm.train.target is not None:
            for split in ("val", "test"):
                z = lin.embed(pred.source, getattr(asm, split).inputs[pred.source])
                preds[split] = {f"{cfg.name}:knn": P.knn_predict(z, train_all[pred.target], asm.train.target)}
        c = lin.cca
        record = {"selection_metric": "retrieval", "best_epoch": 0, "epochs_run": 0,
                  "best_val_score": round(recall_at_k(emb["val"][a], emb["val"][b]), 4),
                  "seconds": round(time.perf_counter() - start, 1),
                  "stopped_because": f"closed form (CCA, {c.Ua.shape[1]} canonical pairs)",
                  "n_trainable_parameters": int(c.px.components_.size + c.py.components_.size + c.Ua.size + c.Ub.size),
                  "final_temperature": None, "train_loss_at_best": None, "val_loss_at_best": None,
                  "val_score": [], "train_loss": []}
        return FitRecord(cfg.name, i, record, emb, preds, False, asm.target_features)

    def _fit_finetuned(self, cfg: ConfigSpec, i: int) -> FitRecord:
        fold, ds = self.folds[i], self.ds
        release_all()                                   # fresh pretrained weights for every fold
        encs = {mp.modality: load_encoder(mp.encoder, self.hw.device) for mp in cfg.modalities}
        labels, classes = None, []
        if self.plan.evaluation.label_column:
            labels, classes = label_codes(ds.obs[self.plan.evaluation.label_column].to_numpy(), fold.train)
        norm = self.normalizer() if any(m.colour_normalisation == "reinhard" for m in cfg.modalities) else None
        ft, res = finetune(cfg, ds, encs, fold.train, fold.val, labels, len(classes), self.hw.device, self.cfg.seed,
                           norm, self.norm_groups, progress=self.say)
        idx = {"train": fold.train[self.train_pos[i]], "val": fold.val, "test": fold.test}
        emb = {split: {m: ft.embed_indices(m, rows) for m in self.pair} for split, rows in idx.items()}
        del ft, encs
        release_all()
        return FitRecord(cfg.name, i, critic_mod.training_record(res), emb, {}, True, [])

    # -- evaluation ----------------------------------------------------------------------------

    def evaluate(self, split: str, primary: str, names: list[str], metrics: list[str]) -> list[dict]:
        """Every representation on one held-out split of every fold: the trained configurations,
        each modality alone, CCA, the pretrained space (zero-shot) and ridge/kNN predictions."""
        out, obs = [], self.ds.obs
        bc = self.plan.evaluation.group_column
        for i, fold in enumerate(self.folds):
            recs = [self.fits[(n, i)] for n in names if (n, i) in self.fits]
            if not any(r.config == primary for r in recs):
                continue
            base = self.baselines(i, self.config(primary))
            asm, pos = base["asm"], self.train_pos[i]
            held = getattr(asm, split)
            tr = {k: v[pos] for k, v in asm.train.inputs.items()}
            reps = [S.Representation(r.config, "joint", r.emb["train"], r.emb[split]) for r in recs]
            reps.append(S.Representation("unimodal", "unimodal", tr, held.inputs, aligned=False))
            if base["cca"] is not None and self.config(primary).strategy != "linear_alignment":
                lin = base["cca"]
                reps.append(S.Representation("cca", "linear", {k: lin.embed(k, tr[k]) for k in self.pair},
                                             {k: lin.embed(k, held.inputs[k]) for k in self.pair}))
            if base["zero_shot"]:
                reps.append(S.Representation("zero_shot", "zero_shot", tr, held.inputs))
            preds = {meth: v for r in recs if r.target_features == asm.target_features
                     for meth, v in r.preds.get(split, {}).items()}
            if split in base["ridge"]:
                preds["ridge"] = base["ridge"][split]
            idx = fold.val if split == "val" else fold.test
            fd = S.FoldData(fold.name, self.pair, None if asm.train.labels is None else asm.train.labels[pos],
                            held.labels, asm.classes,
                            obs[bc].astype(str).to_numpy()[idx] if bc else None,
                            obs[self.within_col].astype(str).to_numpy()[idx] if self.within_col else None,
                            held.target, asm.target_variance_order, asm.target_features)
            out.append(S.evaluate_fold(reps, fd, metrics, tuple(self.plan.evaluation.retrieval_ks), preds,
                                       self.cfg.seed))
        return out

    def validation_metrics(self) -> list[str]:
        """The critic acts on the primary metric; probes are fitted on validation data only
        when the primary metric needs them (they are the slowest part of an evaluation)."""
        pm = self.plan.evaluation.primary_metric
        keep = VALIDATION_METRICS | {METRIC_FAMILY[pm]}
        return [m for m in self.plan.evaluation.metrics if m in keep]

    def review(self) -> None:
        plan, pm = self.plan, self.plan.evaluation.primary_metric
        primary = plan.primary.name
        self.primary_name = primary
        names = self.trained()
        with self.stage("validation evaluation"):
            self.val_folds = self.evaluate("val", primary, names, self.validation_metrics())
        boot = S.bootstrap(self.val_folds[0], self.cfg.seed) if len(self.val_folds) == 1 else None
        self.val_comparisons = S.integration_comparisons(self.val_folds, primary, COMPARISON_METRICS, boot)
        self.val_flags = S.flags(self.val_folds, primary, self.val_comparisons, pm, names)
        if self.cfg.max_critic_revisions < 1:
            self.log.add("critic", "revise the plan?", "critic disabled", "max_critic_revisions is 0.", source="system")
            return
        training = {n: [self.fits[(n, i)].record for i in range(len(self.folds)) if (n, i) in self.fits]
                    for n in names}
        remaining = self.budget.total_train_seconds - self.planned_seconds
        with self.stage("critic"):
            crit, revised = critic_mod.critique(plan, self.prof, self.val_folds, self.val_comparisons, self.val_flags,
                                                training, self.client, self.log, self.budget, self.cfg.quick,
                                                remaining)
        self.critique = crit
        original = [S.metric_value(f, primary, pm) for f in self.val_folds]
        self.revision = {"action": crit.action, "ran": False, "kept": False, "original_val": original}
        mean = lambda xs: float(np.mean([x for x in xs if x is not None])) if any(x is not None for x in xs) else None  # noqa: E731
        self.revision["original_val_mean"] = mean(original)
        if crit.action == "switch_primary":
            new = [S.metric_value(f, crit.target_config, pm) for f in self.val_folds]
            kept = critic_mod.keep_revision(original, new, pm, self.log)
            self.revision.update(ran=True, kept=kept, target=crit.target_config, revised_val=new,
                                 revised_val_mean=mean(new))
            if kept:
                self.primary_name = crit.target_config
        elif crit.action == "modify_primary" and revised is not None:
            with self.stage("critic revision"):
                for i in range(len(self.folds)):
                    self.fit_config(revised, i)
                folds = self.evaluate("val", revised.name, [primary, revised.name], self.validation_metrics())
            new = [S.metric_value(f, revised.name, pm) for f in folds]
            kept = critic_mod.keep_revision(original, new, pm, self.log)
            self.revision.update(ran=True, kept=kept, target=revised.name, revised_val=new,
                                 revised_val_mean=mean(new), changes={c.path: c.value for c in crit.changes})
            if kept:
                self.primary_name = revised.name

    def final_evaluation(self) -> None:
        pm, names = self.plan.evaluation.primary_metric, self.trained()
        with self.stage("test evaluation"):
            self.test_folds = self.evaluate("test", self.primary_name, names, self.plan.evaluation.metrics)
        self.test_boot = S.bootstrap(self.test_folds[0], self.cfg.seed) if len(self.test_folds) == 1 else None
        self.test_comparisons = S.integration_comparisons(self.test_folds, self.primary_name, COMPARISON_METRICS,
                                                          self.test_boot)
        self.test_flags = S.flags(self.test_folds, self.primary_name, self.test_comparisons, pm, names)
        self.test_aggregate = S.aggregate(self.test_folds)
        vals = [S.metric_value(f, self.primary_name, pm) for f in self.test_folds]
        self.log.add("evaluation", "held-out test result", f"{pm} {np.mean([v for v in vals if v is not None]):.4f}",
                     f"Test data were scored once, after the critic, for '{self.primary_name}' and every baseline "
                     f"({len(self.test_folds)} fold(s)).", [], {"per_fold": vals}, source="system")

    # -- downstream --------------------------------------------------------------------------

    def downstream_fold(self) -> int:
        """The fold whose test set has the most labelled samples (first fold on ties)."""
        counts = []
        for i in range(len(self.folds)):
            labels = self.baselines(i, self.config(self.primary_name))["asm"].test.labels
            counts.append(0 if labels is None else int((labels >= 0).sum()))
        return int(np.argmax(counts))

    def run_downstream(self) -> None:
        plan, ds = self.plan, self.ds
        d = self.downstream_fold()
        fold, rec = self.folds[d], self.fits[(self.primary_name, d)]
        asm = self.baselines(d, self.config(self.primary_name))["asm"]
        res = next(f for f in self.test_folds if f["fold"] == fold.name)
        idx, labels, classes = fold.test, asm.test.labels, asm.classes
        spatial = ds.by_kind("spatial")
        xy = np.asarray(spatial[0].data, float)[idx] if spatial else None
        grp_col = self.within_col or self.prof.cv_group
        groups = ds.obs[grp_col].astype(str).to_numpy()[idx] if grp_col else np.array(["all"] * len(idx))
        a, b = self.pair[0], self.pair[-1]
        out: dict = {"fold": fold.name, "configuration": self.primary_name}
        with self.stage("downstream analyses"):
            if "retrieval_examples" in plan.downstream and len(self.pair) == 2:
                out["retrieval_examples"] = D.retrieval_examples(rec.emb["test"][a], rec.emb["test"][b], ds.ids[idx],
                                                                 labels, classes, 4, 5, self.cfg.seed)
            if "cross_modal_prediction" in plan.downstream and res["prediction"]:
                best = S.prediction_method(res["prediction"], self.primary_name)
                if best and best in rec.preds.get("test", {}):
                    out["prediction"] = {"method": best, "best_features": res["prediction"][best]["best_genes"]}
                    if xy is not None:
                        maps = D.prediction_maps(rec.preds["test"][best], asm.test.target, groups, xy,
                                                 asm.target_features, asm.target_variance_order)
                        pred = plan.task.prediction
                        maps.update(source=pred.source, target=pred.target, group_column=grp_col)
                        out["prediction_maps"] = maps
            if "classification" in plan.downstream and classes:
                view = next((m for m in self.pair if ds.modalities[m].kind == "image"), "fused")
                pr = res["probe"].get(self.primary_name, {}).get(view)
                if pr:
                    out["confusion"] = {**D.confusion(pr, classes), "view": view}
            if {"clustering", "spatial_maps"} & set(plan.downstream) and len(self.pair) == 2:
                z = l2n(l2n(rec.emb["test"][a]) + l2n(rec.emb["test"][b]))
                k_free = res.get("cluster_k", {}).get("k")           # the k chosen when there are no labels
                cm = D.cluster_map(z, labels, classes, groups, xy, k_free, self.cfg.seed,
                                   {a: rec.emb["test"][a], b: rec.emb["test"][b]})
                cm.update(spatial=xy is not None, group_column=grp_col)
                out["cluster_map"] = cm
        self.downstream_results = out
        self.downstream_index = d

    # -- outputs -------------------------------------------------------------------------------

    def write_outputs(self) -> dict:
        with self.stage("report"):
            result = report.write_all(self)
        self.say(f"done: {self.out}")
        return result


def run(data_path, objective: str, out_dir, **overrides) -> dict:
    """Run the agent. Overrides: config (YAML path), quick, no_llm, replay, label_column,
    group_column, device, seed, llm={...}; ``client`` injects an LLM client (tests)."""
    client = overrides.pop("client", None)
    config_path = overrides.pop("config", None)
    no_llm = overrides.pop("no_llm", False)
    cfg = load_config(config_path, **overrides)
    if no_llm:
        cfg.llm.enabled = False
    return Agent(data_path, objective, out_dir, cfg, client).execute()
