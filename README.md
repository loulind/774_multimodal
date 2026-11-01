# Multimodal representation-learning agent

An AI agent that takes a multimodal biomedical dataset and a free-text objective and decides **how** to learn a joint representation. In one command it:

1. inspects the modalities (image, text, expression matrix, table, spatial coordinates), how they are paired, and which labels, groups and missing modalities exist;
2. interprets the objective (retrieval, cross-modal prediction, classification, clustering);
3. chooses the preprocessing for each modality, the integration strategy, the encoders, the training objectives and the evaluation split;
4. screens candidate pretrained encoders on the data before committing to one;
5. trains or fine-tunes within a compute budget;
6. evaluates the representations against unimodal, fusion, linear (CCA, ridge), zero-shot and chance baselines;
7. runs the downstream analyses the objective asks for;
8. lets a critic order at most one revision, which is kept only if the validation metric improves;
9. writes a report.

Every decision is logged with its rationale, the evidence behind it and the alternatives it rejected (`decision_log.json`).

BIOS 774, Project 3.

```
load (format-generic) -> profile modalities -> parse objective -> screen encoders (frozen, subsample)
  -> plan (LLM | rules) -> validate / repair once / fall back to rules
  -> split (predefined | grouped CV | random) -> preprocess (fitted on training rows only)
  -> encode (cached frozen features, or partial fine-tuning) -> train heads (time and epoch caps)
  -> validation evaluation -> critic (<= 1 revision, kept only if validation improves)
  -> test evaluation (scored once) vs baselines -> downstream analyses
  -> report (Markdown + HTML), methods summary, embeddings, decision log
```

**Division of labour.** The LLM (Claude, through the local Claude Code CLI) acts only as planner, critic and report writer, and answers in JSON. Deterministic Python does the profiling, plan validation, training, evaluation and number checking. The planner sees the data profile without the dataset's name or path, so its choices rest on the measured properties. A rule-based planner, critic and writer cover the whole pipeline when no LLM is available.

**What the agent can choose.**
- Strategies: CLIP-style contrastive alignment, supervised representation learning (cross-entropy on the fused embedding of both modalities), self-supervised learning (one modality), and closed-form linear alignment (CCA). Early and late fusion are evaluated as baselines.
- Encoder regimes: frozen pretrained encoders with trainable projection heads, partial fine-tuning of the last blocks, and MLPs trained from scratch for matrices and tables.
- Objectives: InfoNCE (hard or soft targets, learned temperature), cross-entropy, cross-modal regression, reconstruction, masked denoising, and two dropout-noised views of the same features (SimCSE-style).
- Splits: the dataset's own split, grouped K-fold cross-validation over patients, slides or sources, or a stratified random split.

## Install

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). A GPU is optional: CUDA or Apple MPS is detected automatically, and time budgets are scaled to the device. Development and all runs used an Apple M2 Pro (MPS); the CUDA path has not been tested.

```bash
uv sync
uv run mmagent hardware
```

Without uv, the pinned requirements (which also install this package) work with pip in a Python 3.12 environment:

```bash
pip install -r requirements.txt
```

**LLM (optional).** The planner, critic and report writer call Claude through the Claude Code CLI and its login; there is no API key. The CLI comes with the `claude-agent-sdk` dependency, and the login is shared with any [Claude Code](https://claude.com/claude-code) installation: run `claude auth login` once (`claude auth status` should then show `loggedIn: true`). Without a login, or with `--no-llm`, the rule-based planner, critic and writer run instead. If an LLM call fails, the run continues with the rules and says so in the decision log.

## Data

```bash
uv run python scripts/prepare_data.py                 # both datasets, full and quick subsets
uv run python scripts/prepare_data.py --only quick    # quick subsets only
```

The script downloads into `data/` (gitignored), checks free disk space before each download and checksums after, and converts each dataset to one of the generic formats the agent reads. It is the only file in the repository that names a dataset.

| dataset | subset used | download | kept on disk |
|---|---|---|---|
| [STimage-1K4M](https://huggingface.co/datasets/jiawennnn/STimage-1K4M) (histology + spatial transcriptomics; MIT) | HER2-positive breast cancer cohort: 36 slides from 8 patients, 13,620 spots; 8 slides carry pathologist tissue labels | 0.76 GB (dataset revision pinned) | 0.84 GB (slide images are read in place) |
| [ROCOv2 v2.0.1](https://zenodo.org/records/10821435) (radiology images + captions; CC BY-NC 4.0) | 20,000 training images (stratified by imaging modality) and 2,000 / 5,000 images from the official validation / test splits | 6.4 GB of image archives, deleted after extraction | 0.76 GB (images stored with a 256-px short side) |
| quick subsets | 4 labelled slides from 4 patients (1,123 spots); 1,000 image–caption pairs (600 / 150 / 250 from the three splits) | 0.23 GB: the 4 slides, the caption tables, and the 1,000 images read directly out of the remote archives (about 3 minutes) | 0.15 GB |

The assignment links ROCOv2 v2.0 (Zenodo record 8333645), whose page asks users not to use that version; v2.0.1 replaces it.

## Quick smoke test

About a minute per dataset, with small encoders, at most 1,000 samples, one fold and at most 5 epochs per fit:

```bash
uv run python scripts/prepare_data.py --only quick
uv run mmagent run --quick --no-llm --data data/stimage_her2_quick --out runs/quick/her2 --objective "Learn joint embeddings of tissue image patches and spot-level gene expression to enable cross-modal retrieval and prediction of gene expression from histology."
uv run mmagent run --quick --no-llm --data data/rocov2_quick --out runs/quick/rocov2 --objective "Learn a joint image–caption embedding for image–text retrieval and imaging-modality classification."
```

Each run ends by printing the path of its `report.html`. The first quick run also downloads the three small encoders (about 0.2 GB). Drop `--no-llm` to let Claude plan, critique and write the report; that costs about $0.6 per quick run.

## Run: one command per dataset

```bash
# generated_report_1: histology + spatial transcriptomics
uv run mmagent run --data data/stimage_her2 --out outputs/generated_report_1 \
  --objective "Learn joint embeddings of tissue image patches and spot-level gene expression to enable cross-modal retrieval and prediction of gene expression from histology."

# generated_report_2: radiology images + captions
uv run mmagent run --data data/rocov2 --out outputs/generated_report_2 \
  --objective "Learn a joint image–caption embedding for image–text retrieval and imaging-modality classification."
```

On the M2 Pro, the committed runs took 377 s and 1,564 s, with the frozen-encoder features already cached from earlier runs. The first runs, which also extracted those features, took 761 s and 2,053 s. Most of the second dataset's time is a partial fine-tuning comparison. The three LLM calls cost $0.69 for the first dataset and $0.71 for the second. The first use of each pretrained encoder also downloads its weights (about 4.8 GB for all of them, into `~/.cache/huggingface`).

A fresh run makes new LLM calls, and their answers can differ from the committed ones. To repeat a committed run's decisions without calling a model, replay its LLM transcript:

```bash
uv run mmagent run --data data/stimage_her2 --out runs/replay_1 --replay outputs/generated_report_1 \
  --objective "Learn joint embeddings of tissue image patches and spot-level gene expression to enable cross-modal retrieval and prediction of gene expression from histology."
```

Options for `mmagent run`:

| flag | effect |
|---|---|
| `--quick` | smoke-test budgets (see `uv run mmagent hardware`) and small encoders |
| `--no-llm` | rule-based planner, critic and writer; no Claude login needed |
| `--replay PATH` | reuse a previous run's LLM answers: a run directory, its `llm_cache/` or its `llm_transcript.json` |
| `--config my.yaml` | budgets, seed and LLM settings (see `configs/default.yaml`); never model choices |
| `--label-column`, `--group-column` | override the detected label column, or the group column (patient, slide, source) used for splitting and batch checks |
| `--device`, `--seed`, `--model` | `auto` / `cuda` / `mps` / `cpu`; random seed (default 774); Claude model id (default `claude-opus-5-5`) |

Other commands:

| command | what it does |
|---|---|
| `mmagent hardware` | detected device, memory, free disk, and the full and quick budgets |
| `mmagent profile --data PATH [--out profile.json]` | load a dataset and print its modality profile |
| `mmagent plan --data PATH --objective "..." [--quick] [--no-llm] [--no-screen] [--out DIR]` | dry run: profile, objective, encoder screening and plan, without training |
| `mmagent schema` | rewrite `schemas/plan.schema.json` from the plan's pydantic model |

**From Python.** `run_agent` is the entry point for other code (Project 4 reuses it):

```python
from multimodal_agent import run_agent

result = run_agent("data/rocov2_quick", "Learn a joint image–caption embedding for image–text retrieval.",
                   "runs/quick/api", quick=True, no_llm=True)
result["primary_metric"], result["test_value"], result["out_dir"]
```

## Using your own data

`--data` takes a file, or a directory holding one. Modalities, pairing and column roles (ID, group, split, label, image path, free text, numeric, categorical) are inferred from the format and the contents:

| layout | modalities |
|---|---|
| spatial AnnData: `.h5ad` with `obsm['spatial']` pixel coordinates and, per slide in `uns['spatial']`, a source-image path and spot diameter | image patches cut around each spot + expression matrix + spatial coordinates |
| AnnData `.h5ad` without coordinates; an `obs` column of image paths adds images | matrix (+ image) |
| table (`.csv`, `.tsv`, `.parquet`) with an image-path column | image + free text and/or a table of numeric and categorical columns |
| 10x Genomics `.h5` with several feature types | one matrix per feature type |
| a directory of `.h5ad` files whose rows share IDs | paired matrices |

An optional `dataset.yaml` in the directory names the file(s) and overrides roles:

```yaml
file: cohort.csv          # or files: [a.h5ad, b.h5ad]
roles: {label: diagnosis, group: patient}
```

## How the agent decides

The rules below are the rule-based planner's defaults. The LLM planner receives them as a proposal, together with their reasons, and may change any field within the validator's allowed values and ranges. Each threshold is a named constant in `planner/heuristic.py`.

| decision | rule | evidence used |
|---|---|---|
| split | the dataset's split column if present; else grouped K-fold CV over a patient/slide/source column (K = min(#groups, 4); every group tested once; each fold holds out one more group for validation, or several if one holds less than 10% of the remaining samples); else a stratified random split | split and group columns; group sizes; label coverage per group |
| strategy | one usable modality: self-supervised; < 500 training pairs: CCA on frozen features; the objective needs a shared space (retrieval, prediction, clustering): CLIP-style contrastive alignment; classification only: supervised fusion | pairs, tasks |
| encoders | candidates matched to the measured image domain (stained tissue, greyscale imaging, other) plus generic encoders; each screened (frozen) on 2,000 training / 500 validation pairs by CCA recall@10, zero-shot recall@10 (pre-aligned pairs), ridge R² and probe macro-F1; the primary task's score picks. Matrices and tables get an MLP | screening table |
| frozen or fine-tuned | frozen backbone + small heads by default; partial fine-tuning of the last 2 blocks as a comparison when a pre-aligned dual encoder exists, there are ≥ 10,000 training pairs and ≥ 2 epochs fit the time budget; the primary method at ≥ 50,000 pairs | pairs, throughput, budget |
| heads | residual heads that start as the identity (CLIP-Adapter style) when both encoders form a pre-aligned pair, so training starts from the pretrained alignment and keeps it unless validation improves | encoder pairing |
| losses | symmetric InfoNCE with a learned temperature; soft targets when ≥ 10% of samples have a near-identical twin; + cross-entropy (weight 0.5) when classification is asked for and ≥ 10% of pairs are labelled; + cross-modal regression (weight 1.0) when prediction is asked for | redundancy, label coverage, tasks |
| image preprocessing | patch = spot footprint (with a 2× context comparison); Reinhard colour normalisation when ≥ 20% of patch-colour variance lies between groups; greyscale replicated to 3 channels; resized to the encoder's input | spot geometry, colour ICC, greyscale share |
| matrix, text, table preprocessing | expression: library-size normalisation, log1p, 2,000 most variable genes, PCA when there are < 5 training samples per feature; text: Unicode and whitespace cleaning, truncation from the caption-length distribution; tables: median imputation with missingness indicators, one-hot coding with rare levels pooled, standardisation. All fitted on training rows only | sparsity, n/p, token lengths, column types |
| comparisons | up to 2 extra configurations, each changing one decision (loss, patch context, fine-tuning, supervision) | the decisions most open to doubt |
| primary metric | retrieval: mean recall@10 over both directions; prediction: mean Pearson r over the 50 most variable target features; classification: linear-probe macro-F1; clustering: ARI | the objective's first measurable task |
| critic | triggers: no gain over the best unimodal baseline, CCA or ridge beats the trained model, over- or underfitting, embedding collapse, batch confounding, a comparison did better. At most one revision, kept only if the mean validation metric improves; the test data are scored once, at the end | validation results, training curves |

## Outputs (per run)

| file | contents |
|---|---|
| `report.md`, `report.html` | final analysis report: data, decisions and reasons, results against baselines, critic and revision, limitations, number check |
| `methods_summary.md` | the selected methods and preprocessing steps |
| `decision_log.json` | every decision: step, choice, rationale, rejected alternatives, evidence, source (heuristic / llm / validator / critic / system) |
| `profile.json`, `screening.json` | the data profile; encoder screening scores |
| `plan.json`, `plan.md` | the validated plan that ran (JSON schema in `schemas/plan.schema.json`) |
| `evaluation.json`, `tables/*.csv` | validation and test metrics per fold and configuration, baselines, comparisons with uncertainty, flags |
| `downstream/*.json`, `figures/` | retrieval examples, predicted-vs-observed maps, confusion matrix, cluster maps; comparison, training-curve and embedding figures |
| `embeddings/held_out_embeddings.npz` (+ `about.json`) | learned held-out embeddings of each modality, float16, capped at 5 MB |
| `model_summary.json` | configurations, parameter counts, training histories |
| `run_manifest.json` | command, configuration, hardware, package versions, git commit, timings, LLM calls and cost |
| `llm_transcript.json` | every LLM prompt and reply (used by `--replay`) |

Each figure and table is described in the report. Scratch files, feature caches and per-run LLM caches (`runs/`, `llm_cache/`) are gitignored.

## The two generated reports

| | `generated_report_1` (STimage-1K4M) | `generated_report_2` (ROCOv2) |
|---|---|---|
| split | grouped 4-fold CV by patient (8 patients) | predefined: 20,000 / 1,999 / 5,000 |
| encoders | frozen Phikon (best of 4 screened) + expression MLP on 100 PCs of 2,000 genes | frozen BiomedCLIP image and text towers, residual 512-d heads |
| objective terms | soft InfoNCE + expression regression; the critic switched to the hard-InfoNCE comparison (better in 4 of 4 validation folds) | InfoNCE + modality cross-entropy (weight 0.5); comparisons: partial fine-tuning, contrastive only |
| retrieval (test) | R@10 0.024 (±1 SD 0.003 over folds) vs CCA 0.012, chance 0.003 | R@10 0.790 (95% bootstrap 0.780–0.800), equal to zero-shot; +0.152 over CCA |
| other tasks | expression prediction: mean r 0.170 vs ridge 0.180 (no gain) | modality probe macro-F1 0.938: +0.017 over image only, tied with early and late fusion |

## Tests

```bash
uv run pytest -q
```

The suite takes about a minute on CPU and runs offline (fake encoders and a fake LLM backend). It covers:
- known-answer tests for the losses and metrics;
- the loaders on synthetic spatial AnnData, image tables with text or clinical columns, and paired matrices;
- planner and validator guardrails;
- LLM fallback, repair and replay paths;
- critic triggers;
- end-to-end runs.

It also includes a syntax-tree check that no dataset name appears in the package's code, configs or prompts.

## Reproducibility notes

- **Seeds and splits.** All seeds are fixed (`seed: 774`); splits and subsamples are exactly reproducible.
- **Training.** The projection heads train on the CPU (`heads_device: cpu`), which is bit-for-bit deterministic. The GPU is used for feature extraction and fine-tuning; GPU kernels (MPS especially) are not guaranteed to be deterministic, so those results may differ in the last digits on other machines or library versions. Frozen features are cached as float16, rounded the same way on the first run and on reruns.
- **LLM.** Model answers are not deterministic across fresh runs. Each run stores every prompt and reply, and `--replay` repeats that run's decisions exactly. Replaying both committed reports with the current code reproduced their plans, decisions, training histories, evaluation tables and embeddings exactly, including the GPU fine-tuning comparison. The only differences came from two report fixes made after those runs: report 2's comparison figure now zooms to the range of the data, and its number check no longer flags the "50" in "ResNet-50".
- **Budgets.** Fits stop at early stopping, an epoch cap or a time cap. When a time cap binds, a slower machine trains fewer epochs and the results differ.
- **Pinned versions.** `pyproject.toml` with `uv.lock`, and an exported `requirements.txt`. `run_manifest.json` records the versions, the hardware and the git commit of each run.

## Pretrained models

Only openly downloadable weights are used. Models that need an account or accepting terms (UNI, CONCH, Virchow, Prov-GigaPath, H-optimus, Hibou) are excluded. PubMedCLIP is excluded because it was trained on ROCO, whose images overlap ROCOv2.

| encoder | weights | pretraining | licence |
|---|---|---|---|
| Phikon | `owkin/phikon` | iBOT self-supervision on 40M histology tiles | Owkin non-commercial |
| PLIP (image, text) | `vinid/plip` | CLIP fine-tuned on OpenPath pathology image–text pairs | not stated (research use) |
| BiomedCLIP (image, text) | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | PMC-15M figure–caption pairs from PubMed Central | MIT |
| PubMedBERT | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` | masked language modelling on PubMed | MIT |
| CLIP ViT-B/16 (image, text) | `openai/clip-vit-base-patch16` | 400M web image–text pairs | MIT |
| DINOv2 base / small* | `facebook/dinov2-base`, `facebook/dinov2-small` | self-supervision on LVD-142M images | Apache-2.0 |
| ResNet-50 / ResNet-18* | `timm/resnet50.a1_in1k`, `timm/resnet18.a1_in1k` | supervised ImageNet-1k | Apache-2.0 |
| MiniLM* | `sentence-transformers/all-MiniLM-L6-v2` | contrastive training on 1B sentence pairs | Apache-2.0 |

\* small encoders used in `--quick` mode.

## Known limitations

- **Possible pretraining overlap.** BiomedCLIP's pretraining pairs and ROCOv2 both come from PubMed Central, so ROCOv2's absolute scores may be optimistic. The agent flags any zero-shot recall@10 ≥ 0.9 as possible overlap.
- **Shared articles across ROCOv2 splits.** 18.7% of the test images come from articles that also supplied validation images, which can make validation-based selection slightly optimistic.
- **Training added nothing on ROCOv2.** Every trained configuration, partial fine-tuning included, kept the residual heads' starting point (epoch 0), so the learned space is the pretrained BiomedCLIP space.
- **Strong patient effects.** In the spatial data, held-out embeddings group more by patient (silhouette 0.38) than by tissue label (0.05). With 8 patients, results vary considerably from fold to fold.
- **Self-supervised pretraining** of a modality with many unpaired samples is recorded as an option in the decision log, but not run.

## Layout

```
multimodal_agent/
  __init__.py      run_agent()
  cli.py           mmagent command: run, plan, profile, hardware, schema
  orchestrator.py  the agent loop
  config.py        budgets, seeds and LLM settings
  hardware.py      device detection, seeding
  data/            format-generic loaders, column roles, image sources, splits
  profiler.py      modality profile: pairing, labels, groups, sparsity, spatial autocorrelation, colour, redundancy
  objective.py     free-text objective -> tasks, modality pair, label column
  schema.py        profile, task, plan, decision and critique models (pydantic)
  planner/         heuristic.py (rules), llm.py (LLM + repair + fallback), validator.py (guardrails), flow.py
  encoders/        registry.py (open-weight encoders), pretrained.py (wrappers, feature cache), screening.py
  preprocess/      image.py, matrix.py, text.py, tabular.py
  models/          heads, joint model, losses, training loop, partial fine-tuning, CCA, input assembly
  evaluate/        retrieval, prediction, probes and clustering, geometry, suite (baselines, comparisons, flags)
  downstream.py    retrieval examples, prediction maps, confusion matrix, cluster maps
  critic.py        triggers + at most one revision
  writer.py        LLM report sections + number check
  report.py        report, tables, manifest, embedding export
  figures.py       figures
  llm.py           Claude CLI backend, cache, replay, spend cap
  decision_log.py  decision log
prompts/           planner.md, critic.md, writer.md
schemas/           plan.schema.json
configs/           default.yaml
scripts/           prepare_data.py
tests/
outputs/           generated_report_1, generated_report_2
```

## Data sources

- STimage-1K4M (Chen et al., 2024), MIT licence: https://huggingface.co/datasets/jiawennnn/STimage-1K4M. The HER2-positive breast cancer cohort in it was first published by Andersson et al. (2021).
- ROCOv2 v2.0.1 (Rückert et al., 2024), CC BY-NC 4.0: https://zenodo.org/records/10821435 (DOI 10.5281/zenodo.10821435).

Please cite the datasets as their pages request.
