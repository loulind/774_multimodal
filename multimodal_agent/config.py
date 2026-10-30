"""Run configuration: budgets, seeds and LLM settings -- never modeling decisions.

A config file may say *how much* compute a run gets, *which* LLM to ask and
*what* seed to use. It may not say which encoder, loss, fusion or split to use;
those come only from the data profile and the objective. Keeping the two apart
is what stops a per-dataset config from becoming a dataset-specific branch in
disguise (adapted from Project 2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

# Multiplier on every time budget, relative to the reference machine (an M2 Pro
# on MPS). A CPU-only run gets less wall-clock per fit so a full run still
# finishes in about an hour; it simply trains for fewer epochs.
DEVICE_BUDGET_SCALE = {"mps": 1.0, "cuda": 1.0, "cpu": 0.5}


class Budget(BaseModel):
    """Compute caps for one mode (full or quick)."""
    train_seconds_per_fit: float = 180     # one configuration on one fold (heads on cached features)
    finetune_seconds_per_fit: float = 1200  # one partial fine-tuning fit
    total_train_seconds: float = 2700      # all fits in a run, including the critic's revision
    max_epochs: int = 300
    max_folds: int = 4                     # cap on grouped cross-validation folds
    max_samples: int | None = None         # subsample cap on samples used at all (quick mode)
    screening_train: int = 2000            # samples used to screen candidate encoders
    screening_val: int = 500
    max_candidates: int = 4                # candidate encoders screened per modality
    eval_max_samples: int = 5000           # rows used by O(n^2) metrics
    feature_batch_size: int = 64
    device: str = "mps"                    # set by RunConfig.budget(); planners read it


class LLMConfig(BaseModel):
    enabled: bool = True
    # Checked against the Models API at startup and against the id each response
    # reports; see llm.py.
    model: str = "claude-opus-5-5"
    max_output_tokens: int = 16_000        # hard cap per call
    effort: dict[str, str] = Field(default_factory=lambda: {
        "planner": "high", "critic": "high", "writer": "medium"})
    max_budget_usd: float = 3.0            # per run; further calls fall back to rules
    refusal_fallback: bool = True          # server-side fallback model on a policy refusal
    timeout_seconds: float = 600
    max_retries: int = 2
    # USD per million tokens, used only to estimate spend against max_budget_usd.
    price_per_mtok: dict[str, float] = Field(default_factory=lambda: {
        "input": 4.0, "output": 20.0, "cache_write": 5.0, "cache_read": 0.2})


class RunConfig(BaseModel):
    seed: int = 774
    device: str = "auto"                   # auto | cuda | mps | cpu
    quick: bool = False
    replay: str | None = None              # directory of cached LLM responses to replay
    runs_dir: str = "runs"
    cache_dir: str = "runs/cache"          # frozen-feature and patch caches, shared across runs
    # Column roles, when automatic detection should be overridden. These describe
    # the data's format, not how to model it.
    label_column: str | None = None
    group_column: str | None = None
    max_critic_revisions: int = 1
    full: Budget = Field(default_factory=Budget)
    quick_budget: Budget = Field(default_factory=lambda: Budget(
        train_seconds_per_fit=30, finetune_seconds_per_fit=60, total_train_seconds=240,
        max_epochs=5, max_folds=1, max_samples=1_000, screening_train=300, screening_val=100,
        max_candidates=2, eval_max_samples=1_000, feature_batch_size=32,
    ))
    llm: LLMConfig = Field(default_factory=LLMConfig)

    def budget(self, device: str) -> Budget:
        """The active budget, scaled for the detected device."""
        base = self.quick_budget if self.quick else self.full
        scale = DEVICE_BUDGET_SCALE.get(device, 1.0)
        return base.model_copy(update={
            "device": device,
            "train_seconds_per_fit": base.train_seconds_per_fit * scale,
            "finetune_seconds_per_fit": base.finetune_seconds_per_fit * scale,
            "total_train_seconds": base.total_train_seconds * scale,
        })


def _deep_update(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, **overrides: Any) -> RunConfig:
    """Load default.yaml, layer an optional file on top, then keyword overrides."""
    data: dict = {}
    if DEFAULT_CONFIG.exists():
        data = yaml.safe_load(DEFAULT_CONFIG.read_text()) or {}
    if path is not None:
        data = _deep_update(data, yaml.safe_load(Path(path).read_text()) or {})
    data = _deep_update(data, {k: v for k, v in overrides.items() if v is not None})
    return RunConfig(**data)
