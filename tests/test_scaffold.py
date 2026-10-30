"""Scaffold: device detection, budgets and configuration."""

from multimodal_agent import hardware
from multimodal_agent.cli import main
from multimodal_agent.config import load_config


def test_detect_returns_known_device():
    info = hardware.detect()
    assert info.device in {"cuda", "mps", "cpu"}
    assert hardware.detect("cpu").device == "cpu"
    assert info.memory_gb is None or info.memory_gb > 0


def test_quick_budget_is_smaller_and_cpu_is_scaled():
    full = load_config().budget("mps")
    quick = load_config(quick=True).budget("mps")
    cpu = load_config().budget("cpu")
    assert quick.train_seconds_per_fit < full.train_seconds_per_fit
    assert quick.max_epochs <= 5 and quick.max_folds == 1 and quick.max_samples
    assert cpu.train_seconds_per_fit < full.train_seconds_per_fit
    assert cpu.total_train_seconds < full.total_train_seconds


def test_config_overrides_and_llm_defaults():
    cfg = load_config(seed=1, llm={"enabled": False})
    assert cfg.seed == 1 and cfg.llm.enabled is False
    default = load_config().llm
    assert default.model and default.max_output_tokens <= 16_000
    assert set(default.effort) == {"planner", "critic", "writer"}


def test_config_holds_no_modeling_choices():
    """Budgets, seeds and LLM settings only (encoders/losses come from the profile)."""
    keys = set(load_config().model_dump())
    banned = {"encoder", "encoders", "loss", "objective", "fusion", "strategy", "model_name"}
    assert not keys & banned


def test_hardware_command_prints_json(capsys):
    main(["hardware"])
    out = capsys.readouterr().out
    assert '"device"' in out and '"full_budget"' in out
