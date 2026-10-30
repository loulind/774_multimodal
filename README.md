# Multimodal representation-learning agent

An AI agent that takes a multimodal biomedical dataset and a free-text objective and decides **how** to learn a joint representation: which preprocessing, integration strategy, encoders and training objectives to use. It then trains within a compute budget, evaluates the representations against unimodal baselines, runs the downstream analyses the objective asks for, and writes a report.

BIOS 774, Project 3. Work in progress; full documentation comes with the final release.

## Install

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). A GPU is optional: CUDA or Apple MPS is detected automatically.

```bash
uv sync
uv run mmagent hardware
```
