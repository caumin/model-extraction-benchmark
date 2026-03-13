# AGENTS.md — Model Extraction Benchmark

## Project Overview

This repository benchmarks model extraction attacks under a unified `Track-B-only` runtime contract.

- Goal: compare attacks against each other under shared budgets, seeds, substitute defaults, and victim settings.
- Non-goal: preserve a second from-scratch retraining protocol for checkpoint evaluation.

## Runtime Contract

- Canonical path: `mebench/core/engine.py -> attack.run(ctx) -> checkpoint/final eval -> artifact logging`
- `1 query = 1 image`
- Victim inference must use `eval()` and `torch.no_grad()`
- Artifact outputs are canonicalized around `track_b`

## Quick Commands

```bash
pip install -e ".[dev]"
python -m mebench run --config configs/experiment.yaml
python -m pytest tests/ -q
ruff check mebench/
ruff format mebench/
```

## Core Files

- `mebench/core/engine.py` — experiment orchestration
- `mebench/core/validate.py` — contract validation
- `mebench/core/logging.py` — run artifacts
- `mebench/attackers/runner.py` — attack interface and shared evaluation hook
- `generate_configs.py` — matrix config generation
- `aggregate_matrix.py` — canonical report aggregation

## Artifact Contract

Per seed run directory:

```text
runs/<run_name>/<timestamp>/seed_<seed>/
  run_config.yaml
  summary.json
  metrics.csv
  metrics_history.csv
```

- `summary.json`: checkpoint metrics keyed under `track_b`
- `metrics.csv`: long-form results with `track=track_b`

## Validation Expectations

- Pool-based attacks must provide substitute training controls like `max_epochs` and `patience`
- Data-free attacks must use `dataset.data_mode: data_free`
- `substitute.trackA` is invalid
- `victim.output_mode` and `attack.output_mode` must match

## Testing Priorities

1. budget accounting
2. config validation
3. track-b checkpoint evaluation
4. artifact schema
5. attack smoke/regression tests
