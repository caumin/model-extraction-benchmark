# Model Extraction Benchmark

PyTorch benchmark for comparing model extraction attacks under a unified experiment envelope.

[한국어](README-ko.md) • [Contract Guide](Model_Extraction_Benchmark_v1.0.1_Contract_and_Implementation_Guide.md)

## Overview

- Goal: compare attacks against each other, not preserve a legacy two-track evaluation protocol.
- Official protocol: `Track B` only.
- Runtime contract: `engine -> attack.run(ctx) -> checkpoint/final eval -> artifact logging`.
- Budget contract: `1 query = 1 image`.
- Oracle contract: victim runs in `eval()` and `torch.no_grad()` with `temperature=1.0` by default.

## What "Track B only" means

- Every attack is evaluated through its native benchmark runtime.
- The benchmark still unifies victim setup, substitute defaults, budgets, seeds, and reporting.
- The benchmark no longer performs a separate from-scratch retraining protocol at each checkpoint.
- Artifacts and reports are canonicalized around `track_b`.

## Quick Start

```bash
pip install -e ".[dev]"

# smoke
bash scripts/launch/run_smoke.sh cuda:0

# generate configs
IMAGENET_ROOT=D:/imagenet python generate_configs.py

# run matrix
IMAGENET_ROOT=D:/imagenet bash scripts/launch/run_matrix.sh

# aggregate
python aggregate_matrix.py
```

## Core Rules

1. `1 query = 1 image`.
2. Victim and attack `output_mode` must match.
3. Default oracle uses `T=1.0`.
4. Pool-based attacks use the shared benchmark substitute defaults defined by configs.
5. Data-free attacks keep their native loop and are evaluated through the same artifact/reporting contract.

## Artifacts

Per seed run directory:

```text
runs/<run_name>/<timestamp>/seed_<seed>/
  run_config.yaml
  summary.json
  metrics.csv
  metrics_history.csv
```

- `summary.json`: checkpoint metrics keyed by budget, recorded under `track_b`
- `metrics.csv`: long-form rows with `track=track_b`
- `run_config.yaml`: exact config used for the run

## Supported Attack Families

- Pool-based: `random`, `activethief`, `marich`, `knockoff_nets`, `cloudleak`, `copycatcnn`, `inversenet`, `blackbox_dissector`, `swiftthief`
- Data-free / generative: `dfme`, `ds`, `dfms`, `disguide`, `maze`, `es`, `game`, `blackbox_ripper`

## Methodology

See `Model_Extraction_Benchmark_v1.0.1_Contract_and_Implementation_Guide.md` for the current benchmark contract.

## Contributing

- Add new attacks under `mebench/attackers/`
- Keep runtime compatible with `AttackRunner.run(ctx)`
- Run `python -m pytest tests/ -q`

## License

MIT. See `LICENSE`.
