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

## Preprocessing Contract

- Victim query path: attacker-produced query tensors are forwarded to the victim without an extra benchmark-side normalization wrapper.
- Victim evaluation path: benchmark metrics are computed on the task's public test set using the dataset's official evaluation normalization.
- Surrogate / pool path: attacker-side samples from public surrogate datasets use the surrogate dataset's official preprocessing or normalization for that attack path.
- Data-free path: attacker-side tensors stay in their native data-free scale (typically `[-1,1]`) unless an attack explicitly owns a different internal convention.
- Benchmark rule: preprocessing policy is path-specific. The benchmark does not force one shared normalization transform across victim query, victim evaluation, surrogate training, and data-free generation.

## Benchmark Policy vs. Paper Parity

- Matrix results are benchmark-policy comparisons: attacks run under a shared runtime contract, shared reporting, shared seeds, and setup-level substitute defaults.
- Benchmark-policy results are designed for controlled cross-attack comparison, not as a claim that every attack is reproduced under its original paper training stack.
- Paper-parity reproduction remains a separate objective under `repro/` and should be reported separately when claiming closeness to original paper numbers.

## Interpreting SET-A/B/C

- `SET-A1`, `SET-B1`, and `SET-C1` are benchmark cells with different victim/task regimes.
- Within-set comparisons are the primary fairness target: attacks in the same set share the same victim family, budget policy, output mode constraints, and reporting contract.
- Cross-set results should not be interpreted as a single absolute leaderboard because budgets, victim domains, and surrogate pool caps differ by set.
- Use SET-level rankings and trends for headline benchmark conclusions; treat cross-set comparisons as contextual rather than directly commensurate.

## SET-C1 Substitute Policy

- `SET-C1` uses a fixed substitute-training schedule for all attacks: `batch=256`, `optimizer=sgd(lr=0.1,momentum=0.9,wd=5e-4)`, `scheduler=multistep([0.5,0.75],gamma=0.1)`, `max_epochs=90`.
- Model selection is by best validation loss.
- Early stopping is disabled in practice by setting `patience=max_epochs`, so runs always complete the full 90 epochs before restoring the best validation-loss checkpoint.

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
