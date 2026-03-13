# Model Extraction Benchmark Contract and Implementation Guide

## Current Contract

This repository uses a `Track-B-only` benchmark contract.

The benchmark goal is to compare model extraction attacks under a unified experimental envelope, not to preserve a second from-scratch retraining protocol.

## Core Guarantees

### Budget

- `1 query = 1 image`
- Budget accounting is image-count only
- Checkpoints are expressed in consumed query images

### Oracle

- Default output mode is `soft_prob`
- Default temperature is `T=1.0`
- Victim inference runs in `eval()` and `torch.no_grad()`
- Query tensors are forwarded according to the benchmark input contract for the configured attack/data mode

### Runtime

- Canonical execution path: `engine -> attack.run(ctx) -> checkpoint/final eval -> artifact logging`
- Attacks own their native extraction loop
- BenchmarkContext owns budget metering and checkpoint notification
- ArtifactLogger owns canonical result persistence

### Reporting

- Canonical runtime track is `track_b`
- `summary.json` stores checkpoint metrics under `track_b`
- `metrics.csv` stores long-form rows with `track=track_b`
- New reports and aggregation scripts must not rely on `track_a`

### Fair Comparison Policy

- Victim profile, substitute defaults, seeds, budgets, and output modes are unified by benchmark config generation and validation
- Pool-based attacks and data-free attacks may keep different native training loops, but they are compared within the same benchmark envelope and artifact contract

## Config Rules

- `victim.output_mode` and `attack.output_mode` must match
- `substitute.trackA` is invalid and rejected
- Pool-based attacks must define substitute training controls such as `max_epochs` and `patience`
- Data-free attacks must use `dataset.data_mode: data_free`

## Required Artifacts

Per seed run directory:

```text
runs/<run_name>/<timestamp>/seed_<seed>/
  run_config.yaml
  summary.json
  metrics.csv
  metrics_history.csv
```

## Recommended Verification

1. `python -m pytest tests/test_contract_validation.py -q`
2. `python -m pytest tests/test_track_b.py tests/test_data_free_checkpoint_eval.py tests/test_artifact_schema.py -q`
3. Run one pool-based config and one data-free config through `python -m mebench run --config ...`
4. Confirm artifacts only contain `track_b`

## Legacy Note

Older repository history may still contain references to a prior two-track protocol. Those references are historical and should not be used for new benchmark work.
