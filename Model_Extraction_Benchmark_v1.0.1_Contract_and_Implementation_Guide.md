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

### Preprocessing Contract

- Victim query path:
  - The benchmark forwards attacker-produced query tensors directly to the victim runtime.
  - The benchmark does not add an extra victim-side normalization wrapper on the query path.
- Victim evaluation path:
  - Benchmark metrics are computed on the public task test set.
  - Test-set evaluation uses the dataset's official evaluation normalization.
- Surrogate / pool path:
  - Public surrogate samples are prepared with the surrogate dataset's official preprocessing or normalization for that attack path.
  - This preprocessing is attacker-side / data-pipeline-side, not an extra victim wrapper.
- Data-free path:
  - Data-free attacks keep their native attacker-side tensor scale.
  - The canonical benchmark convention is to preserve the attack's `[-1,1]`-style query scale unless the attack explicitly owns a different internal convention.
- Interpretation:
  - The benchmark uses path-specific preprocessing contracts.
  - Victim query preprocessing, victim test evaluation preprocessing, surrogate preprocessing, and data-free generation preprocessing are intentionally not collapsed into one universal transform.

### Runtime

- Canonical execution path: `engine -> attack.run(ctx) -> checkpoint/final eval -> artifact logging`
- Attacks own their native extraction loop
- BenchmarkContext owns budget metering and checkpoint notification
- ArtifactLogger owns canonical result persistence

### Reporting

- Canonical runtime track is `track_b`
- `summary.json` stores checkpoint metrics under `track_b`
- `metrics.csv` stores long-form rows with `track=track_b`
- New reports and aggregation scripts must use `track_b` directly

### Fair Comparison Policy

- Victim profile, substitute defaults, seeds, budgets, and output modes are unified by benchmark config generation and validation
- Pool-based attacks and data-free attacks may keep different native training loops, but they are compared within the same benchmark envelope and artifact contract

### Benchmark Policy vs. Paper Parity

- Matrix outputs are benchmark-policy results.
- Benchmark-policy results prioritize controlled cross-attack comparison under a shared runtime contract over paper-faithful reproduction of every original training stack.
- Shared benchmark controls may include setup-level substitute defaults, canonical budget accounting, shared seeds, and canonical reporting.
- Paper-parity claims must be evaluated separately through per-paper reproduction work and should not be inferred automatically from matrix benchmark results.

### SET Interpretation

- `SET-A1`, `SET-B1`, and `SET-C1` are benchmark cells, not a single absolute scale.
- The benchmark's primary fairness goal is within-set comparison.
- Cross-set comparisons require caution because victim domain, budget policy, and surrogate-pool limits differ across sets.
- A paper should treat cross-set results as contextual evidence unless it explicitly normalizes those differences.

### SET-C1 Fixed-Length Substitute Training

- `SET-C1` uses a setup-level fixed substitute profile across attacks: `batch=256`, `optimizer=sgd(lr=0.1,momentum=0.9,wd=5e-4)`, `scheduler=multistep([0.5,0.75],gamma=0.1)`, `max_epochs=90`.
- Substitute checkpoint selection is based on minimum validation loss.
- The run is fixed-length rather than early-stopped: generated configs set `patience=max_epochs`, which keeps the existing attack training paths running through all 90 epochs while still restoring the best validation-loss checkpoint at the end.

## Config Rules

- `victim.output_mode` and `attack.output_mode` must match
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

Older repository history may also contain legacy preprocessing or reproduction notes. When conflicts appear, follow the current path-specific preprocessing contract in this document and the active benchmark runtime.
