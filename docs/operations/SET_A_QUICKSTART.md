# SET-A Quickstart (Fast Execution Playbook)

This runbook is optimized for quickly executing SET-A while preserving benchmark contracts.

## 1) Preflight

### Environment

```bash
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Success criteria:
- Python and PyTorch are importable.
- CUDA availability matches intended device plan.

### Disk and Runtime Paths

- Ensure enough free space for intermediate run artifacts (`runs/`, temporary checkpoints, logs).
- Confirm excluded runtime paths remain out of source control (`runs/`, `checkpoints/`, `results/`, `reports/`).

### Data and Victim Requirements

- SET-A surrogate source: ImageNet subset 100k.
- SET-A preprocessing policy: grayscale + resize to `28x28` (MNIST victim profile alignment).
- Verify victim checkpoint path exists and is readable by config.

## 2) Generate and Validate Configs

```bash
python generate_configs.py
```

Then verify SET-A targets are generated and usable.

Recommended checks:
- output mode compatibility (`victim.output_mode` == `attack.output_mode`)
- budget values are expected for stage (`20k` pool, `20m` data-free for Stage-1)
- data scale contract is consistent by attack family (`[0,1]` pool-based, `[-1,1]` data-free)

## 3) Smoke Run (Fast Failure Detection)

Run one reduced-budget SET-A config first (single seed).

Example:
```bash
python -m mebench run --config configs/smoke/dfme_mnist_200.yaml --device cuda:0
```

Smoke must produce:
- `run_config.yaml`
- `summary.json`
- `metrics.csv`

Smoke acceptance criteria:
- run completes without config/asset failure
- budget accounting is monotonic and not above `budget.max_budget`
- no NaN propagation in key metrics

## 4) Stage-1 Main Run (SET-A)

Stage-1 plan:
- Pool attacks: budget `20k`, seeds `0 -> 1 -> 2`
- Data-free attacks: budget `20m`, seeds `0 -> 1 -> 2`

Execution policy:
- sequential seed order for reproducibility and easier triage
- keep skip-if-exists behavior for resumable runs
- preserve per-seed logs for failure recovery

## 5) Throughput Tuning (Safe)

Use only non-contract-breaking optimizations:
- dataloader workers/pin-memory tuning
- evaluation cadence tuning where protocol allows
- temporary caching only under run-scoped location and cleanup after run

Stop conditions:
- deterministic repeated failure signatures
- invalid budget accounting
- persistent NaN divergence after one controlled retry

## 6) Aggregate and Validate

```bash
python aggregate_matrix.py
```

Post-run checks:
- seed completeness (0/1/2 all present)
- checkpoint completeness for each attack
- no duplicate seed-checkpoint rows
- contract checks:
  - budget = image count
  - victim inference in eval/no_grad path
  - Track A reproducibility assumptions respected

Deliverables:
- aggregated CSV/Markdown/LaTeX table
- run notes with failures, retries, and exclusions

## 7) Failure Triage Quick Map

- Missing surrogate path -> fix `dataset.surrogate_root` or environment variable.
- Missing victim checkpoint -> verify path in config and filesystem.
- Output mode mismatch -> align victim/attack output modes.
- Over/under budget consumption -> inspect attack loop budget accounting.
- OOM or severe slowdown -> reduce worker count and run single-seed sequentially.
