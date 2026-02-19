# Reproduction Set and Pipeline

This folder is optimized for **low-resource local environments** where strict paper-level full runs may be expensive or blocked by missing datasets.

## Goals

- Keep per-paper reproduction artifacts in a standard layout.
- Make runs reproducible with staged commands.
- Support quick `smoke` runs first, then `full` runs when hardware/data are available.

## Layout

Each paper uses this structure under `repro/papers/<paper_id>/`:

- `paper_meta.yaml`
- `extracted_spec.yaml`
- `evidence.md`
- `mapping.md`
- `configs/` (`victim_train.yaml`, `victim_eval.yaml`, `attack.yaml`, `experiment.yaml`)
- `logs/`, `checkpoints/`, `results/`

## Pipeline CLI

Use `repro/run_experiment.py`.

Priority queue runner: `repro/run_priority_queue.py` (DFME -> MAZE -> DFMS -> InverseNet).

### 1) Bootstrap all paper folders

```bash
python repro/run_experiment.py bootstrap
```

### 2) Dry-run one paper (command plan only)

```bash
python repro/run_experiment.py run --paper-id 2020_pal_activethief --profile smoke --dry-run
```

### 3) Execute smoke profile (recommended first)

```bash
python repro/run_experiment.py run --paper-id 2020_pal_activethief --profile smoke --device cuda:0
```

### 4) Execute full profile (if feasible)

```bash
python repro/run_experiment.py run --paper-id 2020_pal_activethief --profile full --device cuda:0
```

### 5) Run 4-priority queue in order

```bash
python repro/run_priority_queue.py --profile smoke --device cuda:0
```

Queue behavior:

- DFME stage trains/evaluates shared CIFAR10 victim checkpoint.
- MAZE/DFMS stages reuse that checkpoint and run `attack,collect,compare` only.
- InverseNet stage trains its own MNIST victim checkpoint.

Remote/high-resource full run:

```bash
python repro/run_priority_queue.py --profile full --device cuda:0
```

## Stages

Default run stages:

- `victim_train`
- `victim_eval`
- `attack`
- `collect`
- `compare`

Custom stage selection example:

```bash
python repro/run_experiment.py run \
  --paper-id 2020_pal_activethief \
  --profile smoke \
  --stages victim_train,victim_eval
```

## Outputs

Per run, pipeline writes/updates:

- `repro/papers/<paper_id>/logs/pipeline_<timestamp>.log`
- `repro/papers/<paper_id>/results/environment.json`
- `repro/papers/<paper_id>/results/reproduced_metrics.csv`
- `repro/papers/<paper_id>/results/reproduced_metrics.json`
- `repro/papers/<paper_id>/results/comparison_table.md`

## Notes for Low GPU

- Start with `--profile smoke` and small epochs (`--smoke-epochs`).
- Keep `num_workers` low in configs to reduce CPU/RAM pressure.
- Run `collect,compare` stages independently after long jobs finish.

## Known blocker example

- ActiveThief strict paper reproduction needs local ILSVRC ImageFolder surrogate (`train/`, `val/`).
- Without that dataset, only an assumption-based local profile is possible.
