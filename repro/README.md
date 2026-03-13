# Reproduction Set and Pipeline

This folder is optimized for **low-resource local environments** where strict paper-level full runs may be expensive or blocked by missing datasets.

## Goals

- Keep per-paper reproduction artifacts in a standard layout.
- Make runs reproducible with staged commands.
- Support quick `smoke` runs first, then `full` runs when hardware/data are available.

## Benchmark Policy vs. Paper Reproduction

- `repro/` exists for paper-specific reproduction and parity debugging, not for the benchmark matrix itself.
- Matrix benchmark results should be interpreted as benchmark-policy results under the shared `mebench` runtime contract.
- Paper reproduction results may intentionally preserve attack-native training choices that are not part of the default benchmark matrix.
- Keep benchmark-policy reporting and paper-parity reporting separate in writeups and tables.

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

Priority queue runner: `repro/run_priority_queue.py` (MAZE -> GAME -> DFMS -> blackbox-dissector -> disguide).

Current status:

- DFME (`2021_truong_dfme`): full 20M rerun complete, practical reproduction success (`acc_gt=0.871` vs paper `0.881`).
- MAZE (`2021_kariyappa_maze`): original MAZE-paper target (`30M`, `acc_gt=0.8985`) is not reproduced, but 20M run behavior (peak `acc_gt=0.4454`) is close to the follow-up DFME-paper MAZE reference (`45.6%`).
- DFMS (`2022_sanyal_dfms`): current artifact is still smoke-only, so a full rerun is required.
- Priority queue still starts from MAZE; run paper-specific updates with `repro/run_experiment.py` when needed.

### 1) Bootstrap all paper folders

```bash
python repro/run_experiment.py bootstrap
```

### 2) Dry-run one paper (command plan only)

```bash
python repro/run_experiment.py run --paper-id 2020_pal_activethief --profile smoke --dry-run
```

### 2.5) Preflight readiness checks (recommended before full runs)

Validate experiment config contracts and required local assets for the tracked set
(`2023_karmakar_marich`, `2021_wang_blackbox_dissector`, `2020_barbalau_blackbox_ripper`, `2023_tan_disguide`, `2021_truong_dfme`, `2023_beetham_dual_students`):

```bash
python repro/run_experiment.py preflight --paper-id all --profile full
```

Run for one paper only:

```bash
python repro/run_experiment.py preflight --paper-id 2023_karmakar_marich --profile full
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

PowerShell sequential runner (recommended if you want live `tqdm`/progress-bar rendering):

```powershell
./repro/run_priority_queue.ps1 -RunProfile smoke -Device cuda:0 -SmokeEpochs 2 -SmokeBatchSize 32
```

`run_experiment.py` supports `--live-output` (default on) so child process output is attached to the terminal for proper progress-bar rendering.

Queue behavior:

- MAZE and DFMS use existing victim checkpoints and run `victim_eval,attack,collect,compare`.
- GAME runs `victim_train,victim_eval,attack,collect,compare`.
- blackbox-dissector and disguide use existing ResNet34 checkpoint and run `victim_eval,attack,collect,compare`.

### Shared ResNet34 Victim Policy (CIFAR10)

For CIFAR10 papers that target the shared ResNet34 victim profile, use one checkpoint path
consistently across configs:

- `runs/victims/cifar10-resnet34_8x.pt`

This applies to `experiment.yaml`, `victim_eval.yaml`, and `victim_train.yaml` (`out`) so
`run_experiment.py` can reuse the checkpoint and auto-skip unnecessary victim retraining.

### Data-free Scale Policy

For data-free attacks in `@repro`, victim query tensors must remain tanh-scale (`[-1,1]`)
at the attacker boundary. Do not add attacker-side tanh->unit conversion on query path.
Runtime victim query path uses direct model input (no attacker-side wrapper normalization).

## Runtime Preprocessing Contract

- Victim query path: no extra benchmark-side normalization wrapper is applied before victim inference.
- Victim evaluation path: dataset test-set evaluation uses the dataset's official normalization.
- Surrogate / pool path: attacker-side surrogate data should follow the surrogate dataset's official preprocessing for that experiment.
- Data-free path: preserve the attack-native tensor scale unless a paper-specific reproduction artifact explicitly documents a different convention.

Remote/high-resource full run:

```bash
python repro/run_priority_queue.py --profile full --device cuda:0
```

```powershell
./repro/run_priority_queue.ps1 -RunProfile full -Device cuda:0 -SmokeEpochs 2 -SmokeBatchSize 32
```

## Stages

Default run stages (when `--stages` is omitted):

- `victim_train,victim_eval,attack,collect,compare`
- If a victim checkpoint already exists at the configured `victim_train.out`, pipeline auto-skips `victim_train` and reuses that checkpoint.

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

- Start with `--profile smoke` and tune both `--smoke-epochs` and `--smoke-batch-size`.
- Keep `num_workers` low in configs to reduce CPU/RAM pressure.
- Run `collect,compare` stages independently after long jobs finish.

Example (OOM-safe smoke):

```bash
python repro/run_priority_queue.py --profile smoke --device cuda:0 --smoke-epochs 2 --smoke-batch-size 32
```

## Known blocker example

- ActiveThief strict paper reproduction needs local ILSVRC ImageFolder surrogate (`train/`, `val/`).
- Without that dataset, only an assumption-based local profile is possible.
