# REPRODUCTION_REPORT

## Scope

- paper_id: `2020_barbalau_blackbox_ripper`
- status: runnable smoke/full profile prepared; preflight-enabled

## What Is Ready

- Reproduction configs for victim train/eval and attack smoke/full runs are defined.
- Attack implementation is wired under `mebench/attackers/blackbox_ripper.py`.
- Generator checkpoint loader supports extension fallback (`.pth`/`.pt`).

## Execution

```bash
# prerequisite: download/generate official ripper checkpoints
python scripts/download_blackbox_ripper_checkpoints.py

python repro/run_experiment.py run --paper-id 2020_barbalau_blackbox_ripper --profile smoke --device cuda:0
python repro/run_experiment.py run --paper-id 2020_barbalau_blackbox_ripper --profile full --device cuda:0

# direct full run
python -m mebench run --config repro/papers/2020_barbalau_blackbox_ripper/configs/experiment.yaml --device cuda:0
```

## Notes

- `blackbox_ripper` requires `soft_prob` oracle mode.
- A pretrained generator checkpoint is mandatory (`attack.generator_checkpoint`).
- Paper-faithful profile uses official checkpoint stem `cifar_100_6_classes_gan`.
- Shared victim checkpoint policy: use `runs/victims/cifar10-resnet34_8x.pt` and prefer
  `--stages victim_eval,attack,collect,compare` when victim retraining is unnecessary.
- Repro preflight command:

```bash
python repro/run_experiment.py preflight --paper-id 2020_barbalau_blackbox_ripper --profile full
```

- `extracted_spec.yaml` includes target rows for compare-stage verdicting.
