# REPRODUCTION_REPORT

## Scope

- paper_id: `2021_wang_blackbox_dissector`
- status: runnable smoke/full profile prepared; preflight-enabled

## What Is Ready

- Victim train/eval configs and attack experiment configs are present.
- Attack implementation is wired under `mebench/attackers/blackbox_dissector.py`.
- Smoke/full reproduction profiles can run via `repro/run_experiment.py`.

## Execution

```bash
python repro/run_experiment.py run --paper-id 2021_wang_blackbox_dissector --profile smoke --device cuda:0
python repro/run_experiment.py run --paper-id 2021_wang_blackbox_dissector --profile full --device cuda:0

# direct full run
python -m mebench run --config repro/papers/2021_wang_blackbox_dissector/configs/experiment.yaml --device cuda:0
```

## Notes

- This path assumes hard-label oracle mode (`hard_top1`) only.
- Strict replication requires ImageNet-style surrogate pool availability.
- Shared victim checkpoint policy: use `runs/victims/cifar10-resnet34_8x.pt` across
  `configs/experiment.yaml`, `configs/victim_eval.yaml`, and `configs/victim_train.yaml` (`out`).
- If you want to hard-disable victim retraining regardless of local files, run with
  `--stages victim_eval,attack,collect,compare`.
- Repro preflight command:

```bash
python repro/run_experiment.py preflight --paper-id 2021_wang_blackbox_dissector --profile full
```

- `extracted_spec.yaml` includes target rows for compare-stage verdicting.
