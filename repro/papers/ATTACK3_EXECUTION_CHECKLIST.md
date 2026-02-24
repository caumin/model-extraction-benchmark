# Attack4 Repro Checklist (MARICH / BlackboxDissector / BlackboxRipper / DisGUIDE)

This checklist is for starting near-paper runs under `@repro` (not `paperlike`).

## 1) Preflight

- [ ] Run full-profile preflight:

```bash
python repro/run_experiment.py preflight --paper-id all --profile full
```

- [ ] All checks must be `[OK]` for:
  - attack name and output mode contract
  - victim checkpoint path
  - budget consistency (`attack.max_budget == budget.max_budget`)
  - extracted-spec targets present
  - ImageNet surrogate split directory (MARICH/Dissector)
  - generator checkpoint (BlackboxRipper, `.pth/.pt` fallback)
  - victim preprocess profile consistency (`official_preprocess_profile: dfme_cifar10_test`)
  - data-free victim scale contract (tanh query tensors normalized in victim wrapper)

## 2) Hard Blockers

- [ ] ImageNet surrogate root is valid for MARICH/Dissector configs:
  - `dataset.surrogate_root` or `MEBENCH_IMAGENET_ROOT`
  - expected split path exists for current profile (`train` if `train_split=true`, else `val`)

- [ ] Victim checkpoint exists:
  - `runs/victims/cifar10-resnet34_8x.pt`

- [ ] Shared victim checkpoint policy is applied in configs:
  - `victim_train.yaml` uses `out: runs/victims/cifar10-resnet34_8x.pt`
  - `victim_eval.yaml`/`experiment.yaml` use `checkpoint_ref: runs/victims/cifar10-resnet34_8x.pt`

- [ ] BlackboxRipper generator checkpoint exists:
  - from config: `attack.generator_checkpoint`
  - extensionless path is allowed if matching `.pth/.pt` exists

## 3) Execution Order (per seed)

Run one seed at a time in this order:

1. `2023_karmakar_marich`
2. `2021_wang_blackbox_dissector`
3. `2020_barbalau_blackbox_ripper`
4. `2023_tan_disguide`

Smoke-first:

```bash
python repro/run_experiment.py run --paper-id 2023_karmakar_marich --profile smoke --device cuda:0 --stages victim_eval,attack,collect,compare
python repro/run_experiment.py run --paper-id 2021_wang_blackbox_dissector --profile smoke --device cuda:0 --stages victim_eval,attack,collect,compare
python repro/run_experiment.py run --paper-id 2020_barbalau_blackbox_ripper --profile smoke --device cuda:0 --stages victim_eval,attack,collect,compare
python repro/run_experiment.py run --paper-id 2023_tan_disguide --profile smoke --device cuda:0 --stages victim_eval,attack,collect,compare
```

Then full:

```bash
python repro/run_experiment.py run --paper-id 2023_karmakar_marich --profile full --device cuda:0 --stages victim_eval,attack,collect,compare
python repro/run_experiment.py run --paper-id 2021_wang_blackbox_dissector --profile full --device cuda:0 --stages victim_eval,attack,collect,compare
python repro/run_experiment.py run --paper-id 2020_barbalau_blackbox_ripper --profile full --device cuda:0 --stages victim_eval,attack,collect,compare
python repro/run_experiment.py run --paper-id 2023_tan_disguide --profile full --device cuda:0 --stages victim_eval,attack,collect,compare
```

## 4) Result Gate

- [ ] For each run, confirm artifacts exist:
  - `repro/papers/<paper_id>/results/reproduced_metrics.csv`
  - `repro/papers/<paper_id>/results/reproduced_metrics.json`
  - `repro/papers/<paper_id>/results/comparison_table.md`

- [ ] Comparison table has target rows and PASS/FAIL verdicts (not INCOMPLETE).

## 5) Typical Fixes

- Preflight says surrogate split missing:
  - fix `dataset.surrogate_root` or export `MEBENCH_IMAGENET_ROOT`
- Preflight says generator checkpoint missing:
  - run `python scripts/download_blackbox_ripper_checkpoints.py`
  - or point config to an existing trained checkpoint
- Compare stage incomplete:
  - ensure run generated expected query-budget checkpoints in `metrics.csv`
