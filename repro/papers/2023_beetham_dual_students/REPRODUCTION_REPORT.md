# Reproduction Report (Template)

## Paper
- ID: `2023_beetham_dual_students`
- Attack: `ds`

## Run Profiles
- `configs/experiment_smoke.yaml`: short local sanity run
- `configs/experiment.yaml`: paper-like long-budget profile

## Status
- [x] smoke/full config aligned to shared ResNet34_8x victim checkpoint
- [x] compare target extracted (`acc_gt` @ 20M = 0.9134)
- [ ] smoke run completed
- [ ] full run completed
- [ ] metrics collected in `results/reproduced_metrics.csv`
- [ ] comparison table updated

## Notes
- Official paper-scale budget (20M on CIFAR-10) is high-cost and should be run on a remote GPU when possible.
- Repro preflight command:

```bash
python repro/run_experiment.py preflight --paper-id 2023_beetham_dual_students --profile full
```
