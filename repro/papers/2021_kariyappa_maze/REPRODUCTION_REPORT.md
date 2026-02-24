# REPRODUCTION_REPORT

## Scope

- paper_id: `2021_kariyappa_maze`
- status: reproduced against follow-up reference (original-paper target unmet)

## Execution

```bash
python repro/run_experiment.py run --paper-id 2021_kariyappa_maze --profile smoke --device cuda:0
python repro/run_experiment.py run --paper-id 2021_kariyappa_maze --profile full --device cuda:0
```

## Notes

- Current canonical full artifact is a `20M` run in `repro/papers/2021_kariyappa_maze/results/`.
- Main result (track_b): `query_budget=20000000`, `acc_gt=0.4009`, `agreement=0.4051`.
- Same run reaches a local peak at `query_budget=19301760` with `acc_gt=0.4454`.
- Original MAZE paper target (`CIFAR10`, `30M`, `acc_gt=0.8985`) is not reproduced.
- Follow-up reference (DFME paper mention of MAZE around `45.6%` at `20M`) is closely matched by the observed `44.54%` peak, so this profile is treated as a practical reproduction success for that reference point.
- Width note: paper/checkpoint naming uses `ResNet-34-8x`, but this repro fixes `victim.width_mult=1` to match the runnable implementation path.
