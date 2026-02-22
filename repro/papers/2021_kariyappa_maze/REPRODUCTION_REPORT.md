# REPRODUCTION_REPORT

## Scope

- paper_id: `2021_kariyappa_maze`
- status: pipeline-ready (smoke/full profiles)

## Execution

```bash
python repro/run_experiment.py run --paper-id 2021_kariyappa_maze --profile smoke --device cuda:0
python repro/run_experiment.py run --paper-id 2021_kariyappa_maze --profile full --device cuda:0
```

## Notes

- Full profile budget is 30M and intended for remote/high-resource runs.
- Smoke profile validates orchestration and artifact generation locally.
- Width note: paper/checkpoint naming uses `ResNet-34-8x`, but this repro fixes `victim.width_mult=1` to match the runnable implementation path.
