# REPRODUCTION_REPORT

## Scope

- paper_id: `2021_truong_dfme`
- status: revalidation planned (rerun pending)

## Execution

```bash
python repro/run_experiment.py run --paper-id 2021_truong_dfme --profile smoke --device cuda:0
python repro/run_experiment.py run --paper-id 2021_truong_dfme --profile full --device cuda:0
```

## Notes

- Previous DFME completion was a project milestone snapshot.
- Current objective is to rerun verification (smoke and full) and refresh reproduced metrics/comparison outputs.
- Width note: paper/checkpoint naming uses `ResNet-34-8x`, but this repro fixes `victim.width_mult=1` to match the runnable implementation path.
