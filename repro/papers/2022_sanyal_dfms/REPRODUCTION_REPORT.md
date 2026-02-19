# REPRODUCTION_REPORT

## Scope

- paper_id: `2022_sanyal_dfms`
- status: pipeline-ready (smoke/full profiles)

## Execution

```bash
python repro/run_experiment.py run --paper-id 2022_sanyal_dfms --profile smoke --device cuda:0
python repro/run_experiment.py run --paper-id 2022_sanyal_dfms --profile full --device cuda:0
```

## Notes

- Full profile preserves official-stage style configuration and is compute-heavy.
- Smoke profile reduces stage epochs for local pipeline verification.
