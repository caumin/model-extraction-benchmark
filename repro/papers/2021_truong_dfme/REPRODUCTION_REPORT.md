# REPRODUCTION_REPORT

## Scope

- paper_id: `2021_truong_dfme`
- status: pipeline-ready (smoke/full profiles)

## Execution

```bash
python repro/run_experiment.py run --paper-id 2021_truong_dfme --profile smoke --device cuda:0
python repro/run_experiment.py run --paper-id 2021_truong_dfme --profile full --device cuda:0
```

## Notes

- Full profile is paperlike-budget oriented and compute-heavy (20M queries).
- Smoke profile is for local validation of end-to-end reproducibility flow.
