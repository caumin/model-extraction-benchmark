# REPRODUCTION_REPORT

## Scope

- paper_id: `2021_gong_inversenet`
- status: pipeline-ready (smoke/full profiles)

## Execution

```bash
python repro/run_experiment.py run --paper-id 2021_gong_inversenet --profile smoke --device cuda:0
python repro/run_experiment.py run --paper-id 2021_gong_inversenet --profile full --device cuda:0
```

## Notes

- Uses MNIST victim and EMNIST letters surrogate for attack pipeline.
- Smoke profile is designed for low-resource local verification.
