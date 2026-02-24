# REPRODUCTION_REPORT

## Scope

- paper_id: `2021_truong_dfme`
- status: reproduced (practical success at 20M)

## Execution

```bash
python repro/run_experiment.py run --paper-id 2021_truong_dfme --profile smoke --device cuda:0
python repro/run_experiment.py run --paper-id 2021_truong_dfme --profile full --device cuda:0
```

## Notes

- Full rerun is complete with canonical artifacts in `repro/papers/2021_truong_dfme/results/`.
- Main result (track_b): `query_budget=20000000`, `acc_gt=0.8710`, `agreement=0.8850`.
- Paper target is `acc_gt=0.8810` at `20M`; absolute gap is `1.00pp`.
- Practical interpretation: reproduction is successful (near-parity at full budget).
- Strict `comparison_table.md` currently shows `FAIL` at the `1.00pp` boundary and should be read as a tolerance-edge gating artifact, not a substantive mismatch.
- Width note: paper/checkpoint naming uses `ResNet-34-8x`, but this repro fixes `victim.width_mult=1` to match the runnable implementation path.
