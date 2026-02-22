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

- Full profile follows paper/offical-query budget settings (CIFAR10: 8M) and official-stage epoch defaults.
- DFMS execution uses a budget planner (`paper_fair`) so stage allocation is query-count driven and final query usage exactly matches `budget.max_budget`.
- Budget semantics used in this benchmark are explicit: we cap **total** oracle queries at 8M for fair comparison across attacks. Paper Sec.3.5 equations (`Total=2*nC+NQ`) with `nC=50,000` and `NQ=8M` imply 8.1M total; this implementation keeps `nC=50,000` and correspondingly reduces effective `NQ` to 7.9M so total remains exactly 8M.
- Smoke profile reduces stage epochs for local pipeline verification.
