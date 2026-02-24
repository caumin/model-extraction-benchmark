# REPRODUCTION_REPORT

## Scope

- paper_id: `2023_karmakar_marich`
- status: runnable smoke/full profile prepared; preflight-enabled

## What Is Ready

- Reproduction folder scaffold and runnable configs are added.
- Attack path is wired to `mebench/attackers/marich.py` with hard-label constraints.
- Smoke/full experiment configs exist for immediate execution.

## Execution

```bash
python repro/run_experiment.py run --paper-id 2023_karmakar_marich --profile smoke --device cuda:0
python repro/run_experiment.py run --paper-id 2023_karmakar_marich --profile full --device cuda:0

# direct run (full)
python -m mebench run --config repro/papers/2023_karmakar_marich/configs/experiment.yaml --device cuda:0
```

## Notes

- Current implementation enforces `hard_top1` output mode for MARICH.
- Shared victim checkpoint policy: use `runs/victims/cifar10-resnet34_8x.pt` and prefer
  `--stages victim_eval,attack,collect,compare` when victim retraining is unnecessary.
- Budget-accounting caveat (critical for score interpretation):
  - Official MARICH image notebook constructs a fixed validation split from the full 50k pool
    (`train_test_split(range(50000), test_size=0.2)`) and trains with `validloader` built from
    that split.
  - Labels for that split are effectively pre-available through precomputed `unlab_y` assets in
    the official pipeline, so reported query counts track active extraction samples and do not
    include validation-label construction cost.
  - In this repo, `validation_source: pool_split` emulates official behavior. This mode may make
    paper-reproduction scores look better than strict black-box billing, because validation labels
    are outside the active-query budget counter by design.
  - For strict `1 query = 1 image` accounting, use `validation_source: queried_holdout` (or raise
    max budget to include validation labeling cost explicitly).
- Repro preflight command:

```bash
python repro/run_experiment.py preflight --paper-id 2023_karmakar_marich --profile full
```

- `extracted_spec.yaml` includes target rows for compare-stage verdicting.
