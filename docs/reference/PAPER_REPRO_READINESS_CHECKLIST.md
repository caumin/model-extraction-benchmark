# Paper Reproduction Readiness Checklist (MARICH / BlackboxDissector / BlackboxRipper)

This checklist is the practical gate for starting near-paper runs for:

- `marich`
- `blackbox_dissector`
- `blackbox_ripper`

## 1) Config Generation Gate

- [ ] Generate paperlike configs:

```bash
python generate_paperlike_configs.py --out configs/paperlike --seeds 0 1 2 --device cuda:0
```

- [ ] Confirm expected files exist per seed:
  - `SET-B1_marich_paper_hard_30k_seed{seed}.yaml`
  - `SET-B1_blackbox_dissector_paper_hard_30k_seed{seed}.yaml`
  - `SET-B1_blackbox_ripper_paper_soft_30k_seed{seed}.yaml`

## 2) Dataset / Artifact Gate

- [ ] ImageNet surrogate root is configured for pool-based attacks:
  - either set `dataset.surrogate_root` in generated configs
  - or export `MEBENCH_IMAGENET_ROOT` to a directory containing `train/` and `val/`

- [ ] Victim checkpoint exists:
  - `runs/victims/cifar10-resnet34_8x.pt`

- [ ] BlackboxRipper generator checkpoint exists:
  - `checkpoints/blackbox_ripper/official/cifar_100_6_classes_gan.pth`

- [ ] If missing, download official ripper checkpoints:

```bash
python scripts/download_blackbox_ripper_checkpoints.py
```

## 3) Contract Gate

- [ ] MARICH config uses hard-label mode:
  - `victim.output_mode: hard_top1`
  - `attack.output_mode: hard_top1`

- [ ] BlackboxDissector config uses hard-label mode:
  - `victim.output_mode: hard_top1`
  - `attack.output_mode: hard_top1`

- [ ] BlackboxRipper config uses soft-prob mode and generator checkpoint:
  - `victim.output_mode: soft_prob`
  - `attack.output_mode: soft_prob`
  - `attack.generator_checkpoint` is set and file exists

## 4) Near-Paper Hyperparameter Gate

- [ ] MARICH paperlike defaults:
  - `sampling=all_elg`, `init_points=1000`, `budget=300`, `gamma1=0.8`, `gamma2=0.8`, `rounds=20`, `epochs=20`

- [ ] BlackboxDissector paperlike defaults:
  - `n_variants=10`, `erase_rate=0.25`, `iterative_budgets=[1000,5000,10000,20000,30000]`, `max_epochs=200`

- [ ] BlackboxRipper paperlike defaults:
  - `population_size=30`, `elite_size=10`, `fitness_threshold=0.02`, `max_evolve_iters=10`, `substitute_epochs=200`, `train_batch_size=64`

## 5) Preflight Run Gate (Smoke)

- [ ] Run one-seed smoke sanity for each attack (reduced budget config or early stop environment), then verify:
  - run completes without config/asset errors
  - `summary.json` and `metrics.csv` are emitted
  - query budget accounting is monotonic and capped by `budget.max_budget`

## 6) Full Run Gate (Paperlike)

- [ ] Run seeds in order `0 -> 1 -> 2`:
  - MARICH 30k
  - BlackboxDissector 30k
  - BlackboxRipper 30k

- [ ] Aggregate and compare:

```bash
python aggregate_matrix.py
```

- [ ] Record deltas vs reference tables in per-paper repro reports under `repro/papers/*/REPRODUCTION_REPORT.md`.

## 7) Failure Triage Checklist

- [ ] `surrogate_root`/ImageNet path unresolved -> fix path or env var
- [ ] Missing ripper checkpoint -> run download script and verify `.pth` exists
- [ ] Output mode mismatch errors -> align `victim.output_mode` with `attack.output_mode`
- [ ] Budget under-consumption/over-consumption -> inspect attack loop and `budget.max_budget`
- [ ] OOM/throughput issues -> lower loader workers and run per-seed sequentially
