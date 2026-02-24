# 2022_xie_game Ultrawork Runbook

## Purpose

Operational checklist to run GAME reproduction after `mebench/attackers/game.py` refactor.

## Preconditions

- Victim checkpoints exist (or can be trained).
- Proxy datasets are available for pair-1/pair-2.
- Runtime uses benchmark metered-query contract.

## Pair-1 (MNIST <- FashionMNIST)

1. Train/eval victim

```bash
python scripts/train_victim.py --config repro/papers/2022_xie_game/configs/victim_train.yaml --device cuda:0
python scripts/eval_victim.py --config repro/papers/2022_xie_game/configs/victim_eval.yaml --device cuda:0
```

2. Smoke repro

```bash
python -m mebench run --config repro/papers/2022_xie_game/configs/experiment_smoke.yaml --device cuda:0
```

3. Paper-profile single seed

```bash
python -m mebench run --config repro/papers/2022_xie_game/configs/experiment_paper_half_lenet_seed0_sgd.yaml --device cuda:0
```

4. Paper-profile multi-seed

```bash
python -m mebench run --config repro/papers/2022_xie_game/configs/experiment_paper_half_lenet.yaml --device cuda:0
```

## Pair-2 (BelgiumTSC <- GTSRB)

1. Train/eval victim

```bash
python scripts/train_victim.py --config repro/papers/2022_xie_game/configs/victim_train_pair2.yaml --device cuda:0
python scripts/eval_victim.py --config repro/papers/2022_xie_game/configs/victim_eval_pair2.yaml --device cuda:0
```

2. Official-aligned run

```bash
python -m mebench run --config repro/papers/2022_xie_game/configs/experiment_pair2_18k_official.yaml --device cuda:0
```

3. Repo-literal run

```bash
python -m mebench run --config repro/papers/2022_xie_game/configs/experiment_pair2_18k_repo_literal.yaml --device cuda:0
```

## Result Collection

- Check `runs/*/seed_*/summary.json` for checkpoint metrics.
- Update `repro/papers/2022_xie_game/GAME_PARITY_CHECKLIST.md` after each run slice.

## Stop Criteria

- Pair-1: stable non-collapsing trajectory and reproducible seed behavior.
- Pair-2: best setting isolated with one-variable ablations and documented gap cause.
