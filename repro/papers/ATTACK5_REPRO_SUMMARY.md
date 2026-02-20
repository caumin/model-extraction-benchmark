# Repro Summary (DFME, DFMS, MAZE, GAME, SwiftThief)

## Scope
- This summary consolidates reproducibility configs and current reproduced artifacts for:
  - DFME (`2021_truong_dfme`)
  - DFMS-HL (`2022_sanyal_dfms`)
  - MAZE (`2021_kariyappa_maze`)
  - GAME (`2022_xie_game`)
  - SwiftThief (`2024_lee_swiftthief`)

## Config status

| Attack | Paper ID | Experiment config | Attack config | Victim train/eval config | Status |
|---|---|---|---|---|---|
| DFME | `2021_truong_dfme` | `repro/papers/2021_truong_dfme/configs/experiment.yaml` | `repro/papers/2021_truong_dfme/configs/attack.yaml` | present | ready |
| DFMS-HL | `2022_sanyal_dfms` | `repro/papers/2022_sanyal_dfms/configs/experiment.yaml` | `repro/papers/2022_sanyal_dfms/configs/attack.yaml` | present | ready |
| MAZE | `2021_kariyappa_maze` | `repro/papers/2021_kariyappa_maze/configs/experiment.yaml` | `repro/papers/2021_kariyappa_maze/configs/attack.yaml` | present | ready |
| GAME | `2022_xie_game` | `repro/papers/2022_xie_game/configs/experiment.yaml` | `repro/papers/2022_xie_game/configs/attack.yaml` | present | ready |
| SwiftThief | `2024_lee_swiftthief` | `repro/papers/2024_lee_swiftthief/configs/experiment.yaml` | `repro/papers/2024_lee_swiftthief/configs/attack.yaml` | present | ready |

## Official-default anchors used

- DFME:
  - `official_repo_clones/datafree-model-extraction/dfme/train.py:182-190` (`batch_size=256`, `lr_S=0.1`, `lr_G=1e-4`, `nz=256`)
  - `official_repo_clones/datafree-model-extraction/dfme/train.py:220-221` (`grad_m=1`, `grad_epsilon=1e-3`)
- DFMS-HL:
  - `official_repo_clones/dfms_hl/run_cifar10_rand_class_resnet.sh:2` (`niter=200`, `batchSize=64`)
  - `official_repo_clones/dfms_hl/run_cifar10_rand_class_resnet.sh:31` (`niter=800`, `batchSize=64`, `d_l=500`)
- MAZE:
  - `official_repo_clones/maze/src/utils/config.py:16,69,85,92,95,98` (batch/budget/ndirs/iter_gen/iter_clone/iter_exp)
  - `official_repo_clones/maze/src/attacks/maze.py:39-43` (budget-per-iter accounting)
- GAME:
  - `official_repo_clones/game_attack/attack.py:23-27` (`batch_size=1024`, `querybudget=2000`, `attack_train_epoch=40`, optimizer)
  - `official_repo_clones/game_attack/GAME/methods.py:83-89` (tail budget slicing loop)
- SwiftThief:
  - `official_repo_clones/swiftthief/scripts/swiftthief/cifar10.sh:4,28-30` (query budget list, `sl_lr=1e-2`, `sl_epoch=500`, `sl_aug_interval=50`)
  - `official_repo_clones/swiftthief/swiftthief.py:50-52,132-160` (10% seed split + periodic sampling)

## Current reproduced results snapshot

| Attack | Result file | Latest populated budget(s) | Notes |
|---|---|---|---|
| DFME | `repro/papers/2021_truong_dfme/results/reproduced_metrics.json` | smoke 200..2000 | full 20M pending |
| DFMS-HL | `repro/papers/2022_sanyal_dfms/results/reproduced_metrics.json` | smoke 200..2000 | full 8M pending |
| MAZE | `repro/papers/2021_kariyappa_maze/results/reproduced_metrics.json` | smoke 200..2000 | full 30M pending |
| GAME | `repro/papers/2022_xie_game/results/reproduced_metrics.json` | none (`[]`) | run not executed yet |
| SwiftThief | `repro/papers/2024_lee_swiftthief/results/reproduced_metrics.json` | none (`[]`) | run not executed yet |

## Pipeline coverage
- Queue runner now includes all five attacks in order: `repro/run_priority_queue.py` and `repro/run_priority_queue.ps1`.
- Stage policy:
  - DFME/DFMS: `victim_eval,attack,collect,compare`
  - MAZE/GAME/SwiftThief: `victim_train,victim_eval,attack,collect,compare`
