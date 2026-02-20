# Attack Parity Report

This report tracks parity status between official implementations in
`official_repo_clones/` and mebench attack runners.

## Checklist (global)

- [x] Same victim query gateway (`BenchmarkContext.query`) for all mebench attacks
- [x] Budget counting is image-based (`1 query image = 1 budget`) in oracle/context
- [x] Attack-mode compatibility guarded in `mebench/core/validate.py`
- [x] Official-compatible preprocessing profiles available (`mebench/data/preprocessing.py`)
- [x] Optional oracle-side preprocessing hook available (`victim.official_preprocess_profile`)
- [ ] Full per-attack metric parity run completed for all datasets/seeds (pending long-run execution)

## Per-attack parity status

| Attack | Official ref | mebench impl | Parity status |
|---|---|---|---|
| DFME | `datafree-model-extraction/dfme/train.py` | `mebench/attackers/dfme.py` | SAME_CORE_LOOP (minor contract adjustments) |
| MAZE | `maze/src/attacks/maze.py` | `mebench/attackers/maze.py` | SAME_CORE_LOOP |
| KnockoffNets | `knockoffnets/knockoff/adversary/*` | `mebench/attackers/knockoff_nets.py` | CLOSE (benchmark-normalized preprocessing by default) |
| SwiftThief | `swiftthief/swiftthief.py` | `mebench/attackers/swiftthief.py` | CLOSE (strict budget gateway + framework adaptation) |
| GAME | `game_attack/attack.py` | `mebench/attackers/game.py` | CLOSE |
| DFMS-HL | `dfms_hl/code/*` | `mebench/attackers/dfms.py` | CLOSE |
| DisGUIDE | `disguide/disguide/train.py` | `mebench/attackers/disguide.py` | SAME_CORE_LOOP (contract-aligned scaling + budget gateway) |
| CloudLeak | `cloudleak/optimize.py` | `mebench/attackers/cloudleak.py` | CLOSE |

## Accepted differences and rationale

1. Oracle contract normalization:
   - Benchmark canonical input is `[0,1]`, while some official repos internally normalize to ImageNet or `[-1,1]`.
   - We preserve benchmark fairness by default and provide opt-in official preprocessing profiles in `mebench/data/preprocessing.py`.

2. Query accounting strictness:
   - Some official scripts use stage/epoch-level querybudget semantics.
   - mebench enforces strict per-image budget decrement in `mebench/oracles/oracle.py:60-65` and `mebench/core/context.py:68-90`.

3. Framework wrapping:
   - Official scripts are often standalone training scripts.
   - mebench ports are wrapped to `AttackRunner.run(ctx)` and artifact/checkpoint conventions.

## Recent default-align updates

- `KnockoffNets`: default `batch_size=8`, default `policy=random`, and random-query refill semantics aligned with `knockoff/adversary/transfer.py`.
- `MAZE`: default scheduler aligned to cosine (`maze/src/attacks/maze.py` uses cosine with SGD path).
- `DFME`: student optimizer defaults aligned to official (`SGD lr=0.1 momentum=0.9 wd=5e-4`).
- `SwiftThief`: KD/SL learning-rate default aligned to official `--sl_lr=1e-2`.
- `GAME`: default query/training batch size aligned to official `--batch_size=1024`.
- `DisGUIDE`: generator objective aligned to disagreement + diversity, with official-style replay/ensemble defaults.
  - Official repository uses `_8x` model names, but `network/resnet_8x.py` defines standard-width channels (effective width multiplier 1).

## 1:1 loop align updates

- `SwiftThief`
  - run schedule aligned to official 10%-seed then periodic 10% query rounds using `sl_aug_interval`.
  - imbalance mode now follows official 5-split sub-round behavior.
  - references: `official_repo_clones/swiftthief/swiftthief.py:50-52`, `official_repo_clones/swiftthief/swiftthief.py:132-133`, `official_repo_clones/swiftthief/swiftthief.py:149-160`.

- `GAME`
  - query loop now follows explicit `querybudget` tail-slicing semantics.
  - official defaults exposed (`querybudget=2000`, `attack_train_epoch=40`) and final full-buffer training phase retained.
  - references: `official_repo_clones/game_attack/attack.py:23-27`, `official_repo_clones/game_attack/GAME/methods.py:83-89`, `official_repo_clones/game_attack/GAME/methods.py:208-214`.

## Preprocessing parity matrix

| Profile | Source | Implemented name |
|---|---|---|
| DFME CIFAR10 test normalize | `official_repo_clones/datafree-model-extraction/dfme/dataloader.py:57-59` | `dfme_cifar10_test` |
| MAZE RGB normalize | `official_repo_clones/maze/src/datasets/datasets.py:184-186` | `maze_rgb_test` |
| SwiftThief CIFAR test | `official_repo_clones/swiftthief/utils/get_datasets.py:28-32` | `swiftthief_cifar_test` |
| KnockoffNets default test transform | `official_repo_clones/knockoffnets/knockoff/utils/transforms.py:26-30` | `knockoffnets_default_test` |
| DFMS-HL student normalize | `official_repo_clones/dfms_hl/code/train_student/train_student.py:244-246` | `dfms_hl_train_student` |

## Output-type parity notes

- Soft-output required attacks are enforced by validation sets in `mebench/core/validate.py:68-96`.
- Hard-output required attacks are also enforced there (including DFMS-HL).
- DisGUIDE is enforced as both-capable (`soft_prob` / `hard_top1`) with hard-label guard (`loss='hl'`) in `mebench/attackers/disguide.py`.

## Remaining work

1. Run full metric parity matrix (official-equivalent config x seed) and append quantitative deltas.
2. Add standalone mebench wrappers for uncovered official variants (`maze-noise`, `maze-jbda`, `fff`) if needed for this benchmark scope.
