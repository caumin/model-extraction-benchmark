# Attack Parity Report

This report tracks official-to-internal parity status for attack ports.

Companion policy document:
- `mebench/attackers/HYPERPARAM_POLICY.md` (fixed-required vs tunable knobs)

## Global parity checklist

- [x] All attacks query victim through `BenchmarkContext.query` (single budget gateway)
- [x] Budget accounting is per-image (`Oracle.query` increments `query_count` by batch size)
- [x] Output-mode compatibility is centrally validated (`mebench/core/validate.py`)
- [x] Official-compatible preprocessing profiles exist (`mebench/data/preprocessing.py`)
- [x] Oracle can apply official preprocessing profile via config (`victim.official_preprocess_profile`)
- [x] Seed reproducibility path exists (`mebench/core/seed.py` + attack-level RNG usage)
- [x] Smoke and preprocess parity tests exist under `tests/test_attack_porting_*.py`
- [ ] Full metric parity matrix completed for all attacks/seeds (pending long-run experiment execution)

## Attack-by-attack parity matrix

| Attack | Official reference | mebench implementation | Preprocess parity | Output type parity | Hyperparameter parity | Seed/repro parity | Notes |
|---|---|---|---|---|---|---|---|
| DFME | `official_repo_clones/datafree-model-extraction/dfme/train.py` | `mebench/attackers/dfme.py` | CLOSE (`dfme_cifar10_test` profile) | SAME (`soft_prob`) | SAME core defaults (`batch_size`, `n_g`, `n_s`, `lr_S`) | SAME (global seed + torch/numpy path) | Data-free query path keeps tanh-scale tensors at oracle boundary |
| MAZE | `official_repo_clones/maze/src/attacks/maze.py` | `mebench/attackers/maze.py` | CLOSE (`maze_rgb_test` profile) | SAME (`soft_prob`) | CLOSE (cosine schedule + iter semantics aligned) | SAME | Query path remains strict image-budgeted |
| KnockoffNets | `official_repo_clones/knockoffnets/knockoff/adversary/transfer.py` | `mebench/attackers/knockoff_nets.py` | CLOSE (`knockoffnets_default_test` profile) | SAME (`soft_prob`) | CLOSE (`batch_size`, policy behavior, transfer semantics) | SAME | Official transfer-set filesystem flow adapted to framework tensors |
| SwiftThief | `official_repo_clones/swiftthief/swiftthief.py` | `mebench/attackers/swiftthief.py` | CLOSE (`swiftthief_cifar_test` profile) | SAME (supports `soft_prob` and `hard_top1`) | CLOSE (10% seed, `sl_aug_interval`, 5-way imbalance round) | SAME | Uses framework pool loaders instead of `unlabeled_dataset.pt` blob |
| GAME | `official_repo_clones/game_attack/attack.py` | `mebench/attackers/game.py` | CLOSE (contract default + optional profile) | SAME (`soft_prob`) | SAME defaults (`batch_size=1024`, `querybudget=2000`, `attack_train_epoch=40`) | SAME | Official baseline-only script structure wrapped into AttackRunner |
| DFMS-HL | `official_repo_clones/dfms_hl/code/train_student/train_student.py` | `mebench/attackers/dfms.py` | CLOSE (`dfms_hl_train_student` profile) | SAME (`hard_top1`) | CLOSE (stage-wise loop and key defaults preserved) | SAME | Query path fixed to tanh-scale; internal model scale remains configurable |
| DisGUIDE | `official_repo_clones/disguide/disguide/train.py` | `mebench/attackers/disguide.py` | CLOSE (official transform assumptions documented) | SAME (`soft_prob` or `hard_top1` with HL loss) | CLOSE (ensemble/replay/diversity defaults) | SAME | Hard-mode guard enforced in attack config validation |
| CloudLeak | `official_repo_clones/cloudleak/optimize.py` | `mebench/attackers/cloudleak.py` | CLOSE (Caffe preprocessing adapted to contract) | SAME (`soft_prob`) | CLOSE (FeatureFool optimization defaults) | SAME | Channel/order conversions explicitly handled in implementation |
| BlackboxDissector | `official_repo_clones/blackbox-dissector/attack.py` | `mebench/attackers/blackbox_dissector.py` | CLOSE (official normalize notes documented) | SAME (`hard_top1`) | CLOSE (iterative split + erase-rate behavior aligned) | SAME | Transfer-set file workflow adapted to in-memory query storage |
| ActiveThief | `official_repo_clones/activethief/README.md` | `mebench/attackers/activethief.py` | CLOSE (`generic_program.py`, `cfg.py`, and `utils/model.py` loop parity) | SAME (`soft_prob`, `hard_top1`) | CLOSE (`initial_seed`, `num_iter`, `k`, iteration scheduling) | SAME | Official-round semantics mapped into framework state machine and retraining schedule |
| CopycatCNN | paper-reference (no direct clone in `official_repo_clones`) | `mebench/attackers/copycatcnn.py` | N/A official clone | SAME (`hard_top1`) | DONE_PAPER | SAME | Offline augmentation parity based on paper/open implementations |
| InverseNet | paper-reference (no direct clone in `official_repo_clones`) | `mebench/attackers/inversenet.py` | N/A official clone | SAME (`hard_top1`) | DONE_PAPER | SAME | Phase-ratio budget semantics implemented in runner |
| BlackboxRipper | paper-reference (no direct clone in `official_repo_clones`) | `mebench/attackers/blackbox_ripper.py` | N/A official clone | SAME (`soft_prob`) | DONE_PAPER | SAME | Evolutionary latent loop aligned to paper spec |
| ESAttack | benchmark implementation | `mebench/attackers/es_attack.py` | N/A official clone | SAME (both modes) | DONE | SAME | Contract-first implementation |
| RandomBaseline | benchmark baseline | `mebench/attackers/random_baseline.py` | N/A official clone | SAME (both modes) | DONE | SAME | Control baseline |

## Accepted differences (and why they are allowed)

1. Input normalization baseline:
   - Benchmark contract keeps pool-based oracle/eval inputs in `[0,1]` and data-free query inputs in `[-1,1]`.
   - Some official repos assume dataset normalization or additional transforms.
   - Resolution: keep contract defaults and expose explicit official preprocess profiles for parity runs.

2. Script-to-framework adaptation:
   - Official code is often standalone multi-script training logic.
   - mebench wraps the same core loops into `AttackRunner.run(ctx)` while preserving query semantics and configurable defaults.

3. Budget semantics unification:
   - Official repos sometimes express budget in rounds/epochs/millions.
   - mebench enforces one canonical rule at runtime: 1 queried image consumes 1 budget.

4. Matrix substitute policy is setup-level unified:
   - SET-B1 (`resnet18`) uses one substitute profile across attacks: `sgd(lr=0.1,momentum=0.9,wd=5e-4)`, `batch=256`, `multistep([0.5,0.75],gamma=0.1)`, `max_epochs=1000`, `patience=100`.
   - SET-A1 (`lenet_mnist`) uses one substitute profile across attacks: `sgd(lr=0.04,momentum=0.9,wd=5e-4)`, `batch=512`, `multistep([0.5,0.75],gamma=0.1)`, `max_epochs=200`, `patience=20`.
   - SET-C1 (`xie2019`) uses one fixed-length substitute profile across attacks: `sgd(lr=0.05,momentum=0.9,wd=5e-4)`, `batch=128`, `val_batch=32`, `eval_batch=32`, `multistep([0.5,0.75],gamma=0.1)`, `max_epochs=90`, `patience=90`, with best-checkpoint selection by validation loss.
   - Matrix generation does not apply per-attack LR/batch alignment overrides; attack-specific knobs remain only when semantic to the attack protocol.

## Evidence: key parity tests in this repo

- `tests/test_attack_porting_smoke.py`
  - Tiny end-to-end smoke for DFME, RandomBaseline, and KnockoffNets.
- `tests/test_attack_porting_preprocess.py`
  - Transform parity checks for official profile chains (`dfme`, `knockoffnets`).
- `tests/test_attack_porting_defaults.py`
  - Default-value parity checks against official defaults (MAZE/DFME/GAME/SwiftThief/KnockoffNets).
- `tests/test_attack_porting_metric_parity.py`
  - Metric-parity scaffold that validates structure and tolerance rules when baseline artifact is provided.
- `tests/test_attack_porting_per_attack_logic.py`
  - Per-attack routing/phase/target-format logic checks for CopycatCNN, InverseNet, and DisGUIDE.

## Remaining work

1. Run full official-equivalent matrix (same seed/hparams/profile) and append quantitative deltas per checkpoint.
2. Decide benchmark scope for currently-unmapped official variants (`maze-jbda`, `maze-noise`, `fff`) and add dedicated runners if required.
3. Store run-level parity metadata with preprocess profile name and hyperparameter snapshot in report artifacts.
