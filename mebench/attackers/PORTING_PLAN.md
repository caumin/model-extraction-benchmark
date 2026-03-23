# Attack Porting Plan (official_repo_clones -> mebench)

This document is the implementation mapping for fairness-preserving attack porting.

Companion policy document:
- `mebench/attackers/HYPERPARAM_POLICY.md` (fixed-required vs tunable knobs and current compliance)

- Official sources (read-only): `official_repo_clones/`
- Internal framework: `mebench/attackers/`
- Goal: keep preprocessing, query/output interface, budget semantics, hyperparameters, and seed behavior aligned with official references.

Current matrix-generation note (`generate_configs.py`, SET-B):
- Matrix generation uses setup-level unified substitute defaults (no per-attack LR/batch alignment overrides).
- SET-B1 (`resnet18`) substitute defaults: `batch=256`, `optimizer=sgd(lr=0.1,momentum=0.9,wd=5e-4)`, `scheduler=multistep([0.5,0.75],gamma=0.1)`, `max_epochs=1000`, `patience=100`.
- SET-A1 (`lenet_mnist`) substitute defaults: `batch=512`, `optimizer=sgd(lr=0.04,momentum=0.9,wd=5e-4)`, `scheduler=multistep([0.5,0.75],gamma=0.1)`, `max_epochs=200`, `patience=20`.
- SET-C1 (`xie2019`) substitute defaults: `batch=64`, `val_batch=32`, `eval_batch=32`, `optimizer=sgd(lr=0.1,momentum=0.9,wd=5e-4)`, `scheduler=multistep([0.5,0.75],gamma=0.1)`, `max_epochs=90`, `patience=90`; selection remains best-by-validation-loss while runs stay fixed at 90 epochs.

## 1) Inventory

### 1.1 Official attack implementations found in `official_repo_clones/`

| Official attack | Primary entrypoint(s) | Default/config evidence |
|---|---|---|
| DFME | `official_repo_clones/datafree-model-extraction/dfme/train.py:57` | `official_repo_clones/datafree-model-extraction/dfme/train.py:182`, `official_repo_clones/datafree-model-extraction/dfme/train.py:185`, `official_repo_clones/datafree-model-extraction/dfme/train.py:186`, `official_repo_clones/datafree-model-extraction/dfme/train.py:188` |
| MAZE | `official_repo_clones/maze/src/attacks/maze.py:30` | `official_repo_clones/maze/src/attacks/maze.py:39`, `official_repo_clones/maze/src/attacks/maze.py:40`, `official_repo_clones/maze/src/attacks/maze.py:55` |
| KnockoffNets | `official_repo_clones/knockoffnets/knockoff/adversary/transfer.py:93` | `official_repo_clones/knockoffnets/knockoff/adversary/transfer.py:101`, `official_repo_clones/knockoffnets/knockoff/adversary/transfer.py:104` |
| SwiftThief | `official_repo_clones/swiftthief/swiftthief.py:44` | `official_repo_clones/swiftthief/swiftthief.py:50`, `official_repo_clones/swiftthief/swiftthief.py:132`, `official_repo_clones/swiftthief/swiftthief.py:150` |
| GAME | `official_repo_clones/game_attack/attack.py:11` | `official_repo_clones/game_attack/attack.py:23`, `official_repo_clones/game_attack/attack.py:24`, `official_repo_clones/game_attack/attack.py:26` |
| DFMS-HL | `official_repo_clones/dfms_hl/code/train_student/train_student.py:81` | `official_repo_clones/dfms_hl/code/train_student/train_student.py:93`, `official_repo_clones/dfms_hl/code/train_student/train_student.py:95`, `official_repo_clones/dfms_hl/code/train_student/train_student.py:245` |
| DisGUIDE | `official_repo_clones/disguide/disguide/train.py`, `official_repo_clones/disguide/disguide/cli_parser.py` | `official_repo_clones/disguide/disguide/cli_parser.py:15`, `official_repo_clones/disguide/disguide/cli_parser.py:99`, `official_repo_clones/disguide/disguide/train.py:152` |
| CloudLeak / FeatureFool | `official_repo_clones/cloudleak/optimize.py` | `official_repo_clones/cloudleak/optimize.py:12`, `official_repo_clones/cloudleak/optimize.py:13` |
| Blackbox Dissector | `official_repo_clones/blackbox-dissector/attack.py` | `official_repo_clones/blackbox-dissector/attack.py:48`, `official_repo_clones/blackbox-dissector/attack.py:684` |
| FFF | `official_repo_clones/fff/train.py` | `official_repo_clones/fff/train.py` |
| ActiveThief | `official_repo_clones/activethief/generic_program.py`, `official_repo_clones/activethief/cfg.py`, `official_repo_clones/activethief/utils/model.py` | `official_repo_clones/activethief/generic_program.py:61-67`, `official_repo_clones/activethief/cfg.py:78-86`, `official_repo_clones/activethief/sss/uncertainty_sss.py` |

### 1.2 Internal attacks found in `mebench/attackers/`

`mebench/attackers/__init__.py:3` exports:

- `ActiveThief`, `DFME`, `MAZE`, `DFMSHL` (`DFMS` alias), `DisGUIDE`, `GAME`, `ESAttack`, `SwiftThief`
- `BlackboxDissector`, `CloudLeak`, `BlackboxRipper`, `CopycatCNN`, `InverseNet`, `KnockoffNets`, `RandomBaseline`

## 2) Mapping Table (required parity fields)

| Official attack | mebench attack | Status | Official code location | Internal code location | Victim query type + budget counting | Hyperparameters + source | Preprocessing pipeline + source | Porting cautions |
|---|---|---|---|---|---|---|---|---|
| DFME | `DFME` | DONE | `official_repo_clones/datafree-model-extraction/dfme/train.py:57` | `mebench/attackers/dfme.py` | Official uses teacher logits/softmax path; mebench uses `BenchmarkContext.query` -> `Oracle.query` (`mebench/core/context.py:68`, `mebench/oracles/oracle.py:66`) with 1-image budget decrement | `batch_size=256`, `g_iter=1`, `d_iter=5`, `lr_S=0.1`, `grad_epsilon=1e-3` (`official_repo_clones/datafree-model-extraction/dfme/train.py:182`, `official_repo_clones/datafree-model-extraction/dfme/train.py:185`, `official_repo_clones/datafree-model-extraction/dfme/train.py:186`, `official_repo_clones/datafree-model-extraction/dfme/train.py:188`, `official_repo_clones/datafree-model-extraction/dfme/train.py:221`) | CIFAR test normalize profile implemented as `dfme_cifar10_test` (`mebench/data/preprocessing.py:54`) from `official_repo_clones/datafree-model-extraction/dfme/dataloader.py` | Keep data-free query tensors in tanh space (`[-1,1]`) at oracle boundary |
| MAZE | `MAZE` | DONE | `official_repo_clones/maze/src/attacks/maze.py:30` | `mebench/attackers/maze.py` | Official computes `budget_per_iter`; mebench decrements on every query image (`mebench/core/context.py:68`, `mebench/oracles/oracle.py:66`) | `iter_clone`, `iter_gen`, `ndirs`, cosine schedule with SGD (`official_repo_clones/maze/src/attacks/maze.py:39`, `official_repo_clones/maze/src/attacks/maze.py:40`, `official_repo_clones/maze/src/attacks/maze.py:55`) | Official RGB normalization mapped as `maze_rgb_test` (`mebench/data/preprocessing.py:68`) | MAZE repo includes jbda/noise/knockoff variants with separate loops; only MAZE core loop mapped to `MAZE` |
| KnockoffNets | `KnockoffNets` | DONE | `official_repo_clones/knockoffnets/knockoff/adversary/transfer.py:93` | `mebench/attackers/knockoff_nets.py` | Official blackbox returns softmax probabilities; mebench enforces soft-prob mode for this attack (`mebench/core/validate.py:77`) and per-image budget | `--budget`, `--batch_size` defaults from transfer CLI (`official_repo_clones/knockoffnets/knockoff/adversary/transfer.py:101`, `official_repo_clones/knockoffnets/knockoff/adversary/transfer.py:104`) | Official ImageNet test transform mapped as `knockoffnets_default_test` (`mebench/data/preprocessing.py:90`) | Preserve dataset-family transform differences; do not silently apply ImageNet normalization unless profile requested |
| SwiftThief | `SwiftThief` | DONE | `official_repo_clones/swiftthief/swiftthief.py:44` | `mebench/attackers/swiftthief.py` | Official budget split is 10% seed + periodic additions; mebench mirrors scheduling while budgeting via oracle image counts | Seed 10% and periodic augmentation/sampling (`official_repo_clones/swiftthief/swiftthief.py:50`, `official_repo_clones/swiftthief/swiftthief.py:132`, `official_repo_clones/swiftthief/swiftthief.py:150`) | Official CIFAR normalization profile mapped as `swiftthief_cifar_test` (`mebench/data/preprocessing.py:74`) | Official repo expects `unlabeled_dataset.pt`; mebench adapts to framework pool loaders and state |
| GAME | `GAME` | DONE | `official_repo_clones/game_attack/attack.py:11` | `mebench/attackers/game.py` | Official uses `querybudget`; mebench honors same semantics with strict context budget gate | `batch_size=1024`, `querybudget=2000`, `attack_train_epoch=40` (`official_repo_clones/game_attack/attack.py:23`, `official_repo_clones/game_attack/attack.py:24`, `official_repo_clones/game_attack/attack.py:26`) | Dataset transform path from official `GAME/datasets.py`; data-free query path fixed to tanh at oracle boundary | Keep class-conditional synthetic loop parity while preserving mebench artifact and evaluation hooks |
| DFMS-HL | `DFMSHL` | DONE | `official_repo_clones/dfms_hl/code/train_student/train_student.py:81` | `mebench/attackers/dfms.py` | Hard-label regime in official scripts; mebench enforces hard-top1 requirement for `dfms` (`mebench/core/validate.py:90`) | `lr=0.1`, `max_epochs=200`, synthetic sample defaults (`official_repo_clones/dfms_hl/code/train_student/train_student.py:93`, `official_repo_clones/dfms_hl/code/train_student/train_student.py:95`, `official_repo_clones/dfms_hl/code/train_student/train_student.py:126`) | Official `(0.5,0.5,0.5)` normalize mapped as `dfms_hl_train_student` (`mebench/data/preprocessing.py:108`) | Official is multi-stage script pipeline; mebench keeps stage semantics in attack runner state machine |
| DisGUIDE | `DisGUIDE` | DONE | `official_repo_clones/disguide/disguide/train.py`, `official_repo_clones/disguide/disguide/cli_parser.py` | `mebench/attackers/disguide.py` | Official updates query budget in-loop; mebench tracks via oracle/context image counting | Query budget in millions + no-logits flags (`official_repo_clones/disguide/disguide/cli_parser.py:15`, `official_repo_clones/disguide/disguide/cli_parser.py:99`) | Official supports pre/post transform assumptions in `disguide/dataloader.py`; mebench keeps canonical input + internal scale mode options | In hard mode, use HL loss only; mebench enforces this guard in `DisGUIDE` config checks |
| CloudLeak | `CloudLeak` | DONE | `official_repo_clones/cloudleak/optimize.py` | `mebench/attackers/cloudleak.py` | Official optimization queries victim with generated candidates; mebench uses context query for each batch | L-BFGS params from official optimization script (`official_repo_clones/cloudleak/optimize.py:12`, `official_repo_clones/cloudleak/optimize.py:13`) | Official code is Caffe-style preprocessing/deprocessing; mebench normalizes to canonical [0,1] before oracle | Channel-order and preprocessing order differences must be explicit in parity report |
| Blackbox Dissector | `BlackboxDissector` | DONE | `official_repo_clones/blackbox-dissector/attack.py` | `mebench/attackers/blackbox_dissector.py` | Official iterative budget splits + hard-label training; mebench enforces hard-top1 and image-count budget | `initial_budget`, split budgets list in script (`official_repo_clones/blackbox-dissector/attack.py:48`, `official_repo_clones/blackbox-dissector/attack.py:684`) | Official includes dataset-specific normalize chains in `attack.py`; mebench keeps parity notes and profile option | File-path transfer set workflow in official is adapted into in-memory framework datasets |
| ActiveThief | `ActiveThief` | DONE | `official_repo_clones/activethief/generic_program.py` | `mebench/attackers/activethief.py` | Official uses iterative budget split (`initial_seed`, `num_iter`, `k`) and strategy-driven selection; mebench tracks budget per-image and maps rounds through runner state | `initial_seed`, `num_iter`, `k`, `iterative`, `sampling_method` equivalents and round scheduling | Official `train_copynet_iter` + `get_queries`/`get_next_batch` flow adapted into `_bootstrap_seed_and_validation_sets` + `_select_query_batch` + `train_substitute` | Keep strategy-specific path for uncertainty/k-center/DFAL aligned with contract batch/image conventions |
| FFF | N/A | NOT_IN_SCOPE | `official_repo_clones/fff/train.py` | N/A | Fast Feature Fool is a universal perturbation method, not the model-extraction attack family used by this benchmark matrix | N/A | N/A | Excluded from benchmark attack matrix scope |
| MAZE-JBDA variant | N/A | NOT_IN_SCOPE | `official_repo_clones/maze/src/attacks/jbda.py` | N/A | Variant attack baseline, not part of current benchmark matrix identifiers | N/A | Uses MAZE dataset transform path | Can be added later as separate attack id if scope expands |
| MAZE-noise baseline | N/A | NOT_IN_SCOPE | `official_repo_clones/maze/src/attacks/noise.py` | N/A | Variant attack baseline, not part of current benchmark matrix identifiers | N/A | Uses MAZE dataset transform path | Can be added later as separate attack id if scope expands |

## 3) Non-official-clone internal attacks (paper-based ports)

The following mebench attacks do not have direct code in `official_repo_clones/`, but are maintained for matrix completeness and paper-faithful behavior.

| mebench attack | Status | Notes |
|---|---|---|
| `CopycatCNN` | DONE_PAPER | Paper-faithful offline augmentation + query-train rounds integrated into AttackRunner interface |
| `InverseNet` | DONE_PAPER | Phase-based extraction loop integrated with strict budget gate |
| `BlackboxRipper` | DONE_PAPER | Evolutionary latent optimization runner integrated with strict oracle budget and output-mode checks |
| `RandomBaseline` | DONE | Framework baseline, no official clone expected |
| `ESAttack` | DONE | Framework data-free baseline aligned to benchmark contract |

## 4) Preprocessing compatibility layer (implemented)

- Module: `mebench/data/preprocessing.py`
  - `get_official_preprocess(profile)`
  - `list_official_preprocess_profiles()`
  - `apply_official_preprocess_batch(x, profile)`
- Oracle integration: `mebench/oracles/oracle.py:73`
  - `victim.official_preprocess_profile` applies official-compatible transform chain right before victim forward.
  - One-time runtime trace log: profile and transform steps.
- Config validation: `mebench/core/validate.py:133`
  - Validates profile name against registered map.

## 5) Parity verification hooks available

- Smoke tests: `tests/test_attack_porting_smoke.py`
- Preprocess parity tests: `tests/test_attack_porting_preprocess.py`
- Default/hparam parity checks: `tests/test_attack_porting_defaults.py`
- Per-attack logic routing checks: `tests/test_attack_porting_per_attack_logic.py`

Additional metric-parity execution is tracked in `mebench/attackers/ATTACK_PARITY_REPORT.md` and requires long-running matrix runs.
