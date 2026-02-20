# Attack Porting Plan (official_repo_clones -> mebench)

This document is the design/reference map for fairness-oriented attack porting.

Scope:
- Official sources: `official_repo_clones/` (read-only)
- Internal targets: `mebench/attackers/`

## 1) Inventory

### 1.1 Official attack implementations discovered

| Official attack | Entrypoint(s) | Key config/hparams source |
|---|---|---|
| DFME | `official_repo_clones/datafree-model-extraction/dfme/train.py:57`, `...:179` | `.../dfme/train.py:207-244` |
| MAZE | `official_repo_clones/maze/src/attacker.py:65`, `official_repo_clones/maze/src/attacks/maze.py:30` | `official_repo_clones/maze/src/utils/config.py`, `.../maze.py:39-43` |
| MAZE-Knockoff variant | `official_repo_clones/maze/src/attacks/knockoff.py:13` | `.../knockoff.py:17-25` |
| MAZE-JBDA variant | `official_repo_clones/maze/src/attacks/jbda.py:104` | `.../jbda.py:119-149` |
| MAZE-Noise baseline | `official_repo_clones/maze/src/attacks/noise.py` | `.../noise.py` |
| KnockoffNets | `official_repo_clones/knockoffnets/knockoff/adversary/transfer.py`, `.../train.py` | `.../transfer.py` CLI `--budget`, `.../victim/blackbox.py` |
| SwiftThief | `official_repo_clones/swiftthief/swiftthief.py:44` | `.../swiftthief.py:260-274`, `.../utils/configs.py` |
| GAME | `official_repo_clones/game_attack/attack.py:11` | `.../attack.py:14-28` |
| DFMS-HL | `official_repo_clones/dfms_hl/code/train_student/train_student.py:81`, `.../train_gen.py` | `.../train_student.py:93-127` |
| CloudLeak / FeatureFool | `official_repo_clones/cloudleak/optimize.py:132`, `.../MCV_query.py` | `.../optimize.py:12-14`, `.../README.md` |
| FFF | `official_repo_clones/fff/train.py`, `.../evaluate.py` | `.../train.py` |

### 1.2 Current mebench attackers

`mebench/attackers/__init__.py:3-39` exports:

- `ActiveThief`, `DFME`, `MAZE`, `DFMSHL` (`DFMS` alias), `GAME`, `ESAttack`, `SwiftThief`,
- `BlackboxDissector`, `CloudLeak`, `BlackboxRipper`, `CopycatCNN`, `InverseNet`, `KnockoffNets`, `RandomBaseline`

## 2) Mapping Table

| Official name | mebench name | Status |
|---|---|---|
| DFME | `DFME` | DONE |
| MAZE | `MAZE` | DONE |
| KnockoffNets | `KnockoffNets` | DONE |
| SwiftThief | `SwiftThief` | DONE |
| GAME | `GAME` | DONE |
| DFMS-HL | `DFMSHL` | DONE |
| CloudLeak (FeatureFool pipeline) | `CloudLeak` | DONE |
| MAZE-JBDA (variant) | N/A (not standalone) | TODO |
| MAZE-noise baseline | N/A (not standalone) | TODO |
| FFF | N/A (no mebench attacker) | TODO |
| ActiveThief (paper ref) | `ActiveThief` | IN_PROGRESS (no direct official clone in scope) |
| CopycatCNN (paper ref) | `CopycatCNN` | IN_PROGRESS (no direct official clone in scope) |
| InverseNet (paper ref) | `InverseNet` | IN_PROGRESS (no direct official clone in scope) |
| Blackbox Dissector (paper ref) | `BlackboxDissector` | IN_PROGRESS (no direct official clone in scope) |
| Blackbox Ripper (paper ref) | `BlackboxRipper` | IN_PROGRESS (no direct official clone in scope) |

## 3) Per-attack porting notes (required fields)

### DFME -> mebench/attackers/dfme.py
- Official code location: `official_repo_clones/datafree-model-extraction/dfme/train.py:57-136`, `.../approximate_gradients.py:15-146`
- Internal code location: `mebench/attackers/dfme.py`
- Victim query type / budget counting:
  - Official: soft probabilities recovered to logits (`train.py` + `approximate_gradients.py`);
  - mebench: oracle `soft_prob` with per-image budget decrement in `mebench/oracles/oracle.py:60-65`
- Hyperparameters (official defaults):
  - `batch_size=256`, `n_g=1`, `n_s=5`, `epsilon=1e-3`, `m=1`, schedule at 10/30/50% (`train.py` and paper-aligned comments)
- Preprocess pipeline (official):
  - CIFAR10 train/test uses normalize `(0.4914,0.4822,0.4465)/(0.2023,0.1994,0.2010)` in `.../dfme/dataloader.py:45-59`
- Porting caution:
  - Official code often expects normalized inputs; benchmark contract uses `[0,1]` canonical oracle inputs.

### MAZE -> mebench/attackers/maze.py
- Official code location: `official_repo_clones/maze/src/attacks/maze.py:30-283`
- Internal code location: `mebench/attackers/maze.py`
- Victim query type / budget counting:
  - Official uses KL over logits/probs and computes `budget_per_iter` in `.../maze.py:39-43`
  - mebench uses `BenchmarkContext.query(...)` and image-count budget in `mebench/core/context.py:60-90`
- Hyperparameters (official defaults):
  - `iter_clone`, `iter_gen`, `ndirs`, `alpha_gan`, optimizer/lr in `.../maze.py:44-63`
- Preprocess pipeline (official):
  - dataset transforms in `official_repo_clones/maze/src/datasets/datasets.py:84-186`
  - default normalize `(0.5,0.5,0.5)` or grayscale equivalents
- Porting caution:
  - MAZE repo includes multiple attack variants (`noise`, `knockoff`, `jbda`) with different assumptions.

### KnockoffNets -> mebench/attackers/knockoff_nets.py
- Official code location: `official_repo_clones/knockoffnets/knockoff/adversary/transfer.py`, `.../adversary/train.py`, `.../victim/blackbox.py`
- Internal code location: `mebench/attackers/knockoff_nets.py`
- Victim query type / budget counting:
  - Official blackbox returns softmax probs in `.../blackbox.py` and transfer uses `--budget` image count in `.../transfer.py`
  - mebench enforces same image-count accounting via oracle/context path.
- Hyperparameters (official defaults):
  - transfer `batch_size`, budget; train epochs/lr in `.../adversary/train.py`
- Preprocess pipeline (official):
  - ImageNet family transforms in `official_repo_clones/knockoffnets/knockoff/utils/transforms.py:15-30`
- Porting caution:
  - Family-specific transforms and output truncation/rounding options in official blackbox should be explicitly documented when disabled.

### SwiftThief -> mebench/attackers/swiftthief.py
- Official code location: `official_repo_clones/swiftthief/swiftthief.py:44-285`, `.../utils/get_datasets.py:20-47`
- Internal code location: `mebench/attackers/swiftthief.py`
- Victim query type / budget counting:
  - Official query budget used by scheduling/splits (`swiftthief.py:50-52`, `:144-159`)
  - mebench uses strict oracle image budget and round-based query batches.
- Hyperparameters (official defaults):
  - `sl_lr`, `sl_epoch`, `sl_aug_interval` in `swiftthief.py:267-270`
- Preprocess pipeline (official):
  - CIFAR group: resize + random crop + flip + normalize in `.../get_datasets.py:21-32`
  - MNIST group: grayscale->3ch + crop/flip + normalize in `.../get_datasets.py:35-47`
- Porting caution:
  - official code mixes CL and KD with dataset-specific assumptions (e.g., unlabeled_dataset.pt layout).
  - 1:1 scheduling anchors now mirrored:
    - official initial split `10%` (`official_repo_clones/swiftthief/swiftthief.py:50-52`)
    - sampling trigger `epoch % sl_aug_interval == 0 and epoch != 0` (`official_repo_clones/swiftthief/swiftthief.py:132-133`)
    - imbalance mode uses 5 sub-rounds (`official_repo_clones/swiftthief/swiftthief.py:149-160`)

### GAME -> mebench/attackers/game.py
- Official code location: `official_repo_clones/game_attack/attack.py:11-99`, `.../GAME/methods.py`
- Internal code location: `mebench/attackers/game.py`
- Victim query type / budget counting:
  - official uses `querybudget` arg (`attack.py:24`) and method-level extraction calls.
  - mebench: strict per-image budget via context/oracle.
- Hyperparameters (official defaults):
  - `batch_size=1024`, `querybudget=2000`, `attack_train_epoch=40` in `attack.py:23-27`
- Preprocess pipeline:
  - defined in `game_attack/GAME/datasets.py` loaders.
- Porting caution:
  - official repo can run baseline-only paths; ensure equivalent mode in mebench configs.
  - 1:1 loop anchors now mirrored:
    - official budget-tail slicing (`official_repo_clones/game_attack/GAME/methods.py:83-89`)
    - official defaults `batch_size=1024`, `querybudget=2000`, `attack_train_epoch=40` (`official_repo_clones/game_attack/attack.py:23-27`)
    - per-round attacker training + final full-buffer train (`official_repo_clones/game_attack/GAME/methods.py:112-114`, `208-214`)

### DFMS-HL -> mebench/attackers/dfms.py
- Official code location: `official_repo_clones/dfms_hl/code/train_student/train_student.py`, `.../train_generator/train_gen.py`, `.../train_generator_clone.py`
- Internal code location: `mebench/attackers/dfms.py`
- Victim query type / budget counting:
  - official primarily hard-label student supervision (teacher argmax) in `train_student.py:394-403`
  - mebench uses hard-top1 oracle mode enforcement in `mebench/core/validate.py:77-95`
- Hyperparameters (official defaults):
  - `lr=0.1`, `max_epochs=200`, ratios and sample sizes in `train_student.py:93-127, 305-306`
- Preprocess pipeline (official):
  - normalization `(0.5,0.5,0.5)` in `train_student.py:241-246`, test `:413-416`
- Porting caution:
  - official scripts are multi-stage and script-driven; preserve stage semantics in attack config.

### CloudLeak -> mebench/attackers/cloudleak.py
- Official code location: `official_repo_clones/cloudleak/optimize.py:132-172`, `.../README.md`
- Internal code location: `mebench/attackers/cloudleak.py`
- Victim query type / budget counting:
  - official pipeline describes synthetic generation + query + label workflow (README)
  - mebench wraps this via context/oracle budget counting.
- Hyperparameters (official defaults):
  - L-BFGS params `factr=1e7`, `pgtol=1e-5` in `optimize.py:12-14`
- Preprocess pipeline (official):
  - Caffe preprocess/deprocess path in `optimize.py:134-149` + `utils.py`
- Porting caution:
  - channel/order differences (Caffe-style) and bound constraints must be explicitly handled.

## 4) Preprocessing compatibility layer plan

Implemented module:
- `mebench/data/preprocessing.py`
  - `get_official_preprocess(profile)`
  - `list_official_preprocess_profiles()`
  - `apply_official_preprocess_batch(x, profile)`

Integrated hook:
- `mebench/oracles/oracle.py`
  - optional `victim.official_preprocess_profile` applied immediately before victim forward
  - one-time runtime log of active profile for traceability

Validation:
- `mebench/core/validate.py`
  - checks `victim.official_preprocess_profile` is one of registered profiles
