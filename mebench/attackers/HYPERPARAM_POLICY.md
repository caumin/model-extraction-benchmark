# Attack Hyperparameter Policy (Fixed vs Tunable)

This document separates per-attack knobs into:

- **Fixed-required**: values/semantics that should stay aligned with paper or official code.
- **Tunable**: benchmark-level knobs that can be changed for scaling/speed/fairness studies.

Scope:

- Evidence source priority: `papers/*.pdf` / `papers/paper_text/*` and `official_repo_clones/*`.
- Implementation status evaluated against `mebench/attackers/*.py` and `generate_configs.py`.

## Matrix

| Attack | Fixed-required (paper/official semantic) | Tunable (benchmark) | Current status in mebench |
|---|---|---|---|
| `activethief` | Image/text batch split is paper-specific (`150` image, `50` text) and optimizer family in official cfg is Adam (`learning_rate=0.01`) | Round count, strategy variants, val split details | **Partial**: image batch is set in matrix configs; training stack differs from TF1 official pipeline (`official_repo_clones/activethief/cfg.py:39-44`) |
| `blackbox_dissector` | Core defaults from official script: `batch_size=128`, `lr=0.02`, `momentum=0.9`, `train_epochs=200` | Selection batch, iterative budget schedule, erase ratio exploration | **Code-aligned / Matrix-unified substitute** (`official_repo_clones/blackbox-dissector/attack.py:44-55`, `mebench/attackers/blackbox_dissector.py:456-463`) |
| `blackbox_ripper` | Upstream SGD train defaults (`lr=0.01`, `momentum=0.9`), evolutionary constants (`population=30`, `elite=10`) | Extra stopping criteria, logging cadence, strict re-label behavior | **Aligned/Partial**: core defaults preserved; optional controls added (`official_repo_clones/blackbox-ripper/trainer/train_or_restore_predictor.py:27-33`, `mebench/attackers/blackbox_ripper.py:117-140`) |
| `cloudleak` | No single stable official tuple across provided artifacts (AlexNet/ResNet/Caffe solvers disagree) | Most training knobs remain benchmark tunables until unified reference chosen | **Heuristic** due conflicting official settings (`official_repo_clones/cloudleak/pre_trained network/alexnet/alexnet_solver.prototxt:4-11`, `official_repo_clones/cloudleak/pre_trained network/resnet50/resnet50_solver.prototxt:7-13`) |
| `copycatcnn` | NPDD-style attack protocol is the key semantic constraint; strong canonical official training script is not available in local clone | Batch, epochs, optimizer details | **Heuristic/Constrained**: NPDD guard enforced, training defaults tunable (`mebench/attackers/copycatcnn.py:101-104`, `mebench/attackers/copycatcnn.py:117-123`) |
| `dfme` | Official defaults: `batch=256`, `g_iter=1`, `d_iter=5`, `lr_S=0.1`, `lr_G=1e-4` | Eval cadence, checkpointing, scheduler granularity | **Code-aligned / Matrix-unified substitute** (`official_repo_clones/datafree-model-extraction/dfme/train.py:182-189`, `mebench/attackers/dfme.py:26-33`, `mebench/attackers/dfme.py:67-73`) |
| `dfms` | Official staged pipeline and stage-specific LRs/epochs are semantic core (init student vs alternate stage) | Budget planner mode, stage query allocation details | **Mostly aligned** in staged implementation; still benchmarkized around planner (`official_repo_clones/dfms_hl/code/train_student/train_student.py:93-96`, `mebench/attackers/dfms.py:72-79`, `mebench/attackers/dfms.py:108-113`) |
| `disguide` | Official CIFAR script: `batch=256`, `lr_S=0.03`, `lr_G=1e-4`, replay-style loop | Replay size, grayscale cadence, eval interval | **Code-aligned / Matrix-unified substitute** (`official_repo_clones/disguide/run_cifar-10.sh:13`, `official_repo_clones/disguide/run_cifar-10.sh:22`, `mebench/attackers/disguide.py:145-154`) |
| `ds` | Official DS defaults: `lr_S=0.3`, `lr_G=1e-4`, `num_students>=2` | Logging/eval intervals, strict budget stepping | **Code-aligned / Matrix-unified substitute** (`official_repo_clones/dual_students/run_cifar_ds.sh:11-14`, `mebench/attackers/ds.py:51-56`) |
| `es_attack` | Local paper-faithful constraints encoded (class-conditional DNN-SYN, init Gaussian step) | Iteration counts, synthesis mode variants, optimization steps | **Constrained heuristic** (official clone mapping is limited locally) (`mebench/attackers/es_attack.py:69-75`, `mebench/attackers/es_attack.py:97-106`) |
| `game` | Official attack script defaults (`batch_size=1024`, `querybudget=2000`, `attack_train_epoch=40`) and beta weights | Proxy data handling, sampler strategy, benchmark normalization controls | **Partial**: major defaults mirrored; benchmark contract adds adaptations (`official_repo_clones/game_attack/attack.py:23-27`, `mebench/attackers/game.py:27-57`) |
| `inversenet` | Paper-style 3-phase schedule and sparse retrain points are semantic core | Batch/lr/epochs around inversion and retraining under budget constraints | **Constrained heuristic** (official repo evidence limited in local clone) (`mebench/attackers/inversenet.py:47-50`, `mebench/attackers/inversenet.py:87-99`) |
| `knockoff_nets` | Official split semantics: query batch in transfer construction (`8`), train batch in knockoff training (`64`), train lr `0.01`, momentum `0.5` | Adaptive reward shaping details, hierarchy parameters, retraining cadence | **Aligned split**: query/update batch pinned to `8`; substitute-train path benchmark-aligned (`official_repo_clones/knockoffnets/knockoff/adversary/transfer.py:104`, `official_repo_clones/knockoffnets/knockoff/adversary/train.py:108-115`, `mebench/attackers/knockoff_nets.py:35-43`) |
| `marich` | Local official sources are inconsistent for optimizer/LR values | Selection strategy controls, round budget growth, training knobs | **Heuristic/Partial** until single reference profile is locked (`official_repo_clones/MARICH/lr_cnn_res_al/utils.py:45-49`, `official_repo_clones/MARICH/lr_cnn_res_al/nets.py:57-70`, `mebench/attackers/marich.py:60-69`) |
| `maze` | Official defaults: `batch=128`, `lr_clone=0.1`, `iter_gen=1`, `iter_clone=5`, ZO params (`eps`, `ndirs`) | Eval cadence, strict budget scheduling, architecture swap experiments | **Code-aligned / Matrix-unified substitute** (`official_repo_clones/maze/src/utils/config.py:16`, `official_repo_clones/maze/src/utils/config.py:53`, `official_repo_clones/maze/src/utils/config.py:92-96`, `mebench/attackers/maze.py:103-109`) |
| `random_baseline` | No paper-anchored fixed hyperparameter profile; role is benchmark control baseline | Query step size, training cadence, substitute train settings | **Heuristic baseline** by design (`mebench/attackers/random_baseline.py:25-35`, `mebench/attackers/random_baseline.py:74`) |
| `swiftthief` | Official script pins key training knobs (`sl_lr=1e-2`, `sl_epoch=500`, interval schedule) | CL/KD epochs, queue sizes, scoring worker settings | **Code-aligned / Matrix-unified substitute** (`official_repo_clones/swiftthief/scripts/swiftthief/cifar10.sh:28-31`, `generate_configs.py`) |

## Config Generator Enforcement (Matrix)

Matrix generation enforces setup-level unified substitute defaults:

- **SET-B1 (`substitute.arch=resnet18`)**: `batch=256`, `optimizer=sgd(lr=0.1,momentum=0.9,wd=5e-4)`, `scheduler=multistep([0.5,0.75],gamma=0.1)`, `max_epochs=1000`, `patience=100`.
- **SET-A1 (`substitute.arch=lenet_mnist`)**: `batch=512`, `optimizer=sgd(lr=0.04,momentum=0.9,wd=5e-4)`, `scheduler=multistep([0.5,0.75],gamma=0.1)`, `max_epochs=200`, `patience=20`.
- **SET-C1 (`substitute.arch=xie2019`)**: `batch=256`, `optimizer=sgd(lr=0.1,momentum=0.9,wd=5e-4)`, `scheduler=multistep([0.5,0.75],gamma=0.1)`, `max_epochs=90`, `patience=90`; best checkpoint is selected by validation loss and the run is intended to complete the full 90 epochs.
- Matrix generation does not apply per-attack LR/batch alignment overrides; attack-level fields are retained only when they are semantic to the attack algorithm.

Reference: `generate_configs.py` matrix policy section and `generate_configs()` default substitute block.

## Practical Rule

- **Do not tune** fixed-required knobs in default benchmark runs unless running explicit ablations.
- **Tune only** tunable knobs for speed/resource studies, and record deviations in run metadata.

## Current Open Risks (Action Items)

1. **ActiveThief optimizer/batch modality split**: official config is TF1-centric (`batch_size=50`, Adam), while image paper profile uses 150; current benchmark path should keep explicit image-profile comments and avoid accidental text-profile reuse.
2. **CloudLeak reference inconsistency**: official artifacts expose multiple solver profiles with conflicting base LR/weight-decay values; choose one canonical profile before hard-locking defaults.
3. **MARICH optimizer ambiguity**: local official files show conflicting LR behaviors; lock one canonical setting before promoting from heuristic to fixed-required.
4. **CopycatCNN source incompleteness**: no stable local official clone script currently available; treat current defaults as benchmark heuristic with NPDD constraint enforced.
5. **InverseNet evidence gap**: direct official clone source is not present in local clones; keep phase semantics fixed but treat LR/batch as tunable pending stronger source evidence.
