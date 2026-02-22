# GAME Parity Checklist (Paper + Official Repo + mebench)

Last updated: 2026-02-21

## Scope

- Paper: `papers/GAME.pdf` (ESORICS 2022)
- Official repo: `official_repo_clones/game_attack/`
- Port target: `mebench/attackers/game.py`

## A. Formula-Level Parity

- [x] **Generator objective terms present** (`L_res`, `L_bou`, `L_adv`, `L_dif`) in mebench AGU path.
- [x] **Generator total loss form** matches Eq.14 style weighted sum.
- [x] **ACGAN pretraining objective** (`L_C +/- L_S`) is represented in TDL.
- [ ] **Student distillation schedule exactly official-like** (official `soft_train` creates fresh optimizer each round with internal LR decay).
- [x] **Critical fix applied**: student loss is no longer multiplied by `beta1` in GMD.
  - Previous behavior: `loss = beta1 * loss_res` (default `beta1=0.002`) made student updates nearly vanish.
  - Current behavior: `loss = loss_res`.
- [x] **AGU loss scaling aligned closer to official**:
  - `L_res`: positive-logit sum (not batch mean)
  - `L_bou`: top1-top2 logit margin sum (not batch mean)
  - `L_dif`: KL uses `reduction="mean"` like official callsite
- [x] **Paper ablation best-loss combo promoted to default**:
  - GAME default AGU terms are now `res+bou+dif` (without `adv`) via `agu_loss_terms`.
  - Configurable override remains available for ablation parity (`adv` can still be enabled explicitly).

## B. Protocol/Order Parity

- [x] **Phase presence**: TDL -> ACS -> GMD/AGU loop -> final train exists.
- [x] **Order in extraction loop**: student update (GMD) occurs before generator update (AGU).
- [x] **Round-wise query accumulation**: queried pairs are buffered and reused.
- [x] **AGU discriminator update parity**: AGU discriminator update path is disabled by default (`agu_update_discriminator=false`) for official-like generator-only adaptive updates.
- [x] **AGU discriminator update now optional and default-off** for official-like behavior.
- [x] **AGU step count default parity**: `agu_steps` default is now `2` to mirror official GAME adaptive loop frequency.
- [x] **Official-per-round optimizer recreation parity**: student full-buffer train now rebuilds optimizer per call with midpoint LR drop behavior.
- [x] **Final retrain epoch parity default**: final retrain now defaults to `attack_train_epoch - 10` (`final_retrain_epoch_offset=10`).
- [x] **ACS sampler-weight update parity (official logic)**: sampler weights now update from last AGU step using official-style rules (`random/confident/unconfident/deviation`) instead of cumulative running-statistics approximation.

## C. Methodology/Experimental Parity

- [x] **Query budget parity (pair-1)**: `8k` configured for `2022_xie_game` full profile.
- [x] **Proxy pair parity**: `MNIST` victim + `FashionMNIST` proxy configured.
- [x] **Input size parity**: GAME repro configs switched to `32x32` for victim/proxy paths.
- [x] **Substitute architecture parity (pair-1 main)**: repro now uses `half_lenet` substitute.
- [x] **Preprocess parity hook**: GAME now supports `repro_input_scale_mode: tanh` (`[-1,1]` internal path).
- [x] **Victim parity profile**: retrained `lenet` victim checkpoint at `32x32`; current official-like repro configs keep `victim.input_scale_mode=unit` (no wrapper conversion).
- [x] **Benchmark query-accounting compliance**: GAME attack path uses metered `ctx.query` only for victim access (1 image = 1 budget unit).
- [x] **AGU victim-query accounting parity-adaptation**: AGU-step victim outputs are fetched via metered oracle calls; no unmetered victim forward remains.
- [x] **ACGAN fake-class semantics parity**: discriminator aux head predicts `n_classes+1`, and fake samples are supervised with dedicated fake-class index in TDL/AGU-discriminator updates.
- [x] **ACGAN architecture parity pass (generator/discriminator)**:
  - Generator switched to official-like upsample+conv path (instead of transposed-conv stack).
  - Discriminator switched to 5-stage conv block with linear source/aux heads.
- [x] **ACGAN low-level parameterization parity pass**:
  - BatchNorm arguments aligned with official code pattern (`BatchNorm2d(..., 0.8)` usage).
  - GAME-specific conv/bn weight initialization (`normal_(0,0.02)` / `normal_(1,0.02)`) applied at model creation.
- [x] **TDL loop parity pass**:
  - TDL now iterates over the full proxy loader for `tdl_steps` epochs (official-style), not one-batch-per-step.
  - Added label-smoothing/flip pattern in TDL source-loss targets using per-epoch batch index (`i % n_output`, `i % 25`) semantics.

## D. Code-Review Checklist (Actionable)

- [x] Verify `GMD` student loss scaling bug is removed in `mebench/attackers/game.py`.
- [ ] Decide whether to disable AGU-time discriminator updates for strict official behavior.
- [ ] Run smoke reproduction and confirm non-random learning curve (accuracy/fidelity should not stay near 10%).
- [ ] Run full pair-1 reproduction and compare against paper target (`acc_gt ~ 0.9036` for half-lenet setting reference).
- [x] Set best-loss default from paper ablation (`res+bou+dif`) and wire repro configs to match.
- [ ] If gap remains large, perform ablation runs:
  - [ ] `acs_strategy`: random vs uncertainty
  - [ ] AGU indicator subsets (`res`, `bou`, `adv`, `dif`) around new default `res+bou+dif`
  - [ ] optimizer/lr sensitivity for current substitute architecture

## E. Verification Log

- [x] Source parity references inspected:
  - `official_repo_clones/game_attack/GAME/methods.py`
  - `official_repo_clones/game_attack/GAME/models.py`
  - `repro/papers/2022_xie_game/configs/experiment.yaml`
  - `mebench/attackers/game.py`
- [x] Runtime sanity check executed:
  - Command: `python -m mebench run --config repro/papers/2022_xie_game/configs/experiment_smoke.yaml --device cuda:0`
  - Result: run completed, but extraction remained near-random (`acc_gt` around `0.097-0.104`).
  - Interpretation: critical scaling bug fixed, but larger parity gaps remain and still block paper-level reproduction.
- [x] Runtime sanity recheck after optimizer/loss parity tweak:
  - Code/config deltas: `L_res` sign aligned to official code path; GAME repro configs switched to SGD (`lr=0.002`, momentum/weight_decay parity), and `agu_steps=2`.
  - Command: `python -m mebench run --config repro/papers/2022_xie_game/configs/experiment_smoke.yaml --device cuda:0`
  - Result: short-lived bump (`acc_gt=0.148` at 500 queries) followed by collapse to near-random (`acc_gt=0.0974`, `agreement=0.0963`) by 1000 queries.
  - Interpretation: remaining parity blocker is likely schedule-level (student per-round LR/recreation and/or final retrain semantics), not only loss sign/optimizer family.
- [x] Runtime sanity recheck after schedule patch (`final_retrain_epoch_offset` + official-like mid-epoch LR drop in full-buffer train):
  - Command: `python -m mebench run --config repro/papers/2022_xie_game/configs/experiment_smoke.yaml --device cuda:0`
  - Result: still unstable; temporary rise to `acc_gt=0.1518` (600-700 queries), then collapse to `acc_gt=0.0974`, `agreement=0.0963` at 800-1000 queries.
  - Interpretation: root mismatch remains in adaptive loop parity (likely AGU/GMD interaction or loss-term exactness), not only optimizer family or final retrain window.
- [x] Runtime sanity recheck after best-loss default migration (`agu_loss_terms=[res,bou,dif]` + `agu_steps=2` default):
  - Command: `python -m mebench run --config repro/papers/2022_xie_game/configs/experiment_smoke.yaml --device cuda:0`
  - Result: behavior remains unstable; brief rise to `acc_gt=0.1524` (600-700 queries) then collapse to `acc_gt=0.0974`, `agreement=0.0963` by 800-1000 queries.
  - Interpretation: paper-ablation best-loss default is now correctly reflected, but major parity gap remains elsewhere (likely protocol-level differences in AGU/GMD/query semantics).
- [x] Query-accounting regression test added (benchmark rule guardrail):
  - Test: `tests/test_game_budget_metering.py::test_game_run_uses_only_metered_oracle_queries`
  - Verification command: `python -m pytest tests/test_game_budget_metering.py tests/test_attack_porting_defaults.py tests/test_game_basic.py tests/test_game_proxy_class_alignment.py -q`
  - Result: `12 passed`.
  - Assertion: victim forward image-count equals `state.query_count` and equals configured budget in attack run path.
- [x] Runtime sanity after strict parity patches (metered AGU victim query + fake-class TDL/AGU + ACGAN arch closer to official):
  - Command: `python -m mebench run --config repro/papers/2022_xie_game/configs/experiment_smoke.yaml --device cuda:0`
  - Result: still low but slightly improved over immediate collapse (`acc_gt` around `0.103~0.116`, `agreement` around `0.103~0.115`).
  - Note: because AGU victim queries are now budget-metered, a 1k total budget includes AGU internal victim calls; effective outer labeled-set growth is lower than official unmetered implementation.
- [x] Runtime sanity after TDL full-epoch parity + BCE source-head alignment:
  - Command: `python -m mebench run --config repro/papers/2022_xie_game/configs/experiment_smoke.yaml --device cuda:0`
  - Result: stable completion with near-random performance (`acc_gt` around `0.1028~0.1032`, `agreement` around `0.1034~0.1038`).
  - Implementation note: official label smoothing formula can produce >1 targets; this caused CUDA BCE target assertions, so targets are clamped to `[0,1]` for stable execution in benchmark runtime.
- [x] Full pair-1 run rechecked after fixes:
  - Command: `python -m mebench run --config repro/papers/2022_xie_game/configs/experiment.yaml --device cuda:0`
  - Result: still near-random at checkpoints (`acc_gt ~ 0.0974`, `agreement ~ 0.0963`) with very large KL divergence.
  - Interpretation: confirmed that unresolved parity gaps are dominant failure factors; more algorithm-level corrections are required.

- [x] Victim retrain/eval for 32x32 tanh profile completed:
  - Train command: `python scripts/train_victim.py --config repro/papers/2022_xie_game/configs/victim_train.yaml --device cuda:0`
  - Eval command: `python scripts/eval_victim.py --config repro/papers/2022_xie_game/configs/victim_eval.yaml --device cuda:0`
  - Victim eval: `acc_gt = 0.9864` (`repro/papers/2022_xie_game/results/victim_eval.json`).

## F. Iteration Cycle (Current)

- [x] Budget-compensated smoke profile created (metered AGU query cost reflected).
  - Config: `repro/papers/2022_xie_game/configs/experiment_smoke_budgetcomp3k.yaml`
- [x] Baseline vs budget-compensated smoke compared with final `acc_gt/agreement`.
  - Baseline 1k (`repro_2022_xie_game_mnist_fmnist_smoke1k_seed0` @ `20260221_202000`): `acc_gt=0.1032`, `agreement=0.1038`
  - Budget-comp 3k (`repro_2022_xie_game_mnist_fmnist_smoke3k_budgetcomp_seed0` @ `20260221_202427`): final `acc_gt=0.0974`, `agreement=0.0963` (mid-run peak at 1500: `0.1506/0.1498`)
- [x] Targeted sweep completed over:
  - [x] `agu_loss_terms`: `[res,bou,dif]` vs `[res,bou,adv,dif]`
  - [x] `round_train_epochs`
  - [x] `student_lr`
  - Sweep configs:
    - `repro/papers/2022_xie_game/configs/experiment_smoke_sweep_loss_adv.yaml`
    - `repro/papers/2022_xie_game/configs/experiment_smoke_sweep_round5.yaml`
    - `repro/papers/2022_xie_game/configs/experiment_smoke_sweep_lr01.yaml`
    - `repro/papers/2022_xie_game/configs/experiment_smoke_sweep_combo.yaml`
- [x] Best smoke profile selected using final checkpoint metric.
  - Best final among current sweeps: `round_train_epochs=5` (`acc_gt=0.1501`, `agreement=0.1494` at 1k).
- [x] Full-budget run executed with selected best profile (if smoke improves).
  - Config: `repro/papers/2022_xie_game/configs/experiment_full_sweep_round5.yaml`
  - Run: `repro_2022_xie_game_mnist_fmnist_8k_round5_seed0` @ `20260221_203308`
  - Final checkpoint (8k): `acc_gt=0.1373`, `agreement=0.1376`.

## G. Iteration Cycle (Loop 2)

- [x] Additional smoke sweep on ACS strategy and step granularity:
  - `repro/papers/2022_xie_game/configs/experiment_smoke_sweep_round5_deviation.yaml`
  - `repro/papers/2022_xie_game/configs/experiment_smoke_sweep_round5_bs64.yaml`
  - `repro/papers/2022_xie_game/configs/experiment_smoke_sweep_round5_bs64_deviation.yaml`
- [x] Loop-2 smoke results:
  - `round5 + deviation`: final `acc_gt=0.1505`, `agreement=0.1501` (best smoke by small margin)
  - `round5 + batch64`: final `acc_gt=0.0974`, `agreement=0.0963`
  - `round5 + batch64 + deviation`: final `acc_gt=0.0974`, `agreement=0.0963`
- [x] Full run using loop-2 smoke winner:
  - Config: `repro/papers/2022_xie_game/configs/experiment_full_sweep_round5_deviation.yaml`
  - Run: `repro_2022_xie_game_mnist_fmnist_8k_round5_deviation_seed0` @ `20260221_204408`
  - Final checkpoint (8k): `acc_gt=0.1150`, `agreement=0.1154` (worse than round5-random full).
- [x] Best-known current full profile remains `round_train_epochs=5, acs_strategy=random` with `acc_gt=0.1373`, `agreement=0.1376` at 8k.

## H. Iteration Cycle (Loop 3: 3x Query Budget)

- [x] Tripled-budget full profile created to compensate metered AGU query usage:
  - `repro/papers/2022_xie_game/configs/experiment_full_sweep_round5_24k.yaml`
- [x] Full 24k run executed:
  - Run: `repro_2022_xie_game_mnist_fmnist_24k_round5_seed0` @ `20260221_205104`
  - Checkpoints (key):
    - `9000`: `acc_gt=0.1379`, `agreement=0.1382`
    - `24000`: `acc_gt=0.0974`, `agreement=0.0963`
- [x] Comparison vs current best 8k full (`round5 + random`):
  - 8k final: `acc_gt=0.1373`, `agreement=0.1376`
  - 24k final: `acc_gt=0.0974`, `agreement=0.0963` (degraded)
  - Interpretation: simply tripling budget does not stabilize late-stage collapse in current benchmark-compliant GAME port.

## I. Remaining Strict-Parity Gaps (Before Next Main Experiment)

- [ ] **Budget semantics mismatch (intentional, benchmark contract)**:
  - Official GAME does not charge AGU-internal victim forwards against `querybudget` (`methods.py` loop).
  - Benchmark requires `1 image == 1 query` for *every* victim call; current port enforces this.
  - This is the largest unavoidable protocol delta and must be called out in every result interpretation.
- [ ] **Victim access path mismatch**:
  - Official calls victim model directly in AGU/outer loop (`victim_model.model(...)`).
  - Port routes through oracle wrapper (`ctx.query`) for accounting and contract safety.
- [~] **Pretrained ACGAN artifact reuse mismatch (non-blocking)**:
  - Official path can `load=True` pre-trained GAN weights in `ACGAN.train(...)`.
  - Port currently re-runs TDL each run; no equivalent external pretrained checkpoint-loading path is wired.
  - Assessment: this is primarily a runtime/repro convenience delta, not the main fidelity-collapse driver.
- [ ] **Dataset/runtime environment parity still incomplete**:
  - Official experiments rely on original dataset wrappers/runtime defaults (workers/transforms/internal data path assumptions).
  - Port runs within benchmark data pipeline/contracts, which may alter dynamics despite matched formulas.

## J. Paper Pair-1 Reproduction Run (Current Code, Metered Query Rule)

- [x] Full reproduction run executed for paper pair-1 half-lenet with seeds `[0,1,2]`:
  - Config: `repro/papers/2022_xie_game/configs/experiment_paper_half_lenet.yaml`
  - Run directories:
    - `runs/repro_2022_xie_game_paper_pair1_half_lenet_8k_s012/20260221_212139/seed_0/summary.json`
    - `runs/repro_2022_xie_game_paper_pair1_half_lenet_8k_s012/20260221_212520/seed_1/summary.json`
    - `runs/repro_2022_xie_game_paper_pair1_half_lenet_8k_s012/20260221_212902/seed_2/summary.json`
- [x] 8k checkpoint results (track_b):
  - seed0: `acc_gt=0.0974`, `agreement=0.0963`
  - seed1: `acc_gt=0.1032`, `agreement=0.1043`
  - seed2: `acc_gt=0.0974`, `agreement=0.0963`
  - mean: `acc_gt=0.0993`, `agreement=0.0990`
- [x] Gap vs paper target (Table 1 half-lenet GAME accuracy `0.9036`):
  - absolute gap at 8k mean: `-0.8043`
  - conclusion: reproduction still fails under benchmark-compliant metered-query setup.

## K. Single-Seed Collapse Audit (No Multi-Seed Sweep)

- [x] Evaluated and removed tanh-eval wrapper path for official-like parity:
  - Problem: wrapping substitute evaluation with tanh conversion conflicts with normalized test-loader evaluation parity.
  - Fix: GAME evaluation now uses the raw substitute model; no tanh wrapper in `_evaluate_current_substitute`.
  - Code: `mebench/attackers/game.py`.
- [x] Re-ran single-seed paper profile after patch:
  - Config: `repro/papers/2022_xie_game/configs/experiment_paper_half_lenet_seed0.yaml`
  - Run: `runs/repro_2022_xie_game_paper_pair1_half_lenet_8k_seed0_evalfix/20260221_214237/seed_0/summary.json`
  - Result: 8k still collapsed (`acc_gt=0.0974`, `agreement=0.0963`), though KL explosion magnitude dropped (from ~`5e5` scale to ~`1.2e4` scale).
- [x] Updated oracle/internal scale routing for tanh path (query remains tanh at oracle boundary) and re-ran:
  - Code: `mebench/attackers/game.py` (`_internal_to_oracle_scale`, `_oracle_to_internal_scale`)
  - Run: `runs/repro_2022_xie_game_paper_pair1_half_lenet_8k_seed0_evalfix/20260221_214855/seed_0/summary.json`
  - Result: collapse persisted (`acc_gt=0.0974`, `agreement=0.0963`), indicating this was not the primary late-stage failure source.
- [x] Conclusion:
  - Evaluation-scale mismatch was a real bug and is fixed.
  - It is not sufficient alone to recover paper-level performance; additional parity gaps remain.

## L. Optimizer-Path Hypothesis (Single Seed)

- [x] Hypothesis run added: if official experiments used SGD path (attack.py default) rather than Adam-paper text, collapse might reduce.
  - Config: `repro/papers/2022_xie_game/configs/experiment_paper_half_lenet_seed0_sgd.yaml`
  - Run: `runs/repro_2022_xie_game_paper_pair1_half_lenet_8k_seed0_sgd_hyp/20260221_215357/seed_0/summary.json`
- [x] Checkpoint comparison vs Adam paper-profile (`...seed0_evalfix/.../summary.json`):
  - 1k: Adam `0.1075` / SGD `0.1075`
  - 2k~4k: both `0.0974`
  - 5k~7k: Adam `0.0974`, SGD `0.2361`
  - 8k: Adam `0.0974`, SGD `0.3679`
- [x] Interpretation:
  - Optimizer choice is a major collapse driver in current port.
  - Even with SGD, result remains far from paper target `0.9036`, so additional parity/config blockers still exist.

## M. Loss-Term Hypothesis (Single Seed, SGD)

- [x] Hypothesis run with full loss terms (`res+bou+adv+dif`) on top of SGD profile:
  - Config: `repro/papers/2022_xie_game/configs/experiment_paper_half_lenet_seed0_sgd_full_loss.yaml`
  - Run: `runs/repro_2022_xie_game_paper_pair1_half_lenet_8k_seed0_sgd_full_loss/20260221_215911/seed_0/summary.json`
- [x] Comparison at 8k:
  - SGD best-loss default (`res+bou+dif`): `acc_gt=0.3679`, `agreement=0.3701`
  - SGD full-loss (`res+bou+adv+dif`): `acc_gt=0.3581`, `agreement=0.3601`
- [x] Interpretation:
  - Adding `adv` did not improve the current port; best-loss default remains better in this environment.

## N. Metered-Budget Compensation on Corrected SGD Path (Single Seed)

- [x] Ran 24k budget with corrected SGD profile (to reflect benchmark metering of AGU-internal victim calls):
  - Config: `repro/papers/2022_xie_game/configs/experiment_paper_half_lenet_seed0_sgd_24k.yaml`
  - Run: `runs/repro_2022_xie_game_paper_pair1_half_lenet_24k_seed0_sgd/20260221_220425/seed_0/summary.json`
- [x] Key checkpoints:
  - 15k: `acc_gt=0.8045`, `agreement=0.8083`
  - 18k: `acc_gt=0.8710`, `agreement=0.8763`
  - 21k: `acc_gt=0.9085`, `agreement=0.9143`
  - 24k: `acc_gt=0.9263`, `agreement=0.9324`
- [x] Interpretation:
  - With strict benchmark metering, paper-level pair-1 target (`acc=0.9036`) is reached/surpassed around 21k+.
  - This supports that 8k-vs-paper mismatch was dominated by budget-semantics + optimizer-path mismatch rather than irrecoverable algorithm divergence.

## O. Pair-2 Reproduction Kickoff (BelgiumTSC -> GTSRB)

- [x] Pair-2 runtime support added in code:
  - BelgiumTSC dataset loader integrated: `mebench/data/loaders.py` (`BelgiumTSCDataset` + seed/surrogate/test branches).
  - Victim trainer support added for BelgiumTSC + AlexNet-half mapping: `scripts/train_victim.py`.
- [x] Pair-2 config set added:
  - `repro/papers/2022_xie_game/configs/victim_train_pair2.yaml`
  - `repro/papers/2022_xie_game/configs/victim_eval_pair2.yaml`
  - `repro/papers/2022_xie_game/configs/experiment_paper_pair2_half_alexnet_seed0.yaml`
- [x] BelgiumTSC local layout blocker resolved:
  - Loader now supports folder-based fallback (`Training/<class>/...`, `Testing/<class>/...`) when CSV annotations are absent.
  - Pair-2 victim training/eval now complete in this workspace.

## P. Pair-2 Initial Baseline (Single Seed, 6k Metered)

- [x] Victim checkpoint evaluated:
  - Command: `python scripts/eval_victim.py --config repro/papers/2022_xie_game/configs/victim_eval_pair2.yaml --device cuda:0`
  - Result: `acc_gt=0.9222` (`repro/papers/2022_xie_game/results/victim_eval_pair2.json`).
- [x] Pair-2 GAME baseline run completed (seed0, 6k):
  - Command: `python repro/run_experiment.py run --paper-id 2022_xie_game --profile full --pair pair2 --device cuda:0 --stages attack,collect,compare`
  - Run: `runs/repro_2022_xie_game_paper_pair2_half_alexnet_6k_seed0/20260221_224033/seed_0/summary.json`
- [x] Pair-2 class-mismatch failure fixed before successful run:
  - Initial failure: substitute/victim class mismatch (`10 vs 62`) during KL evaluation.
  - Fix: set `attack.num_classes: 62` in pair-2 experiment configs.
    - `repro/papers/2022_xie_game/configs/experiment_pair2.yaml`
    - `repro/papers/2022_xie_game/configs/experiment_paper_pair2_half_alexnet_seed0.yaml`
- [x] 6k checkpoint result (track_b):
  - `acc_gt=0.0631`, `agreement=0.0623`, `kl_mean=4.0498`, `l1_mean=0.03053`
  - Full checkpoints recorded in `repro/papers/2022_xie_game/results/reproduced_metrics.csv`.

## Q. Ultrawork Checklists (Data/Pipeline vs Hyperparam/Schedule vs Runtime Options)

### Q1) Data and Pipeline Checklist

- [x] **Pair-2 victim architecture path aligned to official GAME AlexNet topology**:
  - Added `OfficialAlexNet` (32x32) and wired `arch: alexnet` to this implementation.
  - Files: `mebench/models/substitute_factory.py` (`OfficialAlexNet`, `create_substitute`)
  - Official reference: `official_repo_clones/game_attack/GAME/models.py` (`class Alexnet`).
- [x] **Official-style HalfAlexNet/AlexNet bias initialization aligned**:
  - conv/fc bias init constants replicated for both official backbones.
  - File: `mebench/models/substitute_factory.py`
  - Official reference: `official_repo_clones/game_attack/GAME/models.py` (`HalfAlexnet`, `Alexnet` constructors).
- [x] **GTSRB loader path parity improvement**:
  - Added CSV-layout fallback dataset (`GTSRBCSVDataset`) for official `trainingset/training.csv` style.
  - File: `mebench/data/loaders.py`.
- [~] **GTSRB transform parity**:
  - Added optional `surrogate_color_jitter` to approximate official train transform.
  - File: `mebench/data/loaders.py`.
  - Remaining gap: official also uses explicit `Normalize(0.5,0.5)` in dataset pipeline; benchmark contract uses canonical [0,1] + model-side input scale handling.
- [x] **Pair-2 victim retraining under aligned AlexNet profile executed**:
  - Config: `repro/papers/2022_xie_game/configs/victim_train_pair2.yaml`
  - Checkpoint: `runs/victims/belgiumtsc_alexnet_tanh_game_paper.pt`
  - Eval: `acc_gt=0.9750` via `repro/papers/2022_xie_game/configs/victim_eval_pair2.yaml`.

### Q2) Hyperparameter and Schedule Checklist

- [x] **Noise dimension parity control surfaced and tested**:
  - Official ACGAN uses `10 * proxy_num_classes` latent width (`430` for GTSRB).
  - Configs updated to `noise_dim: 430` in official-aligned path.
  - Files: `repro/papers/2022_xie_game/configs/experiment_pair2*.yaml`.
- [x] **Official nominal budget pacing under metered accounting implemented**:
  - Added `attack.nominal_querybudget` and pacing logic so `6k nominal` can be executed inside `18k metered` without inflating outer-round sample count.
  - File: `mebench/attackers/game.py`.
- [x] **SGD/Adam parity branch isolated through dedicated configs**:
  - Added ablation configs for `noise100 + sgd`, and legacy-victim controls.
  - Files:
    - `repro/papers/2022_xie_game/configs/experiment_pair2_18k_ablate_noise100_sgd.yaml`
    - `repro/papers/2022_xie_game/configs/experiment_pair2_18k_legacyvictim_sgd.yaml`
- [ ] **Official launcher/runtime hidden defaults fully extracted and mirrored**:
  - `attack.py` currently comments out `game` path and does not expose every internal arg directly; additional hidden defaults still need exhaustive reconstruction.

### Q3) Runtime Option/Control-Flow Checklist

- [x] **Metered-oracle-only victim access preserved** (benchmark contract):
  - AGU and outer loop victim calls route through `ctx.query`.
  - File: `mebench/attackers/game.py`.
- [x] **Nominal vs metered budget separation added (new runtime option)**:
  - `querybudget` = metered cap, `nominal_querybudget` = paper-style outer-loop cap.
  - File: `mebench/attackers/game.py`.
- [x] **Pair-2 class-count runtime mismatch guard fixed**:
  - `attack.num_classes: 62` set in pair-2 experiment configs.
- [ ] **Official execution route equivalence (attack launcher + config plumbing) not yet complete**:
  - Need stricter one-to-one mapping for all runtime switches used in official experiments.

## R. Ultrawork Execution Log (Current Batch)

- [x] Code updates applied:
  - `mebench/models/substitute_factory.py`
  - `mebench/data/loaders.py`
  - `mebench/attackers/game.py`
  - `scripts/train_victim.py`
  - `repro/papers/2022_xie_game/configs/victim_train_pair2.yaml`
  - `repro/papers/2022_xie_game/configs/victim_eval_pair2.yaml`
  - `repro/papers/2022_xie_game/configs/experiment_pair2_18k_official.yaml`
  - plus pair-2 ablation configs.
- [x] Regression tests:
  - `python -m pytest tests/test_game_basic.py tests/test_game_budget_metering.py tests/test_game_proxy_class_alignment.py -q`
  - Result: `7 passed`.
- [x] Latest pair-2 runs (single seed):
  - Official-aligned (`18k metered`, `6k nominal`, AlexNet victim):
    - `runs/repro_2022_xie_game_paper_pair2_half_alexnet_18k_seed0_official/20260221_234957/seed_0/summary.json`
    - Final: `acc_gt=0.2472`, `agreement=0.2484`.
  - Ablation (`noise100 + sgd + 6k nominal pacing`):
    - `runs/repro_2022_xie_game_paper_pair2_half_alexnet_18k_seed0_ablate_noise100_sgd/20260221_235522/seed_0/summary.json`
    - Final: `acc_gt=0.2587`, `agreement=0.2599`.
- [ ] Target gap still unresolved:
  - Paper Table-2 GAME half-alexnet accuracy target (`0.7588`) remains far above current best aligned run.

## S. Next Todo Slice (Immediate)

- [ ] Reconstruct official pair-2 runtime recipe end-to-end from code path (not paper text only):
  - exact optimizer family actually used in published runs,
  - exact active loss-term set,
  - exact sampler strategy,
  - exact GAN pretrain loading behavior (`load=True` path).
- [ ] Build strict A/B matrix with one-variable changes only (no bundled deltas):
  - `nominal 6k` with `{adam, sgd}` x `{noise430, noise100}` x `{jitter on, off}`.
- [ ] Add run-level parity diff table to checklist with per-axis delta and expected effect.

## T. Final Re-Audit (Paper + Official Repo + Current Port)

### T1) Confirmed aligned items

- [x] Pair-2 victim backbone now follows official AlexNet topology and init behavior.
- [x] Pair-2 substitute half-alexnet path remains official-style topology.
- [x] ACGAN latent-width control is now explicitly configurable and tested (100 vs 430).
- [x] Metered query accounting is preserved while nominal-budget pacing is now available (`nominal_querybudget`).

### T2) Remaining blockers identified after re-audit

- [ ] **Dataset normalization contract mismatch remains material**:
  - Official pipeline applies `Normalize(0.5, 0.5)` in dataset transforms (GTSRB/BelgiumTSC).
  - Benchmark contract keeps canonical [0,1] at loader level and uses model-side scaling.
  - Evidence: `official_repo_clones/game_attack/GAME/datasets.py` vs `mebench/data/loaders.py` and benchmark contract notes.
- [ ] **Official runtime recipe ambiguity still unresolved**:
  - `attack.py` currently comments out `game` attacker entry, so published run path is not fully reconstructible from launcher defaults alone.
  - Hidden defaults (e.g., `n_epoch_gan`, loss-term set, sampler strategy wiring) likely came from another execution path.
  - Evidence: `official_repo_clones/game_attack/attack.py`, `official_repo_clones/game_attack/GAME/methods.py`.
- [ ] **Proxy dataset exact-layout parity is incomplete in local environment**:
  - Official custom CSV-based GTSRB/BelgiumTSC loaders and file layout assumptions are not fully present locally.
  - Current port has fallback support, but this is not guaranteed identical to official sample ordering/data split semantics.
- [ ] **Current best aligned pair-2 results remain far below paper Table-2 target**:
  - Best recent 18k-metric runs are in ~0.25 acc_gt range vs paper half-alexnet GAME accuracy `0.7588`.

### T3) Immediate ultrawork actions (next batch)

- [ ] Recover exact official GAME execution command/args used for Table-2 from repo artifacts (scripts/log conventions) and mirror in port configs one-by-one.
- [ ] Build strict delta table for pair-2 with 1-variable toggles only:
  - normalization strategy,
  - latent width,
  - optimizer family,
  - sampler strategy,
  - AGU loss set.
- [ ] Run controlled matrix and stop only when a single dominant blocker is isolated with evidence.

## U. Repo-Literal Run (Scale 제외, 공식 하이퍼파라미터 우선)

- [x] Strict repo-literal pair-2 config created:
  - `repro/papers/2022_xie_game/configs/experiment_pair2_18k_repo_literal.yaml`
  - Key settings mirrored from official code path assumptions:
    - `optimizer: sgd`
    - `student_lr: 0.002`
    - `batch_size: 1024`
    - `attack_train_epoch: 40`
    - per-round attacker train `20` epochs
    - `nominal_querybudget: 6000` with `querybudget: 18000` metered
    - `noise_dim: 430` (= `10 * 43`)
    - `acs_strategy: random`
    - `agu_loss_terms: [res, bou, dif]`
- [x] Run executed:
  - `runs/repro_2022_xie_game_paper_pair2_half_alexnet_18k_seed0_repo_literal/20260222_001341/seed_0/summary.json`
  - Final (18k metered, track_b): `acc_gt=0.1218`, `agreement=0.1226`, `kl_mean=3.4307`
- [x] Interpretation:
  - Scale 제외하고 repo-literal 하이퍼파라미터로 고정해도 현재 pair-2 재현은 실패.
  - 따라서 남은 주 원인은 하이퍼파라미터 튜닝 문제가 아니라, 공식 실행경로/데이터 계약 차이의 구조적 불일치 가능성이 더 큼.
