# REPRODUCTION_REPORT

## Scope

- paper_id: `2022_xie_game`
- status: iterative parity tuning in progress (pair-1 runnable, pair-2 runnable with initial baseline)

## What Was Rewritten

- Paper pair-1 basis is wired and runnable in this repo:
  - victim/origin: `MNIST`
  - proxy: `FashionMNIST`
  - budget: `8k`
  - victim profile: `lenet`, 15 epochs, Adam lr=0.001 (`victim_train.yaml`)
- GAME parity alignment work added in code:
  - metered AGU victim queries (benchmark rule: 1 image = 1 query)
  - ACGAN fake-class semantics (`n_classes+1` aux head + fake-class target)
  - official-like ACGAN G/D structure path and TDL full-loader looping
  - official-like student full-buffer retrain schedule hooks
- Sweep configs were added for iterative diagnosis under metered-budget constraints.

## Execution

```bash
python repro/run_experiment.py run --paper-id 2022_xie_game --profile smoke --device cuda:0
python repro/run_experiment.py run --paper-id 2022_xie_game --profile full --device cuda:0

# budget-compensation / sweep / full-best-loop (direct mebench configs)
python -m mebench run --config repro/papers/2022_xie_game/configs/experiment_smoke_budgetcomp3k.yaml --device cuda:0
python -m mebench run --config repro/papers/2022_xie_game/configs/experiment_smoke_sweep_round5.yaml --device cuda:0
python -m mebench run --config repro/papers/2022_xie_game/configs/experiment_full_sweep_round5.yaml --device cuda:0
```

## Notes

- Latest smoke baseline (`experiment_smoke.yaml`): final `acc_gt=0.1032`, `agreement=0.1038`.
- Budget-compensated smoke (`3k`) peaked mid-run but finished near-random (`acc_gt=0.0974`, `agreement=0.0963`).
- Best current smoke variant: `round_train_epochs=5` (`experiment_smoke_sweep_round5.yaml`) with final `acc_gt=0.1501`, `agreement=0.1494`.
- Loop-2 smoke extensions:
  - `round5 + deviation`: `acc_gt=0.1505`, `agreement=0.1501`
  - `round5 + batch64` and `round5 + batch64 + deviation`: both collapsed to `acc_gt=0.0974`, `agreement=0.0963`
- Full run with round5-random profile (`experiment_full_sweep_round5.yaml`) gave best current 8k result: `acc_gt=0.1373`, `agreement=0.1376`.
- Full run with round5-deviation profile (`experiment_full_sweep_round5_deviation.yaml`) was worse: `acc_gt=0.1150`, `agreement=0.1154`.
- 3x-budget compensation run (`experiment_full_sweep_round5_24k.yaml`) did not improve final outcome:
  - peak at 9k: `acc_gt=0.1379`, `agreement=0.1382`
  - final at 24k: `acc_gt=0.0974`, `agreement=0.0963`
  - conclusion: increasing metered query budget alone is insufficient; collapse remains unresolved.
- Full paper-profile pair-1 run (half-lenet, seeds `0/1/2`, 8k) completed:
  - Config: `repro/papers/2022_xie_game/configs/experiment_paper_half_lenet.yaml`
  - 8k `acc_gt`: `[0.0974, 0.1032, 0.0974]` -> mean `0.0993`
  - 8k `agreement`: `[0.0963, 0.1043, 0.0963]` -> mean `0.0990`
  - Compared to paper Table-1 GAME half-lenet accuracy `0.9036`, current gap remains very large.
- Single-seed collapse audit note:
  - Evaluating GAME substitutes with tanh wrappers did not resolve collapse and was removed for official-like parity.
  - Re-run config: `repro/papers/2022_xie_game/configs/experiment_paper_half_lenet_seed0.yaml`
  - Run result at 8k remained collapsed (`acc_gt=0.0974`, `agreement=0.0963`).
- Additional routing parity update:
  - GAME query routing now keeps tanh tensors for tanh profiles at oracle boundary (official-like synthetic query path).
  - Re-run kept final 8k result collapsed (`acc_gt=0.0974`, `agreement=0.0963`).
- Optimizer-path single-seed audit:
  - Added SGD hypothesis profile: `repro/papers/2022_xie_game/configs/experiment_paper_half_lenet_seed0_sgd.yaml`
  - Result improved substantially over Adam profile at 8k (`acc_gt=0.3679`, `agreement=0.3701` vs `0.0974/0.0963`).
  - This indicates optimizer path mismatch is a major factor, but reproduction is still far from paper target (`0.9036`).
- Loss-term single-seed audit on SGD path:
  - Full-loss profile (`res+bou+adv+dif`) at 8k: `acc_gt=0.3581`, `agreement=0.3601`.
  - It underperformed the best-loss default (`res+bou+dif`) on this port.
- 24k metered-budget run on corrected SGD profile:
  - Config: `repro/papers/2022_xie_game/configs/experiment_paper_half_lenet_seed0_sgd_24k.yaml`
  - Trajectory improved monotonically after mid-budget and crossed paper pair-1 target range:
    - 21k: `acc_gt=0.9085`, `agreement=0.9143`
    - 24k: `acc_gt=0.9263`, `agreement=0.9324`
  - Practical conclusion: under strict benchmark query accounting (including AGU-internal victim calls), paper-equivalent performance emerges at ~21k+ metered queries rather than 8k.

## Pair-2 Kickoff Status

- Implemented BelgiumTSC data-path support in benchmark loaders and victim training script.
- Added pair-2 configs for victim train/eval and GAME run (`half_alexnet`, seed0).
- BelgiumTSC victim was trained and evaluated successfully in current workspace:
  - checkpoint: `runs/victims/belgiumtsc_alexnet_half_tanh_game_paper.pt`
  - eval (`victim_eval_pair2.yaml`): `acc_gt=0.9222`
- Pair-2 GAME single-seed baseline (6k, metered) was executed:
  - run: `runs/repro_2022_xie_game_paper_pair2_half_alexnet_6k_seed0/20260221_224033/seed_0/summary.json`
  - final 6k (track_b): `acc_gt=0.0631`, `agreement=0.0623`, `kl_mean=4.0498`, `l1_mean=0.03053`
- During first pair-2 run, a class-count mismatch surfaced (`10 vs 62`) at eval time; fixed by setting `attack.num_classes: 62` in pair-2 experiment configs before rerun.
