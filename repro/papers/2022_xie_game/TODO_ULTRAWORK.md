# TODO Ultrawork - 2022_xie_game

## Refactor Validation

- [ ] `python -m pytest tests/test_game_basic.py -q`
- [ ] `python -m pytest tests/test_game_budget_metering.py -q`
- [ ] `python -m pytest tests/test_game_proxy_class_alignment.py -q`
- [ ] `python -m pytest tests/test_attack_porting_defaults.py -q`

## Pair-1 Runs

- [ ] smoke: `experiment_smoke.yaml`
- [ ] single-seed: `experiment_paper_half_lenet_seed0_sgd.yaml`
- [ ] multi-seed: `experiment_paper_half_lenet.yaml`
- [ ] summarize checkpoints and collapse/no-collapse behavior

## Pair-2 Runs

- [ ] victim train/eval (`victim_train_pair2.yaml`, `victim_eval_pair2.yaml`)
- [ ] official-aligned: `experiment_pair2_18k_official.yaml`
- [ ] repo-literal: `experiment_pair2_18k_repo_literal.yaml`
- [ ] one-variable ablation matrix

## Reporting

- [ ] update `GAME_PARITY_CHECKLIST.md`
- [ ] update `REPRODUCTION_REPORT.md`
- [ ] add run-to-run delta notes (what changed / observed effect)
