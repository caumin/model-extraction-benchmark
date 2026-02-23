# Code Hygiene Candidates (Release Prep)

This file captures non-breaking cleanup candidates discovered during release preparation.

## 1) Legacy Wrapper Consolidation

Potentially redundant root wrappers (public docs already prefer `scripts/launch/`):

- `run_matrix.sh`
- `run_full_matrix.sh`
- `run_matrix_SET-A1_20k_100k_seed0.sh`
- `run_matrix_SET-A1_A2_20k_100k.sh`
- `run_matrix.ps1`
- `run_parallel.ps1`
- `run_swiftthief.ps1`

Proposed action:
- Keep one compatibility wrapper per platform or remove all root wrappers and keep only `scripts/launch/`.

## 2) Analysis Scripts Scope Clarification

Scripts that appear to be ad-hoc analysis utilities (not core runtime):

- `analyze_results.py`
- `analyze_soft_attacks_results.py`

Proposed action:
- Either move under `scripts/analysis/` with explicit scope docs, or exclude from public release workflow docs.

## 3) Logging/Print Policy Review

Frequent `print(...)` usage was found in runtime code and tests.

High-impact runtime paths to review:
- `mebench/core/engine.py`
- `mebench/core/context.py`
- `mebench/core/query_storage.py`
- `mebench/oracles/victim_loader.py`
- `mebench/eval/evaluator.py`

Proposed action:
- Keep user-visible progress messages in CLI entry points.
- Gradually replace internal diagnostic prints with structured logging where needed.
- Avoid removing prints in this release pass unless behavior contracts are unaffected.

## 4) Generated/Temporary Folders Watchlist

Ensure these remain excluded and untracked:

- `configs/matrix/`
- `configs/paper_mode/`
- `configs/tmp_channel_check/`
- `runs/`, `results/`, `reports/`, `analysis_results/`

## 5) Safe-Cleanup Rule

- No functional refactor during bugfix/release hygiene pass.
- No change to budget accounting, victim query path, or Track A determinism logic.
- Cleanup candidates should be isolated and validated with core tests.
