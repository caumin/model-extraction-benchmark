# Release Scope (Draft)

## Purpose

This document defines what is included in the public open-source release for this repository.

- Public release target: the `mebench` experiment framework.
- Explicitly excluded from release: `papers/`, `repro/`, `official_repo_clones/`.
- Rule priority: legal/safety constraints > reproducible framework runtime > optional research notes.

## Scope Principles

1. Ship code required to install, run, and validate `mebench` experiments.
2. Do not ship local research assets, paper artifacts, cloned upstream repositories, or runtime outputs.
3. Keep release boundaries path-based and deterministic (no ad-hoc exceptions).

## Must Include

The following paths are in scope for public release:

- `mebench/`
  - Core framework code (attackers, engine, state, oracle wrappers, training/eval utilities).
- `configs/` (only tracked templates/examples)
  - Keep hand-maintained configs used for examples or smoke runs.
- `scripts/launch/`
  - Public launcher scripts and usage wrappers.
- `tests/`
  - Contract and regression tests required for trust and reproducibility.
- `docs/design/`
- `docs/reference/`
- Root project files
  - `README.md`, `README-ko.md`, `pyproject.toml`, `.gitignore`, `AGENTS.md`, `Model_Extraction_Benchmark_v1.0.1_Contract_and_Implementation_Guide.md`.
- Utility entry scripts used by public benchmark workflows
  - `generate_configs.py`, `aggregate_matrix.py`, `run_smoke.sh`, `run_tests.sh`.

## Must Exclude

The following paths are out of scope and must not be included in the public release:

- Research and paper assets
  - `papers/`
  - `repro/`
  - `official_repo_clones/`
- Local/runtime artifacts
  - `data/`, `runs/`, `analysis_results/`, `logs/`, `checkpoints/`, `results/`, `reports/`, `wandb/`.
- Generated or temporary config outputs
  - `configs/matrix/`, `configs/paper_mode/`, `configs/tmp_channel_check/`.
- Legacy and archive materials
  - `archive/`.
- Temporary and cache files
  - `__pycache__/`, `.pytest_cache/`, `temp/`, `temp_configs/`, `temp_*/`, `TEMP_*/`.

## Maybe (Include Only with Explicit Decision)

These paths require explicit maintainer decision before release:

- `docs/technical_reports/`
  - Include only if the release goal includes deep experimental analysis, not just framework usage.
- `docs/archive/`
  - Include only if historical design context is needed by external contributors.
- `reproduction_specs/`
  - Include if used as public methodology docs; exclude if they are internal reconciliation notes.
- Root compatibility wrappers (`run_matrix*.sh`, `run_matrix*.ps1`, `run_parallel.ps1`)
  - Include only if they do not depend on excluded paths (`repro/`, local-only assets).

## Pre-Release Checklist

Run this checklist before creating a public tag:

1. Path boundary check
   - Confirm excluded directories are not tracked for release: `papers/`, `repro/`, `official_repo_clones/`, `data/`, `runs/`.
2. Documentation check
   - Remove or rewrite README commands that depend on excluded paths.
   - Keep only public workflow commands (`python -m mebench run`, smoke/matrix public scripts).
3. Config check
   - Ensure committed configs do not reference excluded paths.
4. Quality gate
   - Run `pytest tests/test_contract_validation.py`.
   - Run `pytest tests/ -v`.
5. Packaging gate
   - Verify `pip install -e ".[dev]"` works from a clean clone.
6. Final audit
   - Reconfirm `.gitignore` still blocks runtime/generated artifacts.

## Release Process (Short)

1. Freeze release branch.
2. Apply README/docs cleanup for public-only workflows.
3. Verify tests and install flow.
4. Perform path audit against this scope file.
5. Tag and publish.

## Maintenance Rule for New Files

When a new file/directory is added, classify it immediately:

- `INCLUDE` if it is required to run/install/test the public `mebench` framework.
- `EXCLUDE` if it is paper asset, local experiment output, cloned external source, or generated runtime artifact.
- `MAYBE` only when both of these are true:
  1. It improves external understanding of the framework.
  2. It has no legal/licensing/distribution risk.

If classification is unclear, default to `EXCLUDE` until a maintainer decision is recorded in this file.
