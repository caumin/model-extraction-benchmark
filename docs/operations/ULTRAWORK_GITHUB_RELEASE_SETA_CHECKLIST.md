# ULTRAWORK: GitHub Release + SET-A Fast Run

## Scope

- Goal 1: Prepare repository for clean GitHub distribution (docs/code hygiene/ignore/release boundaries).
- Goal 2: Execute SET-A experiments quickly and reproducibly.
- Principle: Contract-first (`1 query = 1 image`, deterministic Track A, no hidden runtime artifacts in git).

## Checklist A: GitHub Release Readiness

### A0. Kickoff and Baseline

- [x] Confirm release scope baseline from `RELEASE_SCOPE.md`.
- [x] Confirm current repository state and key meta files (`README.md`, `.gitignore`, `pyproject.toml`, `LICENSE`).
- [ ] Snapshot current branch status for release work log (`git status`, `git diff --name-only`).

### A1. Release Boundary and Asset Policy

- [ ] Finalize INCLUDE set (framework/runtime/test/docs required for public users).
- [ ] Finalize EXCLUDE set (runtime outputs, local data, private/internal assets, temp artifacts).
- [ ] Finalize MAYBE set with explicit maintainer decisions:
  - [ ] `docs/technical_reports/`
  - [ ] `docs/archive/`
  - [ ] `reproduction_specs/`
  - [ ] root compatibility launch wrappers
- [ ] Document large artifact policy (`*.pt`, `*.pth`, `*.ckpt`): source git exclusion + release artifact strategy.

### A1-Decision Draft (Proposed Default)

- [x] `*.pt`, `*.pth`, `*.ckpt`, large result CSVs are excluded from source history by default.
- [x] Runtime products live outside source control (`runs/`, `checkpoints/`, `results/`, `reports/`).
- [x] If sharing is required, publish via Release assets or external artifact storage, then reference path/checksum in docs.
- [x] Keep only lightweight reproducibility metadata in git (configs, scripts, aggregate summaries, schema docs).

### A2. `.gitignore` and Tracking Hygiene

- [ ] Audit `.gitignore` against current runtime outputs and generated config paths.
- [ ] Ensure all known generated folders are excluded:
  - [ ] `runs/`, `analysis_results/`, `logs/`, `reports/`, `results/`, `wandb/`
  - [ ] `configs/matrix/`, `configs/paper_mode/`, `configs/tmp_channel_check/`
  - [ ] temp/cache (`__pycache__/`, `.pytest_cache/`, `temp*/`)
- [ ] Detect accidentally tracked binary or oversized files and remove from tracking plan.

### A3. Documentation Cleanup for Public Release

- [ ] Refactor `README.md` into quick-start-first structure:
  - [ ] What this repo is
  - [ ] Install
  - [ ] Smoke run
  - [ ] Matrix run
  - [ ] Result aggregation
  - [ ] Public release boundary
- [ ] Align `README-ko.md` structure with `README.md`.
- [ ] Add/refresh contributor-facing docs:
  - [ ] `CONTRIBUTING.md` (missing today)
  - [ ] Optional `CHANGELOG.md` template
- [ ] Verify all README commands only reference public paths.

### A4. Code Hygiene (Non-Behavioral Cleanup)

- [ ] Identify removable leftovers (obsolete wrappers, stale temporary scripts, dead helper files).
- [ ] Remove debug-only patterns (`print`, ad-hoc temp flags) where no longer needed.
- [ ] Keep bugfix rule: no opportunistic refactor mixed into release hygiene patches.

### A5. Quality Gates and CI

- [ ] Run local quality gates:
  - [ ] `ruff check mebench/`
  - [ ] `ruff format mebench/`
  - [ ] `mypy mebench/`
  - [ ] `pytest tests/test_contract_validation.py`
  - [ ] `pytest tests/test_budget_accounting.py`
  - [ ] `pytest tests/test_reproducibility.py`
- [ ] Add/verify GitHub Actions for lint/type/test minimal workflow.
- [ ] Clean-clone install validation (`pip install -e \".[dev]\"`).

### A6. Release Execution

- [ ] Final release audit against `RELEASE_SCOPE.md`.
- [ ] Prepare release notes focusing on framework scope and exclusions.
- [ ] Tag/release checklist ready (without pushing artifacts into source history).

## Checklist B: SET-A Fast Experiment Execution

### B0. Preflight

- [ ] Confirm runtime environment:
  - [ ] CUDA availability and selected device
  - [ ] Python/PyTorch versions
  - [ ] Disk capacity for run artifacts
- [ ] Confirm SET-A surrogate data assumptions (ImageNet subset 100k, grayscale + 28x28 policy).
- [ ] Confirm victim checkpoint availability and path validity.

### B1. Config Preparation

- [ ] Generate configs (`python generate_configs.py`).
- [ ] Filter/identify SET-A run targets only.
- [ ] Verify generated configs honor contract:
  - [ ] output mode compatibility
  - [ ] budget correctness
  - [ ] data scale policy (`[0,1]` vs `[-1,1]` by attack family)

### B2. Smoke Path (Fast Failure Detection)

- [ ] Run one reduced-budget SET-A smoke config (single seed).
- [ ] Validate artifact outputs for smoke run:
  - [ ] `run_config.yaml`
  - [ ] `summary.json`
  - [ ] `metrics.csv`
- [ ] Validate budget/accounting monotonicity and cap adherence.

### B3. Stage-1 Main Run (SET-A)

- [ ] Execute Stage-1 plan:
  - [ ] Pool attacks budget `20k`, seeds `0 -> 1 -> 2`
  - [ ] Data-free attacks budget `20m`, seeds `0 -> 1 -> 2`
- [ ] Keep skip-if-exists behavior for resumability.
- [ ] Capture per-seed run log and failure reason if interrupted.

### B4. Fast Throughput Controls

- [ ] Tune dataloader and runtime knobs safely (workers/pin-memory/eval cadence).
- [ ] Enable only contract-allowed temporary caching; verify cleanup after run.
- [ ] Monitor NaN/divergence and terminate early on deterministic failure signatures.

### B5. Aggregation and Validation

- [ ] Aggregate outputs (`python aggregate_matrix.py`).
- [ ] Validate seed completeness and no duplicate/missing checkpoints.
- [ ] Validate contract-critical metrics/reporting:
  - [ ] budget=image count
  - [ ] victim eval/no_grad
  - [ ] reproducibility expectations for Track A
- [ ] Export publication-ready summary table (CSV/Markdown/LaTeX as needed).

## Execution Todo List (Ordered)

### Immediate (Now)

- [x] T1. Create integrated ULTRAWORK checklist document.
- [ ] T2. Run release-boundary file audit and produce include/exclude table snapshot.
- [ ] T3. Draft `.gitignore` delta proposal from observed repository state.
- [ ] T4. Draft `README.md` restructuring patch (public runbook first).

### Next (After Release Hygiene Draft)

- [ ] T5. Build SET-A preflight command checklist and expected outputs.
- [ ] T6. Execute one SET-A smoke run and validate artifact schema.
- [ ] T7. Finalize Stage-1 run command set for seed order execution.

### Finish (Verification and Hand-off)

- [ ] T8. Run quality gates and record pass/fail with evidence.
- [ ] T9. Publish final execution log and unresolved risk list.

## Definition of Done

- Release docs are user-facing, minimal, and executable from a clean clone.
- Runtime/generated artifacts are excluded from source tracking policy.
- SET-A smoke passes and Stage-1 run plan is executable without ambiguity.
- Aggregation and contract validation paths are documented and reproducible.
