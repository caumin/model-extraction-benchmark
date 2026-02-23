# Quality Gates (Release + Experiment Safety)

This plan defines minimal verification before public release and before long SET-A runs.

## 1) Local Environment Gate

```bash
pip install -e ".[dev]"
```

Pass criteria:
- install completes from a clean clone

## 2) Static Checks Gate

```bash
ruff check mebench/
ruff format --check mebench/
mypy mebench/
```

Pass criteria:
- no ruff errors
- formatter check passes (or formatting applied in separate patch)
- mypy exits 0 for configured scope

## 3) Contract-Critical Tests Gate

```bash
pytest tests/test_contract_validation.py -v
pytest tests/test_budget_accounting.py -v
pytest tests/test_reproducibility.py -v
pytest tests/test_cache_cleanup.py -v
```

Pass criteria:
- all listed tests pass
- failures are either fixed or clearly marked as pre-existing and unrelated

## 4) Smoke Runtime Gate

```bash
bash scripts/launch/run_smoke.sh cuda:0
```

Pass criteria:
- run completes
- expected artifacts exist (`summary.json`, `metrics.csv`)

## 5) Release Blocking Rules

Release should be blocked when:
- contract tests fail
- budget accounting regression is detected
- output mode validation is bypassed
- generated runtime artifacts are being added to git history

## 6) Evidence Capture

For each gate, capture:
- command used
- exit code
- timestamp
- short note (pass/fail + follow-up)

Store evidence in release notes or PR description.
