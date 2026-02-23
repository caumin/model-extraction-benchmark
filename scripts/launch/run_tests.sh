#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "${1:-}" == "--full" ]]; then
  "$PYTHON_BIN" -m pytest tests/ -v
  exit 0
fi

"$PYTHON_BIN" -m pytest tests/test_contract_validation.py -v
"$PYTHON_BIN" -m pytest tests/test_budget_accounting.py -v
"$PYTHON_BIN" -m pytest tests/test_reproducibility.py -v
"$PYTHON_BIN" -m pytest tests/test_cache_cleanup.py -v

echo "Core test suite complete. Use --full for all tests."
