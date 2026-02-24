#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v py >/dev/null 2>&1; then
    PYTHON_BIN="py"
  else
    echo "Python executable not found. Set PYTHON_BIN explicitly (e.g., PYTHON_BIN=py)."
    exit 127
  fi
fi

if [[ "${1:-}" == "--full" ]]; then
  "$PYTHON_BIN" -m pytest tests/ -v
  exit 0
fi

"$PYTHON_BIN" -m pytest tests/test_contract_validation.py -v
"$PYTHON_BIN" -m pytest tests/test_budget_accounting.py -v
"$PYTHON_BIN" -m pytest tests/test_reproducibility.py -v
"$PYTHON_BIN" -m pytest tests/test_cache_cleanup.py -v

echo "Core test suite complete. Use --full for all tests."
