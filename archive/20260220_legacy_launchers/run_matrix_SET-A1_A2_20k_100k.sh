#!/usr/bin/env bash
set -euo pipefail

DEVICE=${1:-cuda:0}
PYTHON_BIN=${PYTHON_BIN:-python}
MATRIX_DIR=${MATRIX_DIR:-configs/matrix}
MATRIX_LIMIT=${MATRIX_LIMIT:-0}
RUN_ANALYSIS=${RUN_ANALYSIS:-0}

export MEBENCH_DEVICE="${DEVICE}"
export PYTHON_BIN
export MATRIX_DIR
export MATRIX_LIMIT
export AGGREGATE_MATRIX=0

run_pattern() {
  local pattern="$1"
  if compgen -G "${MATRIX_DIR}/${pattern}" > /dev/null; then
    export MATRIX_PATTERN="${pattern}"
    bash run_matrix.sh
  else
    echo "[SKIP] No configs for pattern: ${pattern}"
  fi
}

run_pattern "SET-A1_*20k*.yaml"
run_pattern "SET-A1_*100k*.yaml"
run_pattern "SET-A2_*20k*.yaml"
run_pattern "SET-A2_*100k*.yaml"

if [ "${RUN_ANALYSIS}" -ne 0 ]; then
  ${PYTHON_BIN} analyze_results.py
fi
