#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MATRIX_DIR="${MATRIX_DIR:-configs/matrix}"
MATRIX_PATTERN="${MATRIX_PATTERN:-*.yaml}"
MEBENCH_DEVICE="${MEBENCH_DEVICE:-cuda:0}"
POOL_BUDGET="${POOL_BUDGET:-20000}"
SYNTHETIC_BUDGET="${SYNTHETIC_BUDGET:-20000000}"
MATRIX_LIMIT="${MATRIX_LIMIT:-0}"
GENERATE_CONFIGS="${GENERATE_CONFIGS:-1}"
INCLUDE_BOTH_HARD="${INCLUDE_BOTH_HARD:-1}"

if [[ "$GENERATE_CONFIGS" != "0" ]]; then
  args=(
    generate_configs.py
    --out "$MATRIX_DIR"
    --device "$MEBENCH_DEVICE"
    --pool-budget "$POOL_BUDGET"
    --synthetic-budget "$SYNTHETIC_BUDGET"
  )
  if [[ "$INCLUDE_BOTH_HARD" != "0" ]]; then
    args+=(--include-both-hard)
  fi
  "$PYTHON_BIN" "${args[@]}"
fi

shopt -s nullglob
configs=("$MATRIX_DIR"/$MATRIX_PATTERN)

if [[ ${#configs[@]} -eq 0 ]]; then
  echo "No configs found in ${MATRIX_DIR} matching ${MATRIX_PATTERN}."
  exit 0
fi

echo "Running matrix from ${MATRIX_DIR} on ${MEBENCH_DEVICE}"
echo "Total configs: ${#configs[@]}"

attempted=0
failed=0

for config in "${configs[@]}"; do
  name="$(basename "$config" .yaml)"

  if ls "runs/${name}"/*/seed_*/summary.json >/dev/null 2>&1; then
    echo "[SKIP] ${name}"
    continue
  fi

  echo "[RUN ] ${name}"
  if "$PYTHON_BIN" -m mebench run --config "$config" --device "$MEBENCH_DEVICE"; then
    echo "[ OK ] ${name}"
  else
    echo "[FAIL] ${name}"
    failed=$((failed + 1))
  fi

  attempted=$((attempted + 1))
  if [[ "$MATRIX_LIMIT" -gt 0 && "$attempted" -ge "$MATRIX_LIMIT" ]]; then
    echo "MATRIX_LIMIT reached: ${MATRIX_LIMIT}"
    break
  fi
done

echo "Matrix run finished. attempted=${attempted} failed=${failed}"
if [[ "$failed" -gt 0 ]]; then
  exit 1
fi
