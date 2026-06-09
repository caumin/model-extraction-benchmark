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
MATRIX_DIR="${MATRIX_DIR:-configs/matrix}"
MATRIX_PATTERN="${MATRIX_PATTERN:-*.yaml}"
MEBENCH_DEVICE="${MEBENCH_DEVICE:-cuda:0}"
IMAGENET_ROOT="${IMAGENET_ROOT:-C:/imagenet}"
SEWERML_ROOT="${SEWERML_ROOT:-D:/Sewer/Sewer-ML}"

# WSL/Linux compatibility: convert Windows-style path (e.g., C:/imagenet)
# to /mnt/c/imagenet for local filesystem checks.
if [[ "$(uname -s)" == "Linux" && "$IMAGENET_ROOT" =~ ^([A-Za-z]):/(.*)$ ]]; then
  drive="${BASH_REMATCH[1],,}"
  rest="${BASH_REMATCH[2]}"
  IMAGENET_ROOT="/mnt/${drive}/${rest}"
fi
SET_A_POOL_BUDGET="${SET_A_POOL_BUDGET:-10000}"
SET_A_SYNTHETIC_BUDGET="${SET_A_SYNTHETIC_BUDGET:-10000000}"
SET_B_POOL_BUDGET="${SET_B_POOL_BUDGET:-20000}"
SET_B_SYNTHETIC_BUDGET="${SET_B_SYNTHETIC_BUDGET:-20000000}"
SET_C_POOL_BUDGET="${SET_C_POOL_BUDGET:-20000}"
SET_C_SYNTHETIC_BUDGET="${SET_C_SYNTHETIC_BUDGET:-20000000}"
# Optional legacy global overrides
POOL_BUDGET="${POOL_BUDGET:-}"
SYNTHETIC_BUDGET="${SYNTHETIC_BUDGET:-}"
MATRIX_LIMIT="${MATRIX_LIMIT:-0}"
GENERATE_CONFIGS="${GENERATE_CONFIGS:-0}"
INCLUDE_BOTH_HARD="${INCLUDE_BOTH_HARD:-1}"
FAST_MODE="${FAST_MODE:-0}"

if [[ "$GENERATE_CONFIGS" != "0" ]]; then
  args=(
    generate_configs.py
    --out "$MATRIX_DIR"
    --device "$MEBENCH_DEVICE"
    --imagenet-root "$IMAGENET_ROOT"
    --sewerml-root "$SEWERML_ROOT"
    --set-a-pool-budget "$SET_A_POOL_BUDGET"
    --set-a-synthetic-budget "$SET_A_SYNTHETIC_BUDGET"
    --set-b-pool-budget "$SET_B_POOL_BUDGET"
    --set-b-synthetic-budget "$SET_B_SYNTHETIC_BUDGET"
    --set-c-pool-budget "$SET_C_POOL_BUDGET"
    --set-c-synthetic-budget "$SET_C_SYNTHETIC_BUDGET"
  )
  if [[ -n "$POOL_BUDGET" ]]; then
    args+=(--pool-budget "$POOL_BUDGET")
  fi
  if [[ -n "$SYNTHETIC_BUDGET" ]]; then
    args+=(--synthetic-budget "$SYNTHETIC_BUDGET")
  fi
  if [[ "$INCLUDE_BOTH_HARD" != "0" ]]; then
    args+=(--include-both-hard)
  fi
  if [[ "$FAST_MODE" != "0" ]]; then
    args+=(--fast-mode)
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

  if compgen -G "runs/${name}/*/seed_*/summary.json" > /dev/null; then
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
