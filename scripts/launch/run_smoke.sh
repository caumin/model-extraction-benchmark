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
DEVICE="${1:-${MEBENCH_DEVICE:-cuda:0}}"
LOG_DIR="${SMOKE_LOG_DIR:-logs/smoke}"
mkdir -p "$LOG_DIR"

CONFIGS=(
  "configs/smoke/knockoff_mnist_200.yaml"
  "configs/smoke/cloudleak_mnist_200.yaml"
  "configs/smoke/dfme_mnist_200.yaml"
)

for config in "${CONFIGS[@]}"; do
  if [[ ! -f "$config" ]]; then
    echo "Missing smoke config: ${config}"
    exit 1
  fi
  name=$(basename "${config}" .yaml)
  echo "[RUN ] ${name}"
  "$PYTHON_BIN" -m mebench run --config "${config}" --device "${DEVICE}" | tee "${LOG_DIR}/${name}.log"
  echo "[ OK ] ${name}"
done

echo "Smoke run complete. Logs: ${LOG_DIR}"
