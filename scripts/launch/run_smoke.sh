#!/usr/bin/env bash
set -euo pipefail

DEVICE=${1:-cuda:0}
mkdir -p logs/smoke

CONFIGS=(
  "configs/smoke/knockoff_mnist_200.yaml"
  "configs/smoke/cloudleak_mnist_200.yaml"
  "configs/smoke/dfme_mnist_200.yaml"
)

for config in "${CONFIGS[@]}"; do
  name=$(basename "${config}" .yaml)
  echo ">>> Running smoke config: ${name}"
  python -m mebench run --config "${config}" --device "${DEVICE}" > "logs/smoke/${name}.log" 2>&1
  echo ">>> ${name} SUCCESS"
done
