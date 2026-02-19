#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs/tests
ATTACKS=(random activethief dfme maze dfms game es swiftthief blackbox_dissector cloudleak blackbox_ripper copycatcnn inversenet knockoff_nets)
for attack in "${ATTACKS[@]}"; do
  echo ">>> Running test for ${attack}..."
  python -m mebench run --config "configs/tests/${attack}.yaml" --device cuda:0 > "logs/tests/${attack}.log" 2>&1
  echo ">>> ${attack} SUCCESS"
done
