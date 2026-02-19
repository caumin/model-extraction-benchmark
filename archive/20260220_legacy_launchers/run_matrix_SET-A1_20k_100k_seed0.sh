#!/bin/bash

matrix_dir=${MATRIX_DIR:-configs/matrix}
device=${MEBENCH_DEVICE:-cuda:0}
python_bin=${PYTHON_BIN:-python}
aggregate=${AGGREGATE_MATRIX:-0}

patterns=("SET-A1_*_20k_seed0.yaml" "SET-A1_*_100k_seed0.yaml")
configs=()

for pattern in "${patterns[@]}"; do
    for config in "${matrix_dir}"/${pattern}; do
        if [ -e "$config" ]; then
            configs+=("$config")
        fi
    done
done

if [ "${#configs[@]}" -eq 0 ]; then
    echo "No configs found for SET-A1 20k/100k seed0 in ${matrix_dir}"
    exit 1
fi

echo "Starting SET-A1 20k/100k seed0 execution..."
echo "Total experiments: ${#configs[@]}"

count=0
for config in "${configs[@]}"; do
    name=$(basename "$config" .yaml)

    if ls "runs/${name}"/*/seed_*/summary.json >/dev/null 2>&1; then
        echo "[SKIP] $name already completed."
        continue
    fi

    echo "=========================================================="
    echo "Running: $name"
    echo "=========================================================="

    $python_bin -m mebench run --config "$config" --device "$device"

    if [ $? -ne 0 ]; then
        echo "[ERROR] $name failed."
    fi

    count=$((count + 1))
done

echo "Matrix execution complete."
if [ "$aggregate" -ne 0 ]; then
    $python_bin aggregate_matrix.py
fi
