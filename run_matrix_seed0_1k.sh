#!/bin/bash

# Configuration
MATRIX_DIR="configs/matrix_A1_1k"
DEVICE=${MEBENCH_DEVICE:-cuda:0}
PYTHON_BIN=${PYTHON_BIN:-python}

echo "Starting Experimental Matrix Execution (SET-A1, Seed 0, 1k Budget Only)..."

# Find all YAML configs in the specific A1_1k directory
configs=$(ls "${MATRIX_DIR}"/*.yaml 2>/dev/null)

if [ -z "$configs" ]; then
    echo "[ERROR] No configurations found in ${MATRIX_DIR}"
    exit 1
fi

total_count=$(echo "$configs" | wc -l)
echo "Total experiments to run: $total_count"

for config in $configs; do
    name=$(basename "$config" .yaml)
    
    # Check if a summary already exists to avoid redundant runs
    # Note: run_name matches the config file name
    if ls "runs/${name}"/*/seed_0/summary.json >/dev/null 2>&1; then
        echo "[SKIP] $name already completed."
        continue
    fi

    echo ""
    echo "=========================================================="
    echo "RUNNING: $name"
    echo "=========================================================="
    
    $PYTHON_BIN -m mebench run --config "$config" --device "$DEVICE"
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] $name failed."
    else
        echo "[SUCCESS] $name completed."
    fi
done

echo ""
echo "=========================================================="
echo "Matrix execution complete (SET-A1, 1k, Seed 0)."
echo "=========================================================="
