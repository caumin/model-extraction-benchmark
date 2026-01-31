#!/bin/bash

# Find all configs in the matrix folder (override with MATRIX_DIR)
matrix_dir=${MATRIX_DIR:-configs/matrix}
device=${MEBENCH_DEVICE:-cuda:0}
pattern=${MATRIX_PATTERN:-*.yaml}
max_runs=${MATRIX_LIMIT:-0}
aggregate=${AGGREGATE_MATRIX:-1}
python_bin=${PYTHON_BIN:-python}
configs=$(ls "${matrix_dir}"/${pattern})

echo "Starting Experimental Matrix Execution..."
echo "Total experiments: $(echo "$configs" | wc -l)"

count=0
for config in $configs; do
    name=$(basename "$config" .yaml)
    
    # Check if a summary already exists to avoid redundant runs
    # We look for any directory starting with the run name in runs/
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
    if [ "$max_runs" -gt 0 ] && [ "$count" -ge "$max_runs" ]; then
        echo "[INFO] MATRIX_LIMIT reached ($max_runs)."
        break
    fi
done

echo "Matrix execution complete."
if [ "$aggregate" -ne 0 ]; then
    $python_bin aggregate_matrix.py
fi
