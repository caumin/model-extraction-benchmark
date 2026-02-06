#!/bin/bash

matrix_dir=${MATRIX_DIR:-configs/matrix}
device=${MEBENCH_DEVICE:-cuda:0}
pattern=${MATRIX_PATTERN:-*.yaml}
max_runs=${MATRIX_LIMIT:-0}
python_bin=${PYTHON_BIN:-python}

# Budgets (fixed for fairness runs)
pool_budget=${POOL_BUDGET:-20000}
synthetic_budget=${SYNTHETIC_BUDGET:-2000000}

# Generate configs (cleans existing YAMLs by default)
include_both_hard=${INCLUDE_BOTH_HARD:-1}
both_flag=""
if [ "$include_both_hard" -ne 0 ]; then
  both_flag="--include-both-hard"
fi

$python_bin generate_configs.py --out "$matrix_dir" --device "$device" --pool-budget "$pool_budget" --synthetic-budget "$synthetic_budget" $both_flag

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
