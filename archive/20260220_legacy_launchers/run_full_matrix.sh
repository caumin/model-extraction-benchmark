#!/bin/bash

# Ensure output directory exists
mkdir -p logs

echo "========================================================"
echo "Model Extraction Benchmark: Full Matrix Execution"
echo "========================================================"
echo "Total Experiments: 168 (4 Sets x 14 Attacks x 3 Seeds)"
echo "Estimated Time: ~24-48 hours (depending on GPU)"
echo "--------------------------------------------------------"

# 1. Ensure Configs are Generated
echo "[Phase 1] Generating configurations..."
python generate_configs.py

# 2. Run Experiments Sequentially
# Note: For multi-GPU, you can split this list and run in parallel.
# e.g., run_subset.sh "SET-A" cuda:0 & run_subset.sh "SET-B" cuda:1

configs=$(ls configs/matrix/*.yaml)
count=0
total=$(echo "$configs" | wc -l)

for config in $configs; do
    count=$((count+1))
    name=$(basename "$config" .yaml)
    
    # Skip if already done
    if ls "runs/${name}"/*/seed_*/summary.json >/dev/null 2>&1; then
        echo "[${count}/${total}] SKIP: $name already completed."
        continue
    fi

    echo "[${count}/${total}] RUNNING: $name"
    
    # Run with timeout (e.g., 2 hours per experiment) to prevent hangs
    # Output logs to logs/ directory
    python -m mebench run --config "$config" --device cuda:0 > "logs/${name}.log" 2>&1
    
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo " -> SUCCESS"
        # Optional: remove log if successful to save space
        # rm "logs/${name}.log"
    else
        echo " -> FAILED (Exit Code: $exit_code). Check logs/${name}.log"
    fi
done

# 3. Aggregate Results
echo "[Phase 3] Aggregating results..."
python aggregate_matrix.py

echo "Full benchmark execution complete."
