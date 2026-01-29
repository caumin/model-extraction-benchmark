#!/bin/bash

# Ensure output directory exists
mkdir -p logs

echo "========================================================"
echo "Model Extraction Benchmark: Seed 0 Execution"
echo "========================================================"
echo "Total Experiments: 64 (4 Sets x 16 Attacks x 1 Seed)"
echo "--------------------------------------------------------"

# 1. Ensure Configs are Generated
echo "[Phase 1] Generating configurations..."
python generate_configs.py

# 2. Run Experiments Sequentially (Seed 0 Only)
configs=$(ls configs/matrix/*_seed0.yaml)
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
    
    # Run with grep filtering to show only minimal progress info
    # Shows: "Queries: X000", "Best", "Track", "Result" - Hides individual epochs
    python -m mebench run --config "$config" --device cuda:0 2>&1 | tee "logs/${name}.log" | grep -E "Queries: [0-9]+000/|Track|Best|Check|track_|Error|Exception|===="
    
    # Check pipe status (pipestatus[0] captures python exit code)
    exit_code=${PIPESTATUS[0]}
    if [ $exit_code -eq 0 ]; then
        echo " -> SUCCESS"
    else
        echo " -> FAILED (Exit Code: $exit_code). Check logs/${name}.log"
    fi
done

# 3. Aggregate Results
echo "[Phase 3] Aggregating results..."
python aggregate_matrix.py

echo "Seed 0 execution complete."
