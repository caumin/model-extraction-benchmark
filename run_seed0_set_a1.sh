#!/bin/bash

# Exit on error
set -e

echo "========================================================"
echo "Model Extraction Benchmark: Seed 0 Execution (SET-A1 Only)"
echo "========================================================"

# 1. Generate configurations (ensure latest configs exist)
echo "[Phase 1] Generating configurations..."
python generate_configs.py

# 2. Find all SET-A1 config files for seed 0
CONFIG_FILES=$(ls configs/matrix/SET-A1_*_seed0.yaml)
COUNT=$(echo "$CONFIG_FILES" | wc -l)

echo "Found $COUNT configurations for SET-A1 (Seed 0)"
echo "--------------------------------------------------------"

i=0
for config_file in $CONFIG_FILES; do
    i=$((i+1))
    
    # Extract run name from filename
    filename=$(basename "$config_file")
    run_name="${filename%.yaml}"
    
    # Check if run already exists and has summary.json
    # We check the 'runs' directory for any folder starting with run_name
    # and containing a 'seed_0' subfolder with 'summary.json'
    
    # Check if run directory exists
    if [ -d "runs/$run_name" ]; then
        # Find latest run directory for this config
        LATEST_RUN=$(ls -td runs/"$run_name"/* | head -n 1)
        
        if [ -n "$LATEST_RUN" ] && [ -f "$LATEST_RUN/seed_0/summary.json" ]; then
            echo "[$i/$COUNT] SKIP: $run_name already completed."
            continue
        fi
    fi
    
    echo "[$i/$COUNT] RUNNING: $run_name"
    python -m mebench run --config "$config_file" --device cuda:0
    
    # Optional: Clear cache if needed (handled by mebench internal cache policy)
done

echo "========================================================"
echo "SET-A1 Execution Completed"
echo "========================================================"
