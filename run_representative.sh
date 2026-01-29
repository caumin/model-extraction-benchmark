#!/bin/bash

# Define specific configs to run (Representative Subset)
configs=(
    "configs/matrix/SET-A1_random_seed0.yaml"
    "configs/matrix/SET-A1_activethief_seed0.yaml"
    "configs/matrix/SET-A1_dfme_seed0.yaml"
    "configs/matrix/SET-A1_cloudleak_seed0.yaml"
    "configs/matrix/SET-A1_knockoff_nets_seed0.yaml"
)

echo "Starting Representative Execution: Random, ActiveThief, DFME, CloudLeak, KnockoffNets"

for config in "${configs[@]}"; do
    name=$(basename "$config" .yaml)
    
    if ls "runs/${name}"/*/seed_*/summary.json >/dev/null 2>&1; then
        echo "[SKIP] $name already completed."
        continue
    fi

    echo "=========================================================="
    echo "Running: $name"
    echo "=========================================================="
    
    # Run with timeout to prevent hang, output to log file
    python -m mebench run --config "$config" --device cuda:0 > "${name}.log" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] $name completed."
        # Show last few lines of log
        tail -n 5 "${name}.log"
        rm "${name}.log"
    else
        echo "[ERROR] $name failed. Check ${name}.log"
        cat "${name}.log"
    fi
done

echo "Subset execution complete."
python aggregate_matrix.py
