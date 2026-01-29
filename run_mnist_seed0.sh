#!/bin/bash

# Find configs for SET-A1 (MNIST) and Seed 0 only
configs=$(ls configs/matrix/SET-A1_*_seed0.yaml)

echo "Starting Subset Execution: SET-A1 (MNIST) Seed 0"
echo "Target experiments: $(echo "$configs" | wc -l)"

for config in $configs; do
    name=$(basename "$config" .yaml)
    
    if ls "runs/${name}"/*/seed_*/summary.json >/dev/null 2>&1; then
        echo "[SKIP] $name already completed."
        continue
    fi

    echo "=========================================================="
    echo "Running: $name"
    echo "=========================================================="
    
    python -m mebench run --config "$config" --device cuda:0
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] $name failed."
    fi
done

echo "Subset execution complete."
python aggregate_matrix.py
