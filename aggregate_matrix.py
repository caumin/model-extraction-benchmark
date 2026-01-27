import json
import os
import pandas as pd
import numpy as np
from pathlib import Path

def aggregate_matrix(root_dir="runs"):
    results = []
    root = Path(root_dir)
    
    # 1. Walk through the runs directory
    # Expected name format: SET-ID_attack_seedS
    for run_dir in root.iterdir():
        if not run_dir.is_dir(): continue
        
        parts = run_dir.name.split('_')
        if len(parts) < 3: continue
        
        set_id = parts[0]
        attack = parts[1]
        
        # 2. Find the latest timestamp
        timestamps = sorted([d for d in run_dir.iterdir() if d.is_dir()])
        if not timestamps: continue
        latest_run = timestamps[-1]
        
        # 3. Iterate through seeds (0, 1, 2)
        for seed_dir in latest_run.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"): continue
            seed = seed_dir.name.split('_')[1]
            
            summary_path = seed_dir / "summary.json"
            if not summary_path.exists(): continue
            
            with open(summary_path, 'r') as f:
                data = json.load(f)
            
            # 4. Extract 10k Budget data
            checkpoint = "10000"
            if checkpoint in data["checkpoints"]:
                metrics = data["checkpoints"][checkpoint]["track_a"]
                results.append({
                    "Set": set_id,
                    "Attack": attack.upper(),
                    "Seed": seed,
                    "Accuracy": metrics["acc_gt"],
                    "Agreement": metrics["agreement"]
                })

    if not results:
        print("No results found to aggregate.")
        return

    df = pd.DataFrame(results)
    
    # 5. Group by Set and Attack to compute Mean and Std
    stats = df.groupby(["Set", "Attack"])[["Accuracy", "Agreement"]].agg(["mean", "std"]).reset_index()
    
    # Flatten columns: Accuracy_mean, Accuracy_std, etc.
    stats.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in stats.columns]
    
    # 6. Format as Mean ± Std
    stats["Accuracy (mean+std)"] = stats.apply(lambda x: f"{x['Accuracy_mean']:.4f} + {x['Accuracy_std']:.4f}" if not pd.isna(x['Accuracy_std']) else f"{x['Accuracy_mean']:.4f}", axis=1)
    stats["Agreement (mean+std)"] = stats.apply(lambda x: f"{x['Agreement_mean']:.4f} + {x['Agreement_std']:.4f}" if not pd.isna(x['Agreement_std']) else f"{x['Agreement_mean']:.4f}", axis=1)
    
    # 7. Pivot for Paper Table
    # Row: Attack, Column: Set
    pivot_acc = stats.pivot(index="Attack", columns="Set", values="Accuracy (mean+std)")
    pivot_agr = stats.pivot(index="Attack", columns="Set", values="Agreement (mean+std)")
    
    print("\n### Extraction Accuracy (mean+std) across Experimental Sets")
    print(pivot_acc.to_markdown())

    print("\n### Extraction Agreement (mean+std) across Experimental Sets")
    print(pivot_agr.to_markdown())
    
    pivot_acc.to_csv("matrix_results_accuracy.csv", encoding='utf-8')
    pivot_agr.to_csv("matrix_results_agreement.csv", encoding='utf-8')
    
    # Save full summary stats
    stats.to_csv("matrix_results_summary.csv", index=False, encoding='utf-8')
    
    # Generate LaTeX for Accuracy
    latex_acc = pivot_acc.to_latex(
        caption="Model Extraction Accuracy (mean $\\pm$ std) across different Victim-Surrogate setups (10k Budget).",
        label="tab:matrix_results_acc",
        escape=False
    )
    with open("matrix_results_accuracy.tex", "w", encoding='utf-8') as f:
        f.write(latex_acc)

    # Generate LaTeX for Agreement
    latex_agr = pivot_agr.to_latex(
        caption="Model Extraction Agreement (mean $\\pm$ std) across different Victim-Surrogate setups (10k Budget).",
        label="tab:matrix_results_agr",
        escape=False
    )
    with open("matrix_results_agreement.tex", "w", encoding='utf-8') as f:
        f.write(latex_agr)
    
    print("\nArtifacts saved: matrix_results_{accuracy,agreement,summary}.csv, matrix_results_{accuracy,agreement}.tex")

if __name__ == "__main__":
    aggregate_matrix()
