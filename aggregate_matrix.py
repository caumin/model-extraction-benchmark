import json
import os
import pandas as pd
import numpy as np
from pathlib import Path

def aggregate_matrix(root_dir="runs"):
    results = []
    root = Path(root_dir)
    
    # 1. Walk through the runs directory
    for run_dir in root.iterdir():
        if not run_dir.is_dir(): continue
        
        # Expected name format: SET-ID_attack[_budget]_seedS
        parts = run_dir.name.split('_')
        if len(parts) < 3: continue
        
        set_id = parts[0]
        # Handle cases like activethief_uncertainty
        if parts[2].endswith('k'):
             attack = parts[1]
             # If attack variant like activethief_uncertainty_1k
             if parts[1] == "activethief" and not parts[2].endswith('k'):
                 # This logic needs to be more robust
                 pass
        
        # Simpler robust parsing
        name_no_seed = "_".join(parts[:-1]) # e.g., SET-A1_activethief_uncertainty_1k
        attack_part = "_".join(parts[1:-1]) # e.g., activethief_uncertainty_1k or dfme
        
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
            
            # 4. Extract data from ALL available checkpoints
            for cp_str, tracks in data.get("checkpoints", {}).items():
                if "track_a" not in tracks: continue
                metrics = tracks["track_a"]
                
                # Check if this is the 'Final' budget for an AL run
                # (to avoid using intermediate checkpoints of 10k/20k runs which are unfair)
                is_al_attack = any(al in attack_part for al in ["activethief", "swiftthief", "cloudleak", "inversenet"])
                if is_al_attack:
                    # For AL, only take the checkpoint that matches the run's max_budget
                    max_b = int(data.get("max_budget", 0))
                    if int(cp_str) != max_b:
                        continue
                
                results.append({
                    "Set": set_id,
                    "Attack": attack_part.replace("_1k","").replace("_10k","").replace("_20k","").upper(),
                    "Budget": int(cp_str),
                    "Seed": seed,
                    "Accuracy": metrics["acc_gt"],
                    "Agreement": metrics["agreement"]
                })

    if not results:
        print("No results found to aggregate.")
        return

    df = pd.DataFrame(results)
    
    # 5. Process for each budget separately
    all_budgets = sorted(df["Budget"].unique())
    print(f"\nFound results for budgets: {all_budgets}")

    for budget in all_budgets:
        b_df = df[df["Budget"] == budget]
        if b_df.empty: continue
        
        # Group by Set and Attack to compute Mean and Std
        stats = b_df.groupby(["Set", "Attack"])[["Accuracy", "Agreement"]].agg(["mean", "std"]).reset_index()
        stats.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in stats.columns]
        
        # Format as Mean ± Std
        stats["Accuracy (mean+std)"] = stats.apply(lambda x: f"{x['Accuracy_mean']:.4f} ± {x['Accuracy_std']:.4f}" if not pd.isna(x['Accuracy_std']) else f"{x['Accuracy_mean']:.4f}", axis=1)
        stats["Agreement (mean+std)"] = stats.apply(lambda x: f"{x['Agreement_mean']:.4f} ± {x['Agreement_std']:.4f}" if not pd.isna(x['Agreement_std']) else f"{x['Agreement_mean']:.4f}", axis=1)
        
        pivot_acc = stats.pivot(index="Attack", columns="Set", values="Accuracy (mean+std)")
        
        print(f"\n### Extraction Accuracy (mean+std) - Budget: {budget//1000}k")
        print(pivot_acc.to_markdown())
        
        pivot_acc.to_csv(f"matrix_results_accuracy_{budget//1000}k.csv", encoding='utf-8')
        
    # Full dump for custom analysis
    df.to_csv("matrix_results_all_raw.csv", index=False, encoding='utf-8')
    print("\nArtifacts saved: matrix_results_accuracy_{Nk}.csv, matrix_results_all_raw.csv")

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
