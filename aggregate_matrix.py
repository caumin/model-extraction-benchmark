import json
import os
import pandas as pd
import numpy as np
from pathlib import Path

def aggregate_matrix(root_dir="runs", output_root="reports"):
    results = []
    root = Path(root_dir)
    output_path = Path(output_root)
    output_path.mkdir(exist_ok=True)
    
    if not root.exists():
        print(f"Root directory {root_dir} does not exist.")
        return

    print("Gathering results from runs...")
    
    # 1. Walk through the runs directory
    for run_dir in root.iterdir():
        if not run_dir.is_dir(): continue
        
        # Expected name format: SET-ID_attack_...
        parts = run_dir.name.split('_')
        if len(parts) < 2: continue
        
        set_id = parts[0]
        # Robust attack name extraction (everything between SET-ID and seed/budget if exists)
        # But we can also get it from summary.json
        
        # 2. Find all seed directories in any timestamp subfolder
        for timestamp_dir in run_dir.iterdir():
            if not timestamp_dir.is_dir(): continue
            
            for seed_dir in timestamp_dir.iterdir():
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"): continue
                
                seed = seed_dir.name.split('_')[1]
                summary_path = seed_dir / "summary.json"
                config_path = seed_dir / "run_config.yaml"
                if not summary_path.exists(): continue
                
                try:
                    with open(summary_path, 'r') as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"Error reading {summary_path}: {e}")
                    continue
                
                attack_name = data.get("attack", "unknown").upper()
                max_budget = data.get("max_budget") # [FIX] Get max_budget of the run
                
                # [FIX] Distinguish ActiveThief by strategy
                if attack_name == "ACTIVETHIEF" and config_path.exists():
                    try:
                        import yaml
                        with open(config_path, 'r') as f:
                            cfg = yaml.safe_load(f)
                        strategy = cfg.get("attack", {}).get("strategy", "unknown")
                        attack_name = f"ACTIVETHIEF ({strategy.upper()})"
                    except Exception as e:
                        print(f"Error reading config {config_path}: {e}")
                
                victim_id = data.get("victim_id", "unknown")
                substitute_arch = data.get("substitute_arch", "unknown")
                
                # 3. Extract data ONLY from the final checkpoint (Max Budget)
                for cp_str, tracks in data.get("checkpoints", {}).items():
                    if "track_a" not in tracks: continue
                    budget = int(cp_str)
                    
                    # [STRICT] Only collect final results to avoid pollution
                    if max_budget is not None and budget != int(max_budget):
                        continue
                    
                    metrics = tracks["track_a"]
                    results.append({
                        "Set": set_id,
                        "Attack": attack_name,
                        "Budget": budget,
                        "Seed": seed,
                        "Accuracy": metrics.get("acc_gt", 0),
                        "Agreement": metrics.get("agreement", 0),
                        "KL": metrics.get("kl_mean", 0),
                        "L1": metrics.get("l1_mean", 0),
                        "Victim": victim_id,
                        "Substitute": substitute_arch,
                        "Timestamp": timestamp_dir.name # [FIX] Track timestamp for deduplication
                    })
                    
    if not results:
        print("No results found to aggregate.")
        return

    df = pd.DataFrame(results)
    
    # [FIX] Deduplicate: If same (Set, Attack, Budget, Seed) exists, keep only the latest timestamp
    if not df.empty:
        # Sort by timestamp descending so the latest is first
        df = df.sort_values("Timestamp", ascending=False)
        # Drop duplicates, keeping the first (latest) one
        df = df.drop_duplicates(subset=["Set", "Attack", "Budget", "Seed"], keep="first")
    
    # Save raw master data
    df.to_csv(output_path / "master_results_raw.csv", index=False)
    
    # 4. Generate Reports per Set
    all_sets = sorted(df["Set"].unique())
    
    for s_id in all_sets:
        s_df = df[df["Set"] == s_id]
        s_dir = output_path / s_id
        s_dir.mkdir(exist_ok=True)
        
        # Save set-specific raw data
        s_df.to_csv(s_dir / "results_raw.csv", index=False)
        
        # Group and compute Mean/Std
        stats = s_df.groupby(["Attack", "Budget"])[["Accuracy", "Agreement", "KL"]].agg(["mean", "std"]).reset_index()
        stats.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in stats.columns]
        
        # Format for table
        stats["Acc_Display"] = stats.apply(lambda x: f"{x['Accuracy_mean']:.4f}" + (f" ± {x['Accuracy_std']:.4f}" if not pd.isna(x['Accuracy_std']) else ""), axis=1)
        stats["Agr_Display"] = stats.apply(lambda x: f"{x['Agreement_mean']:.4f}" + (f" ± {x['Agreement_std']:.4f}" if not pd.isna(x['Agreement_std']) else ""), axis=1)
        
        # Pivot for easy reading: Row=Attack, Column=Budget
        pivot_acc = stats.pivot(index="Attack", columns="Budget", values="Acc_Display").fillna("-")
        pivot_acc.columns = [f"{c//1000}k" if c >= 1000 else str(c) for c in pivot_acc.columns]
        
        # Determine winners
        winners = {}
        for budget in sorted(s_df["Budget"].unique()):
            b_df = s_df[s_df["Budget"] == budget]
            if not b_df.empty:
                # Group by attack and get mean
                means = b_df.groupby("Attack")["Accuracy"].mean()
                if not means.empty:
                    best_attack = means.idxmax()
                    best_val = means.max()
                    winners[budget] = (best_attack, best_val)

        # Write Markdown Report
        with open(s_dir / "REPORT.md", "w", encoding='utf-8') as f:
            f.write(f"# Experimental Report: {s_id}\n\n")
            
            # Metadata
            v_id = s_df["Victim"].iloc[0]
            s_arch = s_df["Substitute"].iloc[0]
            f.write(f"**Victim ID:** `{v_id}`  \n")
            f.write(f"**Substitute Arch:** `{s_arch}`\n\n")
            
            f.write("## 🏆 Winners by Budget\n\n")
            f.write("| Budget | Best Attack | Accuracy |\n")
            f.write("|:---|:---|:---:|\n")
            for b in sorted(winners.keys()):
                atk, val = winners[b]
                label = f"{b//1000}k" if b >= 1000 else str(b)
                f.write(f"| {label} | **{atk}** | {val:.4f} |\n")
            f.write("\n")

            f.write("## 📊 Extraction Accuracy (Mean ± Std)\n\n")
            f.write(pivot_acc.to_markdown() + "\n\n")
            
            f.write("## 🔍 Detailed Metrics (Agreement)\n\n")
            pivot_agr = stats.pivot(index="Attack", columns="Budget", values="Agr_Display").fillna("-")
            pivot_agr.columns = [f"{c//1000}k" if c >= 1000 else str(c) for c in pivot_agr.columns]
            f.write(pivot_agr.to_markdown() + "\n\n")
            
            f.write("---\n*Generated by Model Extraction Benchmark Aggregator*")
            
        print(f"Generated report for {s_id} in {s_dir}")

    # 5. Generate Master Summary
    with open(output_path / "MASTER_SUMMARY.md", "w", encoding='utf-8') as f:
        f.write("# 🏆 Model Extraction Benchmark Master Summary\n\n")
        
        # For each budget, show a comparison across all sets
        common_budgets = sorted(df["Budget"].unique())
        for budget in common_budgets:
            b_df = df[df["Budget"] == budget]
            b_stats = b_df.groupby(["Set", "Attack"])["Accuracy"].mean().reset_index()
            # Pivot: Set as Columns, Attack as Rows
            master_pivot = b_stats.pivot(index="Attack", columns="Set", values="Accuracy")
            
            label = f"{budget//1000}k" if budget >= 1000 else str(budget)
            f.write(f"## Budget: {label}\n\n")
            f.write(master_pivot.to_markdown() + "\n\n")
            
    print(f"\nMaster summary available at {output_path / 'MASTER_SUMMARY.md'}")

if __name__ == "__main__":
    aggregate_matrix()
