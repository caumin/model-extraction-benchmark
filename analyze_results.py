import json
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_results(root_dir="runs", output_dir="analysis_results"):
    print(f"Loading results from {root_dir}...")
    root = Path(root_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    preferred_tracks = ["track_b_unified", "track_b", "track_a"]
    
    final_results = []
    
    # 1. Load Final Metrics Only
    for run_dir in root.iterdir():
        if not run_dir.is_dir(): continue
        
        parts = run_dir.name.split('_')
        if len(parts) < 3: continue
        
        set_id = parts[0]
        name_parts = parts[1:-1]
        attack_name = "_".join(name_parts)
        
        # Clean suffix
        for suffix in ["_1k", "_10k", "_20k", "_50k", "_100k"]:
            if attack_name.endswith(suffix):
                attack_name = attack_name[:-len(suffix)]
                
        # Find latest timestamp
        timestamps = sorted([d for d in run_dir.iterdir() if d.is_dir()])
        if not timestamps: continue
        latest_run = timestamps[-1]
        
        # Iterate Seeds
        for seed_dir in latest_run.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"): continue
            try:
                seed_value = int(seed_dir.name.replace("seed_", ""))
            except ValueError:
                continue
            
            summary_path = seed_dir / "summary.json"
            if not summary_path.exists(): continue
            
            with open(summary_path, 'r') as f:
                data = json.load(f)
            
            # Use MAX budget checkpoint as final result
            checkpoints = data.get("checkpoints", {})
            if not checkpoints: continue
            
            # Find max budget
            max_cp = max(int(cp) for cp in checkpoints.keys())
            tracks = checkpoints[str(max_cp)]

            track_to_use = next((t for t in preferred_tracks if t in tracks), None)
            if track_to_use not in tracks: continue
            
            metrics = tracks[track_to_use]
            
            final_results.append({
                "Set": set_id,
                "Attack": attack_name.upper(),
                "Seed": seed_value,
                "Track": track_to_use,
                "Budget": max_cp, # [ADDED] Track budget
                "Final_Accuracy": metrics.get("acc_gt", np.nan),
                "Final_Agreement": metrics.get("agreement", np.nan),
            })

    if not final_results:
        print("No results found.")
        return

    df = pd.DataFrame(final_results)
    df.to_csv(f"{output_dir}/final_metrics_raw.csv", index=False)

    # Keep a single track per (Set, Attack, Budget):
    # prefer track_b_unified, then track_b, then track_a.
    track_rank = {name: i for i, name in enumerate(preferred_tracks)}
    selected_groups = []
    for (set_id, attack, budget), group in df.groupby(["Set", "Attack", "Budget"]):
        tracks_available = sorted(group["Track"].unique(), key=lambda t: track_rank.get(t, 999))
        selected_track = tracks_available[0]
        if len(tracks_available) > 1:
            print(
                f"[WARN] Mixed tracks for {set_id}/{attack}/{budget}: {tracks_available} -> using {selected_track}"
            )
        selected_groups.append(group[group["Track"] == selected_track])

    if selected_groups:
        df = pd.concat(selected_groups, ignore_index=True)
    df.to_csv(f"{output_dir}/final_metrics_selected.csv", index=False)
    
    # 2. Aggregate Mean/Std (Safe Explicit Method)
    # Avoid MultiIndex hell by iterating groups explicitly
    grouped = df.groupby(["Set", "Attack", "Budget"]) # [FIX] Group by Budget too
    
    table_rows = []
    for (set_id, attack, budget), group in grouped:
        track_label = group["Track"].iloc[0]
        acc_mean = group["Final_Accuracy"].mean()
        acc_std = group["Final_Accuracy"].std()
        agr_mean = group["Final_Agreement"].mean()
        agr_std = group["Final_Agreement"].std()
        
        # Safe string formatting
        acc_str = f"{acc_mean:.4f}"
        if pd.notna(acc_std):
            acc_str += f" ± {acc_std:.4f}"
            
        agr_str = f"{agr_mean:.4f}"
        if pd.notna(agr_std):
            agr_str += f" ± {agr_std:.4f}"
            
        table_rows.append({
            "Set": str(set_id),
            "Attack": str(attack),
            "Budget": int(budget), # Keep budget for filtering
            "Track": str(track_label),
            "Accuracy": acc_str,
            "Agreement": agr_str
        })
        
    summary_df = pd.DataFrame(table_rows)
    summary_df.to_csv(f"{output_dir}/final_summary.csv", index=False)
    
    # Pivot for better view (Attack x Set) PER BUDGET
    # Now guaranteed to be scalar strings, so pivot handles them safely
    unique_budgets = sorted(summary_df["Budget"].unique())
    latex_path = Path(output_dir) / "report.tex"
    
    def _latex_pm(value: str) -> str:
        if not isinstance(value, str):
            return value
        return value.replace(" ± ", " $\\pm$ ")
    
    with open(f"{output_dir}/report.md", "w") as f:
        f.write("# Final Benchmark Results\n\n")
        
        for b in unique_budgets:
            print(f"\n=== Budget: {b} Queries ===")
            f.write(f"## Budget: {b} Queries\n\n")
            
            subset = summary_df[summary_df["Budget"] == b]
            if subset.empty: continue
            
            try:
                pivot_acc = subset.pivot(index="Attack", columns="Set", values="Accuracy")
                pivot_acc.to_csv(f"{output_dir}/final_accuracy_matrix_{b}.csv")
                
                pivot_agr = subset.pivot(index="Attack", columns="Set", values="Agreement")
                pivot_agr.to_csv(f"{output_dir}/final_agreement_matrix_{b}.csv")
                
                print("[Accuracy]")
                print(pivot_acc.to_markdown())
                
                f.write("### Accuracy (Mean ± Std)\n")
                f.write(pivot_acc.to_markdown())
                f.write("\n\n")
                
                f.write("### Agreement (Mean ± Std)\n")
                f.write(pivot_agr.to_markdown())
                f.write("\n\n")
                
            except Exception as e:
                print(f"ERROR: Pivot failed for budget {b}: {e}")

    # LaTeX (Overleaf-style) report: Acc/Fidelity per attack
    with open(latex_path, "w") as f:
        f.write("% Auto-generated. Acc/Fidelity = Accuracy / Agreement\n\n")
        for b in unique_budgets:
            subset = summary_df[summary_df["Budget"] == b]
            if subset.empty:
                continue

            subset = subset.copy()
            subset["Acc/Fidelity"] = subset["Accuracy"] + " / " + subset["Agreement"]
            pivot_combined = subset.pivot(index="Attack", columns="Set", values="Acc/Fidelity")
            pivot_combined = pivot_combined.applymap(_latex_pm)
            pivot_combined.to_csv(f"{output_dir}/final_acc_fidelity_matrix_{b}.csv")

            col_format = "l" + "c" * len(pivot_combined.columns)
            table = pivot_combined.to_latex(
                escape=False,
                na_rep="-",
                column_format=col_format,
            )

            f.write("\\begin{table}[t]\n")
            f.write("\\centering\n")
            f.write(f"\\caption{{Final results at budget {b} queries (Acc/Fidelity).}}\n")
            f.write(f"\\label{{tab:final-{b}}}\n")
            f.write(table)
            f.write("\\end{table}\n\n")

    print(f"\nSaved analysis to {output_dir}/")

if __name__ == "__main__":
    analyze_results()
