import json
from pathlib import Path

import pandas as pd


BINARY_METRICS = [
    ("Binary_Precision", "binary_precision"),
    ("Binary_Recall", "binary_recall"),
    ("Binary_F1", "binary_f1"),
    ("Binary_ROC_AUC", "binary_roc_auc"),
]

BASE_METRICS = ["Accuracy", "Agreement", "KL", "L1"]


def _format_mean_std(mean_value: float, std_value: float) -> str:
    value = f"{mean_value:.4f}"
    if pd.notna(std_value):
        value += f" +- {std_value:.4f}"
    return value


def _budget_label(value: int) -> str:
    return f"{value // 1000}k" if value >= 1000 else str(value)


def aggregate_matrix(root_dir: str = "runs", output_root: str = "reports") -> None:
    results = []
    root = Path(root_dir)
    output_path = Path(output_root)
    output_path.mkdir(exist_ok=True)

    if not root.exists():
        print(f"Root directory {root_dir} does not exist.")
        return

    print("Gathering results from runs...")

    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue

        parts = run_dir.name.split("_")
        if len(parts) < 2:
            continue

        set_id = parts[0]

        for timestamp_dir in run_dir.iterdir():
            if not timestamp_dir.is_dir():
                continue

            for seed_dir in timestamp_dir.iterdir():
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                    continue

                seed = seed_dir.name.split("_")[1]
                summary_path = seed_dir / "summary.json"
                config_path = seed_dir / "run_config.yaml"
                if not summary_path.exists():
                    continue

                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"Error reading {summary_path}: {e}")
                    continue

                attack_name = data.get("attack", "unknown").upper()
                max_budget = data.get("max_budget")

                if attack_name == "ACTIVETHIEF" and config_path.exists():
                    try:
                        import yaml

                        with open(config_path, "r", encoding="utf-8") as f:
                            cfg = yaml.safe_load(f)
                        strategy = cfg.get("attack", {}).get("strategy", "unknown")
                        attack_name = f"ACTIVETHIEF ({strategy.upper()})"
                    except Exception as e:
                        print(f"Error reading config {config_path}: {e}")

                victim_id = data.get("victim_id", "unknown")
                substitute_arch = data.get("substitute_arch", "unknown")

                for cp_str, tracks in data.get("checkpoints", {}).items():
                    if "track_b" not in tracks:
                        continue

                    try:
                        budget = int(cp_str)
                    except (TypeError, ValueError):
                        continue

                    if max_budget is not None and budget != int(max_budget):
                        continue

                    metrics = tracks["track_b"]
                    row = {
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
                        "Timestamp": timestamp_dir.name,
                    }

                    for display_name, metric_key in BINARY_METRICS:
                        if metric_key in metrics:
                            row[display_name] = metrics.get(metric_key)

                    results.append(row)

    if not results:
        print("No results found to aggregate.")
        return

    df = pd.DataFrame(results)

    if not df.empty and "Timestamp" in df.columns:
        df = df.sort_values("Timestamp", ascending=False)
        df = df.drop_duplicates(subset=["Set", "Attack", "Budget", "Seed"], keep="first")

    df.to_csv(output_path / "master_results_raw.csv", index=False)

    all_sets = sorted(df["Set"].unique())

    for s_id in all_sets:
        s_df = df[df["Set"] == s_id]
        s_dir = output_path / s_id
        s_dir.mkdir(exist_ok=True)
        s_df.to_csv(s_dir / "results_raw.csv", index=False)

        metric_columns = [m for m in BASE_METRICS if m in s_df.columns]
        for display_name, _ in BINARY_METRICS:
            if display_name in s_df.columns:
                metric_columns.append(display_name)

        if not metric_columns:
            continue

        stats = s_df.groupby(["Attack", "Budget"])[metric_columns].agg(["mean", "std"]).reset_index()
        stats.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in stats.columns]

        for metric_name in metric_columns:
            stats[f"{metric_name}_Display"] = stats.apply(
                lambda row: _format_mean_std(row[f"{metric_name}_mean"], row[f"{metric_name}_std"]),
                axis=1,
            )

        winners = {}
        for budget in sorted(s_df["Budget"].unique()):
            b_df = s_df[s_df["Budget"] == budget]
            if not b_df.empty and "Accuracy" in b_df.columns:
                means = b_df.groupby("Attack")["Accuracy"].mean()
                if not means.empty:
                    best_attack = means.idxmax()
                    winners[budget] = (best_attack, means.max())

        with open(s_dir / "REPORT.md", "w", encoding="utf-8") as f:
            f.write(f"# Experimental Report: {s_id}\n\n")
            f.write("Generated by Model Extraction Benchmark Aggregator\n\n")

            v_id = s_df["Victim"].iloc[0]
            s_arch = s_df["Substitute"].iloc[0]
            f.write(f"Victim ID: `{v_id}`  \n")
            f.write(f"Substitute Arch: `{s_arch}`\n\n")

            if winners:
                f.write("Winners by Budget\n\n")
                f.write("| Budget | Best Attack | Accuracy |\n")
                f.write("|:---|:---|:---:|\n")
                for budget in sorted(winners.keys()):
                    atk, val = winners[budget]
                    f.write(f"| {_budget_label(budget)} | {atk} | {val:.4f} |\n")
                f.write("\n")

            if "Accuracy" in s_df.columns:
                f.write("Extraction Accuracy (Mean +- Std)\n\n")
                pivot_acc = stats.pivot(index="Attack", columns="Budget", values="Accuracy_Display").fillna("-")
                pivot_acc.columns = [_budget_label(int(c)) for c in pivot_acc.columns]
                f.write(pivot_acc.to_markdown() + "\n\n")

            for metric_name in ["Agreement", "KL", "L1"]:
                if metric_name not in s_df.columns:
                    continue
                f.write(f"Detailed Metrics ({metric_name})\n\n")
                pivot = stats.pivot(index="Attack", columns="Budget", values=f"{metric_name}_Display").fillna("-")
                pivot.columns = [_budget_label(int(c)) for c in pivot.columns]
                f.write(pivot.to_markdown() + "\n\n")

            for metric_name, metric_label in BINARY_METRICS:
                if metric_name in s_df.columns:
                    f.write(f"Optional Metric ({metric_label})\n\n")
                    pivot = stats.pivot(index="Attack", columns="Budget", values=f"{metric_name}_Display").fillna("-")
                    pivot.columns = [_budget_label(int(c)) for c in pivot.columns]
                    f.write(pivot.to_markdown() + "\n\n")

        print(f"Generated report for {s_id} in {s_dir}")

    with open(output_path / "MASTER_SUMMARY.md", "w", encoding="utf-8") as f:
        f.write("# Master Summary\n\n")

        if "Accuracy" in df.columns:
            f.write("## Best Accuracy by Set\n\n")
            for s_id in sorted(df["Set"].unique()):
                s_df = df[df["Set"] == s_id]
                means = s_df.groupby("Attack")["Accuracy"].mean()
                if means.empty:
                    continue
                best_attack = means.idxmax()
                f.write(f"- `{s_id}`: {best_attack} ({means.max():.4f})\n")
            f.write("\n")

        for metric_name in ["Agreement", "KL", "L1"]:
            if metric_name not in df.columns:
                continue
            f.write(f"## Best {metric_name} by Set\n\n")
            for s_id in sorted(df["Set"].unique()):
                s_df = df[df["Set"] == s_id]
                means = s_df.groupby("Attack")[metric_name].mean()
                if means.empty:
                    continue
                best_attack = means.idxmax() if metric_name == "Agreement" else means.idxmin()
                direction = "(higher better)" if metric_name == "Agreement" else "(lower better)"
                f.write(
                    f"- `{s_id}`: {best_attack} ({means[best_attack]:.4f}) {direction}\n"
                )
            f.write("\n")
