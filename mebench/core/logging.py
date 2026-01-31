"""Artifact logging for benchmark results."""

import json
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List


def setup_console_logging():
    """Configure logging to show INFO level and above to console."""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        root_logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(message)s'))
        root_logger.addHandler(handler)


class ArtifactLogger:
    """Log benchmark results to files."""

    def __init__(self, run_dir: Path):
        """Initialize artifact logger.

        Args:
            run_dir: Directory to store artifacts
        """
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.summary_data = {
            "run_name": "",
            "seeds": [],
            "victim_id": "",
            "attack": "",
            "data_mode": "",
            "output_mode": "",
            "checkpoints": {},
        }

        self.metrics_rows: List[Dict[str, Any]] = []
        self.history_rows: List[Dict[str, Any]] = []  # New: Time-series history
        self.checkpoint_metrics: Dict[int, Dict[str, Any]] = {}

    def log_history(self, step: int, metrics: Dict[str, float]) -> None:
        """Log time-series metrics (e.g., loss, val_f1) at a specific step.
        
        Args:
            step: Current query count or iteration
            metrics: Dictionary of metric names and values
        """
        row = {"step": step, "timestamp": datetime.now().isoformat(), **metrics}
        self.history_rows.append(row)
        
        # Flush to disk immediately for real-time monitoring
        self._append_csv("metrics_history.csv", row)

    def _append_csv(self, filename: str, row: Dict[str, Any]) -> None:
        """Append a single row to a CSV file."""
        file_path = self.run_dir / filename
        file_exists = file_path.exists()
        
        fieldnames = list(row.keys())
        with open(file_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def set_run_metadata(self, config: Dict[str, Any]) -> None:
        """Set run metadata from config.

        Args:
            config: Experiment configuration
        """
        self.summary_data["run_name"] = config["run"]["name"]
        self.summary_data["seeds"] = config["run"]["seeds"]
        self.summary_data["victim_id"] = config["victim"]["victim_id"]
        self.summary_data["attack"] = config["attack"]["name"]
        self.summary_data["data_mode"] = config["dataset"]["data_mode"]
        self.summary_data["output_mode"] = config["victim"]["output_mode"]
        self.summary_data["substitute_arch"] = config["substitute"].get("arch", "")

    def log_checkpoint(
        self,
        seed: int,
        checkpoint: int,
        track: str,
        metrics: Dict[str, float],
    ) -> None:
        """Log metrics for a checkpoint.

        Args:
            seed: Random seed
            checkpoint: Checkpoint budget
            track: Track name (track_a or track_b)
            metrics: Dictionary of metrics
        """
        # Add to metrics rows
        row = {
            "seed": seed,
            "checkpoint_B": checkpoint,
            "track": track,
            "acc_gt": metrics.get("acc_gt", 0.0),
            "agreement": metrics.get("agreement", 0.0),
            "kl_mean": metrics.get("kl_mean"),
            "l1_mean": metrics.get("l1_mean"),
            "attack": self.summary_data["attack"],
            "data_mode": self.summary_data["data_mode"],
            "output_mode": self.summary_data["output_mode"],
            "victim_id": self.summary_data["victim_id"],
            "substitute_arch": self.summary_data["substitute_arch"],
        }
        self.metrics_rows.append(row)

        # Update checkpoint summary
        if checkpoint not in self.checkpoint_metrics:
            self.checkpoint_metrics[checkpoint] = {}
        self.checkpoint_metrics[checkpoint][track] = metrics

    def save_summary(self) -> None:
        """Save summary.json file."""
        self.summary_data["checkpoints"] = self.checkpoint_metrics

        summary_path = self.run_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(self.summary_data, f, indent=2)

    def save_metrics_csv(self) -> None:
        """Save metrics.csv file."""
        metrics_path = self.run_dir / "metrics.csv"

        if not self.metrics_rows:
            return

        fieldnames = list(self.metrics_rows[0].keys())
        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.metrics_rows)

    def save_config(self, config: Dict[str, Any]) -> None:
        """Save run_config.yaml file.

        Args:
            config: Experiment configuration
        """
        import yaml

        config_path = self.run_dir / "run_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    def finalize(self) -> None:
        """Finalize logging (save all artifacts)."""
        self.save_summary()
        self.save_metrics_csv()


def create_run_dir(
    base_dir: Path,
    run_name: str,
    seed: int,
) -> Path:
    """Create run directory for a seed.

    Args:
        base_dir: Base runs directory
        run_name: Experiment run name
        seed: Random seed

    Returns:
        Path to run directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / run_name / timestamp / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
