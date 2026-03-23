"""Artifact logging for benchmark results."""

import copy
import csv
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import torch


def setup_console_logging():
    """Configure logging to show INFO level and above to console."""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        root_logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
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
        self.checkpoint_metrics: Dict[int, Dict[str, Any]] = {}
        self.log_path = self.run_dir / "experiment.log"
        self._file_handler: logging.Handler | None = None
        self._attach_file_logging()

    def _attach_file_logging(self) -> None:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.log_path, encoding="utf-8")
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        root_logger.addHandler(handler)
        self._file_handler = handler

    def log_event(self, step: int, name: str, payload: Dict[str, Any] | None = None) -> None:
        details = self._format_fields(payload or {})
        suffix = f" {details}" if details else ""
        logging.getLogger("mebench.event").info("[Event] step=%d name=%s%s", int(step), name, suffix)

    def log_progress(self, step: int, metrics: Dict[str, Any]) -> None:
        details = self._format_fields(metrics)
        suffix = f" {details}" if details else ""
        logging.getLogger("mebench.progress").info("[Progress] step=%d%s", int(step), suffix)

    def log_resource_snapshot(
        self,
        step: int,
        phase: str,
        *,
        device: str | None = None,
        payload: Dict[str, Any] | None = None,
        reset_peak: bool = False,
    ) -> None:
        snapshot = dict(payload or {})
        snapshot["phase"] = str(phase)
        snapshot["device"] = str(device or "unknown")

        cuda_device = self._resolve_cuda_device(device)
        if cuda_device is not None:
            try:
                if reset_peak:
                    torch.cuda.reset_peak_memory_stats(cuda_device)
                snapshot.update(
                    {
                        "cuda_available": True,
                        "alloc_mb": round(torch.cuda.memory_allocated(cuda_device) / (1024 ** 2), 2),
                        "reserved_mb": round(torch.cuda.memory_reserved(cuda_device) / (1024 ** 2), 2),
                        "peak_alloc_mb": round(torch.cuda.max_memory_allocated(cuda_device) / (1024 ** 2), 2),
                        "peak_reserved_mb": round(torch.cuda.max_memory_reserved(cuda_device) / (1024 ** 2), 2),
                    }
                )
            except RuntimeError as exc:
                snapshot["cuda_available"] = False
                snapshot["cuda_error"] = str(exc)
        else:
            snapshot["cuda_available"] = False

        self.log_event(step, "resource_snapshot", snapshot)

    def _resolve_cuda_device(self, device: str | None) -> torch.device | None:
        if not torch.cuda.is_available():
            return None
        try:
            resolved = torch.device(device or "cuda")
        except (TypeError, RuntimeError, ValueError):
            return None
        if resolved.type != "cuda":
            return None
        if resolved.index is None:
            return torch.device("cuda", torch.cuda.current_device())
        return resolved

    def _format_fields(self, fields: Dict[str, Any]) -> str:
        parts: List[str] = []
        for key in sorted(fields.keys()):
            value = fields[key]
            if isinstance(value, float):
                parts.append(f"{key}={value:.6f}")
            else:
                parts.append(f"{key}={value}")
        return " ".join(parts)

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
            track: Canonical runtime track name
            metrics: Dictionary of metrics
        """
        track = "track_b"

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
        for key, value in metrics.items():
            if key not in row:
                row[key] = value
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

        fieldnames: List[str] = []
        for row in self.metrics_rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.metrics_rows)

    def save_config(self, config: Dict[str, Any]) -> None:
        """Save run_config.yaml file.

        Args:
            config: Experiment configuration
        """
        import yaml

        config_path = self.run_dir / "run_config.yaml"
        config_to_save = copy.deepcopy(config)

        run_section = config_to_save.setdefault("run", {})
        if "code_version" not in run_section:
            run_section["code_version"] = _get_git_commit()

        with open(config_path, "w") as f:
            yaml.dump(config_to_save, f, default_flow_style=False)

    def finalize(self) -> None:
        """Finalize logging (save all artifacts)."""
        self.save_summary()
        self.save_metrics_csv()

    def close(self) -> None:
        """Detach per-run file logging handler."""
        if self._file_handler is None:
            return
        root_logger = logging.getLogger()
        root_logger.removeHandler(self._file_handler)
        self._file_handler.close()
        self._file_handler = None


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


def _get_git_commit() -> str:
    """Best-effort git commit hash for run provenance."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        return "unknown"
    return "unknown"
