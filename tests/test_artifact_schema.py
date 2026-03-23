"""Artifact schema tests for summary.json, metrics.csv, and experiment.log."""

import csv
import json
import logging
from pathlib import Path

import pytest

import mebench.core.engine as engine
from mebench.core.logging import ArtifactLogger


def test_artifact_schema_fields(tmp_path: Path) -> None:
    logger = ArtifactLogger(tmp_path)
    try:
        config = {
            "run": {"name": "artifact_test", "seeds": [0]},
            "victim": {"victim_id": "test_victim", "output_mode": "soft_prob"},
            "attack": {"name": "random"},
            "dataset": {"data_mode": "seed"},
            "substitute": {"arch": "resnet18"},
        }

        logger.set_run_metadata(config)
        logger.log_event(step=5, name="checkpoint_reached", payload={"checkpoint": 5})
        logger.log_progress(step=10, metrics={"acc_gt": 0.1, "agreement": 0.2})
        logger.log_resource_snapshot(step=10, phase="train_start", device="cpu", payload={"seed": 0})
        logger.log_checkpoint(
            seed=0,
            checkpoint=10,
            track="track_b",
            metrics={"acc_gt": 0.1, "agreement": 0.2, "kl_mean": 0.3, "l1_mean": 0.4},
        )
        logger.finalize()

        summary_path = tmp_path / "summary.json"
        metrics_path = tmp_path / "metrics.csv"
        experiment_log_path = tmp_path / "experiment.log"

        assert summary_path.exists()
        assert metrics_path.exists()
        assert experiment_log_path.exists()

        with open(summary_path) as f:
            summary = json.load(f)

        for key in ["run_name", "seeds", "victim_id", "attack", "data_mode", "output_mode", "checkpoints"]:
            assert key in summary

        assert "10" in summary["checkpoints"]
        assert "track_b" in summary["checkpoints"]["10"]

        with open(metrics_path, newline="") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 1
        required_columns = {
            "seed",
            "checkpoint_B",
            "track",
            "acc_gt",
            "agreement",
            "kl_mean",
            "l1_mean",
            "attack",
            "data_mode",
            "output_mode",
            "victim_id",
            "substitute_arch",
        }
        assert required_columns.issubset(rows[0].keys())

        log_text = experiment_log_path.read_text(encoding="utf-8")
        assert "[Event] step=5 name=checkpoint_reached checkpoint=5" in log_text
        assert "[Progress] step=10 acc_gt=0.100000 agreement=0.200000" in log_text
        assert "[Event] step=10 name=resource_snapshot cuda_available=False device=cpu phase=train_start seed=0" in log_text
    finally:
        logger.close()


def test_run_experiment_closes_file_handler_on_setup_failure(tmp_path: Path, monkeypatch) -> None:
    def fake_create_run_dir(base_dir: Path, run_name: str, seed: int) -> Path:
        run_dir = tmp_path / run_name / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    monkeypatch.setattr(engine, "validate_config", lambda config: None)
    monkeypatch.setattr(engine, "create_run_dir", fake_create_run_dir)
    monkeypatch.setattr(engine, "set_seed", lambda seed: None)

    def fail_load_victim(victim_cfg, device: str):
        raise RuntimeError("victim load failed")

    monkeypatch.setattr(engine, "load_victim_from_config", fail_load_victim)

    config = {
        "run": {"name": "artifact_failure_test", "seeds": [0]},
        "victim": {
            "victim_id": "test_victim",
            "output_mode": "soft_prob",
            "num_classes": 10,
            "channels": 3,
            "input_size": [32, 32],
        },
        "attack": {"name": "random", "output_mode": "soft_prob"},
        "dataset": {"data_mode": "seed"},
        "substitute": {"arch": "resnet18"},
        "budget": {"max_budget": 10, "checkpoints": [10]},
    }

    with pytest.raises(RuntimeError, match="victim load failed"):
        engine.run_experiment(config, device="cpu")

    root_logger = logging.getLogger()
    leaked_handlers = []
    for handler in root_logger.handlers:
        base_filename = getattr(handler, "baseFilename", None)
        if base_filename and str(tmp_path) in str(base_filename):
            leaked_handlers.append(base_filename)
    assert leaked_handlers == []
