"""Artifact schema tests for summary.json and metrics.csv."""

import csv
import json
from pathlib import Path

from mebench.core.logging import ArtifactLogger


def test_artifact_schema_fields(tmp_path: Path) -> None:
    logger = ArtifactLogger(tmp_path)
    config = {
        "run": {"name": "artifact_test", "seeds": [0]},
        "victim": {"victim_id": "test_victim", "output_mode": "soft_prob"},
        "attack": {"name": "random"},
        "dataset": {"data_mode": "seed"},
        "substitute": {"arch": "resnet18"},
    }

    logger.set_run_metadata(config)
    logger.log_checkpoint(
        seed=0,
        checkpoint=10,
        track="track_b",
        metrics={"acc_gt": 0.1, "agreement": 0.2, "kl_mean": 0.3, "l1_mean": 0.4},
    )
    logger.finalize()

    summary_path = tmp_path / "summary.json"
    metrics_path = tmp_path / "metrics.csv"

    assert summary_path.exists()
    assert metrics_path.exists()

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
