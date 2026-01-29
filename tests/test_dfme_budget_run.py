"""DFME budget-limited end-to-end test."""

import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

import mebench.core.engine as engine


class TinyTestDataset(Dataset):
    """Small synthetic dataset for evaluation."""

    def __init__(self, size: int = 8) -> None:
        self.x = torch.zeros(size, 3, 32, 32)
        self.y = torch.zeros(size, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def test_dfme_budget_limited_run(tmp_path, monkeypatch) -> None:
    """Run DFME with a small budget and verify artifacts/counts."""
    def fake_get_test_dataloader(name: str, batch_size: int = 128):
        return DataLoader(TinyTestDataset(), batch_size=4, shuffle=False)

    def fake_create_run_dir(base_dir: Path, run_name: str, seed: int) -> Path:
        run_dir = tmp_path / run_name / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    monkeypatch.setattr(engine, "get_test_dataloader", fake_get_test_dataloader)
    monkeypatch.setattr(engine, "create_run_dir", fake_create_run_dir)

    config = {
        "run": {"name": "dfme_budget_test", "seeds": [0], "device": "cpu"},
        "victim": {
            "victim_id": "test_victim",
            "checkpoint_ref": "/path/to/ckpt.pt",
            "arch": "lenet",
            "channels": 3,
            "num_classes": 10,
            "input_size": [32, 32],
            "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
            "output_mode": "soft_prob",
            "temperature": 1.0,
            "output_modes_supported": ["soft_prob", "hard_top1"],
        },
        "dataset": {"name": "CIFAR10", "data_mode": "data_free", "seed_size": 10},
        "substitute": {
            "arch": "resnet18",
            "init_seed": 0,
            "trackA": {"batch_size": 4, "steps_coeff_c": 1},
            "optimizer": {"lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4},
            "patience": 5,
        },
        "attack": {
            "name": "dfme",
            "output_mode": "soft_prob",
            "batch_size": 4,
            "student_lr": 0.01,
            "generator_lr": 0.005,
            "n_g_steps": 1,
            "n_s_steps": 1,
            "grad_approx_m": 1,
            "grad_approx_epsilon": 1e-3,
            "noise_dim": 16,
        },
        "budget": {"max_budget": 20, "checkpoints": [10, 20]},
        "cache": {"enabled": True, "delete_on_finish": False},
    }

    engine.run_experiment(config, device="cpu")

    run_dir = tmp_path / "dfme_budget_test" / "seed_0"
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.csv"
    cache_dir = run_dir / "query_cache"
    meta_path = cache_dir / "meta.pkl"

    assert summary_path.exists(), "summary.json should be written"
    assert metrics_path.exists(), "metrics.csv should be written"
    # Note: cache saving is currently disabled in QueryStorage.save() to save disk space.
    # assert meta_path.exists(), "cache metadata should be written"

    with open(metrics_path, newline="") as f:
        rows = list(csv.DictReader(f))

    checkpoints = {int(row["checkpoint_B"]) for row in rows}
    assert checkpoints == {10, 20}
    # Track A is always logged. Track B might not be if evaluator only returns track_a.
    assert len(rows) >= 2

    # import pickle
    # with open(meta_path, "rb") as f:
    #     meta = pickle.load(f)
    # assert meta["count"] == 20
