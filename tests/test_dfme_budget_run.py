"""DFME budget-limited end-to-end test."""

import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

import mebench.core.engine as engine
import mebench.attackers.runner as runner


def test_dfme_budget_limited_run(tmp_path, monkeypatch) -> None:
    """Run DFME with a small budget and verify artifacts/counts."""
    def fake_create_run_dir(base_dir: Path, run_name: str, seed: int) -> Path:
        run_dir = tmp_path / run_name / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    monkeypatch.setattr(engine, "create_run_dir", fake_create_run_dir)

    # Avoid slow/full dataset IO in unit tests. Victim accuracy verification is
    # orthogonal to this budget/accounting test.
    def fake_get_test_dataloader(
        name: str,
        batch_size: int = 128,
        num_workers=None,
        *,
        input_size=None,
        channels=None,
        **kwargs,
    ) -> DataLoader:
        c = int(channels) if channels is not None else 3
        if input_size is None:
            h, w = (32, 32)
        else:
            h, w = (int(input_size[0]), int(input_size[1]))
        x = torch.rand(256, c, h, w)
        y = torch.randint(0, 10, (256,))
        return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False, num_workers=0)

    monkeypatch.setattr(engine, "get_test_dataloader", fake_get_test_dataloader)
    monkeypatch.setattr(runner, "get_test_dataloader", fake_get_test_dataloader)

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

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    engine.run_experiment(config, device=device)

    run_dir = tmp_path / "dfme_budget_test" / "seed_0"
    summary_path = run_dir / "summary.json"
    history_path = run_dir / "metrics_history.csv"

    assert summary_path.exists(), "summary.json should be written"
    assert history_path.exists(), "metrics_history.csv should be written"

    with open(history_path, newline="") as f:
        rows = list(csv.DictReader(f))

    checkpoints = {
        int(row["checkpoint"])
        for row in rows
        if row.get("event") == "checkpoint_reached" and row.get("checkpoint")
    }
    assert checkpoints == {10, 20}

    # import pickle
    # with open(meta_path, "rb") as f:
    #     meta = pickle.load(f)
    # assert meta["count"] == 20
