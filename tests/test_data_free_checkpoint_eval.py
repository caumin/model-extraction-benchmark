"""Data-free Track B checkpoint evaluation test."""

import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

import mebench.attackers.runner as runner
import mebench.core.engine as engine


def test_data_free_track_b_evaluates_each_checkpoint(tmp_path, monkeypatch) -> None:
    def fake_create_run_dir(base_dir: Path, run_name: str, seed: int) -> Path:
        run_dir = tmp_path / run_name / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    monkeypatch.setattr(engine, "create_run_dir", fake_create_run_dir)

    def fake_get_test_dataloader(
        name: str,
        batch_size: int = 128,
        num_workers=None,
        *,
        input_size=None,
        channels=None,
    ) -> DataLoader:
        c = int(channels) if channels is not None else 3
        if input_size is None:
            h, w = (32, 32)
        else:
            h, w = (int(input_size[0]), int(input_size[1]))
        x = torch.rand(128, c, h, w)
        y = torch.randint(0, 10, (128,))
        return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False, num_workers=0)

    monkeypatch.setattr(engine, "get_test_dataloader", fake_get_test_dataloader)
    monkeypatch.setattr(runner, "get_test_dataloader", fake_get_test_dataloader)

    def fake_load_victim_from_config(victim_cfg, device: str):
        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 32 * 32, int(victim_cfg.get("num_classes", 10))),
        )
        return model.to(device)

    monkeypatch.setattr(engine, "load_victim_from_config", fake_load_victim_from_config)

    config = {
        "run": {"name": "dfme_checkpoint_eval_test", "seeds": [0], "device": "cpu"},
        "victim": {
            "victim_id": "test_victim",
            "checkpoint_ref": "runs/victims/cifar10_resnet18_seed0.pt",
            "arch": "resnet18",
            "channels": 3,
            "num_classes": 10,
            "input_size": [32, 32],
            "normalization": None,
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

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    engine.run_experiment(config, device=device)

    run_dir = tmp_path / "dfme_checkpoint_eval_test" / "seed_0"
    metrics_path = run_dir / "metrics.csv"
    assert metrics_path.exists(), "metrics.csv should be written"

    with open(metrics_path, newline="") as f:
        rows = list(csv.DictReader(f))

    track_b_checkpoints = {
        int(row["checkpoint_B"])
        for row in rows
        if row.get("track") == "track_b" and row.get("checkpoint_B")
    }
    assert track_b_checkpoints == {10, 20}
