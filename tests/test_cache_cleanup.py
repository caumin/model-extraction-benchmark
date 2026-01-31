"""Cache cleanup policy tests."""

from pathlib import Path

import mebench.core.engine as engine


def _run_with_cleanup(tmp_path: Path, monkeypatch, delete_on_finish: bool) -> Path:
    def fake_create_run_dir(base_dir: Path, run_name: str, seed: int) -> Path:
        run_dir = tmp_path / run_name / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    monkeypatch.setattr(engine, "create_run_dir", fake_create_run_dir)

    config = {
        "run": {"name": f"cache_cleanup_{delete_on_finish}", "seeds": [0], "device": "cpu"},
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
            "trackA": {"batch_size": 2, "steps_coeff_c": 1},
            "optimizer": {"lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4},
            "patience": 2,
        },
        "attack": {
            "name": "dfme",
            "output_mode": "soft_prob",
            "batch_size": 2,
            "student_lr": 0.01,
            "generator_lr": 0.005,
            "n_g_steps": 1,
            "n_s_steps": 1,
            "grad_approx_m": 1,
            "grad_approx_epsilon": 1e-3,
            "noise_dim": 8,
        },
        "budget": {"max_budget": 4, "checkpoints": [4]},
        "cache": {"enabled": True, "delete_on_finish": delete_on_finish},
    }

    engine.run_experiment(config, device="cpu")

    return tmp_path / f"cache_cleanup_{delete_on_finish}" / "seed_0"


def test_cache_cleanup_deletes_on_finish(tmp_path, monkeypatch) -> None:
    run_dir = _run_with_cleanup(tmp_path, monkeypatch, delete_on_finish=True)
    cache_dir = run_dir / "query_cache"
    assert not cache_dir.exists()


def test_cache_cleanup_preserves_when_disabled(tmp_path, monkeypatch) -> None:
    run_dir = _run_with_cleanup(tmp_path, monkeypatch, delete_on_finish=False)
    cache_dir = run_dir / "query_cache"
    assert not cache_dir.exists()
