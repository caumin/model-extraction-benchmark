"""Test Track B metrics consistency."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mebench.core.state import BenchmarkState
from mebench.eval.evaluator import Evaluator


class TinyDataset(Dataset):
    """Minimal dataset for evaluation tests."""

    def __init__(self, size: int = 4) -> None:
        self.x = torch.zeros(size, 3, 4, 4)
        self.y = torch.zeros(size, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def test_track_b_metrics_when_substitute_missing() -> None:
    """Track B should return numeric metrics when substitute is missing."""
    config = {
        "run": {"device": "cpu"},
        "victim": {"output_mode": "soft_prob", "temperature": 1.0, "channels": 3},
        "substitute": {
            "arch": "lenet",
            "init_seed": 0,
            "trackA": {"batch_size": 1, "steps_coeff_c": 1},
            "optimizer": {"lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4},
        },
    }

    state = BenchmarkState()
    state.metadata = {"device": "cpu"}
    evaluator = Evaluator(config, state, query_storage=None)

    victim = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 10))
    test_loader = DataLoader(TinyDataset(), batch_size=2, shuffle=False)

    metrics = evaluator._evaluate_track_b(victim, test_loader, checkpoint_budget=10)

    assert metrics["acc_gt"] == 0.0
    assert metrics["agreement"] == 0.0
    assert metrics["kl_mean"] == 0.0
    assert metrics["l1_mean"] == 0.0
