"""Test Track B metrics consistency."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mebench.eval.metrics import evaluate_substitute


class TinyDataset(Dataset):
    """Minimal dataset for evaluation tests."""

    def __init__(self, size: int = 4) -> None:
        self.x = torch.zeros(size, 3, 4, 4)
        self.y = torch.zeros(size, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def test_track_b_metrics_are_numeric() -> None:
    """Track B evaluation should emit numeric metrics."""
    victim = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 10))
    substitute = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 10))
    test_loader = DataLoader(TinyDataset(), batch_size=2, shuffle=False)

    metrics = evaluate_substitute(
        substitute=substitute,
        victim=victim,
        test_loader=test_loader,
        device="cpu",
        output_mode="soft_prob",
    )

    assert isinstance(metrics["acc_gt"], float)
    assert isinstance(metrics["agreement"], float)
    assert isinstance(metrics["kl_mean"], float)
    assert isinstance(metrics["l1_mean"], float)
