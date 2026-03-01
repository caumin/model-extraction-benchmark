"""Tests for evaluation mode side effects."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mebench.eval.metrics import evaluate_substitute


def _tiny_loader() -> DataLoader:
    x = torch.zeros(4, 3, 4, 4)
    y = torch.zeros(4, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False, num_workers=0)


def _tiny_model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(inplace=True),
        nn.Flatten(),
        nn.Linear(4 * 4 * 4, 10),
    )


def test_evaluate_substitute_restores_training_modes() -> None:
    substitute = _tiny_model()
    victim = _tiny_model()
    loader = _tiny_loader()

    substitute.train(True)
    victim.train(False)

    metrics = evaluate_substitute(
        substitute=substitute,
        victim=victim,
        test_loader=loader,
        device="cpu",
        output_mode="soft_prob",
        temperature=1.0,
    )

    assert substitute.training is True
    assert victim.training is False
    assert "acc_gt" in metrics
    assert "agreement" in metrics
    assert "kl_mean" in metrics
    assert "l1_mean" in metrics
