"""Tests for CIFAR10 test-loader preprocessing."""

from __future__ import annotations

from typing import Callable

from torch.utils.data import Dataset
from torchvision import transforms

from mebench.data.loaders import CIFAR10_MEAN, CIFAR10_STD, get_test_dataloader


class _DummyCIFAR10(Dataset):
    def __init__(self, transform: Callable | None = None) -> None:
        self.transform = transform

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int):  # pragma: no cover - not needed for this test
        raise RuntimeError("This dataset should not be iterated in this test")


def test_get_test_dataloader_applies_cifar10_normalization(monkeypatch) -> None:
    def _fake_cifar10(*args, **kwargs):
        return _DummyCIFAR10(transform=kwargs.get("transform"))

    monkeypatch.setattr("mebench.data.loaders.torchvision.datasets.CIFAR10", _fake_cifar10)

    loader = get_test_dataloader(name="CIFAR10", batch_size=8, num_workers=0)

    transform = loader.dataset.transform
    assert isinstance(transform, transforms.Compose)

    normalize_ops = [op for op in transform.transforms if isinstance(op, transforms.Normalize)]
    assert len(normalize_ops) == 1

    normalize = normalize_ops[0]
    assert tuple(float(x) for x in normalize.mean) == CIFAR10_MEAN
    assert tuple(float(x) for x in normalize.std) == CIFAR10_STD
