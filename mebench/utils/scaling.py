"""Shared scaling helpers for the benchmark input contract.

Benchmark contract (v1.x):
- Oracle inputs are assumed to be in [0, 1] scale.
- No additional dataset mean/std normalization is applied by the Oracle.

To reduce accidental inconsistencies across attacks, we centralize common scale
conversions here.
"""

from __future__ import annotations

import torch


def clamp_unit(x: torch.Tensor) -> torch.Tensor:
    """Clamp an image tensor into [0, 1]."""

    if not torch.isfinite(x).all():
        return torch.zeros_like(x)
    return torch.clamp(x, 0.0, 1.0)


def tanh_to_unit(x: torch.Tensor) -> torch.Tensor:
    """Convert tanh-like generator output from [-1, 1] to [0, 1] and clamp."""

    if not torch.isfinite(x).all():
        return torch.zeros_like(x)
    return torch.clamp(x * 0.5 + 0.5, 0.0, 1.0)


def unit_to_tanh(x: torch.Tensor) -> torch.Tensor:
    """Convert [0, 1] to [-1, 1] (no clamp)."""

    return x * 2.0 - 1.0
