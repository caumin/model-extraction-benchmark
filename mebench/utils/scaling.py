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


def normalize_input_scale(x: torch.Tensor, mode: str = "unit") -> torch.Tensor:
    """Normalize model input scale according to mode.

    Modes:
    - "unit": keep canonical [0, 1] scale (with clamp).
    - "tanh": convert canonical [0, 1] to [-1, 1].
    """

    mode_norm = str(mode).strip().lower()
    if mode_norm in {"unit", "0_1", "01"}:
        return clamp_unit(x)
    if mode_norm in {"tanh", "neg1_1", "-1_1", "-11"}:
        if not torch.isfinite(x).all():
            return torch.zeros_like(x)
        if x.numel() == 0:
            return x

        # If caller already provides tanh-space tensors ([-1, 1]), keep that
        # scale to avoid accidental double conversion in parity runs.
        x_min = float(x.detach().amin().item())
        x_max = float(x.detach().amax().item())
        if x_min < -1e-6 or x_max > 1.0 + 1e-6:
            return torch.clamp(x, -1.0, 1.0)

        return unit_to_tanh(clamp_unit(x))
    raise ValueError(f"Unsupported input scale mode: {mode!r}")
