"""Shared scaling helpers for benchmark input-scale handling.

Benchmark contract:
- Pool-based query paths use unit-scale tensors ([0,1]).
- Data-free query paths use tanh-scale tensors ([-1,1]) at oracle boundary.

To reduce accidental inconsistencies across attacks, common conversions are
centralized here.
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
        if not torch.isfinite(x).all():
            return torch.zeros_like(x)
        if x.numel() == 0:
            return x

        # Heuristic conversion used by callers that explicitly request unit scale.
        # - If incoming tensor is tanh-space (contains negatives), convert to unit.
        # - Otherwise, treat input as unit and clamp.
        x_min = float(x.detach().amin().item())
        if x_min < -1e-6:
            return tanh_to_unit(torch.clamp(x, -1.0, 1.0))
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
