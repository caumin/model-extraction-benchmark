"""Factory for upstream Black-Box Ripper generators."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from mebench.models.blackbox_ripper.cifar_progan import Generator as CIFARProGANGenerator
from mebench.models.blackbox_ripper.cifar_sngan import Generator as CIFARSNGANGenerator


_KNOWN_GENERATORS = {
    "cifar_sngan": CIFARSNGANGenerator,
    "cifar_progan": CIFARProGANGenerator,
}


def _resolve_checkpoint_path(checkpoint_path: str) -> Path:
    path = Path(checkpoint_path)
    if path.exists():
        return path
    if path.suffix:
        raise FileNotFoundError(f"Generator checkpoint not found at {checkpoint_path}")

    for suffix in (".pth", ".pt"):
        candidate = path.with_suffix(suffix)
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Generator checkpoint not found at {checkpoint_path}")


def create_blackbox_ripper_generator(
    name: str,
    device: str,
    *,
    eval_mode: bool = True,
) -> nn.Module:
    name_l = str(name).lower()
    if name_l not in _KNOWN_GENERATORS:
        raise ValueError(
            f"Unknown BlackboxRipper generator '{name}'. "
            f"Supported: {sorted(_KNOWN_GENERATORS.keys())}"
        )
    gen = _KNOWN_GENERATORS[name_l]()
    gen.to(device)
    if eval_mode:
        gen.eval()
    return gen


def load_blackbox_ripper_generator_weights(
    generator: nn.Module,
    checkpoint_path: str,
    device: str,
    *,
    strict: bool = True,
) -> None:
    path = _resolve_checkpoint_path(checkpoint_path)

    # Prefer a safe load (PyTorch 2.0+). The official checkpoints are state_dict files.
    try:
        checkpoint: Any = torch.load(
            str(path),
            map_location=torch.device(device),
            weights_only=True,
        )
    except TypeError:
        # Older PyTorch: no weights_only.
        checkpoint = torch.load(str(path), map_location=torch.device(device))

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError(
            "Unsupported generator checkpoint format: expected a state_dict-like mapping."
        )

    # Strip common prefixes.
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    if any(k.startswith("generator.") for k in state_dict.keys()):
        state_dict = {k.replace("generator.", ""): v for k, v in state_dict.items()}

    generator.load_state_dict(state_dict, strict=strict)
