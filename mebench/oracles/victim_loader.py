"""Victim model loading from checkpoint with best practices."""

import re
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from mebench.models.substitute_factory import create_substitute
from mebench.utils.scaling import normalize_input_scale


class _InputScaleWrapper(nn.Module):
    """Apply configured input scaling before victim forward pass."""

    def __init__(self, model: nn.Module, input_scale_mode: str) -> None:
        super().__init__()
        self.model = model
        self.input_scale_mode = str(input_scale_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_scaled = normalize_input_scale(x, self.input_scale_mode)
        return self.model(x_scaled)


def _wrap_victim_input_scale(model: nn.Module, input_scale_mode: str, device: str) -> nn.Module:
    mode = str(input_scale_mode).strip().lower()
    if mode in {"unit", "0_1", "01"}:
        model.to(device)
        model.eval()
        return model

    wrapped = _InputScaleWrapper(model, mode)
    wrapped.to(device)
    wrapped.eval()
    return wrapped


def _canonicalize_state_dict_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Map common upstream checkpoint key styles to local model keys.

    Supported canonicalizations:
    - DataParallel prefix: ``module.`` -> removed
    - CIFAR-ResNet head: ``linear.`` -> ``fc.``
    - CIFAR-ResNet shortcut block: ``layerX.Y.shortcut.`` -> ``layerX.Y.downsample.``
    """

    out: Dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = str(key)
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        if new_key.startswith("model."):
            new_key = new_key[len("model.") :]

        new_key = new_key.replace("linear.", "fc.")
        new_key = re.sub(r"^layer(\d+)\.(\d+)\.shortcut\.", r"layer\1.\2.downsample.", new_key)
        out[new_key] = value
    return out


def _infer_width_mult_from_state_dict(arch: str, state_dict: Dict[str, Any]) -> Optional[int]:
    """Infer width_mult from checkpoint tensor shapes for supported CNN families."""

    conv1 = state_dict.get("conv1.weight")
    if conv1 is None or not hasattr(conv1, "shape"):
        return None

    out_channels = int(conv1.shape[0])
    arch_norm = str(arch).lower().strip()
    if arch_norm in {"resnet18", "resnet34"}:
        if out_channels % 64 != 0:
            return None
        return max(1, out_channels // 64)
    if arch_norm in {"resnet20", "wideresnet22", "wrn22", "wideresnet-22"}:
        if out_channels % 16 != 0:
            return None
        return max(1, out_channels // 16)
    return None


def load_victim_checkpoint(
    checkpoint_path: str,
    arch: str,
    num_classes: int,
    input_channels: int = 3,
    width_mult: int = 1,
    dropout_prob: float = 0.0,
    input_scale_mode: str = "unit",
    device: str = "cpu",
    strict: bool = True,
) -> nn.Module:
    """Load victim model from checkpoint with best practices.

    This implements:
    1. Security: weights_only=True (PyTorch 2.6+)
    2. Device mapping: map_location for cross-device loading
    3. Prefix handling: Strip 'module.' from DataParallel models
    4. State dict loading: Load into pre-initialized model
    5. eval() mode: Set model to evaluation mode

    Args:
        checkpoint_path: Path to checkpoint file (.pt, .pth, .pth.tar)
        arch: Model architecture name (resnet18, lenet, etc.)
        num_classes: Number of output classes
        input_channels: Number of input channels
        device: Target device ('cuda:0', 'cpu', etc.)
        strict: Whether to strictly enforce state dict key matching

    Returns:
        Loaded victim model in eval mode on specified device

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint loading fails
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Victim checkpoint not found at {checkpoint_path}")

    # Load checkpoint with security and device mapping
    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device(device),
        weights_only=True,  # Security: prevent arbitrary code execution
    )

    # Extract state dict (handle different checkpoint formats)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get(
            "state_dict",
            checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint)),
        )
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise RuntimeError("Unsupported checkpoint payload: expected state_dict mapping")

    state_dict = _canonicalize_state_dict_keys(state_dict)

    requested_width = int(width_mult)
    inferred_width = _infer_width_mult_from_state_dict(arch, state_dict)
    resolved_width = requested_width
    if inferred_width is not None and inferred_width != requested_width:
        print(
            "WARNING: checkpoint appears to use "
            f"width_mult={inferred_width}, but config requested width_mult={requested_width}. "
            f"Using inferred width_mult={inferred_width}."
        )
        resolved_width = inferred_width

    # Initialize model architecture
    model = create_substitute(
        arch=arch,
        num_classes=num_classes,
        input_channels=input_channels,
        width_mult=resolved_width,
        dropout_prob=dropout_prob,
    )

    # Load state dict into model
    model.load_state_dict(state_dict, strict=strict)

    # Move to target device, apply optional input scale wrapper, and set eval mode.
    wrapped_model = _wrap_victim_input_scale(model, input_scale_mode, device)

    print(
        f"Loaded victim model from {checkpoint_path} to {device} "
        f"(input_scale_mode={str(input_scale_mode).lower()})"
    )
    return wrapped_model


def load_victim_from_config(
    victim_config: Dict[str, Any],
    device: str = "cpu",
) -> nn.Module:
    """Load victim model from configuration.

    Handles both checkpoint loading and placeholder creation.

    Args:
        victim_config: Victim configuration dict from YAML
            - checkpoint_ref: Path to checkpoint file (or None for placeholder)
            - arch: Model architecture (if checkpoint not provided)
            - channels: Input channels
            - num_classes: Number of classes (default 10)
        device: Target device

    Returns:
        Loaded victim model in eval mode
    """
    checkpoint_ref = victim_config.get("checkpoint_ref", None)
    num_classes = victim_config.get("num_classes")
    if num_classes is None:
        raise ValueError("victim.num_classes is required")

    if checkpoint_ref and checkpoint_ref != "/path/to/ckpt.pt":
        # Load from actual checkpoint
        return load_victim_checkpoint(
            checkpoint_path=checkpoint_ref,
            arch=victim_config.get("arch", "resnet18"),
            num_classes=num_classes,
            input_channels=victim_config.get("channels", 3),
            width_mult=int(victim_config.get("width_mult", 1)),
            dropout_prob=float(victim_config.get("dropout_prob", 0.0)),
            input_scale_mode=str(victim_config.get("input_scale_mode", "unit")),
            device=device,
        )
    else:
        # Create placeholder victim for testing
        print("WARNING: Using placeholder victim model (checkpoint_ref not set or is placeholder)")
        model = create_substitute(
            arch=victim_config.get("arch", "resnet18"),
            num_classes=num_classes,
            input_channels=victim_config.get("channels", 3),
            width_mult=int(victim_config.get("width_mult", 1)),
            dropout_prob=float(victim_config.get("dropout_prob", 0.0)),
        )
        return _wrap_victim_input_scale(
            model,
            str(victim_config.get("input_scale_mode", "unit")),
            device,
        )
