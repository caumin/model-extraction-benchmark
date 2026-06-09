"""Augmentation pipeline for substitute training.

Provides a single-view augmentation `preprocess_fn` compatible with
`SubstituteTrainer.TrainRequest.preprocess_fn`. Mirrors SwiftThief's CL-stage
augmentation (RandomResizedCrop + HFlip + ColorJitter + Grayscale) but emits
one view per sample (other attacks have no contrastive objective).

Pipeline order (operates in [0,1] image domain):
    de-normalize  →  RandomResizedCrop / HFlip / ColorJitter / Grayscale  →  re-normalize

The pool dataloader returns already-normalized tensors (mebench common policy),
so this module de-normalizes first, applies augmentation in [0,1] domain, and
re-normalizes before returning to the trainer.

Reference: SwiftThief.TwoCropTransform (mebench/attackers/swiftthief.py:411).
"""
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torchvision.transforms.v2 as v2


_KNOWN_OPS = {
    "random_resized_crop",
    "random_crop",
    "hflip",
    "horizontal_flip",
    "color_jitter",
    "grayscale",
    "random_grayscale",
}


def _make_op(name: str, params: Dict[str, Any], input_size: Tuple[int, int]) -> Any:
    name = name.lower().strip()
    H, W = input_size
    if name == "random_resized_crop":
        scale = tuple(params.get("scale", (0.2, 1.0)))
        return v2.RandomResizedCrop((H, W), scale=scale, antialias=True)
    if name == "random_crop":
        # Padding-based crop (standard supervised aug). Output size defaults
        # to input_size; padding defaults to 4 (CIFAR-style) but can be
        # overridden per dataset (e.g. 2 for MNIST 28x28).
        size = tuple(params.get("size", (H, W)))
        padding = int(params.get("padding", 4))
        padding_mode = str(params.get("padding_mode", "reflect"))
        return v2.RandomCrop(size, padding=padding, padding_mode=padding_mode)
    if name in ("hflip", "horizontal_flip"):
        p = float(params.get("p", 0.5))
        return v2.RandomHorizontalFlip(p=p)
    if name == "color_jitter":
        brightness = float(params.get("brightness", 0.4))
        contrast = float(params.get("contrast", 0.4))
        saturation = float(params.get("saturation", 0.4))
        hue = float(params.get("hue", 0.1))
        p = float(params.get("p", 0.8))
        cj = v2.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )
        return v2.RandomApply([cj], p=p)
    if name in ("grayscale", "random_grayscale"):
        p = float(params.get("p", 0.2))
        return v2.RandomGrayscale(p=p)
    raise ValueError(f"Unknown augmentation op '{name}'. Known: {sorted(_KNOWN_OPS)}")


class SingleViewAugment:
    """Single-view aug callable: tensor[B,C,H,W] (normalized) → tensor[B,C,H,W] (normalized, augmented)."""

    def __init__(
        self,
        aug_compose: v2.Compose,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
    ):
        self.aug = aug_compose
        if mean is None or std is None:
            self._mean = None
            self._std = None
        else:
            self._mean = torch.tensor(list(mean), dtype=torch.float32).view(1, -1, 1, 1)
            self._std = torch.tensor(list(std), dtype=torch.float32).view(1, -1, 1, 1)

    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        if self._mean is None or self._std is None:
            return x
        return (x * self._std.to(x.device) + self._mean.to(x.device)).clamp(0.0, 1.0)

    def _renorm(self, x: torch.Tensor) -> torch.Tensor:
        if self._mean is None or self._std is None:
            return x
        return (x - self._mean.to(x.device)) / self._std.to(x.device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Expect [B, C, H, W] float tensor (already on device, normalized).
        x_raw = self._denorm(x.float())
        x_aug = self.aug(x_raw)
        return self._renorm(x_aug)


def build_augmentation_pipeline(
    spec: Optional[Dict[str, Any]],
    *,
    norm_stats: Optional[Tuple[Sequence[float], Sequence[float]]],
    input_size: Tuple[int, int],
) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
    """Build a single-view augmentation `preprocess_fn`, or return None if disabled.

    Args:
        spec: augmentation spec dict, e.g.
            {
              "pipeline": ["random_resized_crop", "hflip", "color_jitter", "grayscale"],
              "random_resized_crop": {"scale": [0.2, 1.0]},
              "color_jitter": {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1, "p": 0.8},
              "grayscale": {"p": 0.2},
            }
          A truthy `spec` with empty/missing "pipeline" is treated as no-op (returns None).
        norm_stats: (mean, std) tuple the pool loader used to normalize inputs. Pass None
            if the pool returns raw [0,1] tensors (no de/re-norm needed).
        input_size: (H, W) of the input crops.

    Returns:
        A callable `preprocess_fn(x: Tensor[B,C,H,W]) -> Tensor[B,C,H,W]` or None.
    """
    if not spec:
        return None
    pipeline: List[str] = list(spec.get("pipeline") or [])
    if not pipeline:
        return None
    ops = []
    for name in pipeline:
        params = spec.get(name, {}) if isinstance(spec.get(name), dict) else {}
        ops.append(_make_op(name, params, input_size))
    aug = v2.Compose(ops)

    mean = std = None
    if norm_stats is not None:
        mean, std = norm_stats
    return SingleViewAugment(aug, mean=mean, std=std)


def resolve_pool_norm_stats(state) -> Optional[Tuple[Sequence[float], Sequence[float]]]:
    """Recover (mean, std) used by the pool loader. Mirrors SwiftThief._get_pool_norm_stats."""
    from mebench.data.loaders import (
        _resolve_surrogate_normalization,
        get_surrogate_standard_normalization,
    )

    dataset_config = state.metadata.get("dataset_config", {}) or {}
    data_mode = str(dataset_config.get("data_mode", "seed"))
    channels = int(state.metadata.get("input_shape", (3, 32, 32))[0])

    try:
        if data_mode == "surrogate":
            stats = _resolve_surrogate_normalization(
                surrogate_name=str(dataset_config.get("surrogate_name", "ImageNet")),
                output_channels=channels,
                surrogate_normalization=dataset_config.get("surrogate_normalization", "standard"),
                surrogate_norm_mean=dataset_config.get("surrogate_norm_mean"),
                surrogate_norm_std=dataset_config.get("surrogate_norm_std"),
            )
        else:
            stats = get_surrogate_standard_normalization(
                str(dataset_config.get("name", "CIFAR10")),
                channels=channels,
            )
    except Exception:
        return None
    return stats
