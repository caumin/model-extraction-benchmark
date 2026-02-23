"""Official-compatible preprocessing profiles for attack parity.

This module provides explicit, named preprocessing callables derived from
official attack repositories under ``official_repo_clones/``.

Design goals:
- Keep benchmark default behavior unchanged unless a profile is explicitly used.
- Make preprocessing steps auditable via ``describe()``.
- Provide a single place to map official transform defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop


TensorTransform = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class OfficialPreprocess:
    """Named preprocessing profile with traceable steps."""

    name: str
    steps: List[str]
    transform: TensorTransform

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"{self.name} expects CHW tensor, got shape={tuple(x.shape)}")
        if x.dtype != torch.float32:
            x = x.float()
        return self.transform(x)

    def describe(self) -> str:
        return f"{self.name}: " + " -> ".join(self.steps)


def _compose(ops: List[TensorTransform]) -> TensorTransform:
    return Compose(ops)


def _identity_clamp() -> TensorTransform:
    return lambda x: x.clamp(0.0, 1.0)


def _profile_map() -> Dict[str, OfficialPreprocess]:
    return {
        # datafree-model-extraction/dfme/dataloader.py:57-59
        "dfme_cifar10_test": OfficialPreprocess(
            name="dfme_cifar10_test",
            steps=[
                "clamp[0,1]",
                "normalize(mean=(0.4914,0.4822,0.4465), std=(0.2023,0.1994,0.2010))",
            ],
            transform=_compose(
                [
                    _identity_clamp(),
                    Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            ),
        ),
        # maze/src/datasets/datasets.py:184-186
        "maze_rgb_test": OfficialPreprocess(
            name="maze_rgb_test",
            steps=["clamp[0,1]", "normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))"],
            transform=_compose([_identity_clamp(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        ),
        # swiftthief/utils/get_datasets.py:28-32
        "swiftthief_cifar_test": OfficialPreprocess(
            name="swiftthief_cifar_test",
            steps=[
                "resize(32,32)",
                "clamp[0,1]",
                "normalize(mean=(0.4914,0.4822,0.4465), std=(0.2023,0.1994,0.2010))",
            ],
            transform=_compose(
                [
                    Resize((32, 32)),
                    _identity_clamp(),
                    Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            ),
        ),
        # MARICH image branch (official_repo_clones/MARICH/lr_cnn_res_al/data.py)
        # uses ImageNet32 pool tensors normalized with these channel stats.
        "marich_imagenet32_cifar10_query": OfficialPreprocess(
            name="marich_imagenet32_cifar10_query",
            steps=[
                "clamp[0,1]",
                "normalize(mean=(0.473,0.450,0.401), std=(0.258,0.251,0.265))",
            ],
            transform=_compose(
                [
                    _identity_clamp(),
                    Normalize((0.473, 0.450, 0.401), (0.258, 0.251, 0.265)),
                ]
            ),
        ),
        # knockoffnets/knockoff/utils/transforms.py:26-30
        "knockoffnets_default_test": OfficialPreprocess(
            name="knockoffnets_default_test",
            steps=[
                "resize(256)",
                "center_crop(224)",
                "clamp[0,1]",
                "normalize(mean=IMAGENET, std=IMAGENET)",
            ],
            transform=_compose(
                [
                    Resize(256),
                    CenterCrop(224),
                    _identity_clamp(),
                    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            ),
        ),
        # dfms_hl/code/train_student/train_student.py:244-246
        "dfms_hl_train_student": OfficialPreprocess(
            name="dfms_hl_train_student",
            steps=["clamp[0,1]", "normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))"],
            transform=_compose([_identity_clamp(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        ),
    }


def list_official_preprocess_profiles() -> List[str]:
    """Return sorted preprocessing profile names."""

    return sorted(_profile_map().keys())


def get_official_preprocess(profile: str) -> OfficialPreprocess:
    """Return a named official-compatible preprocessing profile."""

    mapping = _profile_map()
    key = str(profile).strip()
    if key not in mapping:
        raise KeyError(
            f"Unknown official preprocess profile '{profile}'. "
            f"Available: {', '.join(sorted(mapping.keys()))}"
        )
    return mapping[key]


def apply_official_preprocess_batch(x: torch.Tensor, profile: str) -> torch.Tensor:
    """Apply official-compatible preprocessing to a batch tensor.

    Args:
        x: NCHW tensor in float space.
        profile: Profile name from ``list_official_preprocess_profiles``.
    """

    if x.ndim != 4:
        raise ValueError(f"Expected NCHW tensor, got shape={tuple(x.shape)}")
    proc = get_official_preprocess(profile)
    return torch.stack([proc(img) for img in x], dim=0)
