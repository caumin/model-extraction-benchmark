"""Equalized learning-rate layers (vendored from upstream Black-Box Ripper).

Upstream source (CC0-1.0):
  https://github.com/antoniobarbalau/black-box-ripper
"""

from __future__ import annotations

from numpy import sqrt
import torch
from torch import nn


class EqualLR:
    def __init__(self, name: str) -> None:
        self.name = name

    def compute_weight(self, module: nn.Module) -> torch.Tensor:
        weight = getattr(module, self.name + "_orig")
        fan_in = weight.data.size(1) * weight.data[0][0].numel()
        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module: nn.Module, name: str) -> "EqualLR":
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + "_orig", nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module: nn.Module, _input: object) -> None:
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module: nn.Module, name: str = "weight") -> nn.Module:
    EqualLR.apply(module, name)
    return module


class PixelNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input / torch.sqrt(torch.mean(input**2, dim=1, keepdim=True) + 1e-8)


class EqualConv2d(nn.Module):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input)


class EqualConvTranspose2d(nn.Module):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__()

        conv = nn.ConvTranspose2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input)


class EqualLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()

        linear = nn.Linear(in_features, out_features)
        linear.weight.data.normal_()
        linear.bias.data.zero_()
        self.linear = equal_lr(linear)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.linear(input)
