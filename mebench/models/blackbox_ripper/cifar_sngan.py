"""CIFAR SNGAN generator (vendored from upstream Black-Box Ripper).

Upstream source (CC0-1.0):
  https://github.com/antoniobarbalau/black-box-ripper

Notes:
- This file intentionally preserves the original module attribute names so that
  official checkpoints can be loaded without state-dict key mismatches.
- We only vendor the generator, as the benchmark attack uses a fixed pretrained
  generator and does not train the GAN during extraction.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GenBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int | None = None,
        ksize: int = 3,
        pad: int = 1,
        activation: nn.Module | None = None,
        upsample: bool = False,
        n_classes: int = 0,
    ) -> None:
        super().__init__()
        self.activation = activation or nn.ReLU()
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.n_classes = n_classes
        self.c1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=ksize, padding=pad)

        self.b1 = nn.BatchNorm2d(in_channels)
        self.b2 = nn.BatchNorm2d(hidden_channels)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def upsample_conv(self, x: torch.Tensor, conv: nn.Module) -> torch.Tensor:
        return conv(nn.UpsamplingNearest2d(scale_factor=2)(x))

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.b1(h)
        h = self.activation(h)
        h = self.upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def shortcut(self, x: torch.Tensor) -> torch.Tensor:
        if self.learnable_sc:
            x = self.upsample_conv(x, self.c_sc) if self.upsample else self.c_sc(x)
            return x
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual(x) + self.shortcut(x)


class _Args:
    pass


_DEFAULT_ARGS = _Args()
_DEFAULT_ARGS.latent_dim = 128
_DEFAULT_ARGS.gf_dim = 256
_DEFAULT_ARGS.bottom_width = 4


class Generator(nn.Module):
    def __init__(
        self,
        args: _Args = _DEFAULT_ARGS,
        activation: nn.Module | None = None,
        n_classes: int = 0,
    ) -> None:
        super().__init__()
        self.bottom_width = args.bottom_width
        self.activation = activation or nn.ReLU()
        self.n_classes = n_classes
        self.ch = args.gf_dim
        self.l1 = nn.Linear(args.latent_dim, (self.bottom_width**2) * self.ch)
        self.block2 = GenBlock(
            self.ch,
            self.ch,
            activation=self.activation,
            upsample=True,
            n_classes=n_classes,
        )
        self.block3 = GenBlock(
            self.ch,
            self.ch,
            activation=self.activation,
            upsample=True,
            n_classes=n_classes,
        )
        self.block4 = GenBlock(
            self.ch,
            self.ch,
            activation=self.activation,
            upsample=True,
            n_classes=n_classes,
        )
        self.b5 = nn.BatchNorm2d(self.ch)
        self.c5 = nn.Conv2d(self.ch, 3, kernel_size=3, stride=1, padding=1)

        # Convenience attribute for integration.
        self.latent_dim = int(args.latent_dim)

    def encoding_size(self) -> int:
        return 128

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = z
        h = self.l1(h).view(-1, self.ch, self.bottom_width, self.bottom_width)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.b5(h)
        h = self.activation(h)
        h = nn.Tanh()(self.c5(h))
        return h
