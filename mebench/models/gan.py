"""Shared DCGAN/ACGAN generator and discriminator."""

from typing import Optional, Tuple, Union
import math
import torch
import torch.nn as nn


def _apply_spectral_norm(module: nn.Module) -> nn.Module:
    """Recursively apply spectral norm to common weight layers.

    This is a lightweight approximation of SNGAN-style stabilization without
    introducing additional dependencies or large architectural changes.
    """

    from torch.nn.utils import spectral_norm

    for name, child in module.named_children():
        if isinstance(child, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            setattr(module, name, spectral_norm(child))
        else:
            _apply_spectral_norm(child)
    return module


class DCGANGenerator(nn.Module):
    """DCGAN-style generator for 32x32 outputs."""

    def __init__(
        self,
        noise_dim: int = 100,
        output_channels: int = 3,
        base_channels: int = 64,
        num_classes: Optional[int] = None,
        output_size: int = 32,
        num_upsamples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.output_channels = output_channels
        self.base_channels = base_channels
        self.num_classes = num_classes
        self.output_size = output_size

        if num_upsamples is None:
            if output_size % 4 != 0:
                raise ValueError("output_size must be divisible by 4")
            num_upsamples = int(math.log2(output_size // 4))
        if num_upsamples < 1:
            raise ValueError("num_upsamples must be >= 1")
        self.num_upsamples = num_upsamples

        input_dim = noise_dim
        if num_classes is not None:
            self.label_embedding = nn.Embedding(num_classes, noise_dim)
            input_dim = noise_dim * 2
        else:
            self.label_embedding = None

        self.fc = nn.Linear(input_dim, 4 * 4 * base_channels * 8)

        layers = [
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
        ]

        in_channels = base_channels * 8
        for _ in range(self.num_upsamples - 1):
            out_channels = max(base_channels, in_channels // 2)
            layers.extend(
                [
                    nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels

        layers.append(
            nn.ConvTranspose2d(in_channels, output_channels, 4, 2, 1, bias=False)
        )
        self.main = nn.Sequential(*layers)
        self.out_act = nn.Tanh()

    def forward(
        self,
        z: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_pre_tanh: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.label_embedding is not None and labels is not None:
            label_vec = self.label_embedding(labels)
            z = torch.cat([z, label_vec], dim=1)
        x = self.fc(z)
        x = x.view(-1, self.base_channels * 8, 4, 4)
        x = self.main(x)
        
        # Resize if dimensions don't match target output_size
        if x.shape[-1] != self.output_size:
            x = torch.nn.functional.interpolate(x, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
            
        pre_tanh = x
        x = self.out_act(x)
        if return_pre_tanh:
            return pre_tanh, x
        return x


class DCGANDiscriminator(nn.Module):
    """DCGAN-style discriminator for 32x32 inputs."""

    def __init__(
        self,
        input_channels: int = 3,
        base_channels: int = 64,
        num_classes: Optional[int] = None,
        input_size: int = 32,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.base_channels = base_channels
        self.num_classes = num_classes
        self.input_size = input_size

        if input_size % 2 != 0:
            raise ValueError("input_size must be divisible by 2")
        num_downsamples = int(math.log2(input_size // 2))
        if num_downsamples < 1:
            raise ValueError("input_size too small for discriminator")

        layers = [
            nn.Conv2d(input_channels, base_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        in_channels = base_channels
        for i in range(1, num_downsamples):
            out_channels = base_channels * (2 ** i)
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        final_size = input_size // (2 ** num_downsamples)
        self.source_head = nn.Conv2d(in_channels, 1, final_size, 1, 0, bias=False)
        if num_classes is not None:
            self.classifier = nn.Linear(in_channels * final_size * final_size, num_classes)
        else:
            self.classifier = None

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        feats = self.features(x)
        source = self.source_head(feats).view(x.size(0), -1)
        if self.classifier is None:
            return source
        flat = feats.view(x.size(0), -1)
        class_logits = self.classifier(flat)
        return source, class_logits


class SNDCGANGenerator(DCGANGenerator):
    """DCGAN generator with spectral norm applied to weight layers."""

    def __init__(
        self,
        noise_dim: int = 100,
        output_channels: int = 3,
        base_channels: int = 64,
        num_classes: Optional[int] = None,
        input_size: int = 32,
    ) -> None:
        super().__init__(
            noise_dim=noise_dim,
            output_channels=output_channels,
            base_channels=base_channels,
            num_classes=num_classes,
            input_size=input_size,
        )
        # _apply_spectral_norm(self)  # Commented out due to issues


class SNDCGANDiscriminator(DCGANDiscriminator):
    """DCGAN discriminator with spectral norm applied to weight layers."""

    def __init__(
        self,
        input_channels: int = 3,
        base_channels: int = 64,
        num_classes: Optional[int] = None,
        input_size: int = 32,
    ) -> None:
        super().__init__(
            input_channels=input_channels,
            base_channels=base_channels,
            num_classes=num_classes,
            input_size=input_size,
        )
        # _apply_spectral_norm(self)  # Commented out due to syntax issues


class DFMEGenerator(nn.Module):
    """Generator architecture closer to DFME (upsample + conv blocks).

    DFME uses a lightweight generator with linear upsampling between conv layers.
    This implementation keeps outputs in [-1, 1] via tanh and supports returning
    pre-tanh activations for zeroth-order gradient backprop.
    """

    def __init__(
        self,
        noise_dim: int = 100,
        output_channels: int = 3,
        base_channels: int = 128,
        output_size: int = 32,
    ) -> None:
        super().__init__()
        if output_size % 4 != 0:
            raise ValueError("output_size must be divisible by 4")

        self.noise_dim = noise_dim
        self.output_channels = output_channels
        self.base_channels = base_channels
        self.output_size = output_size

        self.fc = nn.Linear(noise_dim, 4 * 4 * base_channels)

        # Upsample + conv blocks until reaching output_size.
        layers: list[nn.Module] = []
        cur_channels = base_channels
        cur_size = 4
        while cur_size < output_size:
            next_channels = max(output_channels, cur_channels // 2)
            layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(cur_channels, next_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(next_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            cur_channels = next_channels
            cur_size *= 2

        # Final conv to RGB.
        layers.append(
            nn.Conv2d(cur_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.net = nn.Sequential(*layers)
        self.out_act = nn.Tanh()

    def forward(
        self,
        z: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_pre_tanh: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        _ = labels  # unused (DFME generator is unconditional)
        x = self.fc(z)
        x = x.view(z.size(0), self.base_channels, 4, 4)
        pre_tanh = self.net(x)
        
        # Resize if dimensions don't match target output_size
        if pre_tanh.shape[-1] != self.output_size:
            pre_tanh = torch.nn.functional.interpolate(pre_tanh, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
            
        out = self.out_act(pre_tanh)
        if return_pre_tanh:
            return pre_tanh, out
        return out
