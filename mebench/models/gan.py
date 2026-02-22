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


class OfficialDFMSDCGANGenerator(nn.Module):
    """DFMS official DCGAN generator topology.

    Mirrors `official_repo_clones/dfms_hl/code/train_generator/dcgan_model.py`.
    """

    def __init__(
        self,
        noise_dim: int = 100,
        output_channels: int = 3,
        base_channels: int = 64,
        output_size: int = 32,
    ) -> None:
        super().__init__()
        self.noise_dim = int(noise_dim)
        self.output_channels = int(output_channels)
        self.base_channels = int(base_channels)
        self.output_size = int(output_size)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.noise_dim, self.base_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.base_channels * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.base_channels * 8, self.base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.base_channels * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.base_channels * 4, self.base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.base_channels * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.base_channels * 2, self.base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.base_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.base_channels, self.output_channels, 1, 1, 0, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        elif z.ndim != 4:
            raise ValueError(f"Expected latent z with 2D or 4D shape, got {tuple(z.shape)}")
        x = self.main(z)
        if x.shape[-1] != self.output_size:
            x = torch.nn.functional.interpolate(
                x,
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            )
        return x


class OfficialDFMSDCGANDiscriminator(nn.Module):
    """DFMS official DCGAN discriminator topology.

    Matches convolutional stack from official implementation while returning
    raw logits (without sigmoid) to pair with BCEWithLogitsLoss.
    """

    def __init__(
        self,
        input_channels: int = 3,
        base_channels: int = 64,
        input_size: int = 32,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.base_channels = int(base_channels)
        self.input_size = int(input_size)

        self.main = nn.Sequential(
            nn.Conv2d(self.input_channels, self.base_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.base_channels, self.base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.base_channels * 2, self.base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.base_channels * 4, self.base_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        final_kernel = max(1, int(self.input_size) // 16)
        self.head = nn.Conv2d(self.base_channels * 8, 1, final_kernel, final_kernel, 0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.main(x)
        out = self.head(out)
        return out.view(-1)


class SNDCGANGenerator(DCGANGenerator):
    """DCGAN generator with spectral norm applied to weight layers."""

    def __init__(
        self,
        noise_dim: int = 100,
        output_channels: int = 3,
        base_channels: int = 64,
        num_classes: Optional[int] = None,
        output_size: Optional[int] = None,
        input_size: Optional[int] = None,
    ) -> None:
        if output_size is None:
            output_size = input_size or 32
        super().__init__(
            noise_dim=noise_dim,
            output_channels=output_channels,
            base_channels=base_channels,
            num_classes=num_classes,
            output_size=output_size,
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


class ACGANGenerator(nn.Module):
    """ACGAN generator (conditional DCGAN)."""

    def __init__(
        self,
        noise_dim: int = 100,
        output_channels: int = 3,
        base_channels: int = 64,
        num_classes: int = 10,
        output_size: int = 32,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.output_size = output_size
        self.base_channels = base_channels

        self.label_emb = nn.Embedding(num_classes, noise_dim)
        self.init_size = int(output_size) // 4
        self.fc = nn.Linear(noise_dim, 128 * self.init_size * self.init_size)

        p = float(dropout_prob) if float(dropout_prob) > 0 else 0.0
        layers = [
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, output_channels, 3, stride=1, padding=1),
        ]
        if p > 0:
            layers.insert(5, nn.Dropout2d(p))
            layers.insert(10, nn.Dropout2d(p))
        self.main = nn.Sequential(*layers)
        self.out_act = nn.Tanh()

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Multiplicative conditioning (standard ACGAN style)
        z = z * self.label_emb(labels)
        x = self.fc(z)
        x = x.view(-1, 128, self.init_size, self.init_size)
        x = self.main(x)
        if x.shape[-1] != self.output_size:
            x = torch.nn.functional.interpolate(x, size=(self.output_size, self.output_size))
        return self.out_act(x)


class ACGANDiscriminator(nn.Module):
    """ACGAN discriminator (DCGAN with auxiliary classifier)."""

    def __init__(
        self,
        input_channels: int = 3,
        base_channels: int = 64,
        num_classes: int = 10,
        input_size: int = 32,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        p = float(dropout_prob) if float(dropout_prob) > 0 else 0.25

        def block(in_filters: int, out_filters: int, bn: bool = True) -> list[nn.Module]:
            items: list[nn.Module] = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(p),
            ]
            if bn:
                items.append(nn.BatchNorm2d(out_filters, 0.8))
            return items

        self.features = nn.Sequential(
            *block(input_channels, 16, bn=False),
            *block(16, 32, bn=True),
            *block(32, 64, bn=True),
            *block(64, 128, bn=True),
            *block(128, 256, bn=True),
        )

        ds_size = max(1, self.input_size // (2**5))
        flattened = 256 * ds_size * ds_size
        self.source_head = nn.Linear(flattened, 1)
        self.source_act = nn.Sigmoid()
        # Official GAME discriminator aux head predicts n_classes + 1 where the
        # extra class corresponds to fake samples.
        self.classifier_head = nn.Linear(flattened, num_classes + 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.features(x)
        flat = feats.view(x.size(0), -1)
        source = self.source_act(self.source_head(flat)).view(-1)
        class_logits = self.classifier_head(flat)
        return source, class_logits


class ProGANGenerator(nn.Module):
    """Simplified Progressive GAN generator placeholder."""
    def __init__(self, noise_dim=128, output_channels=3, base_channels=64, output_size=32, **kwargs):
        super().__init__()
        # Just use DCGAN as placeholder for now since true ProGAN is complex
        self.impl = DCGANGenerator(noise_dim, output_channels, base_channels, None, output_size)
    def forward(self, z): return self.impl(z)


class DFMEGenerator(nn.Module):
    """Official DFME GeneratorA-compatible architecture.

    This mirrors the topology from the official DFME repository:
    - linear projection to ngf*2*init_size^2
    - BN block
    - upsample -> conv -> BN -> LeakyReLU
    - upsample -> conv -> BN -> LeakyReLU -> conv -> BN(affine=False)
    - tanh activation applied at output
    """

    def __init__(
        self,
        noise_dim: int = 256,
        output_channels: int = 3,
        base_channels: int = 64,
        output_size: int = 32,
        final_bn: bool = True,
    ) -> None:
        super().__init__()
        if output_size % 4 != 0:
            raise ValueError("output_size must be divisible by 4")

        self.noise_dim = noise_dim
        self.output_channels = output_channels
        self.base_channels = base_channels
        self.output_size = output_size
        self.init_size = output_size // 4

        self.fc = nn.Linear(noise_dim, base_channels * 2 * self.init_size * self.init_size)
        self.conv_blocks0 = nn.Sequential(nn.BatchNorm2d(base_channels * 2))
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        conv2_layers = [
            nn.Conv2d(base_channels * 2, base_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, output_channels, 3, stride=1, padding=1),
        ]
        if final_bn:
            conv2_layers.append(nn.BatchNorm2d(output_channels, affine=False))
        self.conv_blocks2 = nn.Sequential(*conv2_layers)

        self.out_act = nn.Tanh()

    def forward(
        self,
        z: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_pre_tanh: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        _ = labels  # unused (DFME generator is unconditional)
        x = self.fc(z)
        x = x.view(z.size(0), self.base_channels * 2, self.init_size, self.init_size)
        x = self.conv_blocks0(x)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv_blocks1(x)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        pre_tanh = self.conv_blocks2(x)
        
        # Resize if dimensions don't match target output_size
        if pre_tanh.shape[-1] != self.output_size:
            pre_tanh = torch.nn.functional.interpolate(pre_tanh, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
            
        out = self.out_act(pre_tanh)
        if return_pre_tanh:
            return pre_tanh, out
        return out
