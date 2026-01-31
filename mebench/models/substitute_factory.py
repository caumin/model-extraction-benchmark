"""Substitute model factory."""

from typing import Dict, Any
import torch
import torch.nn as nn


class ResNet(nn.Module):
    """Simplified ResNet for substitute model."""
    
    def __init__(
        self,
        num_classes: int,
        width_mult: int = 1,
        input_channels: int = 3,
        dropout_prob: float = 0.0,  # [P0 FIX] Add dropout for ActiveThief
    ):
        """Initialize ResNet.
        
        Args:
            num_classes: Number of output classes
            width_mult: Width multiplier (1=ResNet-18, 8=ResNet-18-8x)
            input_channels: Number of input channels
            dropout_prob: Dropout probability for ActiveThief paper compliance
        """
        super().__init__()
        self.width_mult = width_mult
        self.dropout_prob = dropout_prob

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64 * width_mult, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64 * width_mult),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(64 * width_mult, 128 * width_mult, 2)

        self.layer2 = self._make_layer(128 * width_mult, 256 * width_mult, 2)

        self.layer3 = self._make_layer(256 * width_mult, 512 * width_mult, 2)

        # [P0 FIX] Add dropout after conv blocks for ActiveThief compliance
        self.dropout1 = nn.Dropout2d(dropout_prob) if dropout_prob > 0 else nn.Identity()
        self.dropout2 = nn.Dropout2d(dropout_prob) if dropout_prob > 0 else nn.Identity()
        self.dropout3 = nn.Dropout2d(dropout_prob) if dropout_prob > 0 else nn.Identity()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * width_mult, num_classes)

    def _make_layer(self, in_channels: int, out_channels: int, stride: int) -> nn.Module:
        """Make a ResNet layer."""
        layers = []
        layers.append(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        )
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.dropout1(x)  # [P0 FIX] Apply dropout
        x = nn.MaxPool2d(3, stride=2, padding=1)(x)
        x = self.layer1(x)
        x = self.dropout2(x)  # [P0 FIX] Apply dropout
        x = self.layer2(x)
        x = self.dropout3(x)  # [P0 FIX] Apply dropout
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(self, layers: nn.Module, shortcut: nn.Module):
        """Initialize residual block."""
        super().__init__()
        self.layers = layers
        self.shortcut = shortcut
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)
        identity = self.shortcut(x)
        out += identity
        out = self.relu(out)
        return out


class LeNet(nn.Module):
    """Simple LeNet for small datasets."""

    def __init__(self, num_classes: int, input_channels: int = 1):
        """Initialize LeNet.

        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels
        """
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 6, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class LeNetMNIST(nn.Module):
    """LeNet variant for native MNIST 28x28 inputs.

    With 28x28 input and valid 5x5 convs:
    28 -> 24 -> pool -> 12 -> 8 -> pool -> 4, so flatten dim is 16*4*4.
    """

    def __init__(self, num_classes: int, input_channels: int = 1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 6, 5),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob),  # [P0 FIX] Add dropout for ActiveThief
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob),  # [P0 FIX] Add dropout for ActiveThief
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.Dropout2d(dropout_prob),  # [P0 FIX] Add dropout for ActiveThief
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_substitute(
    arch: str,
    num_classes: int,
    input_channels: int = 3,
    width_mult: int = 1,
    dropout_prob: float = 0.0,  # [P0 FIX] Add dropout parameter for ActiveThief
) -> nn.Module:
    """Create substitute model from architecture name.

    Args:
        arch: Architecture name (resnet18, lenet, etc.)
        num_classes: Number of output classes
        input_channels: Number of input channels
        width_mult: Width multiplier for ResNet

    Returns:
        Substitute model
    """
    if arch == "resnet18":
        return ResNet(num_classes=num_classes, width_mult=width_mult, input_channels=input_channels, dropout_prob=dropout_prob)
    elif arch == "resnet18-8x":
        return ResNet(num_classes=num_classes, width_mult=8, input_channels=input_channels, dropout_prob=dropout_prob)
    elif arch == "lenet":
        return LeNet(num_classes=num_classes, input_channels=input_channels, dropout_prob=dropout_prob)
    elif arch == "lenet_mnist":
        return LeNetMNIST(num_classes=num_classes, input_channels=input_channels, dropout_prob=dropout_prob)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def get_model_info(arch: str) -> Dict[str, Any]:
    """Get model information.

    Args:
        arch: Architecture name

    Returns:
        Dictionary with model info
    """
    info = {
        "resnet18": {
            "num_params": 11173962,  # ~11M
            "default_width": 1,
        },
        "resnet18-8x": {
            "num_params": 11173962 * 64,  # ~714M (8x channels)
            "default_width": 8,
        },
        "lenet": {
            "num_params": 61706,
            "default_width": 1,
        },
        "lenet_mnist": {
            "num_params": 61706,
            "default_width": 1,
        },
    }

    return info.get(arch, {"num_params": 0, "default_width": 1})
