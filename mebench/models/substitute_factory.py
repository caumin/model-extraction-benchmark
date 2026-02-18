"""Substitute model factory."""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torchvision import models


from mebench.models.activethief_cnn import ActiveThiefCNN

def _require_torchvision_3ch_input(arch: str, input_channels: int) -> None:
    """Torchvision classification models assume 3-channel images.

    We intentionally avoid conv1 adaptation to keep architectures "as-is".
    """

    if int(input_channels) != 3:
        raise ValueError(
            f"Torchvision '{arch}' expects input_channels=3. "
            "Do not modify model conv1; instead set dataset transforms/config to output 3 channels. "
            f"Got input_channels={input_channels}"
        )


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock (2 convs + skip)."""

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = int(stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet-18 (CIFAR-style stem) with optional width multiplier.

    Notes:
    - Uses a 3x3, stride-1 conv stem and no maxpool to avoid aggressive early
      downsampling on small inputs (e.g., 32x32 CIFAR).
    - `width_mult` scales the channel width (used for resnet18-8x variants).
    """

    def __init__(
        self,
        num_classes: int,
        width_mult: int = 1,
        input_channels: int = 3,
        dropout_prob: float = 0.0,
        layers: Optional[tuple[int, int, int, int]] = None,
    ) -> None:
        super().__init__()
        self.width_mult = int(width_mult)
        self.dropout_prob = float(dropout_prob)
        block_layers = layers or (2, 2, 2, 2)

        if self.width_mult <= 0:
            raise ValueError(f"width_mult must be >= 1, got {self.width_mult}")

        stem_width = 64 * self.width_mult
        self.inplanes = stem_width

        self.conv1 = _conv3x3(int(input_channels), stem_width, stride=1)
        self.bn1 = nn.BatchNorm2d(stem_width)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64 * self.width_mult, blocks=int(block_layers[0]), stride=1)
        self.layer2 = self._make_layer(128 * self.width_mult, blocks=int(block_layers[1]), stride=2)
        self.layer3 = self._make_layer(256 * self.width_mult, blocks=int(block_layers[2]), stride=2)
        self.layer4 = self._make_layer(512 * self.width_mult, blocks=int(block_layers[3]), stride=2)

        self.dropout = nn.Dropout2d(self.dropout_prob) if self.dropout_prob > 0 else nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.width_mult * BasicBlock.expansion, int(num_classes))

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Module:
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * BasicBlock.expansion, stride=stride),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )

        layers = [
            BasicBlock(
                inplanes=self.inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
            )
        ]
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, int(blocks)):
            layers.append(BasicBlock(inplanes=self.inplanes, planes=planes, stride=1, downsample=None))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.dropout(x)
        x = self.layer4(x)
        x = self.dropout(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet20CIFAR(nn.Module):
    """CIFAR-style ResNet-20 (6n+2 with n=3)."""

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        width_mult: int = 1,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        base = 16 * int(width_mult)
        self.inplanes = base
        self.dropout = nn.Dropout2d(float(dropout_prob)) if float(dropout_prob) > 0 else nn.Identity()

        self.conv1 = _conv3x3(int(input_channels), base, stride=1)
        self.bn1 = nn.BatchNorm2d(base)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(base, blocks=3, stride=1)
        self.layer2 = self._make_layer(base * 2, blocks=3, stride=2)
        self.layer3 = self._make_layer(base * 4, blocks=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base * 4, int(num_classes))

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Module:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes, stride=stride),
                nn.BatchNorm2d(planes),
            )

        layers = [BasicBlock(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes
        for _ in range(1, int(blocks)):
            layers.append(BasicBlock(self.inplanes, planes, stride=1, downsample=None))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.dropout(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class WideBasicBlock(nn.Module):
    """WideResNet basic block for CIFAR-style networks."""

    def __init__(self, in_planes: int, out_planes: int, stride: int, dropout_prob: float) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = _conv3x3(in_planes, out_planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(float(dropout_prob)) if float(dropout_prob) > 0 else nn.Identity()
        self.conv2 = _conv3x3(out_planes, out_planes, stride=1)
        self.shortcut = (
            _conv1x1(in_planes, out_planes, stride=stride)
            if (stride != 1 or in_planes != out_planes)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.bn1(x))
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        out = self.dropout(out)
        out = self.conv2(out)
        return out + self.shortcut(x)


class WideResNet22CIFAR(nn.Module):
    """WideResNet-22 for CIFAR-style inputs (depth=22 => n=3)."""

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        width_mult: int = 2,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.n = 3
        self.k = int(width_mult)
        if self.k <= 0:
            raise ValueError(f"wideresnet22 width_mult must be >= 1, got {self.k}")

        widths = [16, 16 * self.k, 32 * self.k, 64 * self.k]
        self.in_planes = widths[0]

        self.conv1 = _conv3x3(int(input_channels), widths[0], stride=1)
        self.block1 = self._make_group(widths[1], stride=1, dropout_prob=float(dropout_prob))
        self.block2 = self._make_group(widths[2], stride=2, dropout_prob=float(dropout_prob))
        self.block3 = self._make_group(widths[3], stride=2, dropout_prob=float(dropout_prob))
        self.bn = nn.BatchNorm2d(widths[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(widths[3], int(num_classes))

    def _make_group(self, out_planes: int, stride: int, dropout_prob: float) -> nn.Module:
        layers = [WideBasicBlock(self.in_planes, out_planes, stride=stride, dropout_prob=dropout_prob)]
        self.in_planes = out_planes
        for _ in range(1, int(self.n)):
            layers.append(WideBasicBlock(self.in_planes, out_planes, stride=1, dropout_prob=dropout_prob))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu(self.bn(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class AlexNetHalf(nn.Module):
    """AlexNet-half style model for CIFAR-sized inputs."""

    def __init__(self, num_classes: int, input_channels: int = 3, dropout_prob: float = 0.5) -> None:
        super().__init__()
        p = float(dropout_prob)
        self.features = nn.Sequential(
            nn.Conv2d(int(input_channels), 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class LeNet(nn.Module):
    """Simple LeNet for small datasets."""

    def __init__(
        self, 
        num_classes: int, 
        input_channels: int = 1,
        dropout_prob: float = 0.0,
    ):
        """Initialize LeNet.

        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels
            dropout_prob: Dropout probability
        """
        super().__init__()
        self.dropout_prob = dropout_prob

        features = [
            nn.Conv2d(input_channels, 6, 5),
            nn.ReLU(inplace=True),
        ]
        if dropout_prob > 0:
            features.append(nn.Dropout2d(dropout_prob))
        features.append(nn.MaxPool2d(2))
        features.extend([
            nn.Conv2d(6, 16, 5),
            nn.ReLU(inplace=True),
        ])
        if dropout_prob > 0:
            features.append(nn.Dropout2d(dropout_prob))
        features.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*features)

        classifier = [
            nn.Linear(16 * 5 * 5, 120),
        ]
        if dropout_prob > 0:
            classifier.append(nn.Dropout(dropout_prob))
        classifier.extend([
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        ])
        self.classifier = nn.Sequential(*classifier)

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

    def __init__(
        self, 
        num_classes: int, 
        input_channels: int = 1,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.dropout_prob = dropout_prob

        features = [
            nn.Conv2d(input_channels, 6, 5),
            nn.ReLU(inplace=True),
        ]
        if dropout_prob > 0:
            features.append(nn.Dropout2d(dropout_prob))
        features.append(nn.MaxPool2d(2))
        features.extend([
            nn.Conv2d(6, 16, 5),
            nn.ReLU(inplace=True),
        ])
        if dropout_prob > 0:
            features.append(nn.Dropout2d(dropout_prob))
        features.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*features)

        classifier = [
            nn.Linear(16 * 4 * 4, 120),
        ]
        if dropout_prob > 0:
            classifier.append(nn.Dropout(dropout_prob))
        classifier.extend([
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        ])
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class LeNet5MNIST(nn.Module):
    """LeNet-5 for MNIST (1x28x28).

    Classic LeNet-5 uses tanh + average pooling. We keep the overall structure:
    conv(6,5) -> avgpool -> conv(16,5) -> avgpool -> fc(120) -> fc(84) -> fc(K)
    """

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 1,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        if int(input_channels) != 1:
            raise ValueError(f"LeNet5MNIST expects input_channels=1, got {input_channels}")

        self.dropout_prob = float(dropout_prob)

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.act = nn.Tanh()

        self.dropout2d = nn.Dropout2d(self.dropout_prob) if self.dropout_prob > 0 else nn.Identity()
        self.dropout = nn.Dropout(self.dropout_prob) if self.dropout_prob > 0 else nn.Identity()

        # 28 -> 24 -> pool -> 12 -> 8 -> pool -> 4 => 16*4*4
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.dropout2d(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.dropout2d(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc3(x)
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
    arch = str(arch).lower()
    if arch == "resnet18":
        _require_torchvision_3ch_input("resnet18", input_channels)
        # [FIX] Use CIFAR-style ResNet (no maxpool, 3x3 stem) defined above 
        # instead of ImageNet-style torchvision.models.resnet18.
        # This is critical for 32x32 inputs to avoid aggressive downsampling.
        return ResNet(
            num_classes=num_classes,
            input_channels=input_channels,
            width_mult=width_mult,
            dropout_prob=dropout_prob,
            layers=(2, 2, 2, 2),
        )
    elif arch == "resnet34":
        _require_torchvision_3ch_input("resnet34", input_channels)
        return ResNet(
            num_classes=num_classes,
            input_channels=input_channels,
            width_mult=width_mult,
            dropout_prob=dropout_prob,
            layers=(3, 4, 6, 3),
        )
    elif arch == "resnet20":
        return ResNet20CIFAR(
            num_classes=num_classes,
            input_channels=input_channels,
            width_mult=width_mult,
            dropout_prob=dropout_prob,
        )
    elif arch in {"wideresnet22", "wrn22", "wideresnet-22"}:
        return WideResNet22CIFAR(
            num_classes=num_classes,
            input_channels=input_channels,
            width_mult=width_mult,
            dropout_prob=dropout_prob,
        )
    elif arch == "alexnet":
        _require_torchvision_3ch_input("alexnet", input_channels)
        if int(width_mult) != 1:
            raise ValueError("Torchvision alexnet does not support width_mult; use width_mult=1")
        if float(dropout_prob) != 0.0:
            raise ValueError("Torchvision alexnet does not support dropout_prob override; use dropout_prob=0.0")
        return models.alexnet(weights=None, num_classes=int(num_classes))
    elif arch in {"alexnet_half", "alexnet-half"}:
        return AlexNetHalf(
            num_classes=num_classes,
            input_channels=input_channels,
            dropout_prob=(dropout_prob if float(dropout_prob) > 0 else 0.5),
        )
    elif arch == "resnet18-8x":
        raise ValueError("resnet18-8x is not supported in torchvision mode; use resnet18")
    elif arch == "lenet":
        return LeNet(num_classes=num_classes, input_channels=input_channels, dropout_prob=dropout_prob)
    elif arch == "lenet_mnist":
        return LeNet5MNIST(num_classes=num_classes, input_channels=input_channels, dropout_prob=dropout_prob)
    elif arch == "activethief_cnn":
        return ActiveThiefCNN(
            num_classes=num_classes, 
            input_channels=input_channels, 
            dropout_prob=dropout_prob
        )
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
            "num_params": 11181642,  # torchvision resnet18 (num_classes=10)
            "default_width": 1,
        },
        "resnet34": {
            "num_params": 21289802,
            "default_width": 1,
        },
        "resnet20": {
            "num_params": 272474,
            "default_width": 1,
        },
        "wideresnet22": {
            "num_params": 0,  # depends on width_mult
            "default_width": 2,
        },
        "alexnet": {
            "num_params": 57044810,  # torchvision alexnet (num_classes=10)
            "default_width": 1,
        },
        "alexnet_half": {
            "num_params": 0,
            "default_width": 1,
        },
        "lenet": {
            "num_params": 61706,
            "default_width": 1,
        },
        "lenet_mnist": {
            "num_params": 61706,
            "default_width": 1,
        },
        "activethief_cnn": {
            "num_params": 0, # TBD
            "default_width": 1,
        },
    }

    return info.get(arch, {"num_params": 0, "default_width": 1})
