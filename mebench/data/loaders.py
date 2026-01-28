"""Data loaders for seed, surrogate, and data-free modes."""

from typing import Dict, Any, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms


class SeedDataset(Dataset):
    """In-domain seed dataset (default 100 images)."""

    def __init__(
        self,
        name: str,
        seed_size: int = 100,
        train_split: bool = True,
        seed_split: str = "balanced",
    ):
        """Initialize seed dataset.

        Args:
            name: Dataset name (CIFAR10, MNIST, EMNIST, etc.)
            seed_size: Number of seed images
            train_split: Whether to use train or test split
            seed_split: 'balanced' (class-balanced) or 'by_class' (exact per class)
        """
        self.name = name
        self.seed_size = seed_size
        self.train_split = train_split
        self.seed_split = seed_split

        # Load full dataset
        if name == "CIFAR10":
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            full_dataset = torchvision.datasets.CIFAR10(
                root="./data",
                train=train_split,
                download=True,
                transform=transform,
            )
        elif name == "MNIST":
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            full_dataset = torchvision.datasets.MNIST(
                root="./data",
                train=train_split,
                download=True,
                transform=transform,
            )
        elif name == "EMNIST":
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            full_dataset = torchvision.datasets.EMNIST(
                root="./data",
                split="balanced",
                train=train_split,
                download=True,
                transform=transform,
            )
        elif name == "FashionMNIST":
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            full_dataset = torchvision.datasets.FashionMNIST(
                root="./data",
                train=train_split,
                download=True,
                transform=transform,
            )
        elif name == "GTSRB":
            transform = transforms.Compose([
                transforms.Resize((32, 32)),  # GTSRB images have variable sizes, resize to 32x32
                transforms.ToTensor(),
            ])
            full_dataset = torchvision.datasets.GTSRB(
                root="./data",
                split="train",
                download=True,
                transform=transform,
            )
        else:
            raise ValueError(f"Unknown dataset: {name}")

        # Deterministic subset for seed (balanced or by_class)
        if seed_split == "balanced":
            # Handle GTSRB which has different structure
            if name == "GTSRB":
                # GTSRB returns (x, y) tuples where y is integer label
                num_classes = 43  # GTSRB has 43 classes
                samples_per_class = (seed_size + num_classes - 1) // num_classes
                indices = []
                
                # Collect all labels first
                all_labels = []
                for i in range(min(len(full_dataset), seed_size * 2)):  # Sample enough to find all classes
                    try:
                        _, label = full_dataset[i]
                        all_labels.append(int(label))
                    except (IndexError, StopIteration):
                        break
                
                unique_labels = sorted(set(all_labels))
                if len(unique_labels) != num_classes:
                    # Fall back to first num_classes labels we found
                    num_classes = len(unique_labels)
                
                # Select samples per class
                for c in unique_labels:
                    class_indices = [i for i, label in enumerate(all_labels) if label == c]
                    chosen = class_indices[:samples_per_class]
                    indices.extend(chosen)
            else:
                num_classes = len(full_dataset.classes)
                samples_per_class = (seed_size + num_classes - 1) // num_classes
                indices = []
                targets = full_dataset.targets if hasattr(full_dataset, "targets") else None
                if targets is not None:
                    for c in range(num_classes):
                        class_indices = [i for i, t in enumerate(targets) if t == c]
                        chosen = class_indices[:samples_per_class]
                        indices.extend(chosen)
            # Trim if we overshoot
            indices = indices[:seed_size]
        elif seed_split == "by_class":
            # Exact per-class selection (useful for EMNIST letters split)
            samples_per_class = seed_size // len(full_dataset.classes)
            indices = []
            targets = full_dataset.targets if hasattr(full_dataset, "targets") else None
            if targets is not None:
                for c in range(len(full_dataset.classes)):
                    class_start = c * samples_per_class
                    class_end = class_start + samples_per_class
                    class_indices = [i for i, t in enumerate(targets) if t == c]
                    indices.extend(class_indices[class_start:class_end])
            indices = indices[:seed_size]
        else:
            raise ValueError(f"Unknown seed_split: {seed_split}")
        self.dataset = Subset(full_dataset, indices)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.dataset[idx]


class SurrogateDataset(Dataset):
    """Out-of-domain surrogate dataset."""

    def __init__(
        self,
        surrogate_name: str,
        train_split: bool = True,
    ):
        """Initialize surrogate dataset.

        Args:
            surrogate_name: Surrogate dataset name (SVHN, EMNIST, etc.)
            train_split: Whether to use train or test split
        """
        self.surrogate_name = surrogate_name
        self.train_split = train_split

        # Load surrogate dataset
        if surrogate_name == "SVHN":
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            self.dataset = torchvision.datasets.SVHN(
                root="./data",
                split="train" if train_split else "test",
                download=True,
                transform=transform,
            )
        elif surrogate_name == "EMNIST":
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            self.dataset = torchvision.datasets.EMNIST(
                root="./data",
                split="balanced",
                train=train_split,
                download=True,
                transform=transform,
            )
        elif surrogate_name == "FashionMNIST":
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            self.dataset = torchvision.datasets.FashionMNIST(
                root="./data",
                train=train_split,
                download=True,
                transform=transform,
            )
        elif surrogate_name == "CIFAR10":
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            self.dataset = torchvision.datasets.CIFAR10(
                root="./data",
                train=train_split,
                download=True,
                transform=transform,
            )
        elif surrogate_name == "GTSRB":
            transform = transforms.Compose([
                transforms.Resize((32, 32)),  # GTSRB images have variable sizes, resize to 32x32
                transforms.ToTensor(),
            ])
            self.dataset = torchvision.datasets.GTSRB(
                root="./data",
                split="train",
                download=True,
                transform=transform,
            )
        else:
            raise ValueError(f"Unknown surrogate dataset: {surrogate_name}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.dataset[idx]


def get_test_dataloader(
    name: str,
    batch_size: int = 128,
    num_workers: int = 2,
) -> DataLoader:
    """Get test dataloader for victim dataset."""
    if name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transform,
        )
    elif name == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transform,
        )
    elif name == "FashionMNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.FashionMNIST(
            root="./data",
            train=False,
            download=True,
            transform=transform,
        )
    elif name == "GTSRB":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.GTSRB(
            root="./data",
            split="test",
            download=True,
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def create_dataloader(
    config: Dict[str, Any],
    batch_size: int = 128,
    shuffle: bool = True,
) -> DataLoader:
    """Create dataloader based on config."""
    name = config.get("name", "CIFAR10")
    data_mode = config.get("data_mode", "surrogate")
    
    if data_mode == "surrogate":
        dataset = SurrogateDataset(
            surrogate_name=config.get("surrogate_name", "SVHN"),
            train_split=config.get("train_split", True),
        )
    elif data_mode == "seed":
        dataset = SeedDataset(
            name=name,
            seed_size=config.get("seed_size", 100),
            train_split=config.get("train_split", True),
        )
    else:
        raise ValueError(f"Unknown data_mode: {data_mode}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
    )