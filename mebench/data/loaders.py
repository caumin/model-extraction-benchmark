"""Data loaders for seed, surrogate, and data-free modes."""

from typing import Dict, Any, Tuple, Optional
import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms


def _get_default_num_workers(*, default: int = 0) -> int:
    raw = os.environ.get("MEBENCH_NUM_WORKERS")
    if raw is None:
        return int(default)
    raw = str(raw).strip()
    if raw == "":
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


class SeedDataset(Dataset):
    """In-domain seed dataset (default 100 images)."""

    def __init__(
        self,
        name: str,
        seed_size: int = 100,
        train_split: bool = True,
        seed_split: str = "balanced",
        *,
        output_size: Optional[Tuple[int, int]] = None,
        output_channels: Optional[int] = None,
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
            tf = []
            if output_size is not None:
                tf.append(transforms.Resize(output_size))
            tf.append(transforms.ToTensor())
            transform = transforms.Compose(tf)
            full_dataset = torchvision.datasets.CIFAR10(
                root="./data",
                train=train_split,
                download=True,
                transform=transform,
            )
        elif name == "MNIST":
            tf = []
            if output_size is not None:
                tf.append(transforms.Resize(output_size))
            if output_channels is not None and int(output_channels) == 3:
                tf.append(transforms.Grayscale(num_output_channels=3))
            tf.append(transforms.ToTensor())
            transform = transforms.Compose(tf)
            full_dataset = torchvision.datasets.MNIST(
                root="./data",
                train=train_split,
                download=True,
                transform=transform,
            )
        elif name == "EMNIST":
            tf = []
            if output_size is not None:
                tf.append(transforms.Resize(output_size))
            if output_channels is not None and int(output_channels) == 3:
                tf.append(transforms.Grayscale(num_output_channels=3))
            tf.append(transforms.ToTensor())
            transform = transforms.Compose(tf)
            full_dataset = torchvision.datasets.EMNIST(
                root="./data",
                split="balanced",
                train=train_split,
                download=True,
                transform=transform,
            )
        elif name == "FashionMNIST":
            tf = []
            if output_size is not None:
                tf.append(transforms.Resize(output_size))
            if output_channels is not None and int(output_channels) == 3:
                tf.append(transforms.Grayscale(num_output_channels=3))
            tf.append(transforms.ToTensor())
            transform = transforms.Compose(tf)
            full_dataset = torchvision.datasets.FashionMNIST(
                root="./data",
                train=train_split,
                download=True,
                transform=transform,
            )
        elif name == "GTSRB":
            size = output_size if output_size is not None else (32, 32)
            transform = transforms.Compose([
                transforms.Resize(size),
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
        *,
        root: str = "./data",
        resize: Optional[Tuple[int, int]] = None,
        max_samples: int = 0,
        subset_seed: int = 42,
        output_channels: Optional[int] = None,
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
            tf = []
            if resize is not None:
                tf.append(transforms.Resize(resize))
            tf.append(transforms.ToTensor())
            transform = transforms.Compose(tf)
            self.dataset = torchvision.datasets.SVHN(
                root="./data",
                split="train" if train_split else "test",
                download=True,
                transform=transform,
            )
        elif surrogate_name == "EMNIST":
            tf = []
            if resize is not None:
                tf.append(transforms.Resize(resize))
            if output_channels is not None and int(output_channels) == 3:
                tf.append(transforms.Grayscale(num_output_channels=3))
            tf.append(transforms.ToTensor())
            transform = transforms.Compose(tf)
            self.dataset = torchvision.datasets.EMNIST(
                root="./data",
                split="balanced",
                train=train_split,
                download=True,
                transform=transform,
            )
        elif surrogate_name == "FashionMNIST":
            tf = []
            if resize is not None:
                tf.append(transforms.Resize(resize))
            if output_channels is not None and int(output_channels) == 3:
                tf.append(transforms.Grayscale(num_output_channels=3))
            tf.append(transforms.ToTensor())
            transform = transforms.Compose(tf)
            self.dataset = torchvision.datasets.FashionMNIST(
                root="./data",
                train=train_split,
                download=True,
                transform=transform,
            )
        elif surrogate_name == "CIFAR10":
            tf = []
            if resize is not None:
                tf.append(transforms.Resize(resize))
            tf.append(transforms.ToTensor())
            transform = transforms.Compose(tf)
            self.dataset = torchvision.datasets.CIFAR10(
                root="./data",
                train=train_split,
                download=True,
                transform=transform,
            )
        elif surrogate_name == "GTSRB":
            size = resize if resize is not None else (32, 32)
            transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
            ])
            self.dataset = torchvision.datasets.GTSRB(
                root="./data",
                split="train",
                download=True,
                transform=transform,
            )
        elif surrogate_name in {"IMAGENET", "ImageNet", "imagenet", "ILSVRC", "ILSVRC2012"}:
            # ImageNet/ILSVRC2012 is expected to be provided locally in ImageFolder format.
            # Directory layout:
            #   <root>/train/<class>/*
            #   <root>/val/<class>/*
            base = os.environ.get("MEBENCH_IMAGENET_ROOT", root)
            split_dir = os.path.join(str(base), "train" if train_split else "val")
            if not os.path.isdir(split_dir):
                raise ValueError(
                    "ImageNet surrogate requires a local ImageFolder directory. "
                    "Set dataset.surrogate_root (or env MEBENCH_IMAGENET_ROOT) to a folder containing train/ and val/. "
                    f"Missing: {split_dir}"
                )

            size = resize if resize is not None else (32, 32)
            tf = [transforms.Resize((int(size[0]), int(size[1])))]
            if output_channels is not None:
                if int(output_channels) == 1:
                    # SET-A: MNIST/LeNet-5 expects grayscale 1xHxW.
                    tf.append(transforms.Grayscale(num_output_channels=1))
                elif int(output_channels) == 3:
                    # SET-B: keep RGB for CIFAR10/ResNet18.
                    pass
                else:
                    raise ValueError(f"Unsupported output_channels for ImageNet surrogate: {output_channels}")
            tf.append(transforms.ToTensor())
            transform = transforms.Compose(tf)
            full_dataset = torchvision.datasets.ImageFolder(root=split_dir, transform=transform)

            max_n = int(max_samples)
            if max_n > 0 and len(full_dataset) > max_n:
                # Deterministic subsample (default: 100k) to cap pool size.
                g = torch.Generator().manual_seed(int(subset_seed))
                indices = torch.randperm(len(full_dataset), generator=g)[:max_n].tolist()
                self.dataset = Subset(full_dataset, indices)
            else:
                self.dataset = full_dataset
        else:
            raise ValueError(f"Unknown surrogate dataset: {surrogate_name}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.dataset[idx]


def get_test_dataloader(
    name: str,
    batch_size: int = 128,
    num_workers: Optional[int] = None,
    *,
    input_size: Optional[Tuple[int, int]] = None,
    channels: Optional[int] = None,
) -> DataLoader:
    """Get test dataloader for victim dataset."""
    if name == "CIFAR10":
        tf = []
        if input_size is not None:
            tf.append(transforms.Resize(input_size))
        tf.append(transforms.ToTensor())
        transform = transforms.Compose(tf)
        dataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transform,
        )
    elif name == "MNIST":
        tf = []
        if input_size is not None:
            tf.append(transforms.Resize(input_size))
        if channels is not None and int(channels) == 3:
            tf.append(transforms.Grayscale(num_output_channels=3))
        tf.append(transforms.ToTensor())
        transform = transforms.Compose(tf)
        dataset = torchvision.datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transform,
        )
    elif name == "FashionMNIST":
        tf = []
        if input_size is not None:
            tf.append(transforms.Resize(input_size))
        if channels is not None and int(channels) == 3:
            tf.append(transforms.Grayscale(num_output_channels=3))
        tf.append(transforms.ToTensor())
        transform = transforms.Compose(tf)
        dataset = torchvision.datasets.FashionMNIST(
            root="./data",
            train=False,
            download=True,
            transform=transform,
        )
    elif name == "GTSRB":
        size = input_size if input_size is not None else (32, 32)
        transform = transforms.Compose([
            transforms.Resize(size),
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

    resolved_workers = (
        int(num_workers) if num_workers is not None else _get_default_num_workers(default=0)
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=resolved_workers,
    )


def create_dataloader(
    config: Dict[str, Any],
    batch_size: int = 128,
    shuffle: bool = True,
) -> DataLoader:
    """Create dataloader based on config."""
    name = config.get("name", "CIFAR10")
    data_mode = config.get("data_mode", "surrogate")
    
    output_channels = config.get("channels")
    if output_channels is not None:
        output_channels = int(output_channels)

    desired_input_size = config.get("input_size")
    desired_resize: Optional[Tuple[int, int]] = None
    if isinstance(desired_input_size, (list, tuple)) and len(desired_input_size) == 2:
        desired_resize = (int(desired_input_size[0]), int(desired_input_size[1]))

    if data_mode == "surrogate":
        surrogate_root = str(config.get("surrogate_root") or "./data")
        surrogate_resize = config.get("surrogate_resize")
        resize: Optional[Tuple[int, int]]
        if desired_resize is not None:
            resize = desired_resize
        elif surrogate_resize is None:
            resize = None
        elif isinstance(surrogate_resize, (list, tuple)) and len(surrogate_resize) == 2:
            resize = (int(surrogate_resize[0]), int(surrogate_resize[1]))
        else:
            resize = (int(surrogate_resize), int(surrogate_resize))

        dataset = SurrogateDataset(
            surrogate_name=config.get("surrogate_name", "SVHN"),
            train_split=config.get("train_split", True),
            root=surrogate_root,
            resize=resize,
            max_samples=int(config.get("surrogate_max_samples", 0)),
            subset_seed=int(config.get("surrogate_subset_seed", 42)),
            output_channels=output_channels,
        )
    elif data_mode == "seed":
        dataset = SeedDataset(
            name=name,
            seed_size=config.get("seed_size", 100),
            train_split=config.get("train_split", True),
            output_size=desired_resize,
            output_channels=output_channels,
        )
    else:
        raise ValueError(f"Unknown data_mode: {data_mode}")

    if "num_workers" in config and config.get("num_workers") is not None:
        num_workers = int(config.get("num_workers"))
    else:
        num_workers = _get_default_num_workers(default=0)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
