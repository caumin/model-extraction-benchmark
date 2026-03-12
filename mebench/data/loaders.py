"""Data loaders for seed, surrogate, and data-free modes.

Surrogate datasets can apply dataset-standard normalization directly in the
pool loader path. This keeps pool-based attacks in a consistent surrogate
feature space for both query and substitute training.
"""

from typing import Dict, Any, Tuple, Optional
import os
import csv
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from PIL import Image


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
SVHN_MEAN = (0.43768206, 0.44376972, 0.47280434)
SVHN_STD = (0.19803014, 0.20101564, 0.19703615)
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
EMNIST_MEAN = (0.1736,)
EMNIST_STD = (0.3248,)
FASHION_MNIST_MEAN = (0.2860,)
FASHION_MNIST_STD = (0.3530,)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
GTSRB_MEAN = (0.3403, 0.3121, 0.3214)
GTSRB_STD = (0.2724, 0.2608, 0.2669)
BELGIUM_TSC_MEAN = GTSRB_MEAN
BELGIUM_TSC_STD = GTSRB_STD
SEWERML_MEAN = (0.523, 0.453, 0.345)
SEWERML_STD = (0.210, 0.199, 0.154)
SEWERML_LABELS = [
    "RB",
    "OB",
    "PF",
    "DE",
    "FS",
    "IS",
    "RO",
    "IN",
    "AF",
    "BE",
    "FO",
    "GR",
    "PH",
    "PB",
    "OS",
    "OP",
    "OK",
]


def _normalize_sewerml_label_mode(raw_mode: Any) -> str:
    """Normalize SewerML label mode to supported canonical value.

    Supported values:
    - "argmax": 17-class single-label via argmax over defect columns.
    - "binary": binary defect classification from Defect column.
    """

    mode = str(raw_mode if raw_mode is not None else "argmax").strip().lower().replace("-", "_")
    if mode in {"argmax", "single_label", "single", "multilabel", "multiclass"}:
        return "argmax"
    if mode in {"binary", "defect", "defect_binary", "binary_label", "is_defect"}:
        return "binary"

    raise ValueError(f"Unsupported Sewerml label mode: {raw_mode!r}")


def _parse_sewerml_labels(gt: pd.DataFrame, label_mode: str) -> list[int]:
    mode = _normalize_sewerml_label_mode(label_mode)

    if mode == "binary":
        if "Defect" not in gt.columns:
            missing = [col for col in SEWERML_LABELS if col not in gt.columns]
            if missing:
                raise ValueError(
                    "SewerML binary mode requires Defect column or all defect label columns: "
                    + ", ".join(missing)
                )

            labels_raw = gt[SEWERML_LABELS].to_numpy(dtype="float32")
            labels_tensor = torch.from_numpy(labels_raw)
            labels_tensor = torch.nan_to_num(labels_tensor, nan=0.0)
            ok_idx = SEWERML_LABELS.index("OK")
            has_non_ok_defect = torch.sum(labels_tensor, dim=1) - labels_tensor[:, ok_idx]
            y = (has_non_ok_defect > 0).long()
            return y.tolist()

        defect_raw = gt["Defect"].astype(str).str.strip()
        defect_clean = pd.to_numeric(defect_raw, errors="coerce")
        if defect_clean.isna().all():
            raise ValueError("SewerML Defect column is empty or non-numeric")

        defect_tensor = torch.from_numpy(defect_clean.to_numpy(dtype="float32"))
        defect_tensor = torch.nan_to_num(defect_tensor, nan=0.0)
        y = defect_tensor.long()
        y = (y != 0).long()
        return y.tolist()

    missing = [col for col in SEWERML_LABELS if col not in gt.columns]
    if missing:
        raise ValueError(f"SewerML annotations missing required columns: {', '.join(missing)}")

    labels_raw = gt[SEWERML_LABELS].to_numpy(dtype="float32")
    labels_tensor = torch.from_numpy(labels_raw)
    labels_tensor = torch.nan_to_num(labels_tensor, nan=0.0)
    labels = torch.argmax(labels_tensor, dim=1)
    return labels.tolist()


def _rgb_to_grayscale_stats(
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> Tuple[Tuple[float], Tuple[float]]:
    # Rec.709-style luminance weights.
    w_r, w_g, w_b = 0.2989, 0.5870, 0.1140
    mean_gray = (w_r * float(mean[0])) + (w_g * float(mean[1])) + (w_b * float(mean[2]))
    std_gray = (
        ((w_r * float(std[0])) ** 2)
        + ((w_g * float(std[1])) ** 2)
        + ((w_b * float(std[2])) ** 2)
    ) ** 0.5
    return (float(mean_gray),), (float(std_gray),)


def _adapt_stats_channels(
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
    channels: Optional[int],
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    if channels is None:
        return mean, std

    ch = int(channels)
    if ch == len(mean):
        return mean, std
    if ch == 1 and len(mean) == 3:
        mean_g, std_g = _rgb_to_grayscale_stats(
            (float(mean[0]), float(mean[1]), float(mean[2])),
            (float(std[0]), float(std[1]), float(std[2])),
        )
        return mean_g, std_g
    if ch == 3 and len(mean) == 1:
        return (float(mean[0]),) * 3, (float(std[0]),) * 3

    raise ValueError(
        "Unsupported surrogate normalization channel adaptation: "
        f"stats_channels={len(mean)}, target_channels={ch}"
    )


def _parse_norm_sequence(name: str, value: Any) -> Tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) <= 0:
        raise ValueError(f"{name} must be a non-empty list/tuple, got {value!r}")
    try:
        parsed = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values, got {value!r}") from exc
    return parsed


def get_surrogate_standard_normalization(
    surrogate_name: str,
    channels: Optional[int] = None,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    key = str(surrogate_name).strip().lower()
    mapping: Dict[str, Tuple[Tuple[float, ...], Tuple[float, ...]]] = {
        "svhn": (SVHN_MEAN, SVHN_STD),
        "mnist": (MNIST_MEAN, MNIST_STD),
        "emnist": (EMNIST_MEAN, EMNIST_STD),
        "fashionmnist": (FASHION_MNIST_MEAN, FASHION_MNIST_STD),
        "cifar10": (CIFAR10_MEAN, CIFAR10_STD),
        "cifar100": (CIFAR100_MEAN, CIFAR100_STD),
        "gtsrb": (GTSRB_MEAN, GTSRB_STD),
        "belgiumtsc": (BELGIUM_TSC_MEAN, BELGIUM_TSC_STD),
        "sewerml": (SEWERML_MEAN, SEWERML_STD),
        "imagenet": (IMAGENET_MEAN, IMAGENET_STD),
        "ilsvrc": (IMAGENET_MEAN, IMAGENET_STD),
        "ilsvrc2012": (IMAGENET_MEAN, IMAGENET_STD),
    }
    if key not in mapping:
        raise ValueError(
            f"Unsupported surrogate dataset for standard normalization: {surrogate_name!r}"
        )

    mean, std = mapping[key]
    return _adapt_stats_channels(mean, std, channels)


def _resolve_surrogate_normalization(
    *,
    surrogate_name: str,
    output_channels: Optional[int],
    surrogate_normalization: Any,
    surrogate_norm_mean: Any,
    surrogate_norm_std: Any,
) -> Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
    mode = str(surrogate_normalization if surrogate_normalization is not None else "standard")
    mode = mode.strip().lower()

    if mode in {"none", "off", "identity", "raw"}:
        return None

    if mode in {"standard", "dataset", "surrogate_standard"}:
        return get_surrogate_standard_normalization(surrogate_name, channels=output_channels)

    if mode == "custom":
        mean = _parse_norm_sequence("surrogate_norm_mean", surrogate_norm_mean)
        std = _parse_norm_sequence("surrogate_norm_std", surrogate_norm_std)
        if len(mean) != len(std):
            raise ValueError(
                "surrogate_norm_mean and surrogate_norm_std must have same length, "
                f"got mean={len(mean)}, std={len(std)}"
            )
        return _adapt_stats_channels(mean, std, output_channels)

    raise ValueError(
        "dataset.surrogate_normalization must be one of "
        "['standard', 'none', 'custom'], "
        f"got {surrogate_normalization!r}"
    )


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


def _normalize_root_path(raw_root: str) -> str:
    """Normalize filesystem roots from config/env across shells.

    Accepts common malformed forms like `C~/path` and `C:~/path` and treats
    them as `~/path`, then applies env/user expansion.
    """
    root = str(raw_root or "").strip()
    if root == "":
        root = "./data"

    if len(root) >= 3 and root[0].isalpha() and root[1] == ":" and root[2] == "~":
        # e.g., C:~/workspace/... -> ~/workspace/...
        root = root[2:]
    elif len(root) >= 2 and root[0].isalpha() and root[1] == "~":
        # e.g., C~/workspace/... -> ~/workspace/...
        root = root[1:]

    root = os.path.expandvars(root)
    root = os.path.expanduser(root)
    return root


class BelgiumTSCDataset(Dataset):
    def __init__(self, root: str = "./data", train: bool = True, transform=None) -> None:
        self.root = Path(str(root))
        self.transform = transform
        self.subdir = "Training" if bool(train) else "Testing"
        self.base_dir = self.root / "BelgiumTSC" / self.subdir
        csv_name = "train_data.csv" if bool(train) else "test_data.csv"
        csv_path = self.base_dir / csv_name

        self.records: list[tuple[Path, int]] = []
        if csv_path.exists():
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 2:
                        continue
                    rel_path = str(row[0]).strip()
                    label_raw = str(row[1]).strip()
                    if rel_path == "":
                        continue
                    try:
                        label = int(label_raw)
                    except ValueError:
                        continue
                    self.records.append((self.base_dir / rel_path, label))
        else:
            if not self.base_dir.exists():
                raise FileNotFoundError(
                    f"BelgiumTSC folder not found: {self.base_dir}. "
                    "Expected official layout under <root>/BelgiumTSC/{Training|Testing}/."
                )
            image_exts = {".png", ".jpg", ".jpeg", ".ppm", ".bmp"}
            for class_dir in sorted(self.base_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                try:
                    label = int(class_dir.name)
                except ValueError:
                    continue
                for image_path in sorted(class_dir.iterdir()):
                    if image_path.is_file() and image_path.suffix.lower() in image_exts:
                        self.records.append((image_path, label))

        if not self.records:
            raise ValueError(
                f"BelgiumTSC has no usable samples under {self.base_dir}. "
                f"Checked CSV={csv_path.exists()} and class folders."
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.records[int(idx)]
        if not img_path.exists():
            raise FileNotFoundError(f"BelgiumTSC image missing: {img_path}")
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(label)


class GTSRBCSVDataset(Dataset):
    def __init__(self, root: str = "./data", train: bool = True, transform=None) -> None:
        self.root = Path(str(root))
        self.transform = transform
        self.subdir = "trainingset" if bool(train) else "testset"
        self.base_dir = self.root / "GTSRB" / self.subdir
        csv_name = "training.csv" if bool(train) else "test.csv"
        csv_path = self.base_dir / csv_name

        self.records: list[tuple[Path, int]] = []
        if not csv_path.exists():
            raise FileNotFoundError(f"GTSRB CSV not found: {csv_path}")

        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                rel_path = str(row[0]).strip()
                label_raw = str(row[1]).strip()
                if rel_path == "":
                    continue
                try:
                    label = int(label_raw)
                except ValueError:
                    continue
                self.records.append((self.base_dir / rel_path, label))

        if not self.records:
            raise ValueError(f"GTSRB CSV dataset has no records: {csv_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.records[int(idx)]
        if not img_path.exists():
            raise FileNotFoundError(f"GTSRB image missing: {img_path}")
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(label)


def _resolve_sewerml_roots(
    ann_root: Optional[str] = None,
    img_root: Optional[str] = None,
) -> Tuple[Path, Path]:
    ann = str(ann_root or os.getenv("SEWERML_ANN_ROOT") or "./data/SewerML/annotations")
    img = str(img_root or os.getenv("SEWERML_DATA_ROOT") or "./data/SewerML/Data")
    ann_path = Path(ann)
    img_path = Path(img)
    return ann_path, img_path


def _resolve_sewerml_csv_path(ann_root: Path, split: str) -> Path:
    split_name = str(split).strip()
    candidates = [
        ann_root / f"{split_name}13.csv",
        ann_root / f"SewerML_{split_name}.csv",
    ]
    if split_name.lower() == "valid":
        candidates.append(ann_root / "SewerML_Val.csv")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "SewerML annotation CSV not found. Checked: "
        + ", ".join(str(c) for c in candidates)
    )


class SewerMLDataset(Dataset):
    """Sewer-ML dataset adapter based on official annotation format.

    The default behavior is single-label for ground-truth accuracy metrics.
    Configure `label_mode="argmax"` (default) to derive a class index by
    argmax over the 17 defect flags, or `label_mode="binary"` to use the
    `Defect` column as a binary target.
    """

    def __init__(
        self,
        ann_root: str,
        img_root: str,
        split: str,
        transform: Optional[transforms.Compose] = None,
        *,
        label_mode: str = "argmax",
    ) -> None:
        self.ann_root = Path(ann_root)
        self.img_root = Path(img_root)
        self.split = str(split)
        self.transform = transform
        mode = _normalize_sewerml_label_mode(label_mode)
        self.classes = ["no_defect", "defect"] if mode == "binary" else list(SEWERML_LABELS)

        csv_path = _resolve_sewerml_csv_path(self.ann_root, self.split)
        try:
            gt = pd.read_csv(csv_path, sep=",", encoding="utf-8")
        except ValueError as exc:
            raise ValueError(
                f"Invalid SewerML annotation format for label_mode={mode}: {csv_path}"
            ) from exc

        if "Filename" not in gt.columns:
            raise ValueError(
                f"Invalid SewerML annotation format for label_mode={mode}: {csv_path}"
            )

        self.img_paths = [str(v) for v in gt["Filename"].values]
        self.targets = _parse_sewerml_labels(gt, mode)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        rel_path = str(self.img_paths[int(index)])
        img_path = self.img_root / rel_path
        if not img_path.exists():
            raise FileNotFoundError(f"SewerML image missing: {img_path}")
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.targets[int(index)])


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
        emnist_split: str = "balanced",
        sewerml_ann_root: Optional[str] = None,
        sewerml_data_root: Optional[str] = None,
        sewerml_split: Optional[str] = None,
        sewerml_label_mode: str = "argmax",
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
                split=str(emnist_split),
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
        elif name == "BelgiumTSC":
            size = output_size if output_size is not None else (32, 32)
            tf = [transforms.Resize(size)]
            if output_channels is not None and int(output_channels) == 1:
                tf.append(transforms.Grayscale(num_output_channels=1))
            tf.append(transforms.ToTensor())
            transform = transforms.Compose(tf)
            full_dataset = BelgiumTSCDataset(
                root="./data",
                train=bool(train_split),
                transform=transform,
            )
        elif name == "SewerML":
            size = output_size if output_size is not None else (224, 224)
            tf = [transforms.Resize(size), transforms.ToTensor(), transforms.Normalize(SEWERML_MEAN, SEWERML_STD)]
            transform = transforms.Compose(tf)
            ann_root, data_root = _resolve_sewerml_roots(
                ann_root=sewerml_ann_root,
                img_root=sewerml_data_root,
            )
            split = str(sewerml_split or ("Train" if train_split else "Test"))
            full_dataset = SewerMLDataset(
                ann_root=str(ann_root),
                img_root=str(data_root),
                split=split,
                label_mode=sewerml_label_mode,
                transform=transform,
            )
        else:
            raise ValueError(f"Unknown dataset: {name}")

        # Deterministic subset for seed (balanced or by_class)
        if seed_split == "balanced":
            # Handle GTSRB which has different structure
            if name in {"GTSRB", "BelgiumTSC"}:
                num_classes = 43 if name == "GTSRB" else 62
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
        class_subset_size: int = 0,
        class_subset_seed: int = 42,
        class_subset_names: Optional[list[str]] = None,
        emnist_split: str = "balanced",
        surrogate_color_jitter: bool = False,
        surrogate_normalization: Any = "standard",
        surrogate_norm_mean: Any = None,
        surrogate_norm_std: Any = None,
    ):
        """Initialize surrogate dataset.

        Args:
            surrogate_name: Surrogate dataset name (SVHN, EMNIST, etc.)
            train_split: Whether to use train or test split
        """
        self.surrogate_name = surrogate_name
        self.train_split = train_split
        norm_stats = _resolve_surrogate_normalization(
            surrogate_name=surrogate_name,
            output_channels=output_channels,
            surrogate_normalization=surrogate_normalization,
            surrogate_norm_mean=surrogate_norm_mean,
            surrogate_norm_std=surrogate_norm_std,
        )

        # Load surrogate dataset
        if surrogate_name == "SVHN":
            tf = []
            if resize is not None:
                tf.append(transforms.Resize(resize))
            tf.append(transforms.ToTensor())
            if norm_stats is not None:
                tf.append(transforms.Normalize(*norm_stats))
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
            if norm_stats is not None:
                tf.append(transforms.Normalize(*norm_stats))
            transform = transforms.Compose(tf)
            self.dataset = torchvision.datasets.EMNIST(
                root="./data",
                split=str(emnist_split),
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
            if norm_stats is not None:
                tf.append(transforms.Normalize(*norm_stats))
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
            if norm_stats is not None:
                tf.append(transforms.Normalize(*norm_stats))
            transform = transforms.Compose(tf)
            self.dataset = torchvision.datasets.CIFAR10(
                root="./data",
                train=train_split,
                download=True,
                transform=transform,
            )
        elif surrogate_name == "CIFAR100":
            tf = []
            if resize is not None:
                tf.append(transforms.Resize(resize))
            tf.append(transforms.ToTensor())
            if norm_stats is not None:
                tf.append(transforms.Normalize(*norm_stats))
            transform = transforms.Compose(tf)
            full_dataset = torchvision.datasets.CIFAR100(
                root="./data",
                train=train_split,
                download=True,
                transform=transform,
            )

            ds: Dataset = full_dataset
            subset_k = int(class_subset_size)
            chosen_set: Optional[set[int]] = None

            if class_subset_names:
                class_to_idx = full_dataset.class_to_idx
                requested = [str(name) for name in class_subset_names]
                missing = [name for name in requested if name not in class_to_idx]
                if missing:
                    raise ValueError(
                        "Unknown CIFAR100 class names in surrogate_class_subset_names: "
                        + ", ".join(sorted(missing))
                    )
                chosen_set = set(int(class_to_idx[name]) for name in requested)
            elif subset_k > 0:
                if subset_k > 100:
                    raise ValueError(f"CIFAR100 class_subset_size must be <= 100, got {subset_k}")
                g = torch.Generator().manual_seed(int(class_subset_seed))
                chosen_classes = torch.randperm(100, generator=g)[:subset_k].tolist()
                chosen_set = set(int(c) for c in chosen_classes)

            if chosen_set is not None:
                targets = full_dataset.targets
                keep_indices = [i for i, t in enumerate(targets) if int(t) in chosen_set]
                ds = Subset(full_dataset, keep_indices)

            max_n = int(max_samples)
            if max_n > 0 and len(ds) > max_n:
                g = torch.Generator().manual_seed(int(subset_seed))
                indices = torch.randperm(len(ds), generator=g)[:max_n].tolist()
                ds = Subset(ds, indices)
            self.dataset = ds
        elif surrogate_name == "GTSRB":
            size = resize if resize is not None else (32, 32)
            tf = [transforms.Resize(size)]
            if bool(train_split) and bool(surrogate_color_jitter):
                tf.append(
                    transforms.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.0,
                    )
                )
            tf.append(transforms.ToTensor())
            if norm_stats is not None:
                tf.append(transforms.Normalize(*norm_stats))
            transform = transforms.Compose(tf)
            csv_root = Path(str(root)) / "GTSRB" / "trainingset" / "training.csv"
            if csv_root.exists():
                self.dataset = GTSRBCSVDataset(
                    root=root,
                    train=bool(train_split),
                    transform=transform,
                )
            else:
                self.dataset = torchvision.datasets.GTSRB(
                    root="./data",
                    split="train" if bool(train_split) else "test",
                    download=True,
                    transform=transform,
                )
        elif surrogate_name == "BelgiumTSC":
            size = resize if resize is not None else (32, 32)
            tf = [transforms.Resize(size)]
            if output_channels is not None and int(output_channels) == 1:
                tf.append(transforms.Grayscale(num_output_channels=1))
            tf.append(transforms.ToTensor())
            if norm_stats is not None:
                tf.append(transforms.Normalize(*norm_stats))
            transform = transforms.Compose(tf)
            self.dataset = BelgiumTSCDataset(
                root=root,
                train=bool(train_split),
                transform=transform,
            )
        elif surrogate_name in {"IMAGENET", "ImageNet", "imagenet", "ILSVRC", "ILSVRC2012"}:
            # ImageNet/ILSVRC2012 is expected to be provided locally in ImageFolder format.
            # Directory layout:
            #   <root>/train/<class>/*
            #   <root>/val/<class>/*
            base = _normalize_root_path(str(root))
            split_dir = os.path.join(str(base), "train" if train_split else "val")
            if not os.path.isdir(split_dir):
                raise ValueError(
                    "ImageNet surrogate requires a local ImageFolder directory. "
                    "Set dataset.surrogate_root to a folder containing train/ and val/. "
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
            if norm_stats is not None:
                tf.append(transforms.Normalize(*norm_stats))
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
    sewerml_label_mode: str = "argmax",
    sewerml_ann_root: Optional[str] = None,
    sewerml_data_root: Optional[str] = None,
    sewerml_eval_split: Optional[str] = None,
) -> DataLoader:
    """Get test dataloader for victim dataset."""
    if name == "CIFAR10":
        tf = []
        if input_size is not None:
            tf.append(transforms.Resize(input_size))
        tf.append(transforms.ToTensor())
        tf.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))
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
        if channels is not None and int(channels) == 3:
            tf.append(transforms.Normalize((MNIST_MEAN[0],) * 3, (MNIST_STD[0],) * 3))
        else:
            tf.append(transforms.Normalize(MNIST_MEAN, MNIST_STD))
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
        if channels is not None and int(channels) == 3:
            tf.append(transforms.Normalize((MNIST_MEAN[0],) * 3, (MNIST_STD[0],) * 3))
        else:
            tf.append(transforms.Normalize(MNIST_MEAN, MNIST_STD))
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
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        csv_test = Path("./data") / "GTSRB" / "testset" / "test.csv"
        if csv_test.exists():
            dataset = GTSRBCSVDataset(
                root="./data",
                train=False,
                transform=transform,
            )
        else:
            dataset = torchvision.datasets.GTSRB(
                root="./data",
                split="test",
                download=True,
                transform=transform,
            )
    elif name == "BelgiumTSC":
        size = input_size if input_size is not None else (32, 32)
        tf = [transforms.Resize(size)]
        if channels is not None and int(channels) == 1:
            tf.append(transforms.Grayscale(num_output_channels=1))
        tf.append(transforms.ToTensor())
        if channels is not None and int(channels) == 1:
            tf.append(transforms.Normalize(MNIST_MEAN, MNIST_STD))
        else:
            tf.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))
        transform = transforms.Compose(tf)
        dataset = BelgiumTSCDataset(
            root="./data",
            train=False,
            transform=transform,
        )
    elif name == "SewerML":
        size = input_size if input_size is not None else (224, 224)
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(SEWERML_MEAN, SEWERML_STD),
        ])
        ann_root, data_root = _resolve_sewerml_roots(
            ann_root=sewerml_ann_root,
            img_root=sewerml_data_root,
        )
        split = str(sewerml_eval_split or "Valid")
        dataset = SewerMLDataset(
            ann_root=str(ann_root),
            img_root=str(data_root),
            split=split,
            label_mode=sewerml_label_mode,
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
    """Create dataloader based on config.

    Surrogate mode applies `dataset.surrogate_normalization` (default:
    `standard`) unless explicitly disabled.
    """
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
        surrogate_root = _normalize_root_path(str(config.get("surrogate_root") or "./data"))
        surrogate_normalization = config.get("surrogate_normalization", "standard")
        surrogate_norm_mean = config.get("surrogate_norm_mean")
        surrogate_norm_std = config.get("surrogate_norm_std")
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
            class_subset_size=int(config.get("surrogate_class_subset_size", 0)),
            class_subset_seed=int(config.get("surrogate_class_subset_seed", 42)),
            class_subset_names=config.get("surrogate_class_subset_names"),
            emnist_split=str(config.get("surrogate_split", config.get("emnist_split", "balanced"))),
            surrogate_color_jitter=bool(config.get("surrogate_color_jitter", False)),
            surrogate_normalization=surrogate_normalization,
            surrogate_norm_mean=surrogate_norm_mean,
            surrogate_norm_std=surrogate_norm_std,
        )
    elif data_mode == "seed":
        dataset = SeedDataset(
            name=name,
            seed_size=config.get("seed_size", 100),
            train_split=config.get("train_split", True),
            seed_split=str(config.get("seed_split", "balanced")),
            output_size=desired_resize,
            output_channels=output_channels,
            emnist_split=str(config.get("emnist_split", "balanced")),
            sewerml_ann_root=config.get("sewerml_ann_root"),
            sewerml_data_root=config.get("sewerml_data_root"),
            sewerml_split=config.get("sewerml_split"),
            sewerml_label_mode=config.get("sewerml_label_mode", "argmax"),
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
