"""Train and export victim checkpoints for the benchmark matrix.

The benchmark expects pretrained victim weights at:
  runs/victims/{victim_id}_seed0.pt

Victim inference in `mebench/oracles/oracle.py` assumes inputs are in [0, 1]
with no additional mean/std normalization, so this script trains victims with
ToTensor() scaling only.
"""

from __future__ import annotations

import argparse
import functools
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Add project root to path to allow importing mebench when run as a script.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from mebench.core.seed import set_seed
from mebench.models.substitute_factory import create_substitute
from mebench.utils.binary import binary_bce_loss, binary_hard_labels_from_logits
from mebench.utils.scaling import normalize_input_scale
from mebench.data.loaders import BelgiumTSCDataset


@dataclass(frozen=True)
class VictimTrainRecipe:
    dataset: str
    arch: str
    epochs: int
    batch_size: int
    optimizer: str
    lr: float
    momentum: float
    weight_decay: float
    scheduler: str
    width_mult: int = 1
    label_smoothing: float = 0.0
    dropout_prob: float = 0.0


def _default_device(user_device: Optional[str]) -> str:
    if user_device is not None:
        return str(user_device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _infer_victim_id(dataset_name: str, arch: str) -> str:
    dataset_name = str(dataset_name).lower()
    return f"{dataset_name}_{str(arch)}"


def _backup_if_exists(path: Path) -> None:
    if not path.exists():
        return

    backup = path.with_suffix(path.suffix + ".bak")
    idx = 1
    while backup.exists():
        backup = path.with_suffix(path.suffix + f".bak{idx}")
        idx += 1
    path.rename(backup)
    print(f"[INFO] Existing checkpoint moved to: {backup}")


def _seed_dataloader_worker(worker_id: int, *, base_seed: int) -> None:
    """Seed DataLoader workers deterministically (Windows-safe).

    Must be top-level to be picklable under multiprocessing 'spawn' on Windows.
    """

    seed = int(base_seed) + int(worker_id)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _worker_init_fn(base_seed: int):
    # functools.partial of a top-level function is picklable on Windows.
    return functools.partial(_seed_dataloader_worker, base_seed=int(base_seed))


def _build_transforms(dataset_name: str, arch: str, train: bool) -> transforms.Compose:
    dataset_name = str(dataset_name)
    arch = str(arch)

    # Contract alignment: keep raw pixel scale in [0, 1].
    if dataset_name == "CIFAR10":
        if train:
            return transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        return transforms.Compose([transforms.ToTensor()])

    if dataset_name == "MNIST":
        if arch in {"lenet", "half_lenet", "half-lenet"}:
            return transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
        return transforms.Compose([transforms.ToTensor()])

    if dataset_name == "BelgiumTSC":
        return transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

    raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported: MNIST, CIFAR10, BelgiumTSC")


def _load_dataset(dataset_name: str, arch: str, train: bool, data_root: Path) -> torch.utils.data.Dataset:
    dataset_name = str(dataset_name)
    transform = _build_transforms(dataset_name, arch=arch, train=train)
    if dataset_name == "MNIST":
        return datasets.MNIST(
            root=str(data_root),
            train=train,
            download=True,
            transform=transform,
        )
    if dataset_name == "CIFAR10":
        return datasets.CIFAR10(
            root=str(data_root),
            train=train,
            download=True,
            transform=transform,
        )
    if dataset_name == "BelgiumTSC":
        return BelgiumTSCDataset(
            root=str(data_root),
            train=bool(train),
            transform=transform,
        )
    raise ValueError(f"Unsupported dataset '{dataset_name}'.")


def _infer_dataset_info(dataset_name: str, arch: str) -> Tuple[int, int]:
    dataset_name = str(dataset_name)
    if dataset_name == "MNIST":
        return 1, 10
    if dataset_name == "CIFAR10":
        return 3, 10
    if dataset_name == "BelgiumTSC":
        return 3, 62
    raise ValueError(f"Unsupported dataset '{dataset_name}'.")


def _build_optimizer(
    name: str,
    params,
    lr: float,
    momentum: float,
    weight_decay: float,
) -> optim.Optimizer:
    name = str(name).lower()
    if name == "adam":
        return optim.Adam(params, lr=float(lr), weight_decay=float(weight_decay))
    if name == "sgd":
        return optim.SGD(
            params,
            lr=float(lr),
            momentum=float(momentum),
            weight_decay=float(weight_decay),
        )
    raise ValueError(f"Unsupported optimizer '{name}'. Supported: adam, sgd")


def _build_scheduler(name: str, optimizer: optim.Optimizer, epochs: int):
    name = str(name).lower()
    if name in {"none", ""}:
        return None
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs))
    if name == "multistep":
        # Reasonable defaults for CIFAR-10 style training.
        milestones = [int(epochs * 0.5), int(epochs * 0.75)]
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    raise ValueError(f"Unsupported scheduler '{name}'. Supported: none, cosine, multistep")


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return float(correct) / float(targets.size(0))


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    show_progress: bool = False,
    progress_desc: str = "eval",
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    iterator = tqdm(
        loader,
        desc=str(progress_desc),
        leave=False,
        dynamic_ncols=True,
        disable=not bool(show_progress),
    )
    for x, y in iterator:
        x = x.to(device)
        y = y.to(device)
        x = normalize_input_scale(x, "unit")
        logits = model(x)
        if logits.dim() == 1 or (logits.dim() == 2 and logits.size(1) == 1):
            loss = binary_bce_loss(logits, y)
            preds = binary_hard_labels_from_logits(logits)
        else:
            loss = nn.CrossEntropyLoss()(logits, y)
            preds = torch.argmax(logits, dim=1)
        total_loss += float(loss.item()) * int(y.size(0))
        total_correct += int((preds == y).sum().item())
        total += int(y.size(0))

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, total_correct / total


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    label_smoothing: float = 0.0,
    show_progress: bool = True,
    progress_desc: str = "train",
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    iterator = tqdm(
        loader,
        desc=str(progress_desc),
        leave=False,
        dynamic_ncols=True,
        disable=not bool(show_progress),
    )
    for x, y in iterator:
        x = x.to(device)
        y = y.to(device)
        x = normalize_input_scale(x, "unit")

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        if logits.dim() == 1 or (logits.dim() == 2 and logits.size(1) == 1):
            loss = binary_bce_loss(logits, y)
            preds = binary_hard_labels_from_logits(logits)
        else:
            loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)(logits, y)
            preds = torch.argmax(logits, dim=1)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * int(y.size(0))
        total_correct += int((preds == y).sum().item())
        total += int(y.size(0))
        if bool(show_progress):
            iterator.set_postfix(
                loss=f"{float(loss.item()):.4f}",
                acc=f"{(float(total_correct) / float(max(1, total))) * 100.0:.2f}%",
            )

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, total_correct / total


def _default_recipe(dataset: str, arch: str) -> VictimTrainRecipe:
    dataset = str(dataset)
    arch = str(arch)

    if dataset == "MNIST" and arch in {"lenet_mnist", "lenet"}:
        return VictimTrainRecipe(
            dataset=dataset,
            arch=arch,
            epochs=20,
            batch_size=128,
            optimizer="adam",
            lr=1e-3,
            momentum=0.9,
            weight_decay=0.0,
            scheduler="none",
        )

    if dataset == "MNIST" and arch == "classifier":
        return VictimTrainRecipe(
            dataset=dataset,
            arch=arch,
            epochs=30,
            batch_size=128,
            optimizer="adam",
            lr=1e-3,
            momentum=0.9,
            weight_decay=0.0,
            scheduler="none",
            dropout_prob=0.2,
        )

    if dataset == "CIFAR10" and arch == "resnet18":
        return VictimTrainRecipe(
            dataset=dataset,
            arch=arch,
            epochs=200,
            batch_size=256,
            optimizer="sgd",
            lr=0.1,
            momentum=0.9,
            weight_decay=1e-4,
            scheduler="cosine",
            width_mult=1,
            label_smoothing=0.1,
        )

    if dataset == "CIFAR10" and arch == "resnet34":
        return VictimTrainRecipe(
            dataset=dataset,
            arch=arch,
            epochs=200,
            batch_size=256,
            optimizer="sgd",
            lr=0.1,
            momentum=0.9,
            weight_decay=1e-4,
            scheduler="cosine",
            width_mult=1,
            label_smoothing=0.1,
        )

    if dataset == "BelgiumTSC" and arch in {"alexnet", "alexnet_half", "half_alexnet", "half-alexnet"}:
        return VictimTrainRecipe(
            dataset=dataset,
            arch=arch,
            epochs=20,
            batch_size=1024,
            optimizer="adam",
            lr=1e-3,
            momentum=0.9,
            weight_decay=0.0,
            scheduler="none",
            width_mult=1,
            label_smoothing=0.0,
            dropout_prob=0.5,
        )

    if dataset == "CIFAR10" and arch == "resnet20":
        return VictimTrainRecipe(
            dataset=dataset,
            arch=arch,
            epochs=200,
            batch_size=128,
            optimizer="sgd",
            lr=0.1,
            momentum=0.9,
            weight_decay=5e-4,
            scheduler="cosine",
            width_mult=1,
            label_smoothing=0.0,
        )

    if dataset == "CIFAR10" and arch == "activethief_cnn":
        return VictimTrainRecipe(
            dataset=dataset,
            arch=arch,
            epochs=300,
            batch_size=150,
            optimizer="adam",
            lr=1e-3,
            momentum=0.9,
            weight_decay=1e-3,
            scheduler="multistep",
            width_mult=1,
            label_smoothing=0.0,
            dropout_prob=0.2,
        )

    # Generic fallback.
    c, _ = _infer_dataset_info(dataset, arch=arch)
    _ = c
    return VictimTrainRecipe(
        dataset=dataset,
        arch=arch,
        epochs=50,
        batch_size=128,
        optimizer="sgd",
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4,
        scheduler="multistep",
        width_mult=1,
        label_smoothing=0.0,
        dropout_prob=0.0,
    )


def _load_yaml_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected mapping at YAML root: {path}")
    return cfg


def _cfg_get(config: dict, *keys, default=None):
    for key in keys:
        if key in config:
            return config[key]
    return default


def train() -> None:
    parser = argparse.ArgumentParser(description="Train a victim model checkpoint for mebench")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config path for victim training",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        choices=["MNIST", "CIFAR10", "BelgiumTSC"],
        help="Victim dataset",
    )
    parser.add_argument(
        "--arch",
        type=str,
        required=False,
        choices=["lenet_mnist", "lenet", "classifier", "resnet18", "resnet20", "resnet34", "activethief_cnn", "alexnet", "alexnet_half", "half_alexnet", "half-alexnet"],
        help="Victim architecture (must match config victim.arch)",
    )
    parser.add_argument(
        "--width-mult",
        type=int,
        default=None,
        help="Override width multiplier (for resnet variants)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Training seed for victim checkpoint")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs (default: recipe default)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size (default: recipe default)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=["adam", "sgd"],
        help="Override optimizer (default: recipe default)",
    )
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--momentum", type=float, default=None, help="SGD momentum override")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay override")
    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        choices=["none", "cosine", "multistep"],
        help="Override LR scheduler (default: recipe default)",
    )
    parser.add_argument(
        "--dropout-prob",
        type=float,
        default=None,
        help="Override model dropout probability (default: recipe default)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm batch progress bars",
    )
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path override (default: runs/victims/{victim_id}_seed{seed}.pt)",
    )
    parser.add_argument(
        "--save-metadata",
        action="store_true",
        default=None,
        help="Also write runs/victims/{victim_id}_seed{seed}.json",
    )
    args = parser.parse_args()

    cfg = _load_yaml_config(args.config) if args.config is not None else {}
    train_cfg = cfg.get("train", {}) if isinstance(cfg.get("train", {}), dict) else {}
    opt_cfg = train_cfg.get("optimizer", {}) if isinstance(train_cfg.get("optimizer", {}), dict) else {}

    dataset = args.dataset or _cfg_get(cfg, "dataset")
    arch = args.arch or _cfg_get(cfg, "arch")
    if dataset is None or arch is None:
        raise ValueError("dataset and arch are required (via CLI args or YAML config)")

    seed = int(args.seed) if args.seed is not None else int(_cfg_get(cfg, "seed", default=0))
    num_workers = (
        int(args.num_workers)
        if args.num_workers is not None
        else int(_cfg_get(train_cfg, "num_workers", default=0))
    )

    save_metadata_cfg = _cfg_get(cfg, "save_metadata", default=False)
    save_metadata = bool(args.save_metadata) if args.save_metadata is not None else bool(save_metadata_cfg)

    # Benchmark defaults: MNIST uses LeNet-family or Classifier; CIFAR10 uses ResNet18/ResNet20/ActiveThief CNN.
    if dataset == "MNIST" and arch not in {"lenet_mnist", "lenet", "classifier"}:
        raise ValueError("For MNIST victim training, use --arch lenet_mnist, --arch lenet, or --arch classifier")
    if dataset == "CIFAR10" and arch not in {"resnet18", "resnet20", "resnet34", "activethief_cnn"}:
        raise ValueError(
            "For CIFAR10 victim training, use --arch resnet18, --arch resnet20, --arch resnet34, or --arch activethief_cnn"
        )
    if dataset == "BelgiumTSC" and arch not in {"alexnet", "alexnet_half", "half_alexnet", "half-alexnet"}:
        raise ValueError(
            "For BelgiumTSC victim training, use --arch alexnet, --arch alexnet_half, or --arch half_alexnet"
        )

    recipe = _default_recipe(dataset, arch)
    epochs = int(args.epochs) if args.epochs is not None else int(_cfg_get(train_cfg, "epochs", default=recipe.epochs))
    batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(_cfg_get(train_cfg, "batch_size", default=recipe.batch_size))
    )
    optimizer_name = (
        str(args.optimizer)
        if args.optimizer is not None
        else str(_cfg_get(opt_cfg, "name", default=recipe.optimizer))
    )
    lr = float(args.lr) if args.lr is not None else float(_cfg_get(opt_cfg, "lr", default=recipe.lr))
    momentum = (
        float(args.momentum)
        if args.momentum is not None
        else float(_cfg_get(opt_cfg, "momentum", default=recipe.momentum))
    )
    weight_decay = (
        float(args.weight_decay)
        if args.weight_decay is not None
        else float(_cfg_get(opt_cfg, "weight_decay", default=recipe.weight_decay))
    )
    scheduler_name = (
        str(args.scheduler)
        if args.scheduler is not None
        else str(_cfg_get(train_cfg, "scheduler", default=recipe.scheduler))
    )
    width_mult = (
        int(args.width_mult)
        if args.width_mult is not None
        else int(_cfg_get(cfg, "width_mult", default=recipe.width_mult))
    )
    label_smoothing = recipe.label_smoothing
    dropout_prob = (
        float(args.dropout_prob)
        if args.dropout_prob is not None
        else float(_cfg_get(train_cfg, "dropout_prob", default=recipe.dropout_prob))
    )
    device = _default_device(args.device)
    print(f"[INFO] Device: {device}")

    set_seed(int(seed))

    victim_id = str(_cfg_get(cfg, "victim_id", default=_infer_victim_id(dataset, arch)))
    out_dir = project_root / "runs" / "victims"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out is not None else Path(_cfg_get(cfg, "out", default=out_dir / f"{victim_id}_seed{seed}.pt"))

    # Preserve previous checkpoints (handy when iterating on recipes).
    _backup_if_exists(out_path)

    data_root = project_root / "data"
    train_dataset = _load_dataset(dataset, arch=arch, train=True, data_root=data_root)
    test_dataset = _load_dataset(dataset, arch=arch, train=False, data_root=data_root)

    loader_gen = torch.Generator()
    loader_gen.manual_seed(int(seed))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=(device.startswith("cuda")),
        generator=loader_gen,
        worker_init_fn=_worker_init_fn(int(seed)),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device.startswith("cuda")),
        worker_init_fn=_worker_init_fn(int(seed)),
    )

    input_channels, num_classes = _infer_dataset_info(dataset, arch=arch)
    model = create_substitute(
        arch=arch,
        num_classes=num_classes,
        input_channels=input_channels,
        width_mult=width_mult,
        dropout_prob=dropout_prob,
    ).to(device)

    optimizer = _build_optimizer(
        name=optimizer_name,
        params=model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = _build_scheduler(scheduler_name, optimizer, epochs)

    print(
        "[INFO] Recipe: "
        + json.dumps(
            {
                "victim_id": victim_id,
                "dataset": dataset,
                "arch": arch,
                "seed": int(seed),
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "width_mult": int(width_mult),
                "optimizer": optimizer_name,
                "lr": float(lr),
                "momentum": float(momentum),
                "weight_decay": float(weight_decay),
                "scheduler": scheduler_name,
                "dropout_prob": float(dropout_prob),
                "num_workers": int(num_workers),
                "out": str(out_path),
            },
            sort_keys=True,
        )
    )

    best_acc = -1.0
    best_epoch = -1
    best_state = None
    for epoch in range(1, epochs + 1):
        show_progress = not bool(args.no_progress)
        train_loss, train_acc = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            label_smoothing=label_smoothing,
            show_progress=show_progress,
            progress_desc=f"train {epoch}/{epochs}",
        )
        test_loss, test_acc = _evaluate(
            model,
            test_loader,
            device,
            show_progress=show_progress,
            progress_desc=f"eval  {epoch}/{epochs}",
        )

        if scheduler is not None:
            scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{epochs} "
            f"lr={lr_now:.6f} "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% "
            f"test_loss={test_loss:.4f} test_acc={test_acc*100:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = float(test_acc)
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training produced no model state to save.")

    torch.save(best_state, out_path)
    print(f"[INFO] Saved best checkpoint (epoch={best_epoch}, acc={best_acc*100:.2f}%) to: {out_path}")

    if save_metadata:
        meta_path = out_path.with_suffix(out_path.suffix + ".json")
        meta = {
            "victim_id": victim_id,
            "dataset": dataset,
            "arch": arch,
            "seed": int(seed),
            "best_epoch": int(best_epoch),
            "best_acc": float(best_acc),
            "recipe": asdict(
                VictimTrainRecipe(
                    dataset=dataset,
                    arch=arch,
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    optimizer=optimizer_name,
                    lr=float(lr),
                    momentum=float(momentum),
                    weight_decay=float(weight_decay),
                    scheduler=scheduler_name,
                    width_mult=int(width_mult),
                    label_smoothing=label_smoothing,
                    dropout_prob=float(dropout_prob),
                )
            ),
        }
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[INFO] Saved metadata to: {meta_path}")


if __name__ == "__main__":
    train()
