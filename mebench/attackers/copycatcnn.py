from typing import Dict, Any, List, Tuple, Optional
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm

from mebench.attackers.runner import AttackRunner
from mebench.core.context import BenchmarkContext
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.data.loaders import create_dataloader
from mebench.models.substitute_factory import create_substitute
from mebench.training import SubstituteTrainer, TrainRequest
from mebench.utils.dataloader import (
    pool_loader_kwargs,
    resolve_train_num_workers,
    resolve_val_num_workers,
)


class _OfflineAugmentedTensorDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        aug_list: list[object],
        multiplier: int,
    ) -> None:
        self.x = x
        self.y = y
        self.aug_list = aug_list
        self.multiplier = max(1, int(multiplier))

    def __len__(self) -> int:
        return int(self.x.size(0)) * int(self.multiplier)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        base_len = int(self.x.size(0))
        base_idx = int(idx % base_len)
        aug_idx = int((idx // base_len) % self.multiplier)
        x = self.x[base_idx]
        if self.aug_list:
            aug = self.aug_list[aug_idx % len(self.aug_list)]
            x = aug(x)
        return x, self.y[base_idx]


class _CopycatGaussianNoise:
    def __init__(self, sigma_max: float = 0.08) -> None:
        self.sigma_max = float(sigma_max)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        sigma = float(np.random.uniform(0.0, self.sigma_max))
        out = x + torch.randn_like(x) * sigma
        return self._clamp(out)

    @staticmethod
    def _clamp(x: torch.Tensor) -> torch.Tensor:
        if float(x.min().item()) < 0.0:
            return x.clamp(-1.0, 1.0)
        return x.clamp(0.0, 1.0)


class _CopycatAddConstant:
    def __init__(self, vmin: float = -20.0 / 255.0, vmax: float = 40.0 / 255.0) -> None:
        self.vmin = float(vmin)
        self.vmax = float(vmax)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        val = float(np.random.uniform(self.vmin, self.vmax))
        out = x + val
        return self._clamp(out)

    @staticmethod
    def _clamp(x: torch.Tensor) -> torch.Tensor:
        if float(x.min().item()) < 0.0:
            return x.clamp(-1.0, 1.0)
        return x.clamp(0.0, 1.0)


class CopycatCNN(AttackRunner):
    """CopycatCNN attack using non-problem domain data."""

    def __init__(self, config: dict, state: BenchmarkState) -> None:
        super().__init__(config, state)

        self.batch_size = int(config.get("batch_size", 128))
        self.train_every = int(config.get("train_every", 1000))
        self.train_checkpoints = config.get("train_checkpoints", [1000, 10000, 100000, 1000000])
        self.substitute_lr = float(config.get("substitute_lr", 0.01))
        self.substitute_momentum = float(config.get("substitute_momentum", 0.9))
        self.substitute_weight_decay = float(config.get("substitute_weight_decay", 5e-4))
        self.substitute_epochs = int(config.get("substitute_epochs", 200))
        # Round-based execution: query -> (re)train substitute, repeated.
        # Default aligns with common "10 round" reporting in prior works.
        self.rounds = int(config.get("rounds", 10))
        # Paper CopycatCNN balances the fake dataset by class by default.
        self.balance_by_class = bool(config.get("balance_by_class", True))
        # Official CopycatCNN code expands each image with ~22 offline augmentation methods.
        self.augmentation_multiplier = int(config.get("augmentation_multiplier", 5))
        self.num_classes = int(
            state.metadata.get("num_classes")
            or config.get("num_classes")
            or state.metadata.get("dataset_config", {}).get("num_classes", 10)
        )

        self.pool_dataset = None
        self.substitute: nn.Module | None = None
        self.substitute_optimizer: torch.optim.Optimizer | None = None

        self._initialize_state(state)
        
        # [P0 FIX] Validate NPDD constraint in initialization
        dataset_config = self._get_pool_dataset_config(state)
        if dataset_config.get("data_mode", "").lower() in ["cifar10", "svhn", "mnist", "fashionmnist"]:
            raise ValueError(
                f"CopycatCNN requires NPDD dataset, but '{dataset_config.get('data_mode')}' is problem-domain. "
                f"Use datasets like ImageNet, Caltech101, or Textures for true NPDD evaluation."
            )

    def run(self, ctx: BenchmarkContext) -> None:
        self.victim = ctx.oracle.model
        device = self.state.metadata.get("device", "cpu")
        total_budget = int(
            self.state.metadata.get("max_budget")
            or self.config.get("max_budget", ctx.budget_remaining)
        )

        rounds = max(1, int(self.rounds))
        round_size = max(1, int(math.ceil(total_budget / rounds)))

        pbar = tqdm(total=total_budget, desc="[CopycatCNN] Extracting")
        round_id = 0
        while ctx.budget_remaining > 0:
            round_id += 1
            step_size = min(round_size, ctx.budget_remaining)
            query_batch = self._select_query_batch(step_size, self.state)
            meta = dict(query_batch.meta or {})
            meta["round"] = round_id
            oracle_output = ctx.query(query_batch.x, meta=meta)
            self._handle_oracle_output(query_batch.x, oracle_output, self.state)
            pbar.update(query_batch.x.size(0))

            # Offline training after each round on accumulated labeled set.
            self._train_substitute(self.state)

        pbar.close()

    def _select_query_batch(self, k: int, state: BenchmarkState) -> QueryBatch:
        if self.pool_dataset is None:
            self._load_pool(state)

        if len(self.pool_dataset) == 0:
            raise ValueError("CopycatCNN requires a non-empty pool dataset (NPD).")

        replace = k > len(self.pool_dataset)
        indices = np.random.choice(len(self.pool_dataset), k, replace=replace).tolist()
        x_list = [self.pool_dataset[idx][0] for idx in indices]
        x = torch.stack(x_list)
        return QueryBatch(x=x, meta={"indices": indices, "synthetic": False})

    def _handle_oracle_output(
        self,
        x_batch: torch.Tensor,
        oracle_output: OracleOutput,
        state: BenchmarkState,
    ) -> None:
        if oracle_output.kind == "soft_prob":
            labels = torch.argmax(oracle_output.y, dim=1)
        else:
            labels = oracle_output.y

        state.attack_state["query_data_x"].append(x_batch.detach().cpu())
        state.attack_state["query_data_y"].append(labels.detach().cpu())

    def _initialize_state(self, state: BenchmarkState) -> None:
        state.attack_state["query_data_x"] = []
        state.attack_state["query_data_y"] = []
        state.attack_state["val_query_data_x"] = []
        state.attack_state["val_query_data_y"] = []
        state.attack_state["substitute"] = None

    def _get_pool_dataset_config(self, state: BenchmarkState) -> dict:
        dataset_config = self.config.get("attack", {}).get("dataset")
        if not dataset_config:
            dataset_config = state.metadata.get("dataset_config", {})
        if "data_mode" not in dataset_config:
            dataset_config = {"data_mode": "seed", **dataset_config}
        if dataset_config.get("data_mode") == "seed" and "name" not in dataset_config:
            dataset_config = {"name": "CIFAR10", **dataset_config}
        if dataset_config.get("data_mode") == "surrogate" and "surrogate_name" not in dataset_config:
            dataset_config = {"surrogate_name": "SVHN", **dataset_config}
        return dataset_config

    def _load_pool(self, state: BenchmarkState) -> None:
        dataset_config = self._get_pool_dataset_config(state)
        self.pool_dataset = create_dataloader(
            dataset_config,
            batch_size=1,
            shuffle=False,
        ).dataset

    def _train_substitute(self, state: BenchmarkState) -> None:
        self._ensure_fixed_validation_holdout(state)
        query_x = state.attack_state["query_data_x"]
        query_y = state.attack_state["query_data_y"]
        val_query_x = state.attack_state.get("val_query_data_x", [])
        val_query_y = state.attack_state.get("val_query_data_y", [])
        if len(query_x) == 0:
            return

        x_all = torch.cat(query_x, dim=0)
        y_all = torch.cat(query_y, dim=0)

        indices = list(range(len(y_all)))
        if self.balance_by_class:
            class_indices = {c: [] for c in range(self.num_classes)}
            for idx, label in enumerate(y_all.tolist()):
                class_indices[int(label)].append(idx)

            # Balance to a uniform class histogram by oversampling minority classes.
            counts = [len(v) for v in class_indices.values() if v]
            target = max(counts) if counts else 0
            if target > 0:
                indices = []
                for cls_indices in class_indices.values():
                    if not cls_indices:
                        continue
                    replace = len(cls_indices) < target
                    chosen = np.random.choice(cls_indices, target, replace=replace).tolist()
                    indices.extend(chosen)

        x_train = x_all[indices]
        y_train = y_all[indices]

        # 22 offline augmentation methods (approximated with torchvision ops on tensors).
        # Reference: https://github.com/jeiks/Stealing_DL_Models (Copycat_CNN/Scripts/image-augmentation.py)
        augs: list[callable] = [
            # 1-6: Affine variants
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.9, 1.0), shear=20),
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.9, 1.0), shear=20),
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.85, 1.0), shear=20),
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.9, 1.0), shear=20),
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.9, 1.0)),
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.9, 1.0)),

            # 7-9: Crop + affine (approximated)
            transforms.Compose(
                [transforms.RandomResizedCrop(size=x_train.shape[-1], scale=(0.9, 1.0)), transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=15)]
            ),
            transforms.Compose(
                [transforms.RandomResizedCrop(size=x_train.shape[-1], scale=(0.9, 1.0)), transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), shear=15)]
            ),
            transforms.Compose(
                [transforms.RandomResizedCrop(size=x_train.shape[-1], scale=(0.9, 1.0)), transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), shear=15), transforms.RandomHorizontalFlip(p=0.9)]
            ),

            # 10-15: Noise / Sharpen / Crop (+ flip variants approximated)
            _CopycatGaussianNoise(sigma_max=0.08),
            transforms.Compose([transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=1.0)]),
            transforms.RandomResizedCrop(size=x_train.shape[-1], scale=(0.85, 1.0)),
            transforms.Compose([_CopycatGaussianNoise(sigma_max=0.08), transforms.RandomHorizontalFlip(p=0.9)]),
            transforms.Compose([transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=1.0), transforms.RandomHorizontalFlip(p=0.9)]),
            transforms.Compose([transforms.RandomResizedCrop(size=x_train.shape[-1], scale=(0.85, 1.0)), transforms.RandomHorizontalFlip(p=0.9)]),

            # 16-18: Blur / Add / Contrast
            transforms.GaussianBlur(kernel_size=3, sigma=(1.0, 1.5)),
            _CopycatAddConstant(),
            transforms.ColorJitter(contrast=(1.0, 1.5)),

            # 19-22: Piecewise affine (approximated via perspective)
            transforms.RandomPerspective(distortion_scale=0.05, p=1.0),
            transforms.RandomPerspective(distortion_scale=0.05, p=1.0),
            transforms.RandomPerspective(distortion_scale=0.05, p=1.0),
            transforms.Compose([
                transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.9, 1.0), shear=20),
                transforms.RandomPerspective(distortion_scale=0.05, p=1.0),
            ]),
        ]

        dataset = _OfflineAugmentedTensorDataset(x_train, y_train, augs, self.augmentation_multiplier)
        sub_config = state.metadata.get("substitute_config", {})
        train_batch_size = int(
            sub_config.get("batch_size")
            or sub_config.get("trackA", {}).get("batch_size", self.batch_size)
        )
        device = state.metadata.get("device", "cpu")
        train_workers = resolve_train_num_workers(sub_config, self.config, default=0)
        val_workers = resolve_val_num_workers(sub_config, self.config, default=train_workers)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=train_batch_size,
            shuffle=True,
            **pool_loader_kwargs(device, {"num_workers": int(train_workers)}),
        )

        val_loader = None
        if len(val_query_x) > 0 and len(val_query_y) > 0:
            x_val = torch.cat(val_query_x, dim=0)
            y_val = torch.cat(val_query_y, dim=0)
            val_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(x_val, y_val),
                batch_size=train_batch_size,
                shuffle=False,
                **pool_loader_kwargs(device, {"num_workers": int(val_workers)}),
            )

        if self.substitute is None:
            width_mult = int(sub_config.get("width_mult", 1))
            dropout_prob = float(sub_config.get("dropout_prob", 0.0))
            self.substitute = create_substitute(
                arch=sub_config.get("arch", "resnet18"),
                num_classes=self.num_classes,
                input_channels=state.metadata.get("input_shape", (3, 32, 32))[0],
                width_mult=width_mult,
                dropout_prob=dropout_prob,
            ).to(device)
        epochs = max(1, int(self.substitute_epochs))
        
        victim_config = state.metadata.get("victim_config", {})
        normalization = victim_config.get("normalization")
        if normalization is None:
            normalization = {"mean": [0.0], "std": [1.0]}
        norm_mean = torch.tensor(normalization["mean"]).view(1, -1, 1, 1).to(device)
        norm_std = torch.tensor(normalization["std"]).view(1, -1, 1, 1).to(device)
        
        def preprocess_fn(x_batch: torch.Tensor) -> torch.Tensor:
            return (x_batch - norm_mean) / norm_std

        def loss_fn(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return F.cross_entropy(outputs, targets.long())

        def eval_fn(model_local: nn.Module, loader_local: torch.utils.data.DataLoader) -> float:
            model_local.eval()
            total_loss = 0.0
            total_count = 0
            with torch.no_grad():
                for x_val_b, y_val_b in loader_local:
                    x_val_b = x_val_b.to(device)
                    y_val_b = y_val_b.to(device)
                    outputs = model_local(preprocess_fn(x_val_b))
                    loss = F.cross_entropy(outputs, y_val_b.long())
                    total_loss += float(loss.item()) * int(x_val_b.size(0))
                    total_count += int(x_val_b.size(0))
            return total_loss / max(1, total_count)

        train_config = dict(sub_config)
        train_config["max_epochs"] = int(sub_config.get("max_epochs", epochs))
        train_config["patience"] = int(sub_config.get("patience", 20))
        trainer = SubstituteTrainer(train_config, device=device, logger=self.logger)
        max_steps = int(train_config["max_epochs"]) * max(1, int(math.ceil(max(1, len(dataset)) / max(1, train_batch_size))))
        validate_every = max(1, int(math.ceil(max(1, len(dataset)) / max(1, train_batch_size))))
        request = TrainRequest(
            model=self.substitute,
            train_loader=loader,
            val_loader=val_loader,
            eval_fn=eval_fn if val_loader is not None else None,
            loss_fn=loss_fn,
            preprocess_fn=preprocess_fn,
            max_steps=max_steps,
            validate_every=validate_every,
            patience=int(train_config["patience"]) * validate_every,
            early_stop_mode="min",
            load_best=True,
        )
        trainer.train(request)

        state.attack_state["substitute"] = self.substitute
        self.logger.info("CopycatCNN substitute trained.")
        self._evaluate_current_substitute(self.substitute, device)

