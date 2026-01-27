import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms

from mebench.attackers.base import BaseAttack
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.data.loaders import create_dataloader
from mebench.models.substitute_factory import create_substitute


class CopycatCNN(BaseAttack):
    """CopycatCNN attack using non-problem domain data."""

    def __init__(self, config: dict, state: BenchmarkState) -> None:
        super().__init__(config, state)

        self.batch_size = int(config.get("batch_size", 128))
        self.train_every = int(config.get("train_every", 1000))
        self.substitute_lr = float(config.get("substitute_lr", 0.01))
        self.substitute_momentum = float(config.get("substitute_momentum", 0.9))
        self.substitute_weight_decay = float(config.get("substitute_weight_decay", 5e-4))
        self.substitute_epochs = int(config.get("substitute_epochs", 5))
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

    def _initialize_state(self, state: BenchmarkState) -> None:
        state.attack_state["query_data_x"] = []
        state.attack_state["query_data_y"] = []
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

    def propose(self, k: int, state: BenchmarkState) -> QueryBatch:
        if self.pool_dataset is None:
            self._load_pool(state)

        if len(self.pool_dataset) == 0:
            raise ValueError("CopycatCNN requires a non-empty pool dataset (NPD).")

        # CopycatCNN queries natural images from a non-problem domain pool.
        # If k exceeds pool size, sample with replacement rather than using noise.
        replace = k > len(self.pool_dataset)
        indices = np.random.choice(len(self.pool_dataset), k, replace=replace).tolist()
        x_list = [self.pool_dataset[idx][0] for idx in indices]
        x = torch.stack(x_list)
        meta = {"indices": indices}
        return QueryBatch(x=x, meta=meta)

    def observe(
        self,
        query_batch: QueryBatch,
        oracle_output: OracleOutput,
        state: BenchmarkState,
    ) -> None:
        if oracle_output.kind == "soft_prob":
            labels = torch.argmax(oracle_output.y, dim=1)
        else:
            labels = oracle_output.y

        state.attack_state["query_data_x"].append(query_batch.x.detach().cpu())
        state.attack_state["query_data_y"].append(labels.detach().cpu())

        if state.query_count % self.train_every == 0 and state.query_count > 0:
            self._train_substitute(state)

    def _train_substitute(self, state: BenchmarkState) -> None:
        query_x = state.attack_state["query_data_x"]
        query_y = state.attack_state["query_data_y"]
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

        def _clamp_inplace(x: torch.Tensor) -> torch.Tensor:
            # Keep range consistent with common dataset conventions.
            if x.min().item() < 0.0:
                return x.clamp(-1.0, 1.0)
            return x.clamp(0.0, 1.0)

        def _gaussian_noise(x: torch.Tensor, sigma_max: float = 0.08) -> torch.Tensor:
            # sigma_max follows imgaug settings (~0.08 * 255). We treat x as [0,1] or [-1,1].
            sigma = float(np.random.uniform(0.0, sigma_max))
            return _clamp_inplace(x + torch.randn_like(x) * sigma)

        def _add_constant(x: torch.Tensor, vmin: float = -20.0 / 255.0, vmax: float = 40.0 / 255.0) -> torch.Tensor:
            val = float(np.random.uniform(vmin, vmax))
            return _clamp_inplace(x + val)

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
            transforms.Lambda(lambda x: _gaussian_noise(x, sigma_max=0.08)),
            transforms.Compose([transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=1.0)]),
            transforms.RandomResizedCrop(size=x_train.shape[-1], scale=(0.85, 1.0)),
            transforms.Compose([transforms.Lambda(lambda x: _gaussian_noise(x, sigma_max=0.08)), transforms.RandomHorizontalFlip(p=0.9)]),
            transforms.Compose([transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=1.0), transforms.RandomHorizontalFlip(p=0.9)]),
            transforms.Compose([transforms.RandomResizedCrop(size=x_train.shape[-1], scale=(0.85, 1.0)), transforms.RandomHorizontalFlip(p=0.9)]),

            # 16-18: Blur / Add / Contrast
            transforms.GaussianBlur(kernel_size=3, sigma=(1.0, 1.5)),
            transforms.Lambda(lambda x: _add_constant(x)),
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

        class OfflineAugmentedTensorDataset(torch.utils.data.Dataset):
            def __init__(self, x: torch.Tensor, y: torch.Tensor, aug_list: list[callable], multiplier: int):
                self.x = x
                self.y = y
                self.aug_list = aug_list
                self.multiplier = max(1, int(multiplier))

            def __len__(self) -> int:
                return self.x.size(0) * self.multiplier

            def __getitem__(self, idx: int):
                base_len = self.x.size(0)
                base_idx = idx % base_len
                aug_idx = (idx // base_len) % self.multiplier
                x = self.x[base_idx]
                if self.aug_list:
                    aug = self.aug_list[aug_idx % len(self.aug_list)]
                    x = aug(x)
                return x, self.y[base_idx]

        dataset = OfflineAugmentedTensorDataset(x_train, y_train, augs, self.augmentation_multiplier)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        sub_config = state.metadata.get("substitute_config", {})
        opt_params = sub_config.get("optimizer", {})
        
        device = state.metadata.get("device", "cpu")
        if self.substitute is None:
            self.substitute = create_substitute(
                arch=sub_config.get("arch", "resnet18"),
                num_classes=self.num_classes,
                input_channels=state.metadata.get("input_shape", (3, 32, 32))[0],
            ).to(device)
            self.substitute_optimizer = torch.optim.SGD(
                self.substitute.parameters(),
                lr=float(opt_params.get("lr", self.substitute_lr)),
                momentum=float(opt_params.get("momentum", self.substitute_momentum)),
                weight_decay=float(opt_params.get("weight_decay", self.substitute_weight_decay)),
            )

        self.substitute.train()
        epochs = max(1, int(self.substitute_epochs))
        
        victim_config = state.metadata.get("victim_config", {})
        normalization = victim_config.get("normalization")
        if normalization is None:
            normalization = {"mean": [0.0], "std": [1.0]}
        norm_mean = torch.tensor(normalization["mean"]).view(1, -1, 1, 1).to(device)
        norm_std = torch.tensor(normalization["std"]).view(1, -1, 1, 1).to(device)
        
        for _ in range(epochs):
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(device)
                # Normalize images for substitute
                x_batch = (x_batch - norm_mean) / norm_std
                
                y_batch = y_batch.long().to(device)
                self.substitute_optimizer.zero_grad()
                logits = self.substitute(x_batch)
                loss = F.cross_entropy(logits, y_batch)
                loss.backward()
                self.substitute_optimizer.step()

        state.attack_state["substitute"] = self.substitute
