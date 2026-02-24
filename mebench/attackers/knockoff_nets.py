"""Knockoff Nets attack implementation."""

from typing import Dict, Any, List, Tuple, Optional
from collections import deque
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from mebench.attackers.runner import AttackRunner
from mebench.core.context import BenchmarkContext
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.data.loaders import create_dataloader
from mebench.utils.dataloader import (
    pool_loader_kwargs,
    resolve_train_num_workers,
    resolve_val_num_workers,
)
from mebench.models.substitute_factory import create_substitute
from mebench.training import SubstituteTrainer, TrainRequest


class KnockoffNets(AttackRunner):
    """Knockoff Nets with a simple gradient-bandit policy."""

    def __init__(self, config: dict, state: BenchmarkState):
        super().__init__(config, state)

        # Official knockoffnets transfer.py defaults to batch_size=8.
        self.batch_size = int(config.get("batch_size", 8))
        self.policy = str(config.get("policy", "adaptive")).strip().lower()
        if self.policy not in {"random", "adaptive"}:
            raise ValueError(f"KnockoffNets policy must be 'random' or 'adaptive', got {self.policy!r}")
        # Update substitute periodically or every batch if policy requires fresh logits
        self.train_every = max(1, int(config.get("train_every", self.batch_size)))
        self.train_epochs = int(config.get("train_epochs", 1))
        self.online_train_epochs = int(config.get("online_train_epochs", self.train_epochs))
        offline_train_epochs = config.get("offline_train_epochs")
        self.offline_train_epochs = (
            int(offline_train_epochs) if offline_train_epochs is not None else None
        )
        self.reward_window = int(config.get("reward_window", 100))
        self.reward_certainty_weight = float(config.get("reward_certainty_weight", 1.0))
        self.reward_diversity_weight = float(config.get("reward_diversity_weight", 1.0))
        self.reward_loss_weight = float(config.get("reward_loss_weight", 1.0))
        self.coarse_clusters = int(config.get("coarse_clusters", 30))
        self.samples_per_class = int(config.get("samples_per_class", 50))
        self.kmeans_iters = int(config.get("kmeans_iters", 100))
        self.kmeans_tol = float(config.get("kmeans_tol", 1e-4))
        feature_arch = str(config.get("feature_arch", "resnet50")).lower()
        if feature_arch in {"resnet18", "resnet34"}:
            feature_arch = "resnet50"
        self.feature_arch = feature_arch
        self.policy_lr = float(config.get("policy_lr", 0.01))
        self.paper_train_lr = float(config.get("paper_train_lr", 0.1))
        self.paper_train_momentum = float(config.get("paper_train_momentum", 0.5))
        self.num_classes = int(
            state.metadata.get("num_classes")
            or config.get("num_classes")
            or state.metadata.get("dataset_config", {}).get("num_classes", 10)
        )
        loss_reward_scale = float(
            config.get("loss_reward_scale", max(1.0, math.log(max(self.num_classes, 2))))
        )
        self.loss_reward_scale = max(loss_reward_scale, 1e-6)

        self.pool_dataset = None
        self.class_to_indices: Dict[int, List[int]] = {}

        self._initialize_state(state)

    def run(self, ctx: BenchmarkContext) -> None:
        self.victim = ctx.oracle.model
        device = self.state.metadata.get("device", "cpu")
        
        pbar = self._create_progress_bar(self.state.budget_remaining, "[KnockoffNets] Extracting")
        while ctx.budget_remaining > 0:
            step_size = min(self.batch_size, ctx.budget_remaining)
            query_batch = self._select_query_batch(step_size, self.state)
            meta = query_batch.meta or {}
            oracle_output = ctx.query(query_batch.x, meta=meta)
            classes = list(meta.get("classes", []))
            self._handle_oracle_output(query_batch.x, oracle_output, classes, self.state)
            pbar.update(query_batch.x.size(0))
        pbar.close()

        self._finalize_attack(self.state)

    def _select_query_batch(self, k: int, state: BenchmarkState) -> QueryBatch:
        if self.policy == "random":
            return self._select_query_batch_random(k, state)

        if self.pool_dataset is None:
            self._load_pool(state)

        pool_len = len(self.pool_dataset) if self.pool_dataset is not None else 0
        if pool_len <= 0:
            raise ValueError(f"{self.__class__.__name__} requires a non-empty pool dataset.")

        # Regression/unit tests may set `pool_dataset` directly (mock) without calling `_load_pool`,
        # leaving bookkeeping uninitialized. Ensure a consistent minimal state.
        state.attack_state.setdefault("queried_indices", [])
        state.attack_state.setdefault("unqueried_indices", [])
        if len(state.attack_state["unqueried_indices"]) == 0:
            state.attack_state["unqueried_indices"] = list(range(pool_len))

        if not self.class_to_indices:
            self.class_to_indices = {i: [] for i in range(int(self.num_classes))}
            for idx in range(pool_len):
                try:
                    _, label = self.pool_dataset[idx]
                    class_id = int(label) % int(self.num_classes)
                except Exception:
                    class_id = 0
                self.class_to_indices.setdefault(class_id, []).append(idx)

        unqueried = state.attack_state["unqueried_indices"]
        if len(unqueried) == 0:
            raise ValueError(
                f"Query pool exhausted for {self.__class__.__name__}. "
                f"Cannot select {k} more queries."
            )

        selected_indices: List[int] = []
        selected_classes: List[int] = []

        attempts = 0
        while len(selected_indices) < k and attempts < k * 5:
            attempts += 1
            class_id = self._sample_class_with_policy(state)
            pool_list = self.class_to_indices.get(class_id, [])
            if not pool_list:
                continue
            idx = pool_list.pop()
            if idx not in unqueried:
                continue
            unqueried.remove(idx)
            state.attack_state["queried_indices"].append(idx)
            selected_indices.append(idx)
            selected_classes.append(class_id)

        if len(selected_indices) < k:
            remaining = [idx for idx in unqueried if idx not in selected_indices]
            extra = min(k - len(selected_indices), len(remaining))
            if extra > 0:
                extra_indices = np.random.choice(remaining, extra, replace=False).tolist()
                for idx in extra_indices:
                    unqueried.remove(idx)
                    state.attack_state["queried_indices"].append(idx)
                    selected_indices.append(idx)
                    selected_classes.append(-1)

        x_list = []
        for idx in selected_indices:
            img, _ = self.pool_dataset[idx]
            x_list.append(img)

        if len(x_list) < k:
            raise ValueError(
                f"Query pool exhausted for {self.__class__.__name__}. "
                f"Requested {k}, found {len(x_list)}."
            )

        x = torch.stack(x_list)
        return QueryBatch(
            x=x,
            meta={"indices": selected_indices, "classes": selected_classes, "synthetic": False},
        )

    def _select_query_batch_random(self, k: int, state: BenchmarkState) -> QueryBatch:
        """Official transfer.py semantics: random no-policy sampling with refill."""
        if self.pool_dataset is None:
            self._load_pool(state)

        pool_len = len(self.pool_dataset) if self.pool_dataset is not None else 0
        if pool_len <= 0:
            raise ValueError(f"{self.__class__.__name__} requires a non-empty pool dataset.")

        state.attack_state.setdefault("queried_indices", [])
        state.attack_state.setdefault("unqueried_indices", list(range(pool_len)))
        unqueried = state.attack_state["unqueried_indices"]
        if len(unqueried) == 0:
            unqueried.extend(list(range(pool_len)))
            state.attack_state["random_refill_count"] = int(
                state.attack_state.get("random_refill_count", 0)
            ) + 1

        selected_indices: List[int] = []
        while len(selected_indices) < int(k):
            if len(unqueried) == 0:
                unqueried.extend(list(range(pool_len)))
                state.attack_state["random_refill_count"] = int(
                    state.attack_state.get("random_refill_count", 0)
                ) + 1

            take = min(int(k) - len(selected_indices), len(unqueried))
            chosen = np.random.choice(unqueried, take, replace=False).tolist()
            for idx in chosen:
                unqueried.remove(idx)
                state.attack_state["queried_indices"].append(int(idx))
            selected_indices.extend(chosen)

        x_list = [self.pool_dataset[idx][0] for idx in selected_indices]
        x = torch.stack(x_list)
        return QueryBatch(
            x=x,
            meta={
                "indices": selected_indices,
                "classes": [-1] * len(selected_indices),
                "synthetic": False,
                "policy": "random",
                "refill_count": int(state.attack_state.get("random_refill_count", 0)),
            },
        )

    def _handle_oracle_output(
        self,
        x_batch: torch.Tensor,
        oracle_output: OracleOutput,
        classes: List[int],
        state: BenchmarkState,
    ) -> None:
        state.attack_state["query_data_x"].append(x_batch.detach().cpu())
        state.attack_state["query_data_y"].append(oracle_output.y.detach().cpu())
        state.attack_state["query_count"] += x_batch.shape[0]

        if oracle_output.kind == "soft_prob":
            probs = oracle_output.y.detach().cpu()
        else:
            probs = F.one_hot(oracle_output.y, num_classes=self.num_classes).float().cpu()

        recent_probs = state.attack_state["recent_victim_probs"]
        for row in probs:
            recent_probs.append(row)

        if self.policy == "random":
            last_train_count = state.attack_state.get("last_train_count", 0)
            if state.attack_state["query_count"] - last_train_count >= self.train_every:
                self._train_substitute(
                    state,
                    reset_model=False,
                    epochs=self.online_train_epochs,
                    store_key="online_substitute",
                )
                state.attack_state["substitute"] = state.attack_state.get("online_substitute")
                state.attack_state["last_train_count"] = state.attack_state["query_count"]
            return

        top2 = torch.topk(probs, k=2, dim=1).values
        certainty_reward = top2[:, 0] - top2[:, 1]

        if len(recent_probs) > 0:
            mean_recent = torch.stack(list(recent_probs)).mean(dim=0)
            diff = probs - mean_recent.unsqueeze(0)
            diversity_reward = torch.clamp(diff, min=0).sum(dim=1)
        else:
            diversity_reward = torch.zeros(probs.size(0))

        substitute = state.attack_state.get("online_substitute")
        if substitute is not None:
            substitute.eval()
            device = state.metadata.get("device", "cpu")
            norm_mean, norm_std = self._get_normalization(state, device)
            with torch.no_grad():
                x_input = (x_batch.to(device) - norm_mean) / norm_std
                logits = substitute(x_input)
                log_probs = F.log_softmax(logits, dim=1)
                loss_reward = -(probs.to(device) * log_probs).sum(dim=1).detach().cpu()
        else:
            loss_reward = torch.zeros(probs.size(0))

        def _normalize_with_history(values: torch.Tensor, history: deque[float]) -> torch.Tensor:
            if values.numel() == 0:
                return values
            if len(history) > 0:
                hist_min = min(history)
                hist_max = max(history)
            else:
                hist_min = float(values.min().item())
                hist_max = float(values.max().item())
            denom = max(hist_max - hist_min, 1e-6)
            normalized = (values - hist_min) / denom
            return torch.clamp(normalized, 0.0, 1.0)

        certainty_reward_norm = _normalize_with_history(
            certainty_reward, state.attack_state["recent_certainty_rewards"]
        )
        diversity_reward_norm = _normalize_with_history(
            diversity_reward, state.attack_state["recent_diversity_rewards"]
        )
        loss_reward_scaled = loss_reward / self.loss_reward_scale
        loss_reward_norm = _normalize_with_history(
            loss_reward_scaled, state.attack_state["recent_loss_rewards"]
        )
        
        rewards = (
            self.reward_certainty_weight * certainty_reward_norm
            + self.reward_diversity_weight * diversity_reward_norm
            + self.reward_loss_weight * loss_reward_norm
        )

        if len(state.attack_state["recent_rewards"]) > 0:
            baseline = float(np.mean(state.attack_state["recent_rewards"]))
        else:
            baseline = 0.0
        reward_mean = float(rewards.mean().item()) if rewards.numel() > 0 else 0.0
        state.attack_state["recent_rewards"].append(reward_mean)
        state.attack_state["reward_baseline"] = baseline

        state.attack_state["recent_certainty_rewards"].extend(
            [float(x) for x in certainty_reward.tolist()]
        )
        state.attack_state["recent_diversity_rewards"].extend(
            [float(x) for x in diversity_reward.tolist()]
        )
        state.attack_state["recent_loss_rewards"].extend(
            [float(x) for x in loss_reward_scaled.tolist()]
        )

        weights = state.attack_state["policy_weights"].clone().float()
        coarse_weights = state.attack_state["coarse_policy_weights"].clone().float()
        class_to_coarse = state.attack_state["class_to_coarse"]

        pi = torch.softmax(weights, dim=0)
        for idx, class_id in enumerate(classes):
            if class_id < 0 or class_id >= weights.numel():
                continue

            state.attack_state["action_counts"][class_id] += 1.0
            alpha = 1.0 / float(state.attack_state["action_counts"][class_id].item())
            adv = float(rewards[idx]) - baseline
            grad = -pi
            grad[class_id] = 1.0 - pi[class_id]
            weights = weights + alpha * adv * grad

            pi = torch.softmax(weights, dim=0)

            if class_id in class_to_coarse and coarse_weights.numel() > 0:
                coarse_id = int(class_to_coarse[class_id])
                if 0 <= coarse_id < coarse_weights.numel():
                    state.attack_state["coarse_action_counts"][coarse_id] += 1.0
                    coarse_alpha = 1.0 / float(state.attack_state["coarse_action_counts"][coarse_id].item())
                    coarse_pi = torch.softmax(coarse_weights, dim=0)
                    coarse_grad = -coarse_pi
                    coarse_grad[coarse_id] = 1.0 - coarse_pi[coarse_id]
                    coarse_weights = coarse_weights + coarse_alpha * adv * coarse_grad

        state.attack_state["policy_weights"] = weights
        if coarse_weights.numel() > 0:
            state.attack_state["coarse_policy_weights"] = coarse_weights

        last_train_count = state.attack_state.get("last_train_count", 0)
        if state.attack_state["query_count"] - last_train_count >= self.train_every:
            self.logger.debug(
                "Training substitute at query count %s...",
                state.attack_state["query_count"],
            )
            self._train_substitute(
                state,
                reset_model=False,
                epochs=self.online_train_epochs,
                store_key="online_substitute",
            )
            state.attack_state["substitute"] = state.attack_state.get("online_substitute")
            state.attack_state["last_train_count"] = state.attack_state["query_count"]

    def _initialize_state(self, state: BenchmarkState) -> None:
        state.attack_state["queried_indices"] = []
        state.attack_state["unqueried_indices"] = []
        state.attack_state["policy_weights"] = torch.zeros(self.num_classes)
        state.attack_state["coarse_policy_weights"] = torch.zeros(0)
        state.attack_state["action_counts"] = torch.zeros(self.num_classes)
        state.attack_state["coarse_action_counts"] = torch.zeros(0)
        state.attack_state["reward_baseline"] = 0.0
        state.attack_state["recent_victim_probs"] = deque(maxlen=self.reward_window)
        state.attack_state["recent_rewards"] = deque(maxlen=self.reward_window)
        state.attack_state["recent_certainty_rewards"] = deque(maxlen=self.reward_window)
        state.attack_state["recent_diversity_rewards"] = deque(maxlen=self.reward_window)
        state.attack_state["recent_loss_rewards"] = deque(maxlen=self.reward_window)
        state.attack_state["query_data_x"] = []
        state.attack_state["query_data_y"] = []
        state.attack_state["val_query_data_x"] = []
        state.attack_state["val_query_data_y"] = []
        state.attack_state["substitute"] = None
        state.attack_state["online_substitute"] = None
        state.attack_state["offline_substitute"] = None
        state.attack_state["query_count"] = 0
        state.attack_state["last_train_count"] = 0
        state.attack_state["class_to_coarse"] = {}
        state.attack_state["coarse_to_classes"] = {}
        state.attack_state["random_refill_count"] = 0

    def _get_dataset_config(self, state: BenchmarkState) -> dict:
        dataset_config = state.metadata.get("dataset_config", {})
        if not dataset_config:
            dataset_config = self.config.get("dataset", {})
        if "data_mode" not in dataset_config:
            dataset_config = {"data_mode": "seed", **dataset_config}
        if dataset_config.get("data_mode") == "seed" and "name" not in dataset_config:
            dataset_config = {"name": "CIFAR10", **dataset_config}
        if dataset_config.get("data_mode") == "surrogate" and "surrogate_name" not in dataset_config:
            dataset_config = {"surrogate_name": "SVHN", **dataset_config}
        return dataset_config

    def _load_pool(self, state: BenchmarkState) -> None:
        dataset_config = self._get_dataset_config(state)
        self.pool_dataset = create_dataloader(
            dataset_config,
            batch_size=1,
            shuffle=False,
        ).dataset

        # IMPORTANT: keep num_classes aligned with the victim, not the surrogate pool.
        # ImageNet/ImageFolder pools may have 1000 classes; we still steal a 10-class victim.
        victim_num_classes = int(state.metadata.get("num_classes", self.num_classes))
        self.num_classes = victim_num_classes
        state.attack_state["policy_weights"] = torch.zeros(self.num_classes)

        self.class_to_indices = {i: [] for i in range(self.num_classes)}
        for idx in range(len(self.pool_dataset)):
            _, label = self.pool_dataset[idx]
            # Map surrogate classes to victim classes via modulo if needed
            class_id = int(label) % self.num_classes
            if class_id not in self.class_to_indices:
                self.class_to_indices[class_id] = []
            self.class_to_indices[class_id].append(idx)

        state.attack_state["unqueried_indices"] = list(range(len(self.pool_dataset)))

        if not state.attack_state["class_to_coarse"]:
            self._build_hierarchy(state)

    def _get_feature_extractor(self, device: str, input_channels: int = 3) -> nn.Module:
        if self.feature_arch == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
        elif self.feature_arch == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT
            model = models.resnet34(weights=weights)
        elif self.feature_arch == "resnet101":
            weights = models.ResNet101_Weights.DEFAULT
            model = models.resnet101(weights=weights)
        elif self.feature_arch == "resnet152":
            weights = models.ResNet152_Weights.DEFAULT
            model = models.resnet152(weights=weights)
        else:
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)

        # Adapt first conv layer if input channels != 3
        if input_channels != 3:
            old_conv = model.conv1
            model.conv1 = nn.Conv2d(
                input_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            # Initialize new weights by averaging old weights across channels
            with torch.no_grad():
                model.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True).repeat(1, input_channels, 1, 1)

        model.fc = nn.Identity()
        model.eval()
        return model.to(device)

    def _build_hierarchy(self, state: BenchmarkState) -> None:
        device = state.metadata.get("device", "cpu")
        input_channels = state.metadata.get("input_shape", (3, 32, 32))[0]
        extractor = self._get_feature_extractor(device, input_channels)

        class_ids = sorted(self.class_to_indices.keys())
        class_means = []
        valid_class_ids = []

        for class_id in class_ids:
            indices = self.class_to_indices.get(class_id, [])
            if not indices:
                continue
            
            # Avoid full-pass over huge pools (e.g., ImageNet). Use a fixed-size sample.
            n = min(len(indices), int(self.samples_per_class))
            if n <= 0:
                continue
            sampled = indices[:n]

            features = []
            for start in range(0, len(sampled), self.batch_size):
                batch_idx = sampled[start : start + self.batch_size]
                imgs = [self.pool_dataset[idx][0] for idx in batch_idx]
                x = torch.stack(imgs).to(device)
                with torch.no_grad():
                    feats = extractor(x)
                features.append(feats.detach().cpu())

            if features:
                mean_feat = torch.cat(features, dim=0).mean(dim=0)
                class_means.append(mean_feat)
                valid_class_ids.append(class_id)

        if not class_means:
            return

        class_means_tensor = torch.stack(class_means)
        num_classes = class_means_tensor.shape[0]
        k = min(self.coarse_clusters, num_classes)

        perm = torch.randperm(num_classes)[:k]
        centroids = class_means_tensor[perm].clone()

        for _ in range(self.kmeans_iters):
            distances = torch.cdist(class_means_tensor, centroids)
            assignments = distances.argmin(dim=1)
            new_centroids = []
            for cluster_id in range(k):
                mask = assignments == cluster_id
                if mask.any():
                    new_centroids.append(class_means_tensor[mask].mean(dim=0))
                else:
                    new_centroids.append(class_means_tensor[torch.randint(0, num_classes, (1,)).item()])
            new_centroids_tensor = torch.stack(new_centroids)
            shift = torch.norm(centroids - new_centroids_tensor)
            centroids = new_centroids_tensor
            if shift < self.kmeans_tol:
                break

        class_to_coarse = {}
        coarse_to_classes: Dict[int, List[int]] = {i: [] for i in range(k)}
        for idx, class_id in enumerate(valid_class_ids):
            coarse_id = int(assignments[idx].item())
            class_to_coarse[class_id] = coarse_id
            coarse_to_classes[coarse_id].append(class_id)

        state.attack_state["class_to_coarse"] = class_to_coarse
        state.attack_state["coarse_to_classes"] = coarse_to_classes
        state.attack_state["coarse_policy_weights"] = torch.zeros(k)
        state.attack_state["coarse_action_counts"] = torch.zeros(k)

    def _sample_class_with_policy(self, state: BenchmarkState) -> int:
        class_weights = state.attack_state["policy_weights"].clone().float()
        class_to_coarse = state.attack_state["class_to_coarse"]
        coarse_to_classes = state.attack_state["coarse_to_classes"]

        if not class_to_coarse or not coarse_to_classes:
            probs = torch.softmax(class_weights, dim=0).cpu().numpy()
            return int(np.random.choice(list(range(len(probs))), p=probs))

        coarse_weights = state.attack_state["coarse_policy_weights"].clone().float()
        coarse_probs = torch.softmax(coarse_weights, dim=0).cpu().numpy()
        coarse_id = int(np.random.choice(list(range(len(coarse_probs))), p=coarse_probs))

        class_ids = [c for c in coarse_to_classes.get(coarse_id, []) if c < self.num_classes]
        if not class_ids:
            probs = torch.softmax(class_weights, dim=0).cpu().numpy()
            return int(np.random.choice(list(range(len(probs))), p=probs))

        class_probs = torch.softmax(class_weights[class_ids], dim=0).cpu().numpy()
        return int(np.random.choice(class_ids, p=class_probs))

    def _train_substitute(
        self,
        state: BenchmarkState,
        *,
        reset_model: bool,
        epochs: int,
        store_key: str,
    ) -> None:
        self._ensure_fixed_validation_holdout(state)
        query_x = state.attack_state["query_data_x"]
        query_y = state.attack_state["query_data_y"]
        val_query_x = state.attack_state.get("val_query_data_x", [])
        val_query_y = state.attack_state.get("val_query_data_y", [])
        if len(query_x) == 0:
            return

        x_all = torch.cat(query_x, dim=0)
        y_all = torch.cat(query_y, dim=0)

        sub_config = state.metadata.get("substitute_config", {})
        dataset = torch.utils.data.TensorDataset(x_all, y_all)
        train_batch_size = int(
            sub_config.get("batch_size")
            or sub_config.get("trackA", {}).get("batch_size", self.batch_size)
        )
        train_dataset = dataset
        val_dataset = None

        if len(val_query_x) > 0 and len(val_query_y) > 0:
            x_val = torch.cat(val_query_x, dim=0)
            y_val = torch.cat(val_query_y, dim=0)
            val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
            train_size = len(train_dataset)
        else:
            total_size = len(dataset)
            val_size = max(1, int(0.2 * total_size))
            train_size = total_size - val_size
            if train_size < 2:
                return
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )

        device = state.metadata.get("device", "cpu")
        train_workers = resolve_train_num_workers(sub_config, self.config, default=0)
        val_workers = resolve_val_num_workers(sub_config, self.config, default=train_workers)

        # Online retraining may be called many times per run. Using multi-worker
        # loaders here can accumulate worker processes/file descriptors on Linux
        # and trigger "Too many open files" under long runs.
        if (not reset_model) and (int(train_workers) > 0 or int(val_workers) > 0):
            self.logger.warning(
                "KnockoffNets online retraining forcing num_workers=0 to avoid FD exhaustion "
                "(requested train=%d, val=%d)",
                int(train_workers),
                int(val_workers),
            )
            train_workers = 0
            val_workers = 0

        loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            **pool_loader_kwargs(device, {"num_workers": int(train_workers)}),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_batch_size,
            shuffle=False,
            **pool_loader_kwargs(device, {"num_workers": int(val_workers)}),
        )
        opt_params = sub_config.get("optimizer", {})
        
        model = None
        if not reset_model:
            model = state.attack_state.get(store_key)
        if model is None:
            width_mult = int(sub_config.get("width_mult", 1))
            dropout_prob = float(sub_config.get("dropout_prob", 0.0))
            model = create_substitute(
                arch=sub_config.get("arch", "resnet18"),
                num_classes=self.num_classes,
                input_channels=state.metadata.get("input_shape", (3, 32, 32))[0],
                width_mult=width_mult,
                dropout_prob=dropout_prob,
            ).to(device)

        output_mode = self.config.get("output_mode", "soft_prob")
        norm_mean, norm_std = self._get_normalization(state, device)

        def preprocess_fn(x_batch: torch.Tensor) -> torch.Tensor:
            return (x_batch - norm_mean) / norm_std

        def loss_fn(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            if output_mode == "soft_prob":
                targets = torch.clamp(targets, min=1e-10)
                targets = targets / targets.sum(dim=1, keepdim=True)
                log_probs = torch.log_softmax(outputs, dim=1)
                return nn.KLDivLoss(reduction="batchmean")(log_probs, targets)

            return nn.CrossEntropyLoss()(outputs, targets.long())

        def eval_fn(model_local: nn.Module, loader_local: DataLoader) -> float:
            model_local.eval()
            total_loss = 0.0
            total_count = 0
            with torch.no_grad():
                for x_val_b, y_val_b in loader_local:
                    x_val_b = x_val_b.to(device)
                    y_val_b = y_val_b.to(device)
                    outputs = model_local(preprocess_fn(x_val_b))
                    if output_mode == "soft_prob":
                        y_val_b = torch.clamp(y_val_b, min=1e-10)
                        y_val_b = y_val_b / y_val_b.sum(dim=1, keepdim=True)
                        loss = nn.KLDivLoss(reduction="batchmean")(
                            torch.log_softmax(outputs, dim=1), y_val_b
                        )
                    else:
                        loss = nn.CrossEntropyLoss()(outputs, y_val_b.long())
                    total_loss += float(loss.item()) * int(x_val_b.size(0))
                    total_count += int(x_val_b.size(0))
            return total_loss / max(1, total_count)

        train_config = dict(sub_config)
        optimizer_config = dict(train_config.get("optimizer", {}))
        optimizer_config.setdefault("name", "sgd")
        optimizer_config.setdefault("lr", self.paper_train_lr)
        optimizer_config.setdefault("momentum", self.paper_train_momentum)
        train_config["optimizer"] = optimizer_config
        train_config["max_epochs"] = int(sub_config.get("max_epochs", epochs))
        train_config["patience"] = int(sub_config.get("patience", 20))
        trainer = SubstituteTrainer(train_config, device=device, logger=self.logger)
        steps_per_epoch = max(1, int(math.ceil(train_size / max(1, train_batch_size))))
        request = TrainRequest(
            model=model,
            train_loader=loader,
            val_loader=val_loader,
            eval_fn=eval_fn,
            loss_fn=loss_fn,
            preprocess_fn=preprocess_fn,
            max_steps=int(train_config["max_epochs"]) * steps_per_epoch,
            validate_every=steps_per_epoch,
            patience=int(train_config["patience"]) * steps_per_epoch,
            early_stop_mode="min",
            load_best=True,
        )
        trainer.train(request)

        state.attack_state[store_key] = model

    def _get_normalization(self, state: BenchmarkState, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        # Benchmark scaling unification (DFME-style): inputs are in [0,1] and we do NOT
        # apply dataset mean/std normalization inside attacks. KnockoffNets paper
        # (knockoffnets.pdf) typically follows dataset preprocessing; we deviate
        # intentionally for benchmark-wide consistency under the oracle contract.
        input_shape = state.metadata.get("input_shape", (3, 32, 32))
        channels = int(input_shape[0])
        norm_mean = torch.zeros((1, channels, 1, 1), device=device)
        norm_std = torch.ones((1, channels, 1, 1), device=device)
        return norm_mean, norm_std

    def _finalize_attack(self, state: BenchmarkState) -> None:
        if len(state.attack_state["query_data_x"]) == 0:
            return

        self.logger.debug("Final offline retraining for KnockoffNets...")
        sub_config = state.metadata.get("substitute_config", {})
        default_epochs = int(sub_config.get("max_epochs", self.train_epochs))
        offline_epochs = self.offline_train_epochs or default_epochs
        self._train_substitute(
            state,
            reset_model=True,
            epochs=offline_epochs,
            store_key="offline_substitute",
        )
        state.attack_state["substitute"] = state.attack_state.get("offline_substitute")
        self._evaluate_current_substitute(state.attack_state.get("substitute"), state.metadata.get("device", "cpu"))
