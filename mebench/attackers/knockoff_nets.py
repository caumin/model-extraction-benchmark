"""Knockoff Nets attack implementation."""

from typing import Dict, Any, List
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

from mebench.attackers.base import BaseAttack
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.data.loaders import create_dataloader
from mebench.models.substitute_factory import create_substitute


class KnockoffNets(BaseAttack):
    """Knockoff Nets with a simple gradient-bandit policy."""

    def __init__(self, config: dict, state: BenchmarkState):
        super().__init__(config, state)

        self.batch_size = int(config.get("batch_size", 128))
        # Update substitute periodically or every batch if policy requires fresh logits
        self.train_every = int(config.get("train_every", self.batch_size))
        self.train_epochs = int(config.get("train_epochs", 1))
        self.reward_window = int(config.get("reward_window", 100))
        self.reward_certainty_weight = float(config.get("reward_certainty_weight", 1.0))
        self.reward_diversity_weight = float(config.get("reward_diversity_weight", 1.0))
        self.reward_loss_weight = float(config.get("reward_loss_weight", 1.0))
        self.coarse_clusters = int(config.get("coarse_clusters", 30))
        self.samples_per_class = int(config.get("samples_per_class", 50))
        self.kmeans_iters = int(config.get("kmeans_iters", 100))
        self.kmeans_tol = float(config.get("kmeans_tol", 1e-4))
        self.feature_arch = config.get("feature_arch", "resnet50")
        self.policy_lr = float(config.get("policy_lr", 0.01))
        self.num_classes = int(
            state.metadata.get("num_classes")
            or config.get("num_classes")
            or state.metadata.get("dataset_config", {}).get("num_classes", 10)
        )

        self.pool_dataset = None
        self.class_to_indices: Dict[int, List[int]] = {}

        self._initialize_state(state)

    def _initialize_state(self, state: BenchmarkState) -> None:
        state.attack_state["queried_indices"] = []
        state.attack_state["unqueried_indices"] = []
        state.attack_state["policy_weights"] = torch.zeros(self.num_classes)
        state.attack_state["coarse_policy_weights"] = torch.zeros(0)
        state.attack_state["reward_baseline"] = 0.0
        state.attack_state["recent_victim_probs"] = deque(maxlen=self.reward_window)
        state.attack_state["query_data_x"] = []
        state.attack_state["query_data_y"] = []
        state.attack_state["substitute"] = None
        state.attack_state["query_count"] = 0
        state.attack_state["class_to_coarse"] = {}
        state.attack_state["coarse_to_classes"] = {}

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

        if hasattr(self.pool_dataset, "classes"):
            self.num_classes = len(self.pool_dataset.classes)
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
            sample_count = min(self.samples_per_class, len(indices))
            sampled = np.random.choice(indices, sample_count, replace=False).tolist()

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

    def propose(self, k: int, state: BenchmarkState) -> QueryBatch:
        if self.pool_dataset is None:
            self._load_pool(state)

        unqueried = state.attack_state["unqueried_indices"]
        if len(unqueried) == 0:
            x = torch.randn(k, *state.metadata.get("input_shape", (3, 32, 32)))
            meta = {"indices": [], "classes": [], "pool_exhausted": True}
            return QueryBatch(x=x, meta=meta)

        weights = state.attack_state["policy_weights"].clone().float()
        probs = torch.softmax(weights, dim=0).cpu().numpy()
        class_ids = list(range(len(probs)))

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
            x_list.extend([torch.randn_like(x_list[0]) for _ in range(k - len(x_list))])

        x = torch.stack(x_list)
        meta = {"indices": selected_indices, "classes": selected_classes}
        return QueryBatch(x=x, meta=meta)

    def observe(
        self,
        query_batch: QueryBatch,
        oracle_output: OracleOutput,
        state: BenchmarkState,
    ) -> None:
        state.attack_state["query_data_x"].append(query_batch.x.detach().cpu())
        state.attack_state["query_data_y"].append(oracle_output.y.detach().cpu())
        state.attack_state["query_count"] += query_batch.x.shape[0]

        if oracle_output.kind == "soft_prob":
            probs = oracle_output.y.detach().cpu()
        else:
            probs = F.one_hot(oracle_output.y, num_classes=self.num_classes).float().cpu()

        recent_probs = state.attack_state["recent_victim_probs"]
        for row in probs:
            recent_probs.append(row)

        top2 = torch.topk(probs, k=2, dim=1).values
        # Certainty reward (paper): margin between top-1 and top-2 probabilities.
        certainty_reward = top2[:, 0] - top2[:, 1]

        if len(recent_probs) > 0:
            mean_recent = torch.stack(list(recent_probs)).mean(dim=0)
            # Diversity reward: encourage dissimilarity to recent average posterior.
            diversity_reward = 1.0 - F.cosine_similarity(probs, mean_recent.unsqueeze(0).expand_as(probs), dim=1)
        else:
            diversity_reward = torch.zeros(probs.size(0))

        substitute = state.attack_state.get("substitute")
        if substitute is not None:
            substitute.eval()
            with torch.no_grad():
                logits = substitute(query_batch.x.to(state.metadata.get("device", "cpu")))
                log_probs = F.log_softmax(logits, dim=1).cpu()
                loss_reward = -(probs * log_probs).sum(dim=1)
        else:
            loss_reward = torch.zeros(probs.size(0))

        rewards = (
            self.reward_certainty_weight * certainty_reward
            + self.reward_diversity_weight * diversity_reward
            + self.reward_loss_weight * loss_reward
        )

        baseline = float(state.attack_state["reward_baseline"])
        reward_mean = float(rewards.mean().item()) if rewards.numel() > 0 else 0.0
        baseline = 0.9 * baseline + 0.1 * reward_mean
        state.attack_state["reward_baseline"] = baseline

        classes = query_batch.meta.get("classes", [])
        weights = state.attack_state["policy_weights"].clone().float()
        coarse_weights = state.attack_state["coarse_policy_weights"].clone().float()
        class_to_coarse = state.attack_state["class_to_coarse"]
        # Gradient bandit update: w_i <- w_i + alpha * (r - b) * (1[i=a] - pi_i)
        pi = torch.softmax(weights, dim=0)
        for idx, class_id in enumerate(classes):
            if class_id < 0 or class_id >= weights.numel():
                continue
            adv = float(rewards[idx]) - baseline
            grad = -pi
            grad[class_id] = 1.0 - pi[class_id]
            weights = weights + self.policy_lr * adv * grad

            pi = torch.softmax(weights, dim=0)

            if class_id in class_to_coarse and coarse_weights.numel() > 0:
                coarse_id = int(class_to_coarse[class_id])
                if 0 <= coarse_id < coarse_weights.numel():
                    coarse_pi = torch.softmax(coarse_weights, dim=0)
                    coarse_grad = -coarse_pi
                    coarse_grad[coarse_id] = 1.0 - coarse_pi[coarse_id]
                    coarse_weights = coarse_weights + self.policy_lr * adv * coarse_grad
        state.attack_state["policy_weights"] = weights
        if coarse_weights.numel() > 0:
            state.attack_state["coarse_policy_weights"] = coarse_weights

        if state.attack_state["query_count"] % self.train_every == 0:
            self._train_substitute(state)

    def _train_substitute(self, state: BenchmarkState) -> None:
        query_x = state.attack_state["query_data_x"]
        query_y = state.attack_state["query_data_y"]
        if len(query_x) == 0:
            return

        x_all = torch.cat(query_x, dim=0)
        y_all = torch.cat(query_y, dim=0)

        class QueryDataset(torch.utils.data.Dataset):
            def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
                self.x = x
                self.y = y

            def __len__(self) -> int:
                return self.x.shape[0]

            def __getitem__(self, idx: int):
                return self.x[idx], self.y[idx]

        dataset = QueryDataset(x_all, y_all)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        device = state.metadata.get("device", "cpu")
        sub_config = state.metadata.get("substitute_config", {})
        opt_params = sub_config.get("optimizer", {})
        
        model = create_substitute(
            arch=sub_config.get("arch", "resnet18"),
            num_classes=self.num_classes,
            input_channels=state.metadata.get("input_shape", (3, 32, 32))[0],
        ).to(device)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(opt_params.get("lr", 0.01)),
            momentum=float(opt_params.get("momentum", 0.9)),
            weight_decay=float(opt_params.get("weight_decay", 5e-4))
        )
        output_mode = self.config.get("output_mode", "soft_prob")
        if output_mode == "soft_prob":
            criterion = nn.KLDivLoss(reduction="batchmean")
        else:
            criterion = nn.CrossEntropyLoss()

        model.train()
        victim_config = state.metadata.get("victim_config", {})
        normalization = victim_config.get("normalization")
        if normalization is None:
            normalization = {"mean": [0.0], "std": [1.0]}
            
        norm_mean = torch.tensor(normalization["mean"]).view(1, -1, 1, 1).to(device)
        norm_std = torch.tensor(normalization["std"]).view(1, -1, 1, 1).to(device)
        
        for _ in range(self.train_epochs):
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(device)
                # Normalize images for substitute
                x_batch = (x_batch - norm_mean) / norm_std
                
                optimizer.zero_grad()
                if output_mode == "soft_prob":
                    y_batch = y_batch.to(device)
                    # Clip probabilities to avoid log(0)
                    y_batch = torch.clamp(y_batch, min=1e-10)
                    y_batch = y_batch / y_batch.sum(dim=1, keepdim=True)
                    logits = model(x_batch)
                    log_probs = torch.log_softmax(logits, dim=1)
                    loss = criterion(log_probs, y_batch)
                else:
                    y_batch = y_batch.long().to(device)
                    logits = model(x_batch)
                    loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

        state.attack_state["substitute"] = model
