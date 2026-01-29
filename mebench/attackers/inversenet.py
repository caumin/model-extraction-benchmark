"""InverseNet attack implementation."""

from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from mebench.attackers.base import BaseAttack
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.data.loaders import create_dataloader
from mebench.models.substitute_factory import create_substitute
from mebench.models.inversion import InversionGenerator


class InverseNet(BaseAttack):
    """InverseNet with minimal inversion and retraining pipeline."""

    def __init__(self, config: dict, state: BenchmarkState):
        super().__init__(config, state)

        self.batch_size = int(config.get("batch_size", 128))
        self.num_classes = int(
            state.metadata.get("num_classes")
            or config.get("num_classes")
            or state.metadata.get("dataset_config", {}).get("num_classes", 10)
        )
        self.inversion_lr = float(config.get("inversion_lr", 1e-3))
        self.substitute_lr = float(config.get("substitute_lr", 0.01))
        self.inversion_epochs = int(config.get("inversion_epochs", 5))
        self.substitute_epochs = int(config.get("substitute_epochs", 5))
        self.phase_ratios = config.get("phase_ratios", [0.45, 0.45, 0.1])
        # Paper: train twice only (end of phase 1, end of phase 3)
        self.train_phase_1 = False
        self.train_phase_3 = False
        
        # Paper commonly uses top-1 truncation.
        self.truncation_k = int(config.get("truncation_k", 1))
        max_pool_eval = config.get("max_pool_eval")
        self.max_pool_eval = None if max_pool_eval is None else int(max_pool_eval)
        self.coreset_seed = int(config.get("coreset_seed", 20))
        self.hcss_xi = float(config.get("hcss_xi", 0.02))

        self.pool_dataset = None
        self.inversion_model: nn.Module | None = None
        self.inversion_optimizer: torch.optim.Optimizer | None = None
        self.substitute: nn.Module | None = None
        self.substitute_optimizer: torch.optim.Optimizer | None = None

        self._initialize_state(state)

    def _initialize_state(self, state: BenchmarkState) -> None:
        state.attack_state["query_data_x"] = []
        state.attack_state["query_data_y"] = []
        state.attack_state["inversion_x"] = []
        state.attack_state["inversion_y"] = []
        state.attack_state["phase"] = 1
        state.attack_state["substitute"] = None
        state.attack_state["coreset_centers"] = []
        state.attack_state["inversion_trained"] = False

    def _get_dataset_config(self, state: BenchmarkState) -> dict:
        dataset_config = self.config.get("attack", {}).get("dataset")
        if not dataset_config:
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

    def _update_phase(self, state: BenchmarkState) -> None:
        total_budget = int(state.metadata.get("max_budget", 0))
        if total_budget <= 0:
            total_budget = int(self.config.get("total_budget", 10000))

        phase1 = int(self.phase_ratios[0] * total_budget)
        phase2 = int(self.phase_ratios[1] * total_budget)
        if state.query_count < phase1:
            state.attack_state["phase"] = 1
        elif state.query_count < phase1 + phase2:
            state.attack_state["phase"] = 2
        else:
            state.attack_state["phase"] = 3

    def propose(self, k: int, state: BenchmarkState) -> QueryBatch:
        if self.pool_dataset is None:
            self._load_pool(state)

        self._update_phase(state)
        phase = state.attack_state["phase"]

        if phase == 3:
            self._ensure_inversion_trained_for_phase3(state)

        if phase == 3 and self.inversion_model is not None:
            # Paper Phase 3: "Average" samples per class.
            # Generate synthetic vectors: one-hot (1.0) and variations (0.9, 0.8)
            # instead of sampling from historical victim outputs.
            device = state.metadata.get("device", "cpu")
            
            # 1. Define confidence levels for diversity
            confidences = [1.0, 0.9, 0.8]
            templates = []
            
            # 2. Generate target vectors for each class
            for c in range(self.num_classes):
                for conf in confidences:
                    y = torch.zeros(self.num_classes, device=device)
                    # Paper uses truncated vectors (m=1): non-top entries are zeros.
                    # We vary the top entry value to generate multiple inversions per class.
                    y[c] = float(conf)
                    templates.append(y)
            
            # 3. Sample from these templates to fill batch k
            templates = torch.stack(templates)
            idx = torch.randint(0, templates.size(0), (k,))
            y_sample = templates[idx]

            with torch.no_grad():
                x = self.inversion_model(y_sample)
            
            # Paper Phase 3: augmentation happens before victim re-query.
            x = self._augment_inversion(x)
            meta = {"phase": phase, "synthetic": True, "augmented": True}
            return QueryBatch(x=x, meta=meta)

        if len(self.pool_dataset) == 0:
            input_shape = state.metadata.get("input_shape", (3, 32, 32))
            x = torch.randn(k, *input_shape)
            return QueryBatch(x=x, meta={"phase": phase, "pool_exhausted": True})

        indices = self._select_phase_indices(k, state, phase)
        x_list = [self.pool_dataset[idx][0] for idx in indices]
        if len(x_list) < k:
            x_list.extend([torch.randn_like(x_list[0]) for _ in range(k - len(x_list))])
        x = torch.stack(x_list)
        meta = {"indices": indices, "phase": phase}
        return QueryBatch(x=x, meta=meta)

    def observe(
        self,
        query_batch: QueryBatch,
        oracle_output: OracleOutput,
        state: BenchmarkState,
    ) -> None:
        if oracle_output.kind != "soft_prob":
            raise ValueError("InverseNet requires soft_prob output mode")

        state.attack_state["query_data_x"].append(query_batch.x.detach().cpu())
        state.attack_state["query_data_y"].append(oracle_output.y.detach().cpu())
        self._update_phase(state)
        phase = state.attack_state["phase"]

        # Train substitute at end of Phase 1
        if phase == 2 and not self.train_phase_1:
            self._train_substitute_from_queries(state)
            self.train_phase_1 = True

        if phase == 2:
            state.attack_state["inversion_x"].append(query_batch.x.detach().cpu())
            trunc = self._truncate_logits(oracle_output.y.detach().cpu())
            state.attack_state["inversion_y"].append(trunc)
            # Paper: train G_V once after Phase 2 completion.

        if phase == 3:
            self._train_substitute_on_batch(query_batch.x, oracle_output.y, state)
            # Phase 3 is retraining loop.
            
        # Removed periodic training to match paper protocol (2-stage training)

    def _train_inversion(self, state: BenchmarkState) -> None:
        if self.inversion_model is None:
            device = state.metadata.get("device", "cpu")
            self.inversion_model = InversionGenerator(
                num_classes=self.num_classes,
                output_channels=state.metadata.get("input_shape", (3, 32, 32))[0],
                base_channels=int(self.config.get("base_channels", 64)),
                output_activation=self.config.get("output_activation", "sigmoid"),
                output_size=state.metadata.get("input_shape", (3, 32, 32))[1],
            ).to(device)
            self.inversion_optimizer = torch.optim.Adam(
                self.inversion_model.parameters(), lr=self.inversion_lr
            )

        x_list = state.attack_state["inversion_x"]
        y_list = state.attack_state["inversion_y"]
        if len(x_list) == 0:
            return

        x_all = torch.cat(x_list, dim=0)
        y_all = torch.cat(y_list, dim=0)
        dataset = torch.utils.data.TensorDataset(x_all, y_all)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        device = state.metadata.get("device", "cpu")
        self.inversion_model.train()
        epochs = max(1, int(self.inversion_epochs))
        for _ in range(epochs):
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                self.inversion_optimizer.zero_grad()
                recon = self.inversion_model(y_batch)
                loss = F.mse_loss(recon, x_batch)
                loss.backward()
                self.inversion_optimizer.step()

        state.attack_state["inversion_trained"] = True

    def _ensure_inversion_trained_for_phase3(self, state: BenchmarkState) -> None:
        if state.attack_state.get("inversion_trained") is True:
            return

        # Train once (paper) using accumulated Phase 2 data.
        if len(state.attack_state.get("inversion_x", [])) == 0:
            return
        if len(state.attack_state.get("inversion_y", [])) == 0:
            return

        self._train_inversion(state)

    def _train_substitute_on_batch(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        state: BenchmarkState,
    ) -> None:
        """Phase 3: train substitute directly on victim re-query outputs.

        Paper Phase 3 requires: synthetic sample -> (augment) -> victim re-query -> retrain substitute
        using the victim outputs. We keep the existing optimizer/LR/epochs configuration.
        """
        if self.substitute is None:
            device = state.metadata.get("device", "cpu")
            self.substitute = create_substitute(
                arch=self.config.get("substitute_arch", "resnet18"),
                num_classes=self.num_classes,
                input_channels=state.metadata.get("input_shape", (3, 32, 32))[0],
            ).to(device)
            self.substitute_optimizer = torch.optim.SGD(
                self.substitute.parameters(), lr=self.substitute_lr, momentum=0.9
            )

        device = state.metadata.get("device", "cpu")
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        self.substitute.train()
        self.substitute_optimizer.zero_grad()
        logits = self.substitute(x_batch)
        log_probs = F.log_softmax(logits, dim=1)
        loss = F.kl_div(log_probs, y_batch, reduction="batchmean")
        loss.backward()
        self.substitute_optimizer.step()

        state.attack_state["substitute"] = self.substitute

    def _truncate_logits(self, probs: torch.Tensor) -> torch.Tensor:
        if probs.ndim == 1:
            probs = probs.unsqueeze(0)
        k = min(self.truncation_k, probs.size(1))
        topk = torch.topk(probs, k=k, dim=1)
        mask = torch.zeros_like(probs)
        mask.scatter_(1, topk.indices, 1.0)
        truncated = probs * mask
        truncated = truncated / (truncated.sum(dim=1, keepdim=True) + 1e-8)
        return truncated

    def _augment_inversion(self, x: torch.Tensor) -> torch.Tensor:
        # Custom Gaussian Noise transform
        class GaussianNoise:
            def __init__(self, mean=0.0, std=0.1):
                self.mean = mean
                self.std = std
                
            def __call__(self, img):
                noise = torch.randn_like(img) * self.std + self.mean
                return torch.clamp(img + noise, 0.0, 1.0)
        
        aug = transforms.Compose(
            [
                transforms.RandomResizedCrop(x.shape[-2:], scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10, scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
                transforms.RandomApply([GaussianNoise(mean=0.0, std=0.05)], p=0.3),
            ]
        )
        return aug(x)

    def _train_substitute_from_queries(self, state: BenchmarkState) -> None:
        query_x = state.attack_state["query_data_x"]
        query_y = state.attack_state["query_data_y"]
        if len(query_x) == 0:
            return

        x_all = torch.cat(query_x, dim=0)
        y_all = torch.cat(query_y, dim=0)

        class QueryDataset(torch.utils.data.Dataset):
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def __len__(self):
                return len(self.x)

            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]

        dataset = QueryDataset(x_all, y_all)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        device = state.metadata.get("device", "cpu")
        if self.substitute is None:
            self.substitute = create_substitute(
                arch=self.config.get("substitute_arch", "resnet18"),
                num_classes=self.num_classes,
                input_channels=state.metadata.get("input_shape", (3, 32, 32))[0],
            ).to(device)
            self.substitute_optimizer = torch.optim.SGD(
                self.substitute.parameters(), lr=self.substitute_lr, momentum=0.9
            )

        self.substitute.train()
        epochs = max(1, int(self.substitute_epochs))
        for _ in range(epochs):
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                self.substitute_optimizer.zero_grad()
                logits = self.substitute(x_batch)
                log_probs = F.log_softmax(logits, dim=1)
                loss = F.kl_div(log_probs, y_batch, reduction="batchmean")
                loss.backward()
                self.substitute_optimizer.step()

        state.attack_state["substitute"] = self.substitute

    def _select_phase_indices(self, k: int, state: BenchmarkState, phase: int) -> List[int]:
        if len(self.pool_dataset) == 0:
            return []

        # Phase 2 (HCSS): Remove coreset overlap as per paper
        if phase == 2:
            all_indices = np.arange(len(self.pool_dataset))
            coreset_centers = state.attack_state.get("coreset_centers", [])
            
            if len(coreset_centers) > 0:
                available = np.setdiff1d(all_indices, np.array(coreset_centers))
            else:
                available = all_indices
                
            max_eval = self.max_pool_eval
            if max_eval is None or max_eval <= 0:
                candidate_count = len(available)
            else:
                candidate_count = min(len(available), max_eval)
            if candidate_count == 0:
                candidates = []
            else:
                candidates = np.random.choice(available, candidate_count, replace=False).tolist()

            substitute = state.attack_state.get("substitute")
            if substitute is None:
                return candidates[: min(k, len(candidates))]
            return self._hcss_select(k, candidates, substitute)

        # Phase 1 (Coreset) or others: Select from full pool
        max_eval = self.max_pool_eval
        if max_eval is None or max_eval <= 0:
            candidate_count = len(self.pool_dataset)
        else:
            candidate_count = min(len(self.pool_dataset), max_eval)
        candidates = np.random.choice(len(self.pool_dataset), candidate_count, replace=False).tolist()

        if phase == 1:
            return self._coreset_select(k, candidates, state)
            
        return candidates[: min(k, len(candidates))]

    def _coreset_select(self, k: int, candidates: List[int], state: BenchmarkState) -> List[int]:
        centers = state.attack_state.get("coreset_centers", [])
        if len(centers) == 0:
            seed_count = min(self.coreset_seed, len(candidates))
            centers = np.random.choice(candidates, seed_count, replace=False).tolist()

        remaining = [idx for idx in candidates if idx not in centers]
        if len(remaining) == 0:
            return centers[:k]

        selected = []
        for _ in range(min(k, len(remaining))):
            max_dist = -1.0
            best_idx = None
            for idx in remaining:
                img, _ = self.pool_dataset[idx]
                x = img.view(-1)
                min_dist = None
                for c_idx in centers:
                    c_img, _ = self.pool_dataset[c_idx]
                    c_x = c_img.view(-1)
                    dist = torch.norm(x - c_x, p=1).item()
                    if min_dist is None or dist < min_dist:
                        min_dist = dist
                if min_dist is not None and min_dist > max_dist:
                    max_dist = min_dist
                    best_idx = idx

            if best_idx is None:
                break
            selected.append(best_idx)
            centers.append(best_idx)
            remaining.remove(best_idx)

        state.attack_state["coreset_centers"] = centers
        return selected

    def _hcss_select(
        self, k: int, candidates: List[int], substitute: nn.Module
    ) -> List[int]:
        device = next(substitute.parameters()).device
        substitute.eval()
        scores = []
        for idx in candidates:
            img, _ = self.pool_dataset[idx]
            x = img.unsqueeze(0).to(device)
            perturb = self._deepfool_distance(substitute, x, overshoot=self.hcss_xi)
            score = (1.0 + self.hcss_xi) * perturb
            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scores[: min(k, len(scores))]]

    def _deepfool_distance(
        self,
        model: nn.Module,
        x: torch.Tensor,
        max_iter: int = 20,
        overshoot: float = 0.02,
    ) -> float:
        x_adv = x.clone().detach()
        perturbation = torch.zeros_like(x_adv)

        with torch.no_grad():
            logits = model(x_adv)
            num_classes = logits.shape[1]
            current = logits.argmax(dim=1).item()

        for _ in range(max_iter):
            x_adv = (x + perturbation).clone().detach().requires_grad_(True)
            logits = model(x_adv)
            current = logits.argmax(dim=1).item()

            grad_current = torch.autograd.grad(
                logits[0, current],
                x_adv,
                retain_graph=True,
                create_graph=False,
            )[0]

            min_r = None
            best_w = None
            for k in range(num_classes):
                if k == current:
                    continue
                grad_k = torch.autograd.grad(
                    logits[0, k],
                    x_adv,
                    retain_graph=True,
                    create_graph=False,
                )[0]
                w_k = grad_k - grad_current
                f_k = logits[0, k] - logits[0, current]
                denom = torch.norm(w_k)
                if denom.item() == 0:
                    continue
                r_k = torch.abs(f_k) / denom
                if min_r is None or r_k.item() < min_r:
                    min_r = r_k.item()
                    best_w = w_k

            if best_w is None:
                break

            r_i = (min_r + 1e-6) * best_w / torch.norm(best_w)
            perturbation = perturbation + r_i
            x_adv = x + (1 + overshoot) * perturbation

            with torch.no_grad():
                new_pred = model(x_adv).argmax(dim=1).item()
            if new_pred != current:
                break

        return perturbation.view(-1).norm().item()
