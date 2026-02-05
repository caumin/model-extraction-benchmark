"""InverseNet attack implementation."""

from typing import Dict, Any, List, Tuple, Optional
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from tqdm import tqdm

from mebench.attackers.runner import AttackRunner
from mebench.core.context import BenchmarkContext
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.data.loaders import create_dataloader
from mebench.models.substitute_factory import create_substitute
from mebench.training import SubstituteTrainer, TrainRequest
from mebench.models.inversion import InversionGenerator


class InverseNet(AttackRunner):
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
        self.coreset_seed = int(config.get("coreset_seed", 20))
        self.hcss_xi = float(config.get("hcss_xi", 0.02))
        self.hcss_step_size = float(config.get("hcss_step_size", 0.01))
        self.hcss_max_iter = int(config.get("hcss_max_iter", 20))

        self.pool_dataset = None
        self.inversion_model: nn.Module | None = None
        self.inversion_optimizer: torch.optim.Optimizer | None = None
        self.substitute: nn.Module | None = None
        self.substitute_optimizer: torch.optim.Optimizer | None = None

        self._initialize_state(state)

    def run(self, ctx: BenchmarkContext) -> None:
        self.victim = ctx.oracle.model
        device = self.state.metadata.get("device", "cpu")
        total_budget = self.state.budget_remaining
        pbar = tqdm(total=total_budget, desc="[InverseNet] Extracting")
        
        while ctx.budget_remaining > 0:
            step_size = self._default_step_size(ctx)
            x_query, meta = self._select_query_batch(step_size, self.state)
            oracle_output = ctx.query(x_query, meta=meta)
            self._handle_oracle_output(x_query, meta, oracle_output, self.state)
            pbar.update(x_query.size(0))
        pbar.close()

    def _select_query_batch(self, k: int, state: BenchmarkState) -> tuple[torch.Tensor, dict]:
        if self.pool_dataset is None:
            self._load_pool(state)

        self._update_phase(state)
        phase = state.attack_state["phase"]

        if phase == 3:
            self._ensure_inversion_trained_for_phase3(state)

        if phase == 3 and self.inversion_model is not None:
            device = state.metadata.get("device", "cpu")

            confidences = [1.0, 0.9, 0.8]
            templates = []

            for c in range(self.num_classes):
                for conf in confidences:
                    y = torch.zeros(self.num_classes, device=device)
                    y[c] = float(conf)
                    templates.append(y)

            templates = torch.stack(templates)
            idx = torch.randint(0, templates.size(0), (k,))
            y_sample = templates[idx]

            with torch.no_grad():
                x = self.inversion_model(y_sample)

            x = self._augment_inversion(x, y_sample)
            meta = {"phase": phase, "synthetic": True, "augmented": True}
            return x, meta

        if len(self.pool_dataset) == 0:
            raise ValueError(
                f"Query pool exhausted for {self.__class__.__name__}. "
                f"Cannot select {k} more queries."
            )

        indices = self._select_phase_indices(k, state, phase)
        x_list = [self.pool_dataset[idx][0] for idx in indices]
        if len(x_list) < k:
            raise ValueError(
                f"Query pool exhausted for {self.__class__.__name__}. "
                f"Requested {k}, found {len(x_list)}."
            )
        x = torch.stack(x_list)
        meta = {"indices": indices, "phase": phase}
        return x, meta

    def _handle_oracle_output(
        self,
        x_query: torch.Tensor,
        meta: dict,
        oracle_output: OracleOutput,
        state: BenchmarkState,
    ) -> None:
        if oracle_output.kind == "soft_prob":
            victim_probs = oracle_output.y.detach().cpu()
            query_targets = victim_probs
        else:
            victim_labels = oracle_output.y.detach().cpu().long()
            victim_probs = F.one_hot(victim_labels, num_classes=self.num_classes).float()
            query_targets = victim_labels

        state.attack_state["query_data_x"].append(x_query.detach().cpu())
        state.attack_state["query_data_y"].append(query_targets)
        self._update_phase(state)
        phase = state.attack_state["phase"]

        if phase == 2 and not self.train_phase_1:
            self._train_substitute_from_queries(state)
            self.train_phase_1 = True

        if phase == 2:
            state.attack_state["inversion_x"].append(x_query.detach().cpu())
            trunc = self._truncate_logits(victim_probs)
            state.attack_state["inversion_y"].append(trunc)

        if phase == 3:
            if oracle_output.kind == "soft_prob":
                targets = oracle_output.y
            else:
                targets = oracle_output.y.long()
            self._train_substitute_on_batch(x_query, targets, state)

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
        sub_config = state.metadata.get("substitute_config", {})
        train_batch_size = int(
            sub_config.get("batch_size")
            or sub_config.get("trackA", {}).get("batch_size", self.batch_size)
        )
        loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, num_workers=0)

        device = state.metadata.get("device", "cpu")
        self.inversion_model.train()
        epochs = max(1, int(self.inversion_epochs))
        inv_pbar = tqdm(range(epochs), desc="[InverseNet] Training Inversion", leave=False)
        for _ in inv_pbar:
            epoch_loss = 0.0
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                self.inversion_optimizer.zero_grad()
                recon = self.inversion_model(y_batch)
                loss = F.mse_loss(recon, x_batch)
                loss.backward()
                self.inversion_optimizer.step()
                epoch_loss += loss.item()
            inv_pbar.set_postfix({"Loss": f"{epoch_loss/len(loader):.4f}"})

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
        if y_batch.ndim == 1 or (y_batch.ndim == 2 and y_batch.size(1) == 1):
            loss = F.cross_entropy(logits, y_batch.long().view(-1))
        else:
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
        return truncated

    def _augment_inversion(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        x_aug = aug(x)

        if labels is None or x_aug.size(0) < 2:
            return x_aug

        if labels.ndim > 1:
            class_ids = labels.argmax(dim=1)
        else:
            class_ids = labels

        class_ids = class_ids.to(x_aug.device)
        x_mix = x_aug.clone()
        for class_id in class_ids.unique():
            idxs = (class_ids == class_id).nonzero(as_tuple=False).view(-1)
            if idxs.numel() < 2:
                continue
            perm = idxs[torch.randperm(idxs.numel())]
            lam = torch.rand(idxs.numel(), 1, 1, 1, device=x_aug.device)
            x_mix[idxs] = lam * x_aug[idxs] + (1.0 - lam) * x_aug[perm]

        return x_mix

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
        sub_config = state.metadata.get("substitute_config", {})
        if self.substitute is None:
            arch = sub_config.get("arch") or self.config.get("substitute_arch", "resnet18")
            width_mult = int(sub_config.get("width_mult", 1))
            dropout_prob = float(sub_config.get("dropout_prob", 0.0))
            self.substitute = create_substitute(
                arch=arch,
                num_classes=self.num_classes,
                input_channels=state.metadata.get("input_shape", (3, 32, 32))[0],
                width_mult=width_mult,
                dropout_prob=dropout_prob,
            ).to(device)
        epochs = max(1, int(self.substitute_epochs))

        def loss_fn(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            if targets.ndim == 1 or (targets.ndim == 2 and targets.size(1) == 1):
                return F.cross_entropy(outputs, targets.long().view(-1))
            log_probs = F.log_softmax(outputs, dim=1)
            return F.kl_div(log_probs, targets, reduction="batchmean")

        train_config = dict(sub_config)
        train_config["max_epochs"] = int(sub_config.get("max_epochs", epochs))
        trainer = SubstituteTrainer(train_config, device=device, logger=self.logger)
        request = TrainRequest(
            model=self.substitute,
            train_loader=loader,
            loss_fn=loss_fn,
            load_best=True,
        )
        trainer.train(request)

        state.attack_state["substitute"] = self.substitute
        self.logger.info("InverseNet substitute trained from queries.")
        self._evaluate_current_substitute(self.substitute, device)

    def _select_phase_indices(self, k: int, state: BenchmarkState, phase: int) -> List[int]:
        if len(self.pool_dataset) == 0:
            return []

        # Phase 2 (HCSS): Remove coreset overlap as per paper
        if phase == 2:
            all_indices = np.arange(len(self.pool_dataset))
            coreset_centers = state.attack_state.get("coreset_centers", [])
            
            if len(coreset_centers) > 0:
                available = np.setdiff1d(all_indices, np.array(coreset_centers)).tolist()
            else:
                available = all_indices.tolist()
                
            # Use ENTIRE available pool (Strict Protocol)
            candidates = available
            if not candidates:
                return []

            substitute = state.attack_state.get("substitute")
            if substitute is None:
                return candidates[: min(k, len(candidates))]
            return self._hcss_select(k, candidates, substitute)

        # Phase 1 (Coreset) or others: Select from full pool (Strict Protocol)
        candidates = list(range(len(self.pool_dataset)))

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
        
        # [OPTIMIZATION] Batched feature extraction & distance computation
        # 1. Extract features (flattened images) for all centers and remaining candidates ONCE
        # Note: InverseNet uses raw image L1 distance as "feature"
        
        # Load all candidate images into memory (batched)
        # This is much faster than single-item access inside the loop
        subset = torch.utils.data.Subset(self.pool_dataset, remaining)
        loader = DataLoader(subset, batch_size=256, shuffle=False, num_workers=0)
        
        candidates_matrix = []
        for x_batch, _ in loader:
            candidates_matrix.append(x_batch.view(x_batch.size(0), -1))
        
        if not candidates_matrix:
            return centers[:k]
            
        # [N_remaining, D]
        candidates_matrix = torch.cat(candidates_matrix, dim=0)
        
        # Load centers
        center_subset = torch.utils.data.Subset(self.pool_dataset, centers)
        center_loader = DataLoader(center_subset, batch_size=256, shuffle=False, num_workers=0)
        
        centers_matrix_list = []
        for x_batch, _ in center_loader:
            centers_matrix_list.append(x_batch.view(x_batch.size(0), -1))
        
        # [N_centers, D]
        centers_matrix = torch.cat(centers_matrix_list, dim=0)
        
        # Move to GPU if available for fast distance computation
        device = state.metadata.get("device", "cpu")
        candidates_matrix = candidates_matrix.to(device)
        centers_matrix = centers_matrix.to(device)
        
        # Initialize min_distances with current centers
        # dists: [N_remaining, N_centers]
        # We need min_dist for each candidate: min_{c in centers} ||x - c||_1
        # L1 distance: sum(|x - y|)
        
        # Memory optimization: Compute initial min_dists in chunks if needed
        # But for typical pool sizes (e.g. 50k), we can do it iteratively to save memory
        
        # Initialize with infinity
        min_dists = torch.full((candidates_matrix.size(0),), float('inf'), device=device)
        
        # Update min_dists against existing centers
        # Chunking centers to avoid OOM [N_rem, N_cen, D] tensor
        chunk_size = 100
        for i in range(0, centers_matrix.size(0), chunk_size):
            c_chunk = centers_matrix[i:i+chunk_size] # [C, D]
            # dists: [N_rem, C]
            # L1: |x - c|
            # Expand: [N_rem, 1, D] - [1, C, D] -> [N_rem, C, D] -> sum(abs) -> [N_rem, C]
            # This is still heavy. 
            # Alternative: Iterate candidates? No, slow.
            # Efficient L1 distance matrix computation is tricky without expansion.
            # Let's simply loop over centers for initialization (N_centers is small initially)
            for j in range(c_chunk.size(0)):
                c_vec = c_chunk[j].unsqueeze(0) # [1, D]
                d = torch.norm(candidates_matrix - c_vec, p=1, dim=1) # [N_rem]
                min_dists = torch.minimum(min_dists, d)

        # Greedy selection loop
        for _ in range(min(k, len(remaining))):
            # Find candidate with MAX min_dist
            max_val, max_idx_in_matrix = torch.max(min_dists, dim=0)
            max_idx = max_idx_in_matrix.item()
            
            best_real_idx = remaining[max_idx]
            selected.append(best_real_idx)
            
            # Update centers and remove from consideration
            new_center_vec = candidates_matrix[max_idx].unsqueeze(0) # [1, D]
            
            # Update min_dists for all candidates using the new center
            # New min_dist = min(old_min_dist, dist(x, new_center))
            new_dists = torch.norm(candidates_matrix - new_center_vec, p=1, dim=1)
            min_dists = torch.minimum(min_dists, new_dists)
            
            # Effectively remove the selected one by setting its dist to -1
            min_dists[max_idx] = -1.0
            
            # Update state (strictly speaking we should append to centers list, but we reconstruct at end)
        
        # Reconstruct centers list
        final_centers = centers + selected
        state.attack_state["coreset_centers"] = final_centers
        return selected

    def _hcss_select(
        self, k: int, candidates: List[int], substitute: nn.Module
    ) -> List[int]:
        device = next(substitute.parameters()).device
        substitute.eval()
        scores = []

        batch_size = min(self.batch_size, 32)
        
        # [OPTIMIZATION] Use DataLoader for efficient batch retrieval (I/O Bound -> GPU Bound)
        # Maps candidates[i] -> dataset[candidates[i]]
        subset = Subset(self.pool_dataset, candidates)
        loader = DataLoader(
            subset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0
        )
        
        current_idx_ptr = 0
        
        # Iterate through pre-fetched batches
        for x_batch, _ in loader:
            batch_len = x_batch.size(0)
            
            # Retrieve original indices corresponding to this batch
            # Subset maintains the order of 'candidates', so we can slice directly
            batch_indices = candidates[current_idx_ptr : current_idx_ptr + batch_len]
            current_idx_ptr += batch_len
            
            x_batch = x_batch.to(device)
            
            # DeepFool distance calculation (unchanged logic)
            distances = self._hcss_noise_distance_batch(
                substitute,
                x_batch,
            )
            
            for i, dist in enumerate(distances):
                scores.append((batch_indices[i], (1.0 + self.hcss_xi) * dist.item()))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scores[: min(k, len(scores))]]

    def _hcss_noise_distance_batch(
        self,
        model: nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        device = x.device
        batch = x.shape[0]
        if batch == 0:
            return torch.empty(0, device=device)

        model.eval()
        with torch.no_grad():
            logits = model(x)
            original = logits.argmax(dim=1)

        perturb = torch.zeros_like(x, device=device)
        noise = torch.randn_like(x, device=device) * self.hcss_step_size
        active = torch.ones(batch, dtype=torch.bool, device=device)

        for _ in range(self.hcss_max_iter):
            if not active.any():
                break

            perturb[active] = perturb[active] + noise[active]
            x_adv = torch.clamp(x + perturb, 0.0, 1.0)
            with torch.no_grad():
                preds = model(x_adv).argmax(dim=1)

            active = active & (preds == original)
            if not active.any():
                break

            noise[active] = noise[active] * (1.0 + self.hcss_xi)

        return torch.norm(perturb.view(batch, -1), dim=1)
