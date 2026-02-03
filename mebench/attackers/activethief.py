"""ActiveThief attack implementation with DFAL support using vectorized DeepFool."""

from typing import Dict, Any, List, Tuple, Optional
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from mebench.attackers.runner import AttackRunner
from mebench.core.context import BenchmarkContext
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.data.loaders import create_dataloader
from mebench.models.substitute_factory import create_substitute


class ActiveThief(AttackRunner):
    """ActiveThief with uncertainty, k-center, and DFAL sampling strategies.
    
    Ref: "ActiveThief: Model Extraction Using Active Learning and Unannotated Public Data" (AAAI 2020)
    
    Algorithm loop:
    1. Initialize: Select random initial seed S0 from thief dataset
    2. Query: Send S_i to victim f to get labels D_i
    3. Train: Train substitute model f~ from scratch on all collected data ∪D_t
    4. Evaluate: Predict on remaining pool (unlabeled thief data)
    5. Select: Use active learning strategy to select next queries S_{i+1}
    6. Repeat: Continue until budget exhausted
    """

    def __init__(self, config: dict, state: BenchmarkState):
        super().__init__(config, state)

        # Basic parameters
        self.batch_size = int(config.get("batch_size", 256))
        self.num_classes = int(
            self.state.metadata.get("num_classes")
            or self.config.get("num_classes")
            or self.state.metadata.get("dataset_config", {}).get("num_classes", 10)
        )
        
        # Active learning strategy
        self.strategy = config.get("strategy", "uncertainty")
        initial_seed_size = config.get("initial_seed_size")
        self.initial_seed_size = (
            int(initial_seed_size) if initial_seed_size is not None else None
        )
        step_size = config.get("step_size")
        self.step_size = int(step_size) if step_size is not None else None
        self.rounds = int(config.get("rounds", 10))
        
        # Store oracle labels to override dataset labels during training
        self.observed_labels = {}
        
        # DeepFool parameters for DFAL
        self.dfal_max_iter = int(config.get("dfal_max_iter", 20))
        dfal_rho = config.get("dfal_rho")
        self.dfal_rho = int(dfal_rho) if dfal_rho is not None else None
        
        # Datasets and model
        self.pool_dataset = None
        self.labeled_indices = []
        self.unlabeled_indices = []
        self.initial_seed_indices = []
        self.substitute = None
        self.substitute_optimizer = None

        self._initialize_state(state)

    def _initialize_state(self, state: BenchmarkState) -> None:
        """Initialize attack-specific state."""
        state.attack_state["labeled_indices"] = []
        state.attack_state["unlabeled_indices"] = []
        state.attack_state["round"] = 0
        state.attack_state["initialized"] = False
        state.attack_state["initial_seed_indices"] = []
        state.attack_state["initial_seed_queried"] = False

    def _setup_datasets(self, state: BenchmarkState) -> None:
        """Setup pool dataset and initial seed selection."""
        if state.attack_state.get("initialized"):
            return

        if self.initial_seed_size is None:
            total_budget = int(
                state.metadata.get("max_budget")
                or self.config.get("max_budget", 0)
                or 0
            )
            rounds = max(1, int(self.rounds))
            if total_budget > 0:
                self.initial_seed_size = max(
                    1, int(math.ceil(total_budget / rounds))
                )
            elif self.step_size is not None:
                self.initial_seed_size = int(self.step_size)
            else:
                self.initial_seed_size = 50
            
        dataset_config = state.metadata.get("dataset_config", {})
        self.pool_dataset = create_dataloader(
            dataset_config,
            batch_size=self.batch_size,
            shuffle=False,
        ).dataset
        
        # Initialize labeled/unlabeled splits
        pool_size = len(self.pool_dataset)
        self.unlabeled_indices = list(range(pool_size))
        
        # Select initial random seed
        if len(self.unlabeled_indices) > self.initial_seed_size:
            seed_indices = np.random.choice(
                self.unlabeled_indices, 
                size=self.initial_seed_size, 
                replace=False
            ).tolist()
        else:
            seed_indices = self.unlabeled_indices.copy()
        
        # Move from unlabeled to labeled
        for idx in seed_indices:
            self.unlabeled_indices.remove(idx)
            self.labeled_indices.append(idx)

        self.initial_seed_indices = seed_indices

        state.attack_state["labeled_indices"] = self.labeled_indices
        state.attack_state["unlabeled_indices"] = self.unlabeled_indices
        state.attack_state["initial_seed_indices"] = seed_indices
        state.attack_state["initialized"] = True

    def _create_substitute(self, input_shape: tuple) -> nn.Module:
        """Create substitute model."""
        sub_config = self.state.metadata.get("substitute_config", {})
        arch = sub_config.get("arch", "resnet18")
        input_channels = int(input_shape[0])
        
        width_mult = int(sub_config.get("width_mult", 1))
        dropout_prob = float(sub_config.get("dropout_prob", self.config.get("dropout_prob", 0.0)))

        return create_substitute(
            arch=arch,
            num_classes=self.num_classes,
            input_channels=input_channels,
            width_mult=width_mult,
            dropout_prob=dropout_prob,
        )

    def _train_substitute(self, state: BenchmarkState) -> None:
        """Train substitute model from scratch on labeled data."""
        device = state.metadata.get("device", "cpu")
        input_shape = self.state.metadata.get("input_shape", (3, 32, 32))
        
        # Create fresh model (from scratch)
        self.substitute = self._create_substitute(input_shape).to(device)
        
        # Setup optimizer from substitute config
        sub_config = self.state.metadata.get("substitute_config", {})
        opt_config = sub_config.get("optimizer", {})
        self.substitute_optimizer = self._build_optimizer(
            self.substitute.parameters(),
            opt_config,
        )
        
        # Create custom dataset that uses observed oracle labels
        class LabeledDataset:
            def __init__(self, pool_dataset, labeled_indices, observed_labels, num_classes, logger):
                self.pool_dataset = pool_dataset
                self.labeled_indices = labeled_indices
                self.observed_labels = observed_labels
                self.num_classes = num_classes
                self.logger = logger
            
            def __len__(self):
                return len(self.labeled_indices)
            
            def __getitem__(self, idx):
                dataset_idx = self.labeled_indices[idx]
                x, _ = self.pool_dataset[dataset_idx]  # Ignore original label
                
                # Use oracle labels when available, otherwise use a temporary valid label
                # This handles the initial seed case before any oracle queries
                if dataset_idx in self.observed_labels:
                    y = self.observed_labels[dataset_idx]
                    # Validate label is in correct range
                    if y >= self.num_classes or y < 0:
                        self.logger.error(
                            "Oracle provided invalid label %s for dataset index %s",
                            y,
                            dataset_idx,
                        )
                        raise ValueError(f"Invalid oracle label {y} for {self.num_classes}-class model")
                else:
                    # Use dummy label (0) for initial seed - will be replaced after first oracle query
                    y = 0
                
                return x, y
        
        labeled_dataset = LabeledDataset(
            self.pool_dataset,
            self.labeled_indices,
            self.observed_labels,
            self.num_classes,
            self.logger,
        )
        labeled_loader = DataLoader(
            labeled_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4
        )
        
        # Training parameters
        max_epochs = int(sub_config.get("max_epochs", 1000))
        patience = int(sub_config.get("patience", 100))
        
        # Train
        self.substitute.train()
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(max_epochs):
            epoch_loss = 0.0
            batch_count = 0
            for x_batch, y_batch in labeled_loader:
                try:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    
                    self.substitute_optimizer.zero_grad()
                    logits = self.substitute(x_batch)
                    
                    # Use cross-entropy loss
                    loss = F.cross_entropy(logits, y_batch)
                    loss.backward()
                    self.substitute_optimizer.step()
                    
                    epoch_loss += loss.item()
                    batch_count += 1
                except ValueError as e:
                    # Skip samples without oracle labels
                    continue
            
            if batch_count == 0:
                self.logger.warning(
                    "No valid labeled samples available for training in epoch %s",
                    epoch,
                )
                break
            
            # Early stopping
            avg_loss = epoch_loss / len(labeled_loader)
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    def _select_uncertainty(self, probs: torch.Tensor, k: int) -> List[int]:
        """Select samples with highest entropy.
        
        Eq: H_n = -sum(y_nj * log(y_nj))
        """
        # Compute entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        _, indices = torch.topk(entropy, k)
        return indices.cpu().tolist()

    def _select_k_center(self, probs: torch.Tensor, k: int) -> List[int]:
        """Select samples using k-center greedy algorithm on probability vectors.
        
        Implements Core-Set approach adapted for probability space (Pal et al. 2020).
        """
        selected = []
        remaining = list(range(probs.shape[0]))
        
        # Initialize with a random point
        if remaining:
            first_idx = np.random.choice(remaining)
            selected.append(first_idx)
            remaining.remove(first_idx)
        
        # Initialize min_distances to infinity
        min_dists = torch.full((probs.shape[0],), float('inf'), device=probs.device)
        
        # Update distances for the first selected point
        dists = torch.norm(probs - probs[selected[0]].unsqueeze(0), dim=1)
        min_dists = torch.min(min_dists, dists)

        for _ in range(min(k - 1, len(remaining))):
            current_min_dists = min_dists[remaining]
            if len(current_min_dists) == 0:
                break
            
            max_val, max_idx_local = torch.max(current_min_dists, dim=0)
            best_idx = remaining[max_idx_local.item()]
            
            selected.append(best_idx)
            remaining.remove(best_idx)
            
            # Update min_distances for the new point
            new_dists = torch.norm(probs - probs[best_idx].unsqueeze(0), dim=1)
            min_dists = torch.min(min_dists, new_dists)
        
        return selected

    def _collect_probs(self, loader: DataLoader, device: str) -> torch.Tensor:
        """Collect softmax probabilities for a loader."""
        all_probs = []
        self.substitute.eval()
        with torch.no_grad():
            for x_batch, _ in loader:
                x_batch = x_batch.to(device)
                logits = self.substitute(x_batch)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu())
        return torch.cat(all_probs, dim=0)

    def _deepfool_distance_dfal_chunk(
        self,
        model: nn.Module,
        x: torch.Tensor,
        max_iter: int,
    ) -> torch.Tensor:
        device = x.device
        batch = x.shape[0]
        if batch == 0:
            return torch.empty(0, device=device)

        model.eval()
        perturb = torch.zeros_like(x, device=device)
        with torch.no_grad():
            logits = model(x)
            original = logits.argmax(dim=1)

        active = torch.ones(batch, dtype=torch.bool, device=device)
        for _ in range(max_iter):
            if not active.any():
                break

            x_adv = (x + perturb).detach()
            x_adv.requires_grad_(True)
            logits = model(x_adv)
            preds = logits.argmax(dim=1)
            active = active & (preds == original)
            if not active.any():
                break

            num_classes = logits.shape[1]
            grads = []
            for k in range(num_classes):
                grad_k = torch.autograd.grad(
                    logits[:, k].sum(),
                    x_adv,
                    retain_graph=(k != num_classes - 1),
                    create_graph=False,
                )[0]
                grads.append(grad_k)

            grads = torch.stack(grads, dim=1)
            grad_current = grads[torch.arange(batch), original]
            w = grads - grad_current.unsqueeze(1)
            f = logits - logits.gather(1, original.unsqueeze(1))
            w_norm = torch.norm(w.view(batch, num_classes, -1), dim=2)
            dist = torch.abs(f) / (w_norm + 1e-8)
            dist = torch.where(
                w_norm > 0,
                dist,
                torch.full_like(dist, float("inf")),
            )
            dist[torch.arange(batch), original] = float("inf")
            min_dist, best_idx = dist.min(dim=1)
            valid = torch.isfinite(min_dist)
            active = active & valid
            if not active.any():
                break

            best_w = w[torch.arange(batch), best_idx]
            best_w_norm = torch.norm(best_w.view(batch, -1), dim=1)
            r_i = (min_dist + 1e-8).view(-1, 1, 1, 1) * best_w
            r_i = r_i / (best_w_norm.view(-1, 1, 1, 1) + 1e-8)
            with torch.no_grad():
                perturb[active] = perturb[active] + r_i[active]

        return torch.norm(perturb.view(batch, -1), dim=1)

    def _deepfool_distance_dfal(
        self,
        model: nn.Module,
        x: torch.Tensor,
        max_iter: int,
        internal_batch_size: int,
    ) -> torch.Tensor:
        if x.shape[0] == 0:
            return torch.empty(0, device=x.device)

        chunk = max(1, internal_batch_size)
        distances = []
        for start in range(0, x.shape[0], chunk):
            x_chunk = x[start : start + chunk]
            distances.append(
                self._deepfool_distance_dfal_chunk(model, x_chunk, max_iter)
            )

        return torch.cat(distances, dim=0)

    def _select_dfal(self, state: BenchmarkState, k: int) -> List[int]:
        """Select samples using DeepFool Active Learning."""
        device = state.metadata.get("device", "cpu")
        unlabeled_dataset = Subset(self.pool_dataset, self.unlabeled_indices)
        unlabeled_loader = DataLoader(
            unlabeled_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        all_distances = []
        self.substitute.eval()
        
        dfal_pbar = tqdm(
            unlabeled_loader,
            total=len(unlabeled_loader),
            desc="[ActiveThief] DFAL Scoring",
            leave=False,
        )
        for x_batch, _ in dfal_pbar:
            x_batch = x_batch.to(device)
            # Enable grad for DeepFool
            with torch.enable_grad():
                distances = self._deepfool_distance_dfal(
                    self.substitute,
                    x_batch,
                    max_iter=self.dfal_max_iter,
                    internal_batch_size=min(self.batch_size, x_batch.shape[0]),
                )
            all_distances.append(distances.detach().cpu())
        
        distances = torch.cat(all_distances, dim=0)
        
        # Select k samples with smallest DeepFool distances (closest to boundary)
        _, selected_local = torch.topk(distances, k, largest=False)
        selected_indices = [self.unlabeled_indices[i] for i in selected_local.tolist()]
        
        return selected_indices

    def _select_dfal_k_center(self, state: BenchmarkState, k: int) -> List[int]:
        """DFAL pre-filtering + K-center selection."""
        # Pre-filter with DFAL to get rho candidates
        base_rho = self.dfal_rho
        if base_rho is None:
            total_budget = int(
                state.metadata.get("max_budget")
                or self.config.get("max_budget", 0)
                or 0
            )
            base_rho = total_budget if total_budget > 0 else len(self.unlabeled_indices)

        rho = max(int(base_rho), k + 1)
        rho = min(rho, len(self.unlabeled_indices))
        self.logger.info(
            "[ActiveThief] DFAL pre-filtering: rho=%s (unlabeled=%s)",
            rho,
            len(self.unlabeled_indices),
        )
        dfal_candidates = self._select_dfal(state, rho)
        
        # Create loader for candidates
        candidate_dataset = Subset(self.pool_dataset, dfal_candidates)
        candidate_loader = DataLoader(
            candidate_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        device = state.metadata.get("device", "cpu")
        if self.substitute is None:
            self._train_substitute(state)
        probs = self._collect_probs(candidate_loader, device)

        # Apply k-center on candidates using probability vectors
        selected_local_in_candidates = self._select_k_center(probs.to(device), k)
        selected_indices = [dfal_candidates[i] for i in selected_local_in_candidates]
        
        return selected_indices

    def _finalize_query_batch(
        self,
        selected_indices: List[int],
        state: BenchmarkState,
        k: int,
    ) -> QueryBatch:
        for idx in selected_indices:
            if idx in self.unlabeled_indices:
                self.unlabeled_indices.remove(idx)
                self.labeled_indices.append(idx)

        state.attack_state["labeled_indices"] = self.labeled_indices
        state.attack_state["unlabeled_indices"] = self.unlabeled_indices
        state.attack_state["round"] += 1

        selected_dataset = Subset(self.pool_dataset, selected_indices)
        query_loader = DataLoader(selected_dataset, batch_size=k, shuffle=False, num_workers=0)
        x_batch, _ = next(iter(query_loader))

        return QueryBatch(
            x=x_batch,
            meta={
                "strategy": self.strategy,
                "selected_indices": selected_indices,
                "indices": selected_indices,  # Add indices for observe method
                "round": state.attack_state["round"],
                "labeled_size": len(self.labeled_indices),
                "unlabeled_size": len(self.unlabeled_indices),
            },
        )

    def _select_query_batch(self, k: int, state: BenchmarkState) -> QueryBatch:
        if not state.attack_state.get("initialized"):
            self._setup_datasets(state)

        if not self.unlabeled_indices:
            return QueryBatch(
                x=torch.empty(0, *self.state.metadata.get("input_shape", (3, 32, 32))),
                meta={"strategy": self.strategy, "status": "exhausted"},
            )

        device = state.metadata.get("device", "cpu")

        if self.strategy == "random":
            selected_local = np.random.choice(
                len(self.unlabeled_indices),
                size=min(k, len(self.unlabeled_indices)),
                replace=False,
            ).tolist()
            selected_indices = [self.unlabeled_indices[i] for i in selected_local]
            return self._finalize_query_batch(selected_indices, state, k)

        if self.substitute is None:
            self._train_substitute(state)

        unlabeled_dataset = Subset(self.pool_dataset, self.unlabeled_indices)
        unlabeled_loader = DataLoader(
            unlabeled_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        probs = self._collect_probs(unlabeled_loader, device)

        if self.strategy == "uncertainty":
            selected_local = self._select_uncertainty(probs, k)
            selected_indices = [self.unlabeled_indices[i] for i in selected_local]
            return self._finalize_query_batch(selected_indices, state, k)

        if self.strategy == "k_center":
            selected_local = self._select_k_center(probs.to(device), k)
            selected_indices = [self.unlabeled_indices[i] for i in selected_local]
            return self._finalize_query_batch(selected_indices, state, k)

        if self.strategy == "dfal":
            selected_indices = self._select_dfal(state, k)
            return self._finalize_query_batch(selected_indices, state, k)

        if self.strategy == "dfal_k_center":
            selected_indices = self._select_dfal_k_center(state, k)
            return self._finalize_query_batch(selected_indices, state, k)

        selected_local = np.random.choice(
            len(self.unlabeled_indices),
            size=min(k, len(self.unlabeled_indices)),
            replace=False,
        ).tolist()
        selected_indices = [self.unlabeled_indices[i] for i in selected_local]
        return self._finalize_query_batch(selected_indices, state, k)

    def run(self, ctx: BenchmarkContext) -> None:
        """Run ActiveThief attack."""
        state = ctx.state
        self.victim = ctx.oracle.model
        device = state.metadata.get("device", "cpu")

        step_size = self.step_size
        if step_size is None:
            total_budget = int(state.metadata.get("max_budget", ctx.budget_remaining))
            rounds = max(1, int(self.rounds))
            step_size = max(1, int(math.ceil(total_budget / rounds)))
            self.step_size = step_size
        elif step_size <= 0:
            raise ValueError("step_size must be positive")

        if self.initial_seed_size is None:
            self.initial_seed_size = step_size
        
        if not state.attack_state.get("initialized"):
            self._setup_datasets(state)

        if not state.attack_state.get("initial_seed_queried") and self.initial_seed_indices:
            seed_k = min(len(self.initial_seed_indices), ctx.budget_remaining)
            if seed_k > 0:
                seed_indices = self.initial_seed_indices[:seed_k]
                selected_dataset = Subset(self.pool_dataset, seed_indices)
                query_loader = DataLoader(
                    selected_dataset,
                    batch_size=seed_k,
                    shuffle=False,
                    num_workers=0,
                )
                x_batch, _ = next(iter(query_loader))
                query_batch = QueryBatch(
                    x=x_batch,
                    meta={
                        "strategy": self.strategy,
                        "selected_indices": seed_indices,
                        "indices": seed_indices,
                        "round": state.attack_state.get("round", 0),
                        "labeled_size": len(self.labeled_indices),
                        "unlabeled_size": len(self.unlabeled_indices),
                    },
                )
                oracle_output = ctx.oracle.query(query_batch.x)
                self.observe(query_batch, oracle_output, state)
                state.attack_state["initial_seed_queried"] = True
                if self.labeled_indices:
                    self._train_substitute(state)
                    self.logger.info(
                        "[ActiveThief] Initial seed training complete (budget_remaining=%s)",
                        ctx.budget_remaining,
                    )
                    self._evaluate_current_substitute(self.substitute, device)
        
        while ctx.budget_remaining > 0 and self.unlabeled_indices:
            step_size = min(step_size, ctx.budget_remaining, len(self.unlabeled_indices))
            
            query_batch = self._select_query_batch(step_size, state)
            round_id = state.attack_state.get("round", 0)
            self.logger.info(
                "[ActiveThief] Round %s selected %s samples (labeled=%s, unlabeled=%s)",
                round_id,
                query_batch.x.shape[0],
                len(self.labeled_indices),
                len(self.unlabeled_indices),
            )
            
            if query_batch.x.shape[0] == 0:
                break
            
            oracle_output = ctx.oracle.query(query_batch.x)
            self.observe(query_batch, oracle_output, state)
            
            if self.labeled_indices:
                self._train_substitute(state)
                self.logger.info(
                    "[ActiveThief] Round %s training complete (budget_remaining=%s)",
                    round_id,
                    ctx.budget_remaining,
                )
                self._evaluate_current_substitute(self.substitute, device)

    def observe(
        self, 
        query_batch: QueryBatch, 
        oracle_output: OracleOutput, 
        state: BenchmarkState
    ) -> None:
        """Observe oracle outputs and update attack state."""
        # Store oracle labels to override dataset labels during training
        # This handles cases where surrogate dataset has different label space than victim
        indices = query_batch.meta.get("indices", [])
        if not indices:
            return

        if oracle_output.kind == "soft_prob":
            labels = oracle_output.y.argmax(dim=1)
        else:
            labels = oracle_output.y

        for idx, label in zip(indices, labels):
            self.observed_labels[int(idx)] = int(label.item())
        
        # ActiveThief state updates are handled in _select_query_batch (index management)
        # and _train_substitute (model update).
