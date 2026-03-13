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
from mebench.training import SubstituteTrainer, TrainRequest
from mebench.utils.config_aliases import resolve_iterations
from mebench.utils.binary import (
    binary_bce_loss,
    binary_entropy_from_positive_probs,
    binary_hard_labels_from_positive_probs,
    binary_positive_probs_from_logits,
    is_single_logit_binary_num_classes,
)
from mebench.utils.dataloader import (
    pool_loader_kwargs,
    resolve_pool_num_workers,
    resolve_train_num_workers,
    resolve_val_num_workers,
)


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
        self.scoring_batch_size = int(config.get("scoring_batch_size", config.get("batch_size", 512)))
        self.num_classes = int(
            self.state.metadata.get("num_classes")
            or self.config.get("num_classes")
            or self.state.metadata.get("dataset_config", {}).get("num_classes", 10)
        )
        self.is_single_logit_binary = is_single_logit_binary_num_classes(self.num_classes)
        
        # Fixed-required semantics: strategy family + pool-based querying.
        # Training knobs (rounds/step size) remain tunable in benchmark runs.
        # Note: paper reports modality-dependent batch sizes (image/text).
        # This image benchmark path uses the image profile from generated configs.
        # Active learning strategy
        self.strategy = config.get("strategy", "uncertainty")
        initial_seed_size = config.get("initial_seed_size")
        self.initial_seed_size = (
            int(initial_seed_size) if initial_seed_size is not None else None
        )
        step_size = config.get("step_size")
        self.step_size = int(step_size) if step_size is not None else None
        self.rounds = resolve_iterations(config, default=10, context="activethief")
        
        # Store oracle labels to override dataset labels during training
        self.observed_labels = {}

        # Cache queried tensors to avoid repeated ImageFolder IO during training.
        # Stored as list-of-batches to minimize Python per-sample overhead.
        self.query_data_x: List[torch.Tensor] = []
        self.query_data_y: List[torch.Tensor] = []
        
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
        state.attack_state.setdefault("query_data_x", [])
        state.attack_state.setdefault("query_data_y", [])
        state.attack_state.setdefault("val_query_data_x", [])
        state.attack_state.setdefault("val_query_data_y", [])
        state.attack_state.setdefault("val_indices", [])
        state.attack_state.setdefault("seed_indices", [])
        dataset_config = state.metadata.get("dataset_config", {})
        seed_size = dataset_config.get("seed_size")
        if seed_size is None and isinstance(dataset_config.get("dataset"), dict):
            seed_size = dataset_config["dataset"].get("seed_size")
        default_pool_size = int(seed_size) if seed_size is not None else 10000
        state.attack_state["unlabeled_indices"] = list(range(default_pool_size))
        state.attack_state["round"] = 0
        state.attack_state["initialized"] = False
        state.attack_state["initial_seed_indices"] = []
        state.attack_state["initial_seed_queried"] = False
        state.attack_state["validation_built"] = False

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
            seed_target, _ = self._resolve_seed_and_validation_targets(
                total_budget=total_budget,
                default_seed_ratio=0.1,
                default_validation_ratio=0.2,
            )
            if seed_target > 0:
                self.initial_seed_size = int(seed_target)
            elif self.step_size is not None:
                self.initial_seed_size = int(self.step_size)
            else:
                self.initial_seed_size = 50
            
        dataset_config = state.metadata.get("dataset_config", {})
        self.pool_dataset = create_dataloader(
            dataset_config,
            batch_size=self.scoring_batch_size,
            shuffle=False,
        ).dataset
        
        # Initialize labeled/unlabeled splits
        pool_size = len(self.pool_dataset)
        self.unlabeled_indices = list(range(pool_size))
        self.initial_seed_indices = []

        state.attack_state["labeled_indices"] = self.labeled_indices
        state.attack_state["unlabeled_indices"] = self.unlabeled_indices
        state.attack_state["initial_seed_indices"] = []
        state.attack_state["initialized"] = True

    def _pop_unlabeled_indices(self, k: int) -> List[int]:
        """Pop k indices from unlabeled pool without replacement."""
        if k <= 0 or not self.unlabeled_indices:
            return []
        k = min(int(k), len(self.unlabeled_indices))
        selected = np.random.choice(self.unlabeled_indices, size=k, replace=False).tolist()
        selected_set = set(int(i) for i in selected)
        self.unlabeled_indices = [i for i in self.unlabeled_indices if int(i) not in selected_set]
        return [int(i) for i in selected]

    def _bootstrap_seed_and_validation_sets(self, ctx: BenchmarkContext, state: BenchmarkState) -> None:
        """Reserve 20% validation + 10% seed budget up-front.

        - Validation set: fixed holdout queried from the victim and never used for training.
        - Seed set: initial labeled set queried from the victim and used to train the first substitute.
        """

        total_budget = int(state.metadata.get("max_budget") or ctx.budget_remaining)
        seed_target, val_target = self._resolve_seed_and_validation_targets(
            total_budget=total_budget,
            default_seed_ratio=0.1,
            default_validation_ratio=0.2,
        )

        # Ensure initial_seed_size matches the seed budget target.
        self.initial_seed_size = int(seed_target)

        device = state.metadata.get("device", "cpu")
        pool_workers = resolve_pool_num_workers(self.config, state.metadata.get("dataset_config", {}))
        loader_kwargs = (
            pool_loader_kwargs(device, {"num_workers": int(pool_workers)})
            if pool_workers is not None
            else pool_loader_kwargs(device)
        )
        pool_workers = resolve_pool_num_workers(self.config, state.metadata.get("dataset_config", {}))
        pool_kwargs = (
            pool_loader_kwargs(str(device), {"num_workers": int(pool_workers)})
            if pool_workers is not None
            else pool_loader_kwargs(str(device), self.config)
        )

        # 1) Build validation holdout (20% of budget)
        if not bool(state.attack_state.get("validation_built", False)) and int(val_target) > 0:
            val_k = min(int(val_target), int(ctx.budget_remaining), len(self.unlabeled_indices))
            val_indices = self._pop_unlabeled_indices(val_k)
            state.attack_state["val_indices"] = list(val_indices)

            if val_indices:
                subset = Subset(self.pool_dataset, val_indices)
                loader = DataLoader(
                    subset,
                    batch_size=min(self.scoring_batch_size, len(val_indices)),
                    shuffle=False,
                    **pool_kwargs,
                )
                ptr = 0
                for x_batch, _ in loader:
                    batch_indices = val_indices[ptr : ptr + int(x_batch.size(0))]
                    ptr += int(x_batch.size(0))
                    query_batch = QueryBatch(
                        x=x_batch,
                        meta={"indices": batch_indices, "is_validation": True},
                    )
                    oracle_output = ctx.query(query_batch.x, meta=query_batch.meta)
                    self.observe(query_batch, oracle_output, state)

            state.attack_state["validation_built"] = True

        # 2) Build seed set (10% of budget)
        if not bool(state.attack_state.get("initial_seed_queried", False)) and int(seed_target) > 0:
            seed_k = min(int(seed_target), int(ctx.budget_remaining), len(self.unlabeled_indices))
            seed_indices = self._pop_unlabeled_indices(seed_k)
            state.attack_state["seed_indices"] = list(seed_indices)
            state.attack_state["initial_seed_indices"] = list(seed_indices)

            if seed_indices:
                self.labeled_indices.extend(seed_indices)
                state.attack_state["labeled_indices"] = self.labeled_indices
                state.attack_state["unlabeled_indices"] = self.unlabeled_indices

                subset = Subset(self.pool_dataset, seed_indices)
                loader = DataLoader(
                    subset,
                    batch_size=min(self.scoring_batch_size, len(seed_indices)),
                    shuffle=False,
                    **pool_kwargs,
                )
                ptr = 0
                for x_batch, _ in loader:
                    batch_indices = seed_indices[ptr : ptr + int(x_batch.size(0))]
                    ptr += int(x_batch.size(0))
                    query_batch = QueryBatch(
                        x=x_batch,
                        meta={"indices": batch_indices, "is_validation": False},
                    )
                    oracle_output = ctx.query(query_batch.x, meta=query_batch.meta)
                    self.observe(query_batch, oracle_output, state)

            state.attack_state["initial_seed_queried"] = True

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
        
        sub_config = self.state.metadata.get("substitute_config", {})
        
        # IO optimization: train on cached queried tensors (exact x used for oracle queries)
        # instead of re-indexing ImageFolder each epoch.
        query_x = state.attack_state.get("query_data_x", [])
        query_y = state.attack_state.get("query_data_y", [])
        if len(query_x) == 0:
            return

        x_all = torch.cat(query_x, dim=0)
        y_all = torch.cat(query_y, dim=0)
        full_dataset = torch.utils.data.TensorDataset(x_all, y_all)
        
        # Paper protocol: reserve a fixed validation set (20% of total query budget).
        # If we already queried a dedicated holdout set, use it as-is.
        val_x = state.attack_state.get("val_query_data_x", [])
        val_y = state.attack_state.get("val_query_data_y", [])
        val_len = 0
        if len(val_x) > 0 and len(val_y) > 0:
            train_dataset = full_dataset
            x_val = torch.cat(val_x, dim=0)
            y_val = torch.cat(val_y, dim=0)
            val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
            val_len = len(val_dataset)
        else:
            # Fallback: split collected training queries into train/val (80/20).
            total_len = len(full_dataset)
            val_len = int(0.2 * total_len)
            train_len = total_len - val_len
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset,
                [train_len, val_len],
            )

        train_batch_size = int(
            sub_config.get("batch_size")
            or self.scoring_batch_size
        )

        train_workers = resolve_train_num_workers(sub_config, self.config, default=0)
        val_workers = resolve_val_num_workers(sub_config, self.config, default=train_workers)
        train_loader_kwargs = pool_loader_kwargs(device, {"num_workers": int(train_workers)})
        val_loader_kwargs = pool_loader_kwargs(device, {"num_workers": int(val_workers)})
        
        # Create loaders
        labeled_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            **train_loader_kwargs,
        )
        
        val_loader = None
        if val_len > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=train_batch_size,
                shuffle=False,
                **val_loader_kwargs,
            )
        
        output_mode = str(self.config.get("output_mode", "soft_prob"))

        def loss_fn(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            if getattr(self, "is_single_logit_binary", False):
                return binary_bce_loss(outputs, targets)
            if output_mode == "soft_prob":
                targets = targets.to(device)
                targets = torch.clamp(targets, min=1e-10)
                targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-12)
                log_probs = F.log_softmax(outputs, dim=1)
                return F.kl_div(log_probs, targets, reduction="batchmean")
            return F.cross_entropy(outputs, targets.long())

        # Validation evaluation function
        def eval_fn(model: nn.Module, loader: DataLoader) -> float:
            model.eval()
            total_loss = 0.0
            total_count = 0
            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)
                    if getattr(self, "is_single_logit_binary", False):
                        loss = binary_bce_loss(outputs, y)
                    elif output_mode == "soft_prob":
                        y = torch.clamp(y, min=1e-10)
                        y = y / y.sum(dim=1, keepdim=True).clamp_min(1e-12)
                        loss = F.kl_div(F.log_softmax(outputs, dim=1), y, reduction="batchmean")
                    else:
                        loss = F.cross_entropy(outputs, y.long())
                    total_loss += loss.item() * x.size(0)
                    total_count += x.size(0)
            return total_loss / total_count if total_count > 0 else float('inf')

        # Ensure optimizer config defaults to Adam if not specified (ActiveThief specific)
        # But SubstituteTrainer pulls from sub_config. We can inject it if missing?
        # Better to handle it in generate_configs.py or let it be.
        # Here we just pass the val_loader.

        # [FEATURE] Enable TQDM for substitute training visualization
        sub_config_with_tqdm = dict(sub_config)
        sub_config_with_tqdm["use_tqdm"] = True
        
        trainer = SubstituteTrainer(sub_config_with_tqdm, device=device, logger=self.logger)
        batch_size = max(1, int(train_batch_size))
        train_size = int(len(train_dataset))
        steps_per_epoch = max(1, int(math.ceil(train_size / batch_size)))
        max_epochs = int(sub_config.get("max_epochs", 200))
        patience_epochs = int(sub_config.get("patience", 20))
        request = TrainRequest(
            model=self.substitute,
            train_loader=labeled_loader,
            val_loader=val_loader,
            eval_fn=eval_fn,
            loss_fn=loss_fn,
            load_best=True,
            max_steps=max_epochs * steps_per_epoch,
            validate_every=steps_per_epoch,
            patience=patience_epochs * steps_per_epoch,
            early_stop_mode="min", # minimizing validation loss
        )
        trainer.train(request)

    def train_substitute(self, state: BenchmarkState) -> None:
        self._train_substitute(state)
        state.attack_state["substitute"] = self.substitute

    def _select_uncertainty(self, probs: torch.Tensor, k: int) -> List[int]:
        """Select samples with highest entropy.
        
        Eq: H_n = -sum(y_nj * log(y_nj))
        """
        # Compute entropy
        if getattr(self, "is_single_logit_binary", False):
            entropy = binary_entropy_from_positive_probs(probs)
        else:
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        # [OPTIMIZATION] topk on GPU is much faster
        _, indices = torch.topk(entropy, k)
        return indices.cpu().tolist()

    def _select_k_center(self, unlabeled_probs: torch.Tensor, labeled_probs: torch.Tensor, k: int) -> List[int]:
        """Select samples using k-center greedy algorithm on probability vectors.
        
        Implements Core-Set approach adapted for probability space (Pal et al. 2020).
        labeled_probs: Probabilities of samples ALREADY in the labeled set (centers).
        unlabeled_probs: Probabilities of samples in the pool.
        """
        if labeled_probs.shape[0] == 0:
            # No labeled data yet, fall back to random initialization within unlabeled
            selected = []
            remaining = list(range(unlabeled_probs.shape[0]))
            
            if remaining:
                first_idx = np.random.choice(remaining)
                selected.append(first_idx)
                remaining.remove(first_idx)
            
            min_dists = torch.full((unlabeled_probs.shape[0],), float('inf'), device=unlabeled_probs.device)
            dists = torch.norm(unlabeled_probs - unlabeled_probs[selected[0]].unsqueeze(0), dim=1)
            min_dists = torch.min(min_dists, dists)

            for _ in range(min(k - 1, len(remaining))):
                current_min_dists = min_dists[remaining]
                if len(current_min_dists) == 0:
                    break
                
                max_val, max_idx_local = torch.max(current_min_dists, dim=0)
                best_idx = remaining[max_idx_local.item()]
                
                selected.append(best_idx)
                remaining.remove(best_idx)
                
                new_dists = torch.norm(unlabeled_probs - unlabeled_probs[best_idx].unsqueeze(0), dim=1)
                min_dists = torch.min(min_dists, new_dists)
            
            return selected
        
        # Paper: "most distant from all existing centers"
        # 1. Initialize min_dists based on distance to CLOSEST existing labeled point
        dists = torch.cdist(unlabeled_probs, labeled_probs)  # Shape: [unlabeled, labeled]
        min_dists, _ = torch.min(dists, dim=1)  # Min distance to ANY existing center
        
        selected = []
        
        # 2. Greedy selection for k iterations
        for _ in range(min(k, unlabeled_probs.shape[0])):
            # Select point with maximum 'min_distance'
            max_val, max_idx = torch.max(min_dists, dim=0)
            if not torch.isfinite(max_val):
                break
            best_idx = max_idx.item()
            selected.append(best_idx)

            # Prevent duplicate picks when many points share the same distance.
            min_dists[best_idx] = float("-inf")
            
            # Update distances relative to the NEWLY selected point
            new_center = unlabeled_probs[best_idx].unsqueeze(0)
            new_dists = torch.norm(unlabeled_probs - new_center, dim=1)
            min_dists = torch.min(min_dists, new_dists)
        
        return selected

    def _collect_probs(self, loader: DataLoader, device: str) -> torch.Tensor:
        """Collect probabilities for a loader."""
        all_probs = []
        self.substitute.eval()
        non_blocking = str(device).startswith("cuda")
        with torch.no_grad():
            for x_batch, _ in loader:
                x_batch = x_batch.to(device, non_blocking=non_blocking)
                logits = self.substitute(x_batch)
                if getattr(self, "is_single_logit_binary", False):
                    probs = binary_positive_probs_from_logits(logits)
                else:
                    probs = F.softmax(logits, dim=1)
                # [OPTIMIZATION] Keep on GPU to avoid Host-Device transfer bottleneck
                all_probs.append(probs.detach())
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
        pin_memory = str(device).startswith("cuda")
        attack_cfg = getattr(self, "config", {})
        pool_workers = resolve_pool_num_workers(attack_cfg, state.metadata.get("dataset_config", {}))
        loader_kwargs = (
            pool_loader_kwargs(device, {"num_workers": int(pool_workers)})
            if pool_workers is not None
            else pool_loader_kwargs(device)
        )
        unlabeled_dataset = Subset(self.pool_dataset, self.unlabeled_indices)
        unlabeled_loader = DataLoader(
            unlabeled_dataset, 
            batch_size=self.scoring_batch_size, 
            shuffle=False, 
            **loader_kwargs,
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
            x_batch = x_batch.to(device, non_blocking=pin_memory)
            # Enable grad for DeepFool
            with torch.enable_grad():
                distances = self._deepfool_distance_dfal(
                    self.substitute,
                    x_batch,
                    max_iter=self.dfal_max_iter,
                    internal_batch_size=min(self.scoring_batch_size, x_batch.shape[0]),
                )
            all_distances.append(distances.detach().cpu())
        
        distances = torch.cat(all_distances, dim=0)
        
        # Select k samples with smallest DeepFool distances (closest to boundary)
        _, selected_local = torch.topk(distances, k, largest=False)
        selected_indices = [self.unlabeled_indices[i] for i in selected_local.tolist()]
        
        return selected_indices

    def _select_dfal_k_center(self, state: BenchmarkState, k: int) -> List[int]:
        """DFAL pre-filtering + K-center selection."""
        device = state.metadata.get("device", "cpu")
        pin_memory = str(device).startswith("cuda")
        attack_cfg = getattr(self, "config", {})
        pool_workers = resolve_pool_num_workers(attack_cfg, state.metadata.get("dataset_config", {}))
        loader_kwargs = (
            pool_loader_kwargs(device, {"num_workers": int(pool_workers)})
            if pool_workers is not None
            else pool_loader_kwargs(device, attack_cfg)
        )
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
            batch_size=self.scoring_batch_size, 
            shuffle=False, 
            **loader_kwargs,
        )

        if self.substitute is None:
            self._train_substitute(state)
        probs = self._collect_probs(candidate_loader, device)

        # Need to get probs for ALREADY LABELED data to act as centers
        if self.labeled_indices:
            labeled_dataset = Subset(self.pool_dataset, self.labeled_indices)
            labeled_loader = DataLoader(
                labeled_dataset,
                batch_size=self.scoring_batch_size,
                shuffle=False,
                **loader_kwargs,
            )
            labeled_probs = self._collect_probs(labeled_loader, device)
        else:
            # No labeled data yet, pass empty tensor
            labeled_probs = torch.empty(0, self.num_classes, device=device)
        
        # Apply k-center on candidates using probability vectors with labeled centers
        selected_local_in_candidates = self._select_k_center(probs.to(device), labeled_probs.to(device), k)
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
        device = state.metadata.get("device", "cpu")
        pool_workers = resolve_pool_num_workers(self.config, state.metadata.get("dataset_config", {}))
        loader_kwargs = (
            pool_loader_kwargs(device, {"num_workers": int(pool_workers)})
            if pool_workers is not None
            else pool_loader_kwargs(device, self.config)
        )
        query_loader = DataLoader(
            selected_dataset,
            batch_size=k,
            shuffle=False,
            **loader_kwargs,
        )
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
        pool_workers = resolve_pool_num_workers(
            self.config, state.metadata.get("dataset_config", {})
        )
        loader_kwargs = (
            pool_loader_kwargs(device, {"num_workers": int(pool_workers)})
            if pool_workers is not None
            else pool_loader_kwargs(device, self.config)
        )

        if self.strategy == "random":
            selected_local = np.random.choice(
                len(self.unlabeled_indices),
                size=min(k, len(self.unlabeled_indices)),
                replace=False,
            ).tolist()
            selected_indices = [self.unlabeled_indices[i] for i in selected_local]
            return self._finalize_query_batch(selected_indices, state, k)

        if self.initial_seed_size is not None and len(self.labeled_indices) < self.initial_seed_size:
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
            batch_size=self.scoring_batch_size,
            shuffle=False,
            **loader_kwargs,
        )
        probs = self._collect_probs(unlabeled_loader, device)

        if self.strategy == "uncertainty":
            selected_local = self._select_uncertainty(probs, k)
            selected_indices = [self.unlabeled_indices[i] for i in selected_local]
            return self._finalize_query_batch(selected_indices, state, k)

        if self.strategy == "k_center":
            # Need to get probs for ALREADY LABELED data to act as centers
            if self.labeled_indices:
                labeled_dataset = Subset(self.pool_dataset, self.labeled_indices)
                labeled_loader = DataLoader(
                    labeled_dataset,
                    batch_size=self.scoring_batch_size,
                    shuffle=False,
                    **loader_kwargs,
                )
                labeled_probs = self._collect_probs(labeled_loader, device)
            else:
                # No labeled data yet, pass empty tensor
                labeled_probs = torch.empty(0, self.num_classes, device=device)
            
            # Pass both to selector
            selected_local = self._select_k_center(probs.to(device), labeled_probs.to(device), k)
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
        if step_size is not None and step_size <= 0:
            raise ValueError("step_size must be positive")
        
        if not state.attack_state.get("initialized"):
            self._setup_datasets(state)

        # Reserve validation (20%) and seed (10%) from total query budget.
        self._bootstrap_seed_and_validation_sets(ctx, state)

        # Compute per-round query step AFTER reserving seed/validation.
        # Semantics: divide the remaining active-learning budget into `rounds` rounds.
        if step_size is None:
            rounds = max(1, int(self.rounds))
            active_budget = int(ctx.budget_remaining)
            step_size = max(1, int(math.ceil(active_budget / rounds)))
            self.step_size = step_size

        # Train initial substitute once after seed collection.
        if self.substitute is None and self.labeled_indices:
            self._train_substitute(state)
            if self.substitute is not None:
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
            
            oracle_output = ctx.query(query_batch.x, meta=query_batch.meta)
            self.observe(query_batch, oracle_output, state)
            
            if self.labeled_indices:
                self._train_substitute(state)
                self.logger.info(
                    "[ActiveThief] Round %s training complete (budget_remaining=%s)",
                    round_id,
                    ctx.budget_remaining,
                )
                self._evaluate_current_substitute(self.substitute, device)

            # Round is already incremented exactly once in _finalize_query_batch().

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

        # Cache queried tensors + oracle outputs.
        # - soft_prob: store full probability vectors for KL distillation
        # - hard_top1: store class indices for CE training
        if oracle_output.kind == "soft_prob":
            probs = oracle_output.y.detach().cpu().float()
            if getattr(self, "is_single_logit_binary", False):
                hard_labels = binary_hard_labels_from_positive_probs(probs)
            else:
                hard_labels = probs.argmax(dim=1)
            y_cpu = probs
        else:
            hard_labels = oracle_output.y.detach().cpu().long()
            y_cpu = hard_labels

        for idx, label in zip(indices, hard_labels):
            self.observed_labels[int(idx)] = int(label.item())

        # Cache queried tensors + targets. Validation holdout is stored separately.
        # Store as batches to keep append overhead low.
        x_cpu = query_batch.x.detach().cpu()
        if bool(query_batch.meta.get("is_validation", False)):
            state.attack_state.setdefault("val_query_data_x", []).append(x_cpu)
            state.attack_state.setdefault("val_query_data_y", []).append(y_cpu)
        else:
            state.attack_state.setdefault("query_data_x", []).append(x_cpu)
            state.attack_state.setdefault("query_data_y", []).append(y_cpu)
        
        # ActiveThief state updates are handled in _select_query_batch (index management)
        # and _train_substitute (model update).

    def _handle_oracle_output(
        self,
        query_batch: QueryBatch,
        oracle_output: OracleOutput,
        state: BenchmarkState,
    ) -> None:
        if not getattr(query_batch, "meta", None):
            return
        if not query_batch.meta.get("indices"):
            return
        self.observe(query_batch, oracle_output, state)
