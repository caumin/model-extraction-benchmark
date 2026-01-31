"""ActiveThief attack implementation."""

from typing import Dict, Any, Optional, List, Tuple
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score
from tqdm import tqdm

from mebench.attackers.runner import AttackRunner
from mebench.core.context import BenchmarkContext
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.data.loaders import create_dataloader, get_test_dataloader
from mebench.models.substitute_factory import create_substitute
from mebench.eval.metrics import evaluate_substitute


class ActiveThief(AttackRunner):
    """ActiveThief: Pool-based active learning for model extraction.

    Algorithm loop:
    1. Initialize with random seed S0 from thief dataset
    2. Query victim f to get labels D_i
    3. Train substitute f~ from scratch on all collected labels
    4. Predict on unlabeled pool
    5. Select next queries using active learning strategy
    6. Repeat until budget exhausted

    Selection strategies:
    - Uncertainty: Select samples with highest entropy
    - K-center: Select samples farthest from labeled set in feature space
    - DFAL: Select samples closest to decision boundary
    - DFAL+K-center: DFAL pre-filter then K-center
    """

    def __init__(self, config: dict, state: BenchmarkState):
        """Initialize ActiveThief attack.

        Args:
            config: Attack configuration with strategy and hyperparameters
            state: Global benchmark state
        """
        super().__init__(config, state)

        # Hyperparameters (from papers/activethief.pdf)
        self.strategy = config.get("strategy", "dfal_k_center")  # uncertainty, k_center, dfal, dfal_k_center
        self.num_rounds = int(config.get("num_rounds", 10))
        
        max_budget = state.metadata.get("max_budget", 1000)
        # 10% initial seed per paper
        self.initial_seed_size = int(config.get("initial_seed_size", max(1, int(0.1 * max_budget))))
        
        # Iteration step size (budget / rounds)
        self.step_size = max(1, max_budget // self.num_rounds)
        
        self.batch_size = config.get("batch_size", 150)
        self.max_epochs = config.get("max_epochs", 1000)
        self.patience = config.get("patience", 100)
        self.dropout = config.get("dropout", 0.1)
        self.l2_reg = config.get("l2_reg", 0.001)

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Pool dataset (loaded during selection)
        self.pool_dataset = None
        self.test_loader = None
        self.victim = None

        # Initialize state (will set up indices but not load pool yet)
        self._initialize_state(state)

    def run(self, ctx: BenchmarkContext) -> None:
        self.victim = ctx.oracle.model
        pbar = tqdm(total=self.state.budget_remaining, desc="[ActiveThief] Extracting")
        while ctx.budget_remaining > 0:
            step_size = self._default_step_size(ctx)
            query_batch = self._select_query_batch(step_size, self.state)
            oracle_output = ctx.query(query_batch.x, meta=getattr(query_batch, "meta", None))
            self._handle_oracle_output(query_batch, oracle_output, self.state)
            pbar.update(query_batch.x.size(0))
        pbar.close()

    def _get_pool_dataset_config(self, state: BenchmarkState) -> dict:
        if "dataset" in self.config.get("attack", {}):
            dataset_config = self.config["attack"]["dataset"]
        else:
            dataset_config = state.metadata.get("dataset_config", {})
        if "data_mode" not in dataset_config:
            dataset_config = {"data_mode": "seed", **dataset_config}
        if dataset_config.get("data_mode") == "seed" and "name" not in dataset_config:
            dataset_config = {"name": "CIFAR10", **dataset_config}
        if dataset_config.get("data_mode") == "surrogate" and "surrogate_name" not in dataset_config:
            dataset_config = {"surrogate_name": "SVHN", **dataset_config}
        return dataset_config

    def _select_query_batch(self, k: int, state: BenchmarkState) -> QueryBatch:
        """Propose k queries using active learning strategy.

        Args:
        k: Number of queries to select
            state: Current benchmark state

        Returns:
            QueryBatch with k queries
        """
        labeled_indices = state.attack_state["labeled_indices"]
        unlabeled_indices = state.attack_state["unlabeled_indices"]

        # Load pool dataset early (selection strategies require access).
        if self.pool_dataset is None:
            dataset_config = self._get_pool_dataset_config(state)
            self.pool_dataset = create_dataloader(
                dataset_config,
                batch_size=1,
                shuffle=False,
            ).dataset

        # Retrieve normalization parameters for consistent inference
        victim_config = state.metadata.get("victim_config", {})
        normalization = victim_config.get("normalization")
        if normalization is None:
            normalization = {"mean": [0.0], "std": [1.0]}
        self.norm_mean = torch.tensor(normalization["mean"]).view(1, -1, 1, 1)
        self.norm_std = torch.tensor(normalization["std"]).view(1, -1, 1, 1)

        # Handle empty pool
        if len(unlabeled_indices) == 0:
            raise ValueError(
                f"Query pool exhausted for {self.__class__.__name__}. "
                f"Cannot select {k} more queries."
            )

        # First round: random selection for initial seed
        if len(labeled_indices) < self.initial_seed_size:
            n_select = min(k, self.initial_seed_size - len(labeled_indices))
            selected = np.random.choice(unlabeled_indices, n_select, replace=False).tolist()
        else:
            # Use active learning strategy
            if self.strategy == "uncertainty":
                selected = self._select_uncertainty(k, state)
            elif self.strategy == "k_center":
                selected = self._select_k_center(k, state)
            elif self.strategy == "dfal":
                selected = self._select_dfal(k, state)
            elif self.strategy in {"dfal_k_center", "dfal+k_center", "dfal_kcenter"}:
                # Strict Protocol: Use the ENTIRE pool for DFAL pre-filter unless specified
                rho = int(self.config.get("dfal_rho", len(unlabeled_indices)))
                selected = self._select_dfal_k_center(k, rho, state)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

        # Update state
        for idx in selected:
            state.attack_state["unlabeled_indices"].remove(idx)
            state.attack_state["labeled_indices"].append(idx)

        x_list = []
        for idx in selected:
            img, _ = self.pool_dataset[idx]  # Ignore pool label, use oracle label
            x_list.append(img)

        x = torch.stack(x_list)
        meta = {"indices": selected, "strategy": self.strategy}

        return QueryBatch(x=x, meta=meta)

    def _get_approx_probs(self, indices: List[int], substitute: nn.Module, batch_size: int = 512) -> torch.Tensor:
        """Step 4: Compute approximate labels (softmax probabilities) for candidate pool.
        
        Uses efficient batch processing to minimize overhead.
        """
        device = next(substitute.parameters()).device
        substitute.eval()
        
        # Create temporary dataloader for efficient batch inference
        subset = Subset(self.pool_dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        
        probs_list = []
        norm_mean = self.norm_mean.to(device)
        norm_std = self.norm_std.to(device)
        
        with torch.no_grad():
            for x_batch, _ in tqdm(loader, desc="[ActiveThief] Predicting Pool", leave=False):
                x_batch = x_batch.to(device)
                x_batch = (x_batch - norm_mean) / norm_std
                logits = substitute(x_batch)
                probs = F.softmax(logits, dim=1)
                probs_list.append(probs.cpu())
                
        return torch.cat(probs_list, dim=0)

    def _select_uncertainty(self, k: int, state: BenchmarkState) -> list:
        """Select k samples with highest entropy (Vectorized implementation)."""
        unlabeled_indices = state.attack_state["unlabeled_indices"]
        substitute = state.attack_state["substitute"]

        if substitute is None:
            return np.random.choice(unlabeled_indices, k, replace=False).tolist()

        # Step 4: Get approximate labels for ALL unlabeled samples (Strict Protocol)
        # For huge datasets, one might sample a subset here, but we stick to strict full-scan.
        self.logger.info(f"Scoring {len(unlabeled_indices)} pool samples via Uncertainty...")
        probs = self._get_approx_probs(unlabeled_indices, substitute)
        
        # Calculate entropy: H = -sum(p * log(p))
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=1)
        
        # Top-k selection
        _, topk_indices = torch.topk(entropy, k=min(k, len(entropy)))
        selected = [unlabeled_indices[i] for i in topk_indices.tolist()]
        
        return selected

    def _select_k_center(self, k: int, state: BenchmarkState) -> list:
        """Select k samples using K-center greedy on Probability Vectors (Strict Protocol).
        
        Metric: L2 distance between softmax probability vectors.
        """
        unlabeled_indices = state.attack_state["unlabeled_indices"]
        labeled_indices = state.attack_state["labeled_indices"]
        substitute = state.attack_state["substitute"]

        if substitute is None:
            return np.random.choice(unlabeled_indices, k, replace=False).tolist()
            
        # Step 4: Get approximate labels (probability vectors) for ALL unlabeled samples
        # Strict implementation: No subsampling.
        candidates = unlabeled_indices
            
        # Get probs for candidates (U) and labeled set (L)
        probs_u = self._get_approx_probs(candidates, substitute) # [M, C]
        
        # For L, we re-compute current probs to be in same space
        probs_l = self._get_approx_probs(labeled_indices, substitute) # [N, C]
        
        return self._k_center_greedy(probs_u, probs_l, candidates, k)

    def _k_center_greedy(self, x_u: torch.Tensor, x_l: torch.Tensor, candidate_indices: List[int], k: int) -> List[int]:
        """Greedy K-Center algorithm using L2 distance on tensors."""
        device = x_u.device
        # If too large, move to CPU
        if x_u.numel() > 1e7: 
            device = 'cpu'
            
        x_u = x_u.to(device)
        x_l = x_l.to(device)
        
        # Initialize min distances: min_dist(u) = min_{l in L} ||u - l||^2
        # Use simple loop to avoid huge memory for distance matrix
        min_dists = torch.full((x_u.size(0),), float('inf'), device=device)
        
        # Initial distances from existing labeled set
        # Process in chunks to save memory
        
        chunk_size = 1000

        
        # Initial distance calculation
        num_chunks = (x_l.size(0) + chunk_size - 1) // chunk_size
        for i in range(0, x_l.size(0), chunk_size):
            chunk = x_l[i : i + chunk_size]
            dists = torch.cdist(x_u, chunk, p=2).min(dim=1).values
            min_dists = torch.minimum(min_dists, dists)
            
        selected_indices = []
        for _ in range(min(k, len(candidate_indices))):
            # Select point with max min_dist
            idx = torch.argmax(min_dists).item()
            selected_indices.append(candidate_indices[idx])
            
            # Update min_dists with the new point
            new_center = x_u[idx].unsqueeze(0)
            new_dists = torch.cdist(x_u, new_center, p=2).squeeze(1)
            min_dists = torch.minimum(min_dists, new_dists)
            
            # Infinite distance for selected to prevent re-selection (though indices are unique)
            min_dists[idx] = -1.0 
            
        return selected_indices

    def _select_dfal_k_center(self, k: int, rho: int, state: BenchmarkState) -> list:
        """DFAL pre-filter (rho) then K-center (k)."""

        unlabeled_indices = state.attack_state["unlabeled_indices"]
        if len(unlabeled_indices) == 0:
            return []

        rho = min(int(rho), len(unlabeled_indices))
        if rho <= 0:
            return []

        # DFAL pre-filter: take rho closest-to-boundary.
        dfal_candidates = self._select_dfal(rho, state)
        if len(dfal_candidates) <= k:
            return dfal_candidates
            
        # K-Center greedy selection from the DFAL candidates
        substitute = state.attack_state["substitute"]
        labeled_indices = state.attack_state["labeled_indices"]
        
        # Get probs for DFAL candidates (U) and labeled set (L)
        probs_u = self._get_approx_probs(dfal_candidates, substitute)
        probs_l = self._get_approx_probs(labeled_indices, substitute)
        
        return self._k_center_greedy(probs_u, probs_l, dfal_candidates, k)

    def _select_dfal(self, k: int, state: BenchmarkState) -> list:
        """Select k samples closest to decision boundary using vectorized DeepFool approximation.
        
        Strictly speaking, DeepFool is iterative. To speed up without losing rigor,
        we run iterative DF but on batched inputs using the helper function.
        """
        unlabeled_indices = state.attack_state["unlabeled_indices"]
        substitute = state.attack_state["substitute"]

        if substitute is None:
            return np.random.choice(unlabeled_indices, k, replace=False).tolist()

        # Strict implementation: Apply DFAL metric to ALL unlabeled samples.
        candidates = unlabeled_indices

        self.logger.info(f"Scoring {len(candidates)} pool samples via DFAL...")
        device = next(substitute.parameters()).device
        substitute.eval()
        
        norm_mean = self.norm_mean.to(device)
        norm_std = self.norm_std.to(device)

        # We process candidates in batches to compute DeepFool perturbations
        subset = Subset(self.pool_dataset, candidates)
        loader = DataLoader(subset, batch_size=64, shuffle=False, num_workers=2)
        
        perturbation_scores = []
        cursor = 0
        
        for x_batch, _ in tqdm(loader, desc="[DFAL] Computing Perturbations", leave=False):
            x_batch = x_batch.to(device)

            
            # DeepFool requires gradient access, so we clone and detach for safety but enable grad
            # Note: _deepfool_perturbation_batch handles the loop internally per sample or batch?
            # DeepFool is hard to batch fully because each sample stops at different iter.
            # We use a loop over batch for simplicity but it's faster than single-item dataloader overhead.
            
            batch_scores = []
            for i in range(x_batch.size(0)):
                x = x_batch[i : i+1]
                # Pre-normalize for substitute inside the helper if needed? 
                # Our helper expects raw input? No, typically models expect normalized.
                # But our _deepfool_perturbation helper takes 'x' and passes it to 'model'.
                # So we must pass normalized 'x'.
                x_norm = (x - norm_mean) / norm_std
                pert = self._deepfool_perturbation(substitute, x_norm)
                batch_scores.append(pert)
                
            for score in batch_scores:
                perturbation_scores.append((candidates[cursor], score))
                cursor += 1

        # Sort by perturbation size (smallest first = closest to boundary)
        perturbation_scores.sort(key=lambda x: x[1])
        selected = [idx for idx, _ in perturbation_scores[:k]]

        return selected

    def _deepfool_perturbation(
        self,
        model: nn.Module,
        x: torch.Tensor,
        max_iter: int = 20,
        overshoot: float = 0.02,
    ) -> float:
        """Compute DeepFool perturbation size for a single sample.

        Args:
            model: Substitute model
            x: Input image [1, C, H, W]
            max_iter: Maximum DeepFool iterations
            overshoot: Overshoot parameter to cross boundary

        Returns:
            L2 norm of perturbation
        """
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

            grad_current = self._compute_input_gradient(model, x_adv, current)

            min_r = None
            best_w = None

            for k in range(num_classes):
                if k == current:
                    continue

                grad_k = self._compute_input_gradient(model, x_adv, k)
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

    def _compute_input_gradient(
        self, model: nn.Module, x: torch.Tensor, target_class: int
    ) -> torch.Tensor:
        return torch.autograd.grad(
            model(x)[0, target_class],
            x,
            retain_graph=True,
            create_graph=False,
        )[0]

    def _handle_oracle_output(
        self,
        query_batch: QueryBatch,
        oracle_output: OracleOutput,
        state: BenchmarkState,
    ) -> None:
        """Observe oracle response and update internal state.

        Args:
            query_batch: The query batch that was sent
            oracle_output: Oracle response
            state: Current benchmark state
        """
        # Store query data for training
        x_batch = query_batch.x
        y_batch = oracle_output.y

        state.attack_state["query_data_x"].append(x_batch.cpu())
        state.attack_state["query_data_y"].append(y_batch.cpu())

        # Train substitute periodically based on round step size
        labeled_count = len(state.attack_state["labeled_indices"])
        if labeled_count % self.step_size == 0 and labeled_count > 0:
            self.logger.info(f"Round limit reached (Step: {self.step_size}). Training substitute...")
            self.train_substitute(state)
            self.logger.info(f"Round Complete. Labeled: {labeled_count}, Budget Remaining: {state.budget_remaining}")

    def train_substitute(self, state: BenchmarkState) -> None:
        """Train substitute model from scratch on collected data.

        Args:
            state: Current benchmark state
        """
        device = state.metadata.get("device", "cpu")
        # Check if we have query data
        query_data_x = state.attack_state["query_data_x"]
        query_data_y = state.attack_state["query_data_y"]

        if len(query_data_x) == 0:
            return

        # Concatenate all query batches
        x_all = torch.cat(query_data_x, dim=0)
        y_all = torch.cat(query_data_y, dim=0)

        # Create dataset from query data
        class QueryDataset(torch.utils.data.Dataset):
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def __len__(self):
                return len(self.x)

            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]

        dataset = QueryDataset(x_all, y_all)

        # 20% validation split
        total_size = len(dataset)
        val_size = max(1, int(0.2 * total_size))
        train_size = total_size - val_size

        if train_size < 2:
            device = state.metadata.get("device", "cpu")
            num_classes = int(
                state.metadata.get("num_classes")
                or state.metadata.get("dataset_config", {}).get("num_classes", 10)
            )
            model = create_substitute(
                arch=self.config.get("substitute", {}).get("arch", "resnet18"),
                num_classes=num_classes,
                input_channels=state.metadata.get("input_shape", (3, 32, 32))[0],
            ).to(device)
            state.attack_state["substitute"] = model
            return

        train_subset, val_subset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Initialize fresh model
        device = state.metadata.get("device", "cpu")
        sub_config = state.metadata.get("substitute_config", {})
        num_classes = int(
            state.metadata.get("num_classes")
            or state.metadata.get("dataset_config", {}).get("num_classes", 10)
        )
        model = create_substitute(
            arch=sub_config.get("arch", "resnet18"),
            num_classes=num_classes,
            input_channels=state.metadata.get("input_shape", (3, 32, 32))[0],
        ).to(device)

        # Optimizer with L2 regularization
        import torch.optim as optim
        opt_params = sub_config.get("optimizer", {})
        # [P0 FIX] Paper mandates Adam optimizer for ActiveThief
        optimizer = optim.Adam(
            model.parameters(),
            lr=float(opt_params.get("lr", 0.001)),
            weight_decay=float(opt_params.get("weight_decay", self.l2_reg)),
        )

        # Loss function
        victim_config = state.metadata.get("victim_config", {})
        normalization = victim_config.get("normalization")
        if normalization is None:
            normalization = {"mean": [0.0], "std": [1.0]}
        
        norm_mean = torch.tensor(normalization["mean"]).view(1, -1, 1, 1).to(device)
        norm_std = torch.tensor(normalization["std"]).view(1, -1, 1, 1).to(device)

        if self.config.get("output_mode", "soft_prob") == "soft_prob":
            criterion = nn.KLDivLoss(reduction="batchmean")
        else:
            criterion = nn.CrossEntropyLoss()

        # Early stopping
        best_f1 = 0.0
        patience_counter = 0
        best_model_state = None

        # Training loop (epoch-based)
        
        epoch_pbar = tqdm(range(self.max_epochs), desc="[ActiveThief] Training Substitute", leave=False)
        for epoch in epoch_pbar:
            model.train()
            train_loss = 0.0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                # Normalize images for substitute
                x_batch = (x_batch - norm_mean) / norm_std
                
                y_batch = y_batch.to(device)

                optimizer.zero_grad()

                # Handle output modes
                if self.config.get("output_mode", "soft_prob") == "soft_prob":
                    # Soft labels: convert to log probs for KL loss
                    y_batch = y_batch.to(device)
                    # Clip probabilities to avoid log(0) and normalize
                    y_batch = torch.clamp(y_batch, min=1e-10)
                    y_batch = y_batch / y_batch.sum(dim=1, keepdim=True)
                    
                    log_outputs = torch.log_softmax(model(x_batch), dim=1)
                    loss = criterion(log_outputs, y_batch)
                else:
                    y_batch = y_batch.long()
                    outputs = model(x_batch)
                    loss = criterion(outputs, y_batch)

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            val_f1 = self._compute_f1(model, val_loader, device)
            
            # Update progress bar
            epoch_pbar.set_postfix({"Loss": f"{train_loss/len(train_loader):.4f}", "F1": f"{val_f1:.4f}"})

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.patience:
                # print(f"Early stopping at epoch {epoch}")
                break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Store in state for Track B evaluation
        state.attack_state["substitute"] = model
        self.logger.info(f"ActiveThief substitute trained. Best Val F1: {best_f1:.4f}")

        # Round Evaluation
        # self._evaluate_current_substitute(model, device)

    def _compute_f1(self, model: nn.Module, val_loader: DataLoader, device: str) -> float:
        """Compute F1 score on validation set.

        Args:
            model: Model to evaluate
            val_loader: Validation data loader
            device: Device to use

        Returns:
            F1 score (macro average)
        """
        victim_config = self.state.metadata.get("victim_config", {})
        normalization = victim_config.get("normalization")
        if normalization is None:
            normalization = {"mean": [0.0], "std": [1.0]}
            
        norm_mean = torch.tensor(normalization["mean"]).view(1, -1, 1, 1).to(device)
        norm_std = torch.tensor(normalization["std"]).view(1, -1, 1, 1).to(device)

        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                # Normalize images for substitute
                x_batch = (x_batch - norm_mean) / norm_std
                
                outputs = model(x_batch)

                # Get predictions
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)

                # Get true labels
                if len(y_batch.shape) == 2:  # Soft labels
                    targets = torch.argmax(y_batch, dim=1).cpu().numpy()
                else:  # Hard labels
                    targets = y_batch.cpu().numpy()
                all_targets.extend(targets)

        return f1_score(all_targets, all_preds, average="macro")

    def _initialize_state(self, state: BenchmarkState) -> None:
        """Initialize attack-specific state.

        Args:
            state: Global benchmark state to update
        """
        # Load pool dataset to get actual size
        if self.pool_dataset is None:
            dataset_config = self._get_pool_dataset_config(state)
            self.pool_dataset = create_dataloader(
                dataset_config,
                batch_size=1,
                shuffle=False,
            ).dataset

        pool_size = len(self.pool_dataset)
        state.attack_state["labeled_indices"] = []
        state.attack_state["unlabeled_indices"] = list(range(pool_size))
        state.attack_state["query_data_x"] = []
        state.attack_state["query_data_y"] = []
        state.attack_state["substitute"] = None
