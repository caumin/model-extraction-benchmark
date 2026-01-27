"""ActiveThief attack implementation."""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score

from mebench.attackers.base import BaseAttack
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.data.loaders import create_dataloader
from mebench.models.substitute_factory import create_substitute


class ActiveThief(BaseAttack):
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

        # Hyperparameters (from AGENTS.md)
        self.strategy = config.get("strategy", "dfal_k_center")  # uncertainty, k_center, dfal, dfal_k_center
        default_seed = config.get("initial_seed_size")
        if default_seed is None:
            max_budget = state.metadata.get("max_budget", 1000)
            default_seed = max(1, int(0.1 * max_budget))
        self.initial_seed_size = int(default_seed)
        self.batch_size = config.get("batch_size", 150)
        self.max_epochs = config.get("max_epochs", 1000)
        self.patience = config.get("patience", 100)
        self.dropout = config.get("dropout", 0.1)
        self.l2_reg = config.get("l2_reg", 0.001)

         # Initialize attack state
        self._initialize_state(state)

        # Pool dataset (will be loaded in propose)
        self.pool_dataset = None

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

    def propose(self, k: int, state: BenchmarkState) -> QueryBatch:
        """Propose k queries using active learning strategy.

        Args:
            k: Number of queries to propose
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

        # Handle empty pool
        if len(unlabeled_indices) == 0:
            # Pool exhausted, return synthetic random queries
            input_shape = state.metadata.get("input_shape", (3, 32, 32))
            x = torch.randn(k, *input_shape)
            meta = {"indices": [], "strategy": self.strategy, "pool_exhausted": True}
            return QueryBatch(x=x, meta=meta)

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
                rho = int(self.config.get("dfal_rho", max(k, 10 * k)))
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

    def _select_uncertainty(self, k: int, state: BenchmarkState) -> list:
        """Select k samples with highest entropy (most uncertain).

        Entropy calculation:
        H_n = -sum_j  ỹ_{n,j} * log(ỹ_{n,j})

        Args:
            k: Number of samples to select
            state: Current benchmark state

        Returns:
            List of selected indices
        """
        unlabeled_indices = state.attack_state["unlabeled_indices"]
        substitute = state.attack_state["substitute"]

        if substitute is None:
            # No substitute trained yet, select random
            return np.random.choice(unlabeled_indices, k, replace=False).tolist()

        # Load unlabeled data and compute predictions
        substitute.eval()
        device = next(substitute.parameters()).device
        victim_config = state.metadata.get("victim_config", {})
        normalization = victim_config.get("normalization")
        if normalization is None:
            normalization = {"mean": [0.0], "std": [1.0]}
        
        norm_mean = torch.tensor(normalization["mean"]).view(1, -1, 1, 1).to(device)
        norm_std = torch.tensor(normalization["std"]).view(1, -1, 1, 1).to(device)
        
        entropy_scores = []

        with torch.no_grad():
            for idx in unlabeled_indices:
                img, _ = self.pool_dataset[idx]
                x = img.unsqueeze(0).to(device)  # Add batch dimension
                # Normalize images for substitute
                x = (x - norm_mean) / norm_std

                # Get probability distribution
                probs = F.softmax(substitute(x), dim=1).squeeze(0)

                # Calculate entropy: H = -sum(p * log(p))
                entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                entropy_scores.append((idx, entropy.item()))

        # Sort by entropy (highest first) and select top-k
        entropy_scores.sort(key=lambda x: x[1], reverse=True)
        selected = [idx for idx, _ in entropy_scores[:k]]

        return selected

    def _select_k_center(self, k: int, state: BenchmarkState) -> list:
        """Select k samples farthest from labeled set (K-center greedy).

        For each unlabeled sample:
        - Compute distance to nearest labeled sample
        - Select sample with maximum distance

        Args:
            k: Number of samples to select
            state: Current benchmark state

        Returns:
            List of selected indices
        """
        unlabeled_indices = state.attack_state["unlabeled_indices"]
        return self._select_k_center_candidates(k, unlabeled_indices, state)

    def _select_k_center_candidates(self, k: int, candidates: List[int], state: BenchmarkState) -> list:
        labeled_indices = state.attack_state["labeled_indices"]

        if len(labeled_indices) == 0:
            # No labeled samples yet, select random
            return np.random.choice(candidates, min(k, len(candidates)), replace=False).tolist()

        substitute = state.attack_state["substitute"]
        if substitute is None:
            return np.random.choice(candidates, min(k, len(candidates)), replace=False).tolist()

        device = next(substitute.parameters()).device
        substitute.eval()

        def extract_vector(x: torch.Tensor, use_features: bool) -> torch.Tensor:
            if use_features and hasattr(substitute, "features"):
                return substitute.features(x).squeeze(0)
            return F.softmax(substitute(x), dim=1).squeeze(0)

        def build_labeled_vecs(use_features: bool) -> List[torch.Tensor]:
            vecs = []
            with torch.no_grad():
                for idx in labeled_indices:
                    img, _ = self.pool_dataset[idx]
                    x = img.unsqueeze(0).to(device)
                    vecs.append(extract_vector(x, use_features).cpu())
            return vecs

        def compute_min_distances(
            use_features: bool, labeled_vecs: List[torch.Tensor], candidates: List[int]
        ) -> List[tuple[int, float]]:
            labeled_stack = torch.stack(labeled_vecs)
            distances = []
            with torch.no_grad():
                for idx in candidates:
                    img, _ = self.pool_dataset[idx]
                    x = img.unsqueeze(0).to(device)
                    vec = extract_vector(x, use_features).cpu()
                    dist = torch.norm(labeled_stack - vec, dim=1).min().item()
                    distances.append((idx, dist))
            return distances

        # Greedy selection of k samples
        selected = []
        remaining_indices = candidates.copy()

        use_features = False
        labeled_vecs = build_labeled_vecs(use_features)

        for _ in range(k):
            if len(remaining_indices) == 0:
                break

            dist_scores = compute_min_distances(use_features, labeled_vecs, remaining_indices)
            dists = [d for _, d in dist_scores]
            if dists and max(dists) - min(dists) < 1e-6 and hasattr(substitute, "features"):
                use_features = True
                labeled_vecs = build_labeled_vecs(use_features)
                dist_scores = compute_min_distances(use_features, labeled_vecs, remaining_indices)

            best_idx = max(dist_scores, key=lambda x: x[1])[0]
            selected.append(best_idx)
            remaining_indices.remove(best_idx)

            img, _ = self.pool_dataset[best_idx]
            x = img.unsqueeze(0).to(device)
            labeled_vecs.append(extract_vector(x, use_features).cpu())

        return selected

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
        return self._select_k_center_candidates(k, dfal_candidates, state)

    def _select_dfal(self, k: int, state: BenchmarkState) -> list:
        """Select k samples closest to decision boundary (DeepFool).

        DeepFool computes minimal perturbation needed to change prediction.
        Smaller perturbation = closer to decision boundary.

        Args:
            k: Number of samples to select
            state: Current benchmark state

        Returns:
            List of selected indices
        """
        unlabeled_indices = state.attack_state["unlabeled_indices"]

        substitute = state.attack_state["substitute"]
        if substitute is None:
            return np.random.choice(unlabeled_indices, k, replace=False).tolist()

        device = next(substitute.parameters()).device
        substitute.eval()

        perturbation_sizes = []

        for idx in unlabeled_indices:
            img, _ = self.pool_dataset[idx]
            x = img.unsqueeze(0).to(device)
            perturbation_size = self._deepfool_perturbation(substitute, x)
            perturbation_sizes.append((idx, perturbation_size))

        # Sort by perturbation size (smallest first = closest to boundary)
        perturbation_sizes.sort(key=lambda x: x[1])
        selected = [idx for idx, _ in perturbation_sizes[:k]]

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

    def observe(
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

        # Train substitute periodically for Track B evaluation
        # Every 10% of accumulated data, retrain
        labeled_count = len(state.attack_state["labeled_indices"])
        if labeled_count % 100 == 0 and labeled_count > 0:
            self.train_substitute(state)

    def train_substitute(self, state: BenchmarkState) -> None:
        """Train substitute model from scratch on collected data.

        Args:
            state: Current benchmark state
        """
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
        optimizer = optim.SGD(
            model.parameters(),
            lr=float(opt_params.get("lr", 0.001)),
            momentum=float(opt_params.get("momentum", 0.9)),
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
        for epoch in range(self.max_epochs):
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

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            print(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, Val F1: {val_f1:.4f}")

            # Early stopping
            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Store in state for Track B evaluation
        state.attack_state["substitute"] = model
        print(f"ActiveThief substitute trained. Best F1: {best_f1:.4f}")

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
        dataset_config = state.metadata.get("dataset_config", {})
        config_dataset = self.config.get("dataset", {})
        if dataset_config:
            pool_config = dataset_config
        elif config_dataset:
            pool_config = config_dataset
        else:
            pool_config = {}

        pool_size = pool_config.get("pool_size")
        if pool_size is None:
            if "seed_size" in pool_config:
                pool_size = pool_config["seed_size"]
            else:
                pool_size = 10000

        state.attack_state["labeled_indices"] = []
        state.attack_state["unlabeled_indices"] = list(range(pool_size))
        state.attack_state["query_data_x"] = []
        state.attack_state["query_data_y"] = []
        state.attack_state["substitute"] = None
