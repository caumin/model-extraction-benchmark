"""ActiveThief attack implementation with DFAL support using vectorized DeepFool."""

from typing import Dict, Any, List, Tuple, Optional
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from mebench.attackers.runner import AttackRunner
from mebench.core.context import BenchmarkContext
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.data.loaders import create_dataloader
from mebench.models.substitute_factory import create_substitute
from mebench.utils.adversarial import deepfool_distance_vectorized, DeepFoolAttack


class ActiveThief(AttackRunner):
    """ActiveThief with uncertainty, k-center, and DFAL sampling strategies.
    
    Algorithm loop (from AGENTS.md):
    1. Initialize: Select random initial seed S0 from thief dataset
    2. Query: Send S_i to victim f to get labels D_i
    3. Train: Train substitute model f~ from scratch on all collected data ∪D_t
    4. Evaluate: Predict on remaining pool (unlabeled thief data)
    5. Select: Use active learning strategy to select next queries S_{i+1}
    6. Repeat: Continue until budget exhausted
    
    Selection strategies:
    - uncertainty: Entropy-based sampling
    - k_center: K-center greedy selection
    - dfal: DeepFool Active Learning (smallest perturbations)
    - dfal_k_center: DFAL pre-filtering + K-center selection
    """

    def __init__(self, config: dict, state: BenchmarkState):
        super().__init__(config, state)

        # Basic parameters
        self.batch_size = int(config.get("batch_size", 128))
        self.num_classes = int(
            self.state.metadata.get("num_classes")
            or self.config.get("num_classes")
            or self.state.metadata.get("dataset_config", {}).get("num_classes", 10)
        )
        
        # Active learning strategy
        self.strategy = config.get("strategy", "uncertainty")
        self.initial_seed_size = int(config.get("initial_seed_size", 50))
        self.step_size = int(config.get("step_size", 50))
        
        # DeepFool parameters for DFAL
        self.dfal_max_iter = int(config.get("dfal_max_iter", 20))
        self.dfal_rho = int(config.get("dfal_rho", 1000))  # Pre-filter size
        
        # Datasets and model
        self.pool_dataset = None
        self.labeled_indices = []
        self.unlabeled_indices = []
        self.substitute = None
        self.substitute_optimizer = None

        self._initialize_state(state)

    def _initialize_state(self, state: BenchmarkState) -> None:
        """Initialize attack-specific state."""
        state.attack_state["labeled_indices"] = []
        state.attack_state["unlabeled_indices"] = []
        state.attack_state["round"] = 0
        state.attack_state["initialized"] = False

    def _setup_datasets(self, state: BenchmarkState) -> None:
        """Setup pool dataset and initial seed selection."""
        if state.attack_state.get("initialized"):
            return
            
        # Get surrogate dataset (thief dataset)
        surrogate_name = self.state.metadata.get("surrogate_name")
        input_shape = self.state.metadata.get("input_shape", (3, 32, 32))
        
        self.pool_dataset = create_dataloader(
            dataset_name=surrogate_name,
            batch_size=self.batch_size,
            train=True,
            input_size=input_shape[1:],
            shuffle=False,
            drop_last=False,
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
        
        state.attack_state["labeled_indices"] = self.labeled_indices
        state.attack_state["unlabeled_indices"] = self.unlabeled_indices
        state.attack_state["initialized"] = True

    def _create_substitute(self, input_shape: tuple) -> nn.Module:
        """Create substitute model."""
        sub_config = self.state.metadata.get("substitute_config", {})
        arch = sub_config.get("arch", "resnet18")
        input_channels = int(input_shape[0])
        
        return create_substitute(
            arch=arch,
            num_classes=self.num_classes,
            input_channels=input_channels,
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
        
        self.substitute_optimizer = optim.SGD(
            self.substitute.parameters(),
            lr=float(opt_config.get("lr", 0.01)),  # Global contract
            momentum=float(opt_config.get("momentum", 0.9)),
            weight_decay=float(opt_config.get("weight_decay", 5e-4))
        )
        
        # Create labeled dataset
        labeled_dataset = Subset(self.pool_dataset, self.labeled_indices)
        labeled_loader = DataLoader(
            labeled_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=0
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
            for x_batch, y_batch in labeled_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                self.substitute_optimizer.zero_grad()
                logits = self.substitute(x_batch)
                
                # Use cross-entropy loss
                loss = F.cross_entropy(logits, y_batch)
                loss.backward()
                self.substitute_optimizer.step()
                
                epoch_loss += loss.item()
            
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
        """Select samples with highest entropy."""
        # Compute entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        _, indices = torch.topk(entropy, k)
        return indices.cpu().tolist()

    def _select_k_center(self, features: torch.Tensor, k: int) -> List[int]:
        """Select samples using k-center greedy algorithm."""
        selected = []
        remaining = list(range(features.shape[0]))
        
        # Initialize with a random point
        if remaining:
            first_idx = np.random.choice(remaining)
            selected.append(first_idx)
            remaining.remove(first_idx)
        
        # Greedy selection
        for _ in range(min(k - 1, len(remaining))):
            if not remaining:
                break
                
            max_dist = -1
            best_idx = None
            
            for idx in remaining:
                # Find minimum distance to selected set
                selected_features = features[selected]
                dists = torch.norm(features[idx:idx+1] - selected_features, dim=1)
                min_dist = torch.min(dists).item()
                
                if min_dist > max_dist:
                    max_dist = min_dist
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
        
        return selected

    def _select_dfal(self, state: BenchmarkState, k: int) -> List[int]:
        """Select samples using DeepFool Active Learning."""
        device = state.metadata.get("device", "cpu")
        
        # Get unlabeled data
        unlabeled_dataset = Subset(self.pool_dataset, self.unlabeled_indices)
        unlabeled_loader = DataLoader(
            unlabeled_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=0
        )
        
        all_features = []
        all_distances = []
        
        self.substitute.eval()
        with torch.no_grad():
            for x_batch, _ in unlabeled_loader:
                x_batch = x_batch.to(device)
                
                # Get features (penultimate layer)
                # This is a simplified approach - in practice, you'd extract from model
                logits = self.substitute(x_batch)
                probs = F.softmax(logits, dim=1)
                all_features.append(probs.cpu())
                
                # Compute DeepFool distances
                distances = deepfool_distance_vectorized(
                    self.substitute, x_batch, 
                    max_iter=self.dfal_max_iter, 
                    batch_size=min(32, x_batch.shape[0])
                )
                all_distances.append(distances.cpu())
        
        features = torch.cat(all_features, dim=0)
        distances = torch.cat(all_distances, dim=0)
        
        # Select k samples with smallest DeepFool distances
        _, selected_local = torch.topk(distances, k, largest=False)
        selected_indices = [self.unlabeled_indices[i] for i in selected_local.tolist()]
        
        return selected_indices

    def _select_dfal_k_center(self, state: BenchmarkState, k: int) -> List[int]:
        """DFAL pre-filtering + K-center selection."""
        # Pre-filter with DFAL to get rho candidates
        rho = min(self.dfal_rho, len(self.unlabeled_indices))
        dfal_candidates = self._select_dfal(state, rho)
        
        # Create candidate index mapping
        candidate_to_local = {idx: i for i, idx in enumerate(dfal_candidates)}
        
        # Get features for candidates only
        device = state.metadata.get("device", "cpu")
        candidate_dataset = Subset(self.pool_dataset, dfal_candidates)
        candidate_loader = DataLoader(
            candidate_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=0
        )
        
        all_probs = []
        self.substitute.eval()
        with torch.no_grad():
            for x_batch, _ in candidate_loader:
                x_batch = x_batch.to(device)
                logits = self.substitute(x_batch)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu())
        
        candidate_features = torch.cat(all_probs, dim=0)
        
        # Apply k-center on candidates
        selected_local = self._select_k_center(candidate_features, k)
        selected_indices = [dfal_candidates[i] for i in selected_local]
        
        return selected_indices

    def _select_query_batch(self, k: int, state: BenchmarkState) -> QueryBatch:
        """Select next batch of queries using active learning strategy."""
        if not state.attack_state.get("initialized"):
            self._setup_datasets(state)
        
        # Train substitute if we have labeled data
        if self.labeled_indices and not self.substitute:
            self._train_substitute(state)
        
        if not self.unlabeled_indices:
            # No more unlabeled samples
            return QueryBatch(
                x=torch.empty(0, *self.state.metadata.get("input_shape", (3, 32, 32))),
                meta={"strategy": self.strategy, "status": "exhausted"}
            )
        
        # Select samples based on strategy
        if self.strategy == "uncertainty":
            selected_indices = self._select_uncertainty_strategy(state, k)
        elif self.strategy == "k_center":
            selected_indices = self._select_k_center_strategy(state, k)
        elif self.strategy == "dfal":
            selected_indices = self._select_dfal(state, k)
        elif self.strategy == "dfal_k_center":
            selected_indices = self._select_dfal_k_center(state, k)
        else:
            # Default: random
            selected_indices = np.random.choice(
                self.unlabeled_indices, 
                size=min(k, len(self.unlabeled_indices)), 
                replace=False
            ).tolist()
        
        # Update labeled/unlabeled splits
        for idx in selected_indices:
            if idx in self.unlabeled_indices:
                self.unlabeled_indices.remove(idx)
                self.labeled_indices.append(idx)
        
        # Update state
        state.attack_state["labeled_indices"] = self.labeled_indices
        state.attack_state["unlabeled_indices"] = self.unlabeled_indices
        state.attack_state["round"] += 1
        
        # Create query batch
        selected_dataset = Subset(self.pool_dataset, selected_indices)
        query_loader = DataLoader(selected_dataset, batch_size=k, shuffle=False, num_workers=0)
        x_batch, _ = next(iter(query_loader))
        
        return QueryBatch(
            x=x_batch,
            meta={
                "strategy": self.strategy,
                "selected_indices": selected_indices,
                "round": state.attack_state["round"],
                "labeled_size": len(self.labeled_indices),
                "unlabeled_size": len(self.unlabeled_indices)
            }
        )

    def _select_uncertainty_strategy(self, state: BenchmarkState, k: int) -> List[int]:
        """Uncertainty sampling implementation."""
        device = state.metadata.get("device", "cpu")
        unlabeled_dataset = Subset(self.pool_dataset, self.unlabeled_indices)
        unlabeled_loader = DataLoader(
            unlabeled_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=0
        )
        
        all_probs = []
        self.substitute.eval()
        with torch.no_grad():
            for x_batch, _ in unlabeled_loader:
                x_batch = x_batch.to(device)
                logits = self.substitute(x_batch)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu())
        
        probs = torch.cat(all_probs, dim=0)
        selected_local = self._select_uncertainty(probs, k)
        selected_indices = [self.unlabeled_indices[i] for i in selected_local]
        
        return selected_indices

    def _select_k_center_strategy(self, state: BenchmarkState, k: int) -> List[int]:
        """K-center sampling implementation."""
        device = state.metadata.get("device", "cpu")
        unlabeled_dataset = Subset(self.pool_dataset, self.unlabeled_indices)
        unlabeled_loader = DataLoader(
            unlabeled_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=0
        )
        
        all_probs = []
        self.substitute.eval()
        with torch.no_grad():
            for x_batch, _ in unlabeled_loader:
                x_batch = x_batch.to(device)
                logits = self.substitute(x_batch)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu())
        
        features = torch.cat(all_probs, dim=0)
        selected_local = self._select_k_center(features, k)
        selected_indices = [self.unlabeled_indices[i] for i in selected_local]
        
        return selected_indices

    def run(self, ctx: BenchmarkContext) -> None:
        """Run ActiveThief attack."""
        state = ctx.state
        
        # Setup datasets on first run
        if not state.attack_state.get("initialized"):
            self._setup_datasets(state)
        
        # Main loop - query until budget exhausted
        while ctx.budget_remaining > 0 and self.unlabeled_indices:
            # Determine step size
            step_size = min(self.step_size, ctx.budget_remaining, len(self.unlabeled_indices))
            
            # Select and query batch
            query_batch = self._select_query_batch(step_size, state)
            
            if query_batch.x.shape[0] == 0:
                break
            
            # Send query and observe output
            oracle_output = ctx.oracle.query(query_batch.x)
            self.observe(query_batch, oracle_output, state)
            
            # Retrain substitute for next round
            if self.labeled_indices:
                self._train_substitute(state)

    def observe(
        self, 
        query_batch: QueryBatch, 
        oracle_output: OracleOutput, 
        state: BenchmarkState
    ) -> None:
        """Observe oracle outputs and update attack state."""
        # Store query results (labels are in oracle_output.logits for hard labels)
        # This is handled by the runner base class
        pass