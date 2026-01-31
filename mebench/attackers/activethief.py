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
        
        # [Corrected] Pass dropout probability from config (Default: 0.1 for ActiveThief)
        dropout_prob = float(self.config.get("dropout_prob", 0.0))

        return create_substitute(
            arch=arch,
            num_classes=self.num_classes,
            input_channels=input_channels,
            dropout_prob=dropout_prob
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
        
        self.substitute_optimizer = optim.Adam(
            self.substitute.parameters(),
            lr=float(opt_config.get("lr", 0.001)),  # Default for Adam per AGENTS.md
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

    def _select_dfal(self, state: BenchmarkState, k: int) -> List[int]:
        """Select samples using DeepFool Active Learning."""
        device = state.metadata.get("device", "cpu")
        unlabeled_dataset = Subset(self.pool_dataset, self.unlabeled_indices)
        unlabeled_loader = DataLoader(
            unlabeled_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=0
        )
        
        all_distances = []
        self.substitute.eval()
        
        # We process in batches
        with torch.no_grad():
             # Note: deepfool_distance_vectorized uses gradients, so we need grad enabled inside the function
             # But here we are in no_grad block. 
             # Actually, deepfool_distance_vectorized usually enables grad on input.
             # Let's check logic. We should NOT be in no_grad context if we need input grad.
             pass
             
        for x_batch, _ in unlabeled_loader:
            x_batch = x_batch.to(device)
            # Enable grad for DeepFool
            with torch.enable_grad():
                distances = deepfool_distance_vectorized(
                    self.substitute, x_batch, 
                    max_iter=self.dfal_max_iter, 
                    batch_size=min(32, x_batch.shape[0])
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
        rho = min(self.dfal_rho, len(self.unlabeled_indices))
        dfal_candidates = self._select_dfal(state, rho)
        
        # Map candidates back to local indices in the pool
        # This is tricky because _extract_features expects a loader.
        
        # Create loader for candidates
        candidate_dataset = Subset(self.pool_dataset, dfal_candidates)
        candidate_loader = DataLoader(
            candidate_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=0
        )
        
        device = state.metadata.get("device", "cpu")
        _, features = self._extract_features(candidate_loader, device)
        
        # Apply k-center on candidates
        selected_local_in_candidates = self._select_k_center(features.to(device), k)
        selected_indices = [dfal_candidates[i] for i in selected_local_in_candidates]
        
        return selected_indices

    def run(self, ctx: BenchmarkContext) -> None:
        """Run ActiveThief attack."""
        state = ctx.state
        
        if not state.attack_state.get("initialized"):
            self._setup_datasets(state)
        
        while ctx.budget_remaining > 0 and self.unlabeled_indices:
            step_size = min(self.step_size, ctx.budget_remaining, len(self.unlabeled_indices))
            
            query_batch = self._select_query_batch(step_size, state)
            
            if query_batch.x.shape[0] == 0:
                break
            
            oracle_output = ctx.oracle.query(query_batch.x)
            self.observe(query_batch, oracle_output, state)
            
            if self.labeled_indices:
                self._train_substitute(state)

    def observe(
        self, 
        query_batch: QueryBatch, 
        oracle_output: OracleOutput, 
        state: BenchmarkState
    ) -> None:
        """Observe oracle outputs and update attack state."""
        # ActiveThief updates are handled in _select_query_batch (state update)
        # and _train_substitute (model update).
        # This method is kept for interface compliance.
        pass
