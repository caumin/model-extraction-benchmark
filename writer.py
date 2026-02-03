f = open("mebench/attackers/activethief.py", "w")
f.write(r'''"""ActiveThief attack implementation."""

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
    - K-center: Select samples farthest from labeled set in probability space (Euclidean)
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

        # Hyperparameters (from official repo: shukla-aditya-csa/activethief)
        self.strategy = config.get("strategy", "dfal_k_center")  # uncertainty, k_center, dfal, dfal_k_center
        self.num_rounds = int(config.get("num_rounds", 10))
        
        max_budget = state.metadata.get("max_budget", 1000)
        # 10% initial seed per paper/repo heuristic
        self.initial_seed_size = int(config.get("initial_seed_size", max(1, int(0.1 * max_budget))))
        
        # Iteration step size (budget / rounds)
        self.step_size = max(1, max_budget // self.num_rounds)
        
        # Official Repo Hyperparameters
        self.batch_size = config.get("batch_size", 50)  # Repo default: 50
        self.max_epochs = config.get("max_epochs", 1000) # Repo default: 1000
        self.patience = config.get("patience", 20)      # Repo default: 20
        self.dropout = config.get("dropout", 0.5)       # Repo default: 0.5 (keep_prob=0.5)
        self.l2_reg = config.get("l2_reg", 0.001)       # Repo default: 0.001

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Pool dataset (loaded during selection)
        self.pool_dataset = None
        self.test_loader = None
        self.victim = None

        # Initialize state (will set up indices but not load pool yet)
        self._initialize_state(state)

''')
f.write(r'''    def run(self, ctx: BenchmarkContext) -> None:
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

''')
f.write(r'''    def _get_approx_probs(self, indices: List[int], substitute: nn.Module, batch_size: int = 512) -> torch.Tensor:
        """Step 4: Compute approximate labels (softmax probabilities) for candidate pool.
        
        Uses efficient batch processing to minimize overhead.
        """
        device = next(substitute.parameters()).device
        substitute.eval()
        
        # Create temporary dataloader for efficient batch inference
        subset = Subset(self.pool_dataset, indices)
        loader = DataLoader(subset, batc
