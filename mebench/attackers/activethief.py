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
from mebench.utils.dataloader import pool_loader_kwargs


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

        # Cache queried tensors to avoid repeated ImageFolder IO during training.
        # Stored as list-of-batches to minimize Python per-sample overhead.
        self.query_data_x: List[torch.Tensor] = []
        self.query_data_y: List[torch.Tensor] = []
        self.use_soft_labels = False
        
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
        self.initial_seed_indices = []

        state.attack_state["labeled_indices"] = self.labeled_indices
        state.attack_state["unlabeled_indices"] = self.unlabeled_indices
        state.attack_state["initial_seed_indices"] = []
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
        
        sub_config = self.state.metadata.get("substitute_config", {})
        
        # IO optimization: train on cached queried tensors (exact x used for oracle queries)
        # instead of re-indexing ImageFolder each epoch.
        query_x = state.attack_state.get("query_data_x", [])
        query_y = state.attack_state.get("query_data_y", [])
        if len(query_x) == 0:
            return

        x_all = torch.cat(query_x, dim=0)
        y_all = torch.cat(query_y, dim=0)
        
        # Check if we have soft labels
        is_soft = (y_all.dim() > 1 and y_all.shape[1] > 1)
        
        full_dataset = torch.utils.data.TensorDataset(x_all, y_all)
        
        # Split into train/val (80/20) as per ActiveThief paper
        total_len = len(full_dataset)
        val_len = int(0.2 * total_len)
        train_len = total_len - val_len
        
        # Use fixed generator for reproducibility if needed, or rely on global seed
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_len, val_len]
        )

        train_batch_size = int(
            sub_config.get("batch_size")
            or sub_config.get("trackA", {}).get("batch_size", self.batch_size)
        )
        
        # Create loaders
        labeled_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=str(device).startswith("cuda"),
        )
        
 
