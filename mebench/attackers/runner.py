"""Attack runner interface (IOC)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict
import math

from mebench.core.context import BenchmarkContext
from mebench.core.state import BenchmarkState


import logging
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from mebench.core.context import BenchmarkContext
from mebench.core.state import BenchmarkState
from mebench.data.loaders import get_test_dataloader
from mebench.eval.metrics import evaluate_substitute

class AttackRunner(ABC):
    """Base class for attack runners (Track B)."""

    def __init__(self, config: Dict[str, Any], state: BenchmarkState) -> None:
        self.config = config
        self.state = state
        self.logger = logging.getLogger(self.__class__.__name__)
        self.test_loader = None
        self.victim = None
        self.ctx = None  # [ADDED] Context for artifact logging

    @abstractmethod
    def run(self, ctx: BenchmarkContext) -> None:
        """Execute attack protocol until budget is exhausted."""
        ...

    def _default_step_size(self, ctx: BenchmarkContext, fallback: int = 1000) -> int:
        if "step_size" in self.config:
            step_size = int(self.config.get("step_size"))
        else:
            total_budget = (
                self.state.metadata.get("max_budget")
                or self.config.get("max_budget")
                or ctx.budget_remaining
                or fallback
            )
            if total_budget <= 0:
                total_budget = fallback
            step_size = max(1, int(math.ceil(int(total_budget) / 10)))
            config_batch_size = self.config.get("batch_size")
            attr_batch_size = getattr(self, "batch_size", None)
            max_step = config_batch_size if config_batch_size is not None else attr_batch_size
            if max_step is not None:
                step_size = min(step_size, int(max_step))
        if step_size <= 0:
            raise ValueError("step_size must be positive.")
        return min(step_size, ctx.budget_remaining)

    def _create_progress_bar(self, total: int, desc: str) -> tqdm:
        miniters = int(self.config.get("log_miniters", 0))
        if miniters <= 0:
            miniters = max(1, int(total // 100))
        mininterval = float(self.config.get("log_mininterval", 1.0))
        return tqdm(total=total, desc=desc, miniters=miniters, mininterval=mininterval)

    def _evaluate_current_substitute(self, substitute: nn.Module, device: str) -> None:
        """Perform periodic evaluation on substitute model."""
        if substitute is None or self.victim is None:
            return

        if self.test_loader is None:
            dataset_name = self.state.metadata.get("dataset_config", {}).get("name", "CIFAR10")
            self.test_loader = get_test_dataloader(dataset_name, batch_size=128)

        metrics = evaluate_substitute(
            substitute=substitute,
            victim=self.victim,
            test_loader=self.test_loader,
            device=device,
            output_mode=self.config.get("output_mode", "soft_prob")
        )
        
        current_queries = self.state.query_count
        # Handle cases where query_count is 0 but we have labeled data (e.g. initial seed)
        if current_queries == 0:
            current_queries = len(self.state.attack_state.get('labeled_indices', []))

        msg = (
            f"[{self.__class__.__name__}] [Evaluation] "
            f"Labeled: {current_queries}, "
            f"Acc: {metrics.get('acc_gt', 0.0):.4f}, "
            f"Agreement: {metrics.get('agreement', 0.0):.4f}, "
            f"KL: {metrics.get('kl_mean', 0.0) or 0.0:.4f}"
        )
        self.logger.info(msg)

        # [ADDED] Log to artifacts if context is available
        if self.ctx:
            # Log history (time-series)
            self.ctx.logger.log_history(step=current_queries, metrics=metrics)
            
            # Log checkpoint (metrics.csv)
            # Use 'track_b' as default since we are running the attacker's native loop
            seed = self.state.metadata.get("seed", 0)
            self.ctx.logger.log_checkpoint(
                seed=seed,
                checkpoint=current_queries,
                track="track_b",
                metrics=metrics,
            )
            
            # Force save to ensure persistence even if crashed later
            self.ctx.logger.save_metrics_csv()

    def _build_optimizer(
        self, params: Any, opt_config: Dict[str, Any]
    ) -> optim.Optimizer:
        name = str(opt_config.get("name", "sgd")).lower()
        lr = float(opt_config.get("lr", 0.01))
        weight_decay = float(opt_config.get("weight_decay", 5e-4))

        if name == "adam":
            betas = opt_config.get("betas")
            if betas is not None:
                return optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=tuple(betas))
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)

        if name == "adamw":
            betas = opt_config.get("betas")
            if betas is not None:
                return optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=tuple(betas))
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        if name != "sgd":
            self.logger.warning("Unknown optimizer '%s', defaulting to SGD", name)

        momentum = float(opt_config.get("momentum", 0.9))
        return optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
