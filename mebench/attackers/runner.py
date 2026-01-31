"""Attack runner interface (IOC)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from mebench.core.context import BenchmarkContext
from mebench.core.state import BenchmarkState


import logging
import torch
import torch.nn as nn
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

    @abstractmethod
    def run(self, ctx: BenchmarkContext) -> None:
        """Execute attack protocol until budget is exhausted."""
        ...

    def _default_step_size(self, ctx: BenchmarkContext, fallback: int = 1000) -> int:
        step_size = int(self.config.get("step_size", fallback))
        if step_size <= 0:
            raise ValueError("step_size must be positive.")
        return min(step_size, ctx.budget_remaining)

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
        msg = (
            f"[{self.__class__.__name__}] [Evaluation] "
            f"Labeled: {len(self.state.attack_state.get('labeled_indices', [])) or self.state.query_count}, "
            f"Acc: {metrics.get('acc_gt', 0.0):.4f}, "
            f"Agreement: {metrics.get('agreement', 0.0):.4f}, "
            f"KL: {metrics.get('kl_mean', 0.0) or 0.0:.4f}"
        )
        self.logger.info(msg)
