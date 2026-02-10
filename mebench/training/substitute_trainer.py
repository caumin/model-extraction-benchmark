from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


@dataclass
class TrainRequest:
    """Request object for SubstituteTrainer."""
    model: nn.Module
    train_loader: DataLoader
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    val_loader: Optional[DataLoader] = None
    eval_fn: Optional[Callable[[nn.Module, DataLoader], float]] = None
    early_stop_mode: str = "max"  # "max" or "min"
    max_steps: int = 1000
    validate_every: int = 100
    load_best: bool = True
    patience: int = 5000  # Default patience in steps (50 checks if validate_every=100)


@dataclass
class TrainResult:
    """Result object for SubstituteTrainer."""
    best_value: Optional[float] = None
    final_step: int = 0
    best_step: int = 0


class SubstituteTrainer:
    """Trainer for substitute models in Track A."""

    def __init__(self, config: Dict[str, Any], device: str = "cpu"):
        """Initialize trainer.

        Args:
            config: Trainer configuration (lr, momentum, etc.)
            device: Computation device
        """
        self.config = config
        self.device = device

    def train(self, request: TrainRequest) -> TrainResult:
        """Execute training loop.

        Args:
            request: Training parameters and data

        Returns:
            TrainResult with metrics
        """
        model = request.model
        model.train()
        
        optimizer = self._setup_optimizer(model)
        
        best_value = -float("inf") if request.early_stop_mode == "max" else float("inf")
        best_state = None
        patience_counter = 0
        step = 0
        best_step = 0

        train_iter = iter(request.train_loader)

        while step < request.max_steps:
            try:
                x_batch, y_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(request.train_loader)
                x_batch, y_batch = next(train_iter)

            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = request.loss_fn(outputs, y_batch)
            loss.backward()

            if "grad_clip" in self.config:
                nn.utils.clip_grad_norm_(model.parameters(), self.config["grad_clip"])

            optimizer.step()
            step += 1

            # Validation
            if (
                request.val_loader 
                and request.eval_fn 
                and (step % request.validate_every == 0 or step == request.max_steps)
            ):
                current_value = request.eval_fn(model, request.val_loader)
                model.train()  # Switch back to train mode

                is_improvement = (
                    current_value > best_value
                    if request.early_stop_mode == "max"
                    else current_value < best_value
                )

                if is_improvement:
                    best_value = current_value
                    best_step = step
                    best_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += request.validate_every

                if patience_counter >= request.patience:
                    break

        # Load best model if requested
        if request.load_best and best_state is not None:
            model.load_state_dict(best_state)

        return TrainResult(
            best_value=best_value if best_state is not None else None,
            final_step=step,
            best_step=best_step
        )

    def _setup_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Configure optimizer based on config."""
        lr = float(self.config.get("lr", 0.01))
        weight_decay = float(self.config.get("weight_decay", 5e-4))
        momentum = float(self.config.get("momentum", 0.9))
        opt_name = self.config.get("optimizer", "SGD")

        if opt_name.lower() == "adam":
            return optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            return optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
