from __future__ import annotations
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

_trainer_logger = logging.getLogger(__name__)


@dataclass
class TrainRequest:
    """Request object for SubstituteTrainer."""
    model: nn.Module
    train_loader: DataLoader
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    preprocess_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    step_fn: Optional[Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]] = None
    val_loader: Optional[DataLoader] = None
    eval_fn: Optional[Callable[[nn.Module, DataLoader], float]] = None
    early_stop_mode: str = "max"  # "max" or "min"
    max_steps: int = 1000
    validate_every: int = 100
    load_best: bool = True
    patience: int = 5000  # Default patience in steps (50 checks if validate_every=100)
    # When set, defer early-stop while the recent train loss is above this
    # threshold. Mirrors the official ActiveThief
    # `utils/model.py:443` rule (`if np.mean(t_loss) > 1.5: no_improvement = 0`)
    # which prevents stopping while training has not yet converged. The mean is
    # computed over the most recent `validate_every` training steps.
    train_loss_threshold: Optional[float] = None


@dataclass
class TrainResult:
    """Result object for SubstituteTrainer."""
    best_value: Optional[float] = None
    final_step: int = 0
    best_step: int = 0


class SubstituteTrainer:
    """Trainer for substitute models used by attack-native loops."""

    def __init__(self, config: Dict[str, Any], device: str = "cpu", logger: Any = None):
        """Initialize trainer.

        Args:
            config: Trainer configuration (lr, momentum, etc.)
            device: Computation device
        """
        self.config = config
        self.device = device
        self.logger = logger

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
        scheduler = self._setup_scheduler(optimizer, request.max_steps)

        best_value = -float("inf") if request.early_stop_mode == "max" else float("inf")
        best_state = None
        patience_counter = 0
        step = 0
        best_step = 0

        recent_train_losses: list[float] = []

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

            if request.step_fn is not None:
                loss = request.step_fn(model, x_batch, y_batch)
            else:
                x_in = request.preprocess_fn(x_batch) if request.preprocess_fn is not None else x_batch
                outputs = model(x_in)
                loss = request.loss_fn(outputs, y_batch)
            loss.backward()

            if "grad_clip" in self.config:
                nn.utils.clip_grad_norm_(model.parameters(), self.config["grad_clip"])

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            step += 1

            # Always track recent train losses so the per-validation log can
            # report the running mean and we can compare training curves.
            recent_train_losses.append(float(loss.detach().cpu().item()))
            if len(recent_train_losses) > int(request.validate_every):
                recent_train_losses = recent_train_losses[-int(request.validate_every):]

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
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += request.validate_every

                # Per-validation training-curve log.
                _trainer_logger.info(
                    "[Trainer] step=%d val=%.6f best=%.6f@%d patience=%d/%d lr=%.6f recent_train=%.6f",
                    step, float(current_value), float(best_value), int(best_step),
                    int(patience_counter), int(request.patience),
                    float(optimizer.param_groups[0]["lr"]),
                    (sum(recent_train_losses)/len(recent_train_losses)) if recent_train_losses else float("nan"),
                )

                if patience_counter >= request.patience:
                    # Official ActiveThief rule: defer stop while training loss
                    # remains above the threshold (model has not yet converged).
                    if (
                        request.train_loss_threshold is not None
                        and recent_train_losses
                        and (sum(recent_train_losses) / len(recent_train_losses))
                        > float(request.train_loss_threshold)
                    ):
                        patience_counter = 0
                    else:
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
        opt_config = self.config.get("optimizer", "SGD")

        if "lr" in self.config and "substitute_lr" in self.config:
            if float(self.config["lr"]) != float(self.config["substitute_lr"]):
                raise ValueError(
                    "Conflicting learning-rate keys: use canonical 'substitute_lr' or"
                    " optimizer.lr; got different values for 'lr' and 'substitute_lr'."
                )
        if "lr" in self.config and "substitute_lr" not in self.config:
            warnings.warn(
                "'lr' is deprecated for substitute training config; use 'substitute_lr' "
                "or optimizer.lr.",
                DeprecationWarning,
                stacklevel=2,
            )

        if isinstance(opt_config, dict):
            opt_name = opt_config.get("name", "SGD")
            lr = float(
                opt_config.get(
                    "lr",
                    self.config.get("substitute_lr", self.config.get("lr", 0.01)),
                )
            )
            weight_decay = float(opt_config.get("weight_decay", self.config.get("weight_decay", 5e-4)))
            momentum = float(opt_config.get("momentum", self.config.get("momentum", 0.9)))
        else:
            opt_name = opt_config
            lr = float(self.config.get("substitute_lr", self.config.get("lr", 0.01)))
            weight_decay = float(self.config.get("weight_decay", 5e-4))
            momentum = float(self.config.get("momentum", 0.9))

        name_lc = str(opt_name).lower()
        if name_lc == "adam":
            betas = (0.9, 0.999)
            if isinstance(opt_config, dict) and "betas" in opt_config:
                betas = tuple(opt_config["betas"])
            return optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas
            )
        if name_lc == "adamw":
            betas = (0.9, 0.999)
            if isinstance(opt_config, dict) and "betas" in opt_config:
                betas = tuple(opt_config["betas"])
            return optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
            )
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )

    def _setup_scheduler(
        self, optimizer: optim.Optimizer, max_steps: int
    ) -> Optional[Any]:
        """Configure LR scheduler based on `config["scheduler"]`.

        Returns None if no scheduler is requested. Supports:
        - {name: multistep, milestones_ratio: [..], gamma: ..}
        - {name: cosine, T_max: <int> | null} (defaults T_max to max_steps)
        - {name: step, step_size_ratio: .., gamma: ..}
        - null / missing → no scheduler (constant LR).

        Milestones are expressed as ratios of `max_steps` so that the same
        config produces the correct step-based milestones regardless of
        dataset size or batch size. Stepping is done per-step inside the
        train loop (not per-epoch), which matches the step-based `max_steps`.
        """
        sch = self.config.get("scheduler")
        if not sch:
            return None
        if isinstance(sch, str):
            sch = {"name": sch}
        name = str(sch.get("name", "")).lower().strip()
        if not name or name == "none" or name == "null":
            return None

        max_steps = max(1, int(max_steps))

        if name == "multistep":
            ratios = sch.get("milestones_ratio") or sch.get("milestones") or [0.5, 0.75]
            # If `milestones` is given as absolute step counts, treat as-is;
            # otherwise interpret as ratios in [0, 1].
            if all(isinstance(r, (int, float)) and 0.0 <= float(r) <= 1.0 for r in ratios):
                milestones = sorted({max(1, int(round(float(r) * max_steps))) for r in ratios})
            else:
                milestones = sorted({max(1, int(r)) for r in ratios})
            gamma = float(sch.get("gamma", 0.1))
            return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

        if name == "cosine":
            t_max = sch.get("T_max") or sch.get("t_max") or max_steps
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(t_max))

        if name == "step":
            ratio = float(sch.get("step_size_ratio", 0.3))
            step_size = max(1, int(round(ratio * max_steps)))
            gamma = float(sch.get("gamma", 0.1))
            return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        warnings.warn(
            f"Unknown scheduler name '{name}' — falling back to constant LR.",
            stacklevel=2,
        )
        return None
