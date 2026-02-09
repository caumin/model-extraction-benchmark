"""Standardized substitute training utilities."""

from dataclasses import dataclass
import logging
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


@dataclass
class TrainRequest:
    """Container for a single training run."""

    model: nn.Module
    train_loader: DataLoader
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
    step_fn: Optional[
        Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None
    val_loader: Optional[DataLoader] = None
    eval_fn: Optional[Callable[[nn.Module, DataLoader], float]] = None
    early_stop_mode: str = "min"  # "min" or "max"
    max_steps: Optional[int] = None
    validate_every: int = 100
    preprocess_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    skip_on_error: bool = False
    load_best: bool = True


@dataclass
class TrainResult:
    """Training outcome summary."""

    best_value: Optional[float]
    epochs_ran: int
    steps_ran: int
    stopped_early: bool


class SubstituteTrainer:
    """Standard training loop for substitute models."""

    def __init__(
        self,
        config: Dict[str, Any],
        device: str,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.device = device
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.max_epochs = int(self.config.get("max_epochs", 1000))
        self.patience = self.config.get("patience", 100)
        if self.patience is not None:
            self.patience = int(self.patience)
        self.grad_clip = self.config.get("grad_clip")

    def _build_optimizer(self, params: Any) -> optim.Optimizer:
        opt_config = self.config.get("optimizer", {})
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

    def train(self, req: TrainRequest) -> TrainResult:
        """Train model using standardized loop."""
        if req.step_fn is None and req.loss_fn is None:
            raise ValueError("Either step_fn or loss_fn must be provided.")

        model = req.model
        model.train()

        optimizer = self._build_optimizer(model.parameters())
        best_value: Optional[float] = None
        best_state: Optional[Dict[str, torch.Tensor]] = None
        patience_counter = 0
        steps_ran = 0
        stopped_early = False
        
        # [FEATURE] Optional tqdm progress bar if requested by caller config
        # We check self.config explicitly for 'use_tqdm'
        use_tqdm = bool(self.config.get("use_tqdm", False))
        
        # Helper for progress bar description
        def get_pbar_desc(val_metric: Optional[float]) -> str:
            val_str = f"{val_metric:.4f}" if val_metric is not None else "N/A"
            return f"[Trainer] Epoch {epochs_ran+1}/{self.max_epochs} | Val: {val_str}"

        def is_improved(value: float) -> bool:
            if best_value is None:
                return True
            if req.early_stop_mode == "max":
                return value > best_value
            return value < best_value

        def run_validation() -> Optional[float]:
            if req.eval_fn is None or req.val_loader is None:
                return None
            model.eval()
            with torch.no_grad():
                value = req.eval_fn(model, req.val_loader)
            model.train()
            return float(value)

        max_steps = req.max_steps
        if max_steps is not None:
            max_steps = int(max_steps)
            if max_steps <= 0:
                return TrainResult(best_value=None, epochs_ran=0, steps_ran=0, stopped_early=False)
            
            pbar = None
            if use_tqdm:
                from tqdm import tqdm
                pbar = tqdm(total=max_steps, desc="[Trainer] Steps", leave=False)

            while steps_ran < max_steps:
                for x_batch, y_batch in req.train_loader:
                    if steps_ran >= max_steps:
                        break

                    try:
                        x_batch = x_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        if req.preprocess_fn is not None:
                            x_batch = req.preprocess_fn(x_batch)

                        if req.step_fn is None:
                            outputs = model(x_batch)
                            loss = req.loss_fn(outputs, y_batch)
                        else:
                            loss = req.step_fn(model, x_batch, y_batch)

                        optimizer.zero_grad()
                        loss.backward()
                        if self.grad_clip is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(self.grad_clip))
                        optimizer.step()
                        steps_ran += 1
                        
                        if pbar:
                            pbar.update(1)
                            
                    except ValueError:
                        if req.skip_on_error:
                            continue
                        raise

                    if steps_ran % req.validate_every == 0 or steps_ran >= max_steps:
                        value = run_validation()
                        if value is None:
                            value = float(loss.item())
                        
                        if pbar:
                            pbar.set_postfix({"val": f"{value:.4f}", "best": f"{best_value:.4f}" if best_value else "None"})

                        if is_improved(value):
                            best_value = value
                            patience_counter = 0
                            if req.load_best:
                                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        else:
                            if self.patience is not None:
                                patience_counter += 1
                                if patience_counter >= self.patience:
                                    stopped_early = True
                                    break
                
                if stopped_early:
                    break
            
            if pbar:
                pbar.close()

            if req.load_best and best_state is not None:
                model.load_state_dict(best_state)

            return TrainResult(
                best_value=best_value,
                epochs_ran=0,
                steps_ran=steps_ran,
                stopped_early=stopped_early,
            )

        epochs_ran = 0
        pbar = None
        if use_tqdm:
            from tqdm import tqdm
            pbar = tqdm(total=self.max_epochs, desc="[Trainer] Epochs", leave=False)
            
        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            batch_count = 0

            for x_batch, y_batch in req.train_loader:
                try:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    if req.preprocess_fn is not None:
                        x_batch = req.preprocess_fn(x_batch)

                    if req.step_fn is None:
                        outputs = model(x_batch)
                        loss = req.loss_fn(outputs, y_batch)
                    else:
                        loss = req.step_fn(model, x_batch, y_batch)

                    optimizer.zero_grad()
                    loss.backward()
                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(self.grad_clip))
                    optimizer.step()

                    epoch_loss += float(loss.item())
                    batch_count += 1
                    steps_ran += 1
                except ValueError:
                    if req.skip_on_error:
                        continue
                    raise

            if batch_count == 0:
                break

            epochs_ran += 1
            value = run_validation()
            if value is None:
                value = epoch_loss / batch_count

            if pbar:
                pbar.update(1)
                pbar.set_postfix({"val": f"{value:.4f}", "patience": f"{patience_counter}/{self.patience}"})

            if is_improved(value):
                best_value = value
                patience_counter = 0
                if req.load_best:
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                if self.patience is not None:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        stopped_early = True
                        break
        
        if pbar:
            pbar.close()

        if req.load_best and best_state is not None:
            model.load_state_dict(best_state)

        return TrainResult(
            best_value=best_value,
            epochs_ran=epochs_ran,
            steps_ran=steps_ran,
            stopped_early=stopped_early,
        )
