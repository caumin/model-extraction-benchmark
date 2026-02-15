"""Unified evaluation for pool-based attacks.

This module provides a paper-agnostic, standardized training recipe to
compare pool-based attacks by the information content of their collected
transfer set.

Protocol (per run):
- Use all oracle queries collected during Track B (pool/surrogate mode).
- Split into train/val (default 80/20) with a fixed RNG seed.
- Train a substitute from scratch with the run's substitute config.
- Early-stop on validation loss and evaluate on the victim test set.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mebench.models.substitute_factory import create_substitute
from mebench.training import SubstituteTrainer, TrainRequest
from mebench.utils.dataloader import (
    pool_loader_kwargs,
    resolve_train_num_workers,
    resolve_val_num_workers,
)


def train_unified_pool_substitute(
    *,
    config: Dict[str, Any],
    state: Any,
    query_dataset: Dataset,
    device: str,
    val_ratio: float = 0.2,
    split_seed: int = 42,
) -> Optional[nn.Module]:
    """Train a substitute from scratch on collected pool queries.

    Args:
        config: Full experiment config.
        state: BenchmarkState-like object with metadata.
        query_dataset: Dataset returning (x, y) for oracle queries.
        device: Torch device string.
        val_ratio: Validation split ratio.
        split_seed: RNG seed for deterministic train/val split.

    Returns:
        Trained substitute model, or None if dataset is empty.
    """

    n_total = int(len(query_dataset))
    if n_total <= 0:
        return None

    victim_cfg = config.get("victim", {})
    sub_cfg = config.get("substitute", {})
    output_mode = str(victim_cfg.get("output_mode", "soft_prob"))

    num_classes = int(victim_cfg.get("num_classes", state.metadata.get("num_classes", 10)))
    input_channels = int(victim_cfg.get("channels", state.metadata.get("input_shape", (3, 32, 32))[0]))

    # Deterministic model init per run.
    init_seed = int(sub_cfg.get("init_seed", 42))
    torch.manual_seed(init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(init_seed)

    width_mult = int(sub_cfg.get("width_mult", 1))
    dropout_prob = float(sub_cfg.get("dropout_prob", 0.0))
    substitute = create_substitute(
        arch=str(sub_cfg.get("arch", "resnet18")),
        num_classes=num_classes,
        input_channels=input_channels,
        width_mult=width_mult,
        dropout_prob=dropout_prob,
    ).to(device)

    # Deterministic train/val split.
    val_ratio = float(val_ratio)
    if val_ratio <= 0.0:
        val_size = 0
    else:
        val_size = int(round(n_total * val_ratio))
    val_size = max(1, val_size) if n_total >= 3 else 0
    val_size = min(val_size, n_total - 2) if n_total >= 3 else 0
    train_size = n_total - val_size
    if train_size < 2:
        return None

    gen = torch.Generator().manual_seed(int(split_seed))
    train_ds, val_ds = torch.utils.data.random_split(
        query_dataset,
        [train_size, val_size],
        generator=gen,
    )

    train_batch_size = int(
        sub_cfg.get("batch_size")
        or sub_cfg.get("trackA", {}).get("batch_size")
        or 128
    )

    train_workers = resolve_train_num_workers(sub_cfg, config.get("attack", {}), default=0)
    val_workers = resolve_val_num_workers(sub_cfg, config.get("attack", {}), default=train_workers)
    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        **pool_loader_kwargs(device, {"num_workers": int(train_workers or 0)}),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_batch_size,
        shuffle=False,
        **pool_loader_kwargs(device, {"num_workers": int(val_workers or 0)}),
    )

    def loss_fn(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if output_mode == "soft_prob":
            targets = targets.to(device)
            targets = torch.clamp(targets, min=1e-10)
            targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-12)
            log_probs = F.log_softmax(outputs, dim=1)
            return F.kl_div(log_probs, targets, reduction="batchmean")
        return F.cross_entropy(outputs, targets.long().view(-1))

    def eval_fn(model_local: nn.Module, loader_local: DataLoader) -> float:
        model_local.eval()
        total_loss = 0.0
        total_count = 0
        with torch.no_grad():
            for x_val_b, y_val_b in loader_local:
                x_val_b = x_val_b.to(device)
                y_val_b = y_val_b.to(device)
                outputs = model_local(x_val_b)
                if output_mode == "soft_prob":
                    y_val_b = torch.clamp(y_val_b, min=1e-10)
                    y_val_b = y_val_b / y_val_b.sum(dim=1, keepdim=True).clamp_min(1e-12)
                    loss = F.kl_div(F.log_softmax(outputs, dim=1), y_val_b, reduction="batchmean")
                else:
                    loss = F.cross_entropy(outputs, y_val_b.long().view(-1))
                total_loss += float(loss.item()) * int(x_val_b.size(0))
                total_count += int(x_val_b.size(0))
        return total_loss / max(1, total_count)

    max_epochs = int(sub_cfg.get("max_epochs", 200))
    patience_epochs = int(sub_cfg.get("patience", 20))
    steps_per_epoch = max(1, int(math.ceil(train_size / max(1, train_batch_size))))

    trainer_config = dict(sub_cfg)
    if "grad_clip" not in trainer_config:
        trainer_config["grad_clip"] = 1.0
    trainer = SubstituteTrainer(trainer_config, device=device)
    request = TrainRequest(
        model=substitute,
        train_loader=train_loader,
        val_loader=val_loader,
        eval_fn=eval_fn,
        loss_fn=loss_fn,
        max_steps=max_epochs * steps_per_epoch,
        validate_every=steps_per_epoch,
        patience=patience_epochs * steps_per_epoch,
        early_stop_mode="min",
        load_best=True,
    )
    trainer.train(request)
    return substitute
