"""Evaluation metrics for model extraction benchmark."""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mebench.utils.binary import (
    binary_distribution_from_logits,
    binary_hard_labels_from_logits,
    binary_positive_probs_from_logits,
    ensure_single_logit_shape,
)


def compute_accuracy(model: nn.Module, loader: DataLoader, device: str) -> float:
    """Compute top-1 accuracy on a dataset."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            if outputs.dim() == 1 or (outputs.dim() == 2 and outputs.size(1) == 1):
                predicted = binary_hard_labels_from_logits(outputs)
            else:
                _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    return correct / total if total > 0 else 0.0


def _to_binary_probabilities(logits: torch.Tensor) -> Optional[torch.Tensor]:
    """Return a 2-class probability tensor for single-logit binary outputs."""
    try:
        return binary_distribution_from_logits(logits)
    except ValueError:
        return None


def compute_agreement(
    substitute: nn.Module, victim: nn.Module, loader: DataLoader, device: str
) -> float:
    """Compute label agreement between substitute and victim."""
    substitute.eval()
    victim.eval()
    matches = 0
    total = 0

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            out_sub = substitute(x)
            out_vic = victim(x)

            if out_sub.dim() == 1 or (out_sub.dim() == 2 and out_sub.size(1) == 1):
                pred_sub = binary_hard_labels_from_logits(out_sub)
            else:
                pred_sub = torch.argmax(out_sub, dim=1)

            if out_vic.dim() == 1 or (out_vic.dim() == 2 and out_vic.size(1) == 1):
                pred_vic = binary_hard_labels_from_logits(out_vic)
            else:
                pred_vic = torch.argmax(out_vic, dim=1)

            matches += (pred_sub == pred_vic).sum().item()
            total += x.size(0)

    return matches / total if total > 0 else 0.0


def compute_kl_divergence(
    substitute: nn.Module,
    victim: nn.Module,
    loader: DataLoader,
    device: str,
    temperature: float = 1.0,
) -> float:
    """Compute KL divergence between victim and substitute probabilities."""
    substitute.eval()
    victim.eval()
    total_kl = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)

            sub_logits = substitute(x) / temperature
            vic_logits = victim(x) / temperature
            sub_probs = _to_binary_probabilities(sub_logits)
            vic_probs = _to_binary_probabilities(vic_logits)

            if sub_probs is None:
                sub_probs = F.softmax(sub_logits, dim=1)
            if vic_probs is None:
                vic_probs = F.softmax(vic_logits, dim=1)

            kl = F.kl_div(
                torch.log(sub_probs + 1e-10),
                vic_probs,
                reduction="sum",
                log_target=False,
            )
            total_kl += kl.item()
            total_samples += x.size(0)

    return total_kl / total_samples if total_samples > 0 else 0.0


def compute_l1_distance(
    substitute: nn.Module,
    victim: nn.Module,
    loader: DataLoader,
    device: str,
    temperature: float = 1.0,
) -> float:
    """Compute L1 distance between victim and substitute probabilities."""
    substitute.eval()
    victim.eval()
    total_l1 = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)

            sub_logits = substitute(x) / temperature
            vic_logits = victim(x) / temperature
            sub_probs = _to_binary_probabilities(sub_logits)
            vic_probs = _to_binary_probabilities(vic_logits)

            if sub_probs is None:
                sub_probs = F.softmax(sub_logits, dim=1)
            if vic_probs is None:
                vic_probs = F.softmax(vic_logits, dim=1)

            l1 = F.l1_loss(sub_probs, vic_probs, reduction="sum")
            total_l1 += l1.item()
            total_samples += x.size(0)

    if total_samples == 0:
        return 0.0

    return total_l1 / total_samples / sub_probs.size(1)


def _binary_labels_from_batch(
    y_batch: torch.Tensor,
    class_count: int,
) -> torch.Tensor | None:
    """Convert labels into binary integer labels for single-logit binaries."""
    if y_batch.dim() > 1 and y_batch.size(-1) == 1:
        y_batch = y_batch.squeeze(-1)

    if y_batch.dim() > 1:
        return None

    y_binary = y_batch.to(torch.long)
    if not torch.all((y_binary == 0) | (y_binary == 1)):
        return None
    if class_count != 1:
        return None

    return y_binary


def compute_binary_metrics(
    substitute: nn.Module,
    test_loader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """Compute optional binary classification metrics for single-logit outputs."""
    substitute.eval()

    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[float] = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            try:
                logits = ensure_single_logit_shape(substitute(x))
            except ValueError:
                return {}

            y_batch = _binary_labels_from_batch(y.to(device), int(logits.size(1)))
            if y_batch is None:
                return {}

            prob_pos = binary_positive_probs_from_logits(logits)[:, 0]
            pred_pos = (prob_pos >= 0.5).long()

            y_true.extend(y_batch.detach().cpu().tolist())
            y_pred.extend(pred_pos.detach().cpu().tolist())
            y_prob.extend(prob_pos.detach().cpu().tolist())

    if not y_true:
        return {}

    import numpy as np
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    y_true_np = np.array(y_true, dtype=int)
    y_pred_np = np.array(y_pred, dtype=int)
    y_prob_np = np.array(y_prob, dtype=float)

    precision = float(precision_score(y_true_np, y_pred_np, zero_division=0))
    recall = float(recall_score(y_true_np, y_pred_np, zero_division=0))
    f1 = float(f1_score(y_true_np, y_pred_np, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true_np, y_prob_np))
    except ValueError:
        roc_auc = float("nan")

    return {
        "binary_precision": precision,
        "binary_recall": recall,
        "binary_f1": f1,
        "binary_roc_auc": roc_auc,
    }


def evaluate_substitute(
    substitute: nn.Module,
    victim: nn.Module,
    test_loader,
    device: str,
    output_mode: str = "soft_prob",
    temperature: float = 1.0,
    normalization: dict = None,
) -> Dict[str, float]:
    """Evaluate substitute model comprehensively."""
    substitute_was_training = bool(substitute.training)
    victim_was_training = bool(victim.training)

    metrics = {}
    try:
        metrics["acc_gt"] = compute_accuracy(substitute, test_loader, device)
        metrics["agreement"] = compute_agreement(substitute, victim, test_loader, device)

        if output_mode == "soft_prob":
            metrics["kl_mean"] = compute_kl_divergence(
                substitute, victim, test_loader, device, temperature
            )
            metrics["l1_mean"] = compute_l1_distance(
                substitute, victim, test_loader, device, temperature
            )
        else:
            metrics["kl_mean"] = None
            metrics["l1_mean"] = None

        metrics.update(compute_binary_metrics(substitute, test_loader, device))
    finally:
        substitute.train(substitute_was_training)
        victim.train(victim_was_training)

    return metrics
