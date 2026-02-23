"""Evaluation metrics for model extraction benchmark."""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def compute_accuracy(model: nn.Module, loader: DataLoader, device: str) -> float:
    """Compute top-1 accuracy on a dataset."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            # Evaluation uses shared normalized test loader inputs.
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    return correct / total if total > 0 else 0.0


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
            # Evaluation uses shared normalized test loader inputs.
            out_sub = substitute(x)
            out_vic = victim(x)

            pred_sub = torch.argmax(out_sub, dim=1)
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
            # Evaluation uses shared normalized test loader inputs.
            
            # Get probability distributions
            sub_logits = substitute(x) / temperature
            victim_probs = F.softmax(victim(x) / temperature, dim=1)

            # Compute KL(victim || substitute) - stable version using log_softmax
            log_sub_probs = F.log_softmax(sub_logits, dim=1)
            kl = F.kl_div(
                log_sub_probs,
                victim_probs,
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
            # Evaluation uses shared normalized test loader inputs.
            
            sub_probs = F.softmax(substitute(x) / temperature, dim=1)
            victim_probs = F.softmax(victim(x) / temperature, dim=1)

            l1 = F.l1_loss(sub_probs, victim_probs, reduction="sum")
            total_l1 += l1.item()
            total_samples += x.size(0)

    return total_l1 / total_samples / sub_probs.size(1) if total_samples > 0 else 0.0


def evaluate_substitute(
    substitute: nn.Module,
    victim: nn.Module,
    test_loader,
    device: str,
    output_mode: str = "soft_prob",
    temperature: float = 1.0,
    normalization: dict = None,
) -> Dict[str, float]:
    """Evaluate substitute model comprehensively.

    Args:
        substitute: Substitute model
        victim: Victim model
        test_loader: Test dataloader with ground truth labels
        device: Device to use
        output_mode: Output mode (soft_prob or hard_top1)
        temperature: Temperature for softmax
        normalization: Victim normalization config (IGNORED, using [0, 1])

    Returns:
        Dictionary with metrics
    """
    metrics = {}

    # Test accuracy (ground truth)
    metrics["acc_gt"] = compute_accuracy(substitute, test_loader, device)

    # Agreement with victim
    metrics["agreement"] = compute_agreement(substitute, victim, test_loader, device)

    # Soft-label metrics (only for soft_prob mode)
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

    return metrics
