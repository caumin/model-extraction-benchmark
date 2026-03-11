from __future__ import annotations

import torch
import torch.nn.functional as F


def is_single_logit_binary_num_classes(num_classes: int | None) -> bool:
    return num_classes is not None and int(num_classes) == 1


def ensure_single_logit_shape(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 1:
        return tensor.unsqueeze(1)
    if tensor.ndim == 2 and tensor.size(1) == 1:
        return tensor
    raise ValueError(f"Expected single-logit tensor with shape [N] or [N, 1], got {tuple(tensor.shape)}")


def binary_positive_probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    logits_2d = ensure_single_logit_shape(logits)
    return torch.sigmoid(logits_2d)


def binary_distribution_from_logits(logits: torch.Tensor) -> torch.Tensor:
    prob_pos = binary_positive_probs_from_logits(logits)[:, 0]
    prob_neg = 1.0 - prob_pos
    return torch.stack((prob_neg, prob_pos), dim=1)


def binary_distribution_from_positive_probs(prob_pos: torch.Tensor) -> torch.Tensor:
    prob_pos_2d = ensure_single_logit_shape(prob_pos).clamp(0.0, 1.0)
    pos = prob_pos_2d[:, 0]
    neg = 1.0 - pos
    return torch.stack((neg, pos), dim=1)


def binary_hard_labels_from_logits(logits: torch.Tensor) -> torch.Tensor:
    logits_2d = ensure_single_logit_shape(logits)
    return (logits_2d[:, 0] >= 0).long()


def binary_hard_labels_from_positive_probs(prob_pos: torch.Tensor) -> torch.Tensor:
    prob_pos_2d = ensure_single_logit_shape(prob_pos)
    return (prob_pos_2d[:, 0] >= 0.5).long()


def binary_soft_targets(targets: torch.Tensor) -> torch.Tensor:
    if targets.ndim == 1:
        return targets.float().unsqueeze(1)
    if targets.ndim == 2 and targets.size(1) == 1:
        return targets.float()
    raise ValueError(
        f"Expected single-logit binary targets with shape [N] or [N, 1], got {tuple(targets.shape)}"
    )


def binary_hard_targets(targets: torch.Tensor) -> torch.Tensor:
    if targets.ndim == 2 and targets.size(1) == 1:
        targets = targets[:, 0]
    if targets.ndim != 1:
        raise ValueError(f"Expected hard binary targets with shape [N] or [N, 1], got {tuple(targets.shape)}")
    return targets.float()


def binary_bce_loss(logits: torch.Tensor, targets: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    logits_2d = ensure_single_logit_shape(logits)
    target_2d = binary_soft_targets(targets).to(logits_2d.device)
    return F.binary_cross_entropy_with_logits(logits_2d, target_2d, reduction=reduction)


def binary_logits_from_positive_probs(prob_pos: torch.Tensor) -> torch.Tensor:
    prob_pos_2d = ensure_single_logit_shape(prob_pos).clamp(1e-10, 1.0 - 1e-10)
    return torch.log(prob_pos_2d / (1.0 - prob_pos_2d))


def binary_distribution_from_labels(labels: torch.Tensor) -> torch.Tensor:
    hard = binary_hard_targets(labels).to(torch.long)
    pos = hard.float()
    neg = 1.0 - pos
    return torch.stack((neg, pos), dim=1)


def binary_distribution_from_targets(targets: torch.Tensor) -> torch.Tensor:
    if targets.ndim == 1:
        return binary_distribution_from_labels(targets)
    if targets.ndim == 2 and targets.size(1) == 1:
        return binary_distribution_from_positive_probs(targets)
    raise ValueError(
        f"Expected single-logit binary targets with shape [N] or [N, 1], got {tuple(targets.shape)}"
    )


def binary_entropy_from_positive_probs(prob_pos: torch.Tensor) -> torch.Tensor:
    prob_pos_2d = ensure_single_logit_shape(prob_pos).clamp(1e-12, 1.0 - 1e-12)
    pos = prob_pos_2d[:, 0]
    neg = 1.0 - pos
    return -(pos * torch.log(pos) + neg * torch.log(neg))
