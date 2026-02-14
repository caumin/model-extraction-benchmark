"""Utility functions for adversarial attacks including vectorized DeepFool."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def deepfool_vectorized(
    model: nn.Module,
    x: torch.Tensor,
    max_iter: int = 20,
    overshoot: float = 0.02,
    batch_size: int = 32,
    max_memory_mb: int = 500,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized DeepFool implementation for batch processing.
    
    Args:
        model: PyTorch model to attack
        x: Input tensor of shape (B, C, H, W)
        max_iter: Maximum number of iterations per sample
        overshoot: Overshoot parameter to ensure adversarial examples cross boundary
        batch_size: Internal batch size for gradient computation (memory management)
        
    Returns:
        Tuple of (perturbations, final_predictions)
        - perturbations: Tensor of shape (B, C, H, W) with adversarial perturbations
        - final_predictions: Tensor of shape (B,) with final predicted classes
    """
    device = x.device
    B, C, H, W = x.shape
    model.eval()
    
    # Initialize perturbations
    perturbations = torch.zeros_like(x, device=device)
    x_adv = x.clone()
    
    with torch.no_grad():
        logits = model(x)
        current_preds = logits.argmax(dim=1)
    
    # Process in chunks to manage memory for large batches
    final_preds = current_preds.clone()
    
    for batch_start in range(0, B, batch_size):
        batch_end = min(batch_start + batch_size, B)
        batch_idx = torch.arange(batch_start, batch_end, device=device)
        
        x_batch = x_adv[batch_start:batch_end]
        perturb_batch = perturbations[batch_start:batch_end]
        
        # Process each sample in the mini-batch
        for i in range(x_batch.shape[0]):
            sample = x_batch[i:i+1].clone()
            sample_perturb = perturb_batch[i:i+1].clone()
            
            current_pred = current_preds[batch_start + i].item()
            converged = False
            
            for iteration in range(max_iter):
                # Current adversarial example
                x_current = sample + sample_perturb
                x_current.requires_grad_(True)
                
                # Forward pass
                logits_current = model(x_current)
                pred_current = logits_current.argmax(dim=1).item()
                
                if pred_current != current_pred:
                    converged = True
                    break
                
                # Gradient w.r.t. current class
                grad_current = torch.autograd.grad(
                    logits_current[0, current_pred],
                    x_current,
                    retain_graph=True,
                    create_graph=False,
                )[0]
                
                # Find minimal distance to decision boundary
                min_distance = float('inf')
                best_direction = None
                
                num_classes = logits_current.shape[1]
                for k in range(num_classes):
                    if k == current_pred:
                        continue
                    
                    # Gradient w.r.t. class k
                    grad_k = torch.autograd.grad(
                        logits_current[0, k],
                        x_current,
                        retain_graph=True,
                        create_graph=False,
                    )[0]
                    
                    # Direction and distance to class k boundary
                    w_k = grad_k - grad_current
                    f_k = logits_current[0, k] - logits_current[0, current_pred]
                    
                    w_k_norm = torch.norm(w_k)
                    if w_k_norm == 0:
                        continue
                    
                    distance = torch.abs(f_k) / w_k_norm
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_direction = w_k
                
                if best_direction is None:
                    break
                
                # Update perturbation
                r_i = (min_distance + 1e-8) * best_direction / torch.norm(best_direction)
                sample_perturb = sample_perturb + r_i
            
            # Apply overshoot and store result
            if converged:
                sample_perturb = sample_perturb * (1 + overshoot)
            
            perturb_batch[i] = sample_perturb
            # Final prediction is computed in a batched pass below.
        
        perturbations[batch_start:batch_end] = perturb_batch

        # Batched final prediction for the chunk (reduces per-sample forward overhead).
        with torch.inference_mode():
            logits_final = model(x[batch_start:batch_end] + perturb_batch)
            final_preds[batch_start:batch_end] = logits_final.argmax(dim=1)
    
    return perturbations, final_preds


def deepfool_distance_vectorized(
    model: nn.Module,
    x: torch.Tensor,
    max_iter: int = 20,
    batch_size: int = 32,
) -> torch.Tensor:
    """Compute DeepFool perturbation distances for a batch of samples.

    This preserves the original DeepFool logic (iterative updates, all classes, stop when
    the predicted label changes) but is implemented in a more GPU-friendly way by:
    - processing samples in batches
    - computing per-class input gradients for the whole batch (one backward per class),
      instead of nested per-sample loops

    Args:
        model: PyTorch model to attack.
        x: Input tensor of shape (B, C, H, W).
        max_iter: Maximum number of DeepFool iterations per sample.
        batch_size: Chunk size for memory control.

    Returns:
        Tensor of shape (B,) containing L2 norms of the DeepFool perturbations.
    """

    device = x.device
    b_total = int(x.shape[0])
    if b_total == 0:
        return torch.empty(0, device=device)

    model.eval()
    with torch.no_grad():
        base_logits = model(x)
        base_labels = base_logits.argmax(dim=1)

    out: list[torch.Tensor] = []

    bs = max(1, int(batch_size))
    eps = 1e-8

    for start in range(0, b_total, bs):
        end = min(start + bs, b_total)
        x_b = x[start:end].detach()
        labels_b = base_labels[start:end]
        b = int(x_b.size(0))

        r = torch.zeros_like(x_b)
        done = torch.zeros((b,), dtype=torch.bool, device=device)

        for _it in range(int(max_iter)):
            if bool(done.all()):
                break

            x_cur = (x_b + r).detach().clone().requires_grad_(True)
            logits = model(x_cur)

            pred = logits.argmax(dim=1)
            done = done | (pred != labels_b)
            active = ~done
            if not bool(active.any()):
                break

            num_classes = int(logits.size(1))
            # Reference class is the initial prediction (matches deepfool_vectorized).
            logit_c = logits.gather(1, labels_b.view(-1, 1)).squeeze(1)

            grad_c = torch.autograd.grad(
                logit_c.sum(),
                x_cur,
                retain_graph=True,
                create_graph=False,
            )[0]

            min_dist = torch.full((b,), float("inf"), device=device)
            best_w = torch.zeros_like(x_cur)

            for k in range(num_classes):
                # Skip k == reference class per-sample by masking distance.
                retain = k != (num_classes - 1)
                grad_k = torch.autograd.grad(
                    logits[:, k].sum(),
                    x_cur,
                    retain_graph=retain,
                    create_graph=False,
                )[0]

                w_k = grad_k - grad_c
                w_flat = w_k.view(b, -1)
                w_norm = torch.norm(w_flat, p=2, dim=1)

                f_k = logits[:, k] - logit_c
                # Distance for this class; if w_norm==0 treat as not a candidate.
                dist_k = torch.where(
                    w_norm > 0,
                    torch.abs(f_k) / w_norm,
                    torch.full_like(w_norm, float("inf")),
                )

                dist_k = dist_k.masked_fill(labels_b == int(k), float("inf"))
                dist_k = dist_k.masked_fill(~active, float("inf"))

                upd = dist_k < min_dist
                min_dist = torch.where(upd, dist_k, min_dist)
                best_w = torch.where(upd.view(-1, 1, 1, 1), w_k, best_w)

            best_norm = torch.norm(best_w.view(b, -1), p=2, dim=1)
            valid = active & torch.isfinite(min_dist) & (best_norm > 0)
            if not bool(valid.any()):
                break

            scale = torch.zeros((b,), device=device)
            scale[valid] = (min_dist[valid] + eps) / best_norm[valid]
            r = r + scale.view(-1, 1, 1, 1) * best_w

        out.append(torch.norm(r.view(b, -1), p=2, dim=1).detach())

    return torch.cat(out, dim=0)


class DeepFoolAttack:
    """Wrapper class for DeepFool attack following standard attack interface."""
    
    def __init__(self, max_iter: int = 20, overshoot: float = 0.02, batch_size: int = 32):
        self.max_iter = max_iter
        self.overshoot = overshoot
        self.batch_size = batch_size
    
    def __call__(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Generate adversarial examples using DeepFool.
        
        Args:
            model: Target model
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Adversarial examples of shape (B, C, H, W)
        """
        perturbations, _ = deepfool_vectorized(
            model, x, max_iter=self.max_iter, overshoot=self.overshoot, batch_size=self.batch_size
        )
        return x + perturbations
    
    def compute_distances(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Compute DeepFool distances for uncertainty sampling.
        
        Args:
            model: Target model
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Distance tensor of shape (B,) - smaller distances = more uncertain
        """
        return deepfool_distance_vectorized(
            model, x, max_iter=self.max_iter, batch_size=self.batch_size
        )
