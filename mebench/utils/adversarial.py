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
                model.zero_grad()
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
            final_preds[batch_start + i] = model(x[batch_start + i:batch_start + i + 1] + sample_perturb).argmax(dim=1)[0]
        
        perturbations[batch_start:batch_end] = perturb_batch
    
    return perturbations, final_preds


def deepfool_distance_vectorized(
    model: nn.Module,
    x: torch.Tensor,
    max_iter: int = 20,
    batch_size: int = 32,
) -> torch.Tensor:
    """Compute DeepFool perturbation distances for a batch of samples.
    
    Args:
        model: PyTorch model to attack
        x: Input tensor of shape (B, C, H, W)
        max_iter: Maximum number of iterations per sample
        batch_size: Internal batch size for gradient computation
        
    Returns:
        Tensor of shape (B,) containing perturbation norms for each sample
    """
    perturbations, _ = deepfool_vectorized(
        model, x, max_iter=max_iter, overshoot=0.0, batch_size=batch_size
    )
    
    # Compute L2 norm of perturbations
    perturbations_flat = perturbations.view(perturbations.shape[0], -1)
    distances = torch.norm(perturbations_flat, dim=2)
    
    return distances


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