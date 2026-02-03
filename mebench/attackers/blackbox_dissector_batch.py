
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional

def generate_gradcam_heatmap_batch(
    model: nn.Module,
    x: torch.Tensor,
    target_class: torch.Tensor = None,
) -> torch.Tensor:
    """Generate Grad-CAM heatmap for attention region (Batched).

    Args:
        model: Substitute model (must have final conv + fc)
        x: Input image [B, C, H, W]
        target_class: Target classes [B] (if None, use predicted class)

    Returns:
        Heatmap [B, H, W] normalized to [0, 1]
    """
    model.eval()
    B = x.size(0)

    def _get_target_layer(net: nn.Module) -> nn.Module:
        if hasattr(net, "layer3"):
            return net.layer3
        if hasattr(net, "dense3"):
            return net.dense3
        if hasattr(net, "features"):
            features = net.features
            if isinstance(features, nn.Sequential) and len(features) > 28:
                return features[28]
            return features
        last_conv = None
        for module in net.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError("Grad-CAM requires a Conv2d layer")
        return last_conv

    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

    target_layer = _get_target_layer(model)

    def forward_hook(_module, _inputs, output):
        activations.append(output)

    def backward_hook(_module, _grad_input, grad_output):
        # grad_output[0] is dL/dA
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    
    # Check for register_full_backward_hook availability (PyTorch 1.8+)
    if hasattr(target_layer, "register_full_backward_hook"):
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
    else:
        backward_handle = target_layer.register_backward_hook(backward_hook)

    device = next(model.parameters()).device
    x = x.to(device)
    output = model(x)  # [B, NumClasses]

    if target_class is None:
        target_class = output.argmax(dim=1)

    # One-hot encoding for gradient target
    one_hot = torch.zeros_like(output)
    one_hot.scatter_(1, target_class.unsqueeze(1), 1.0)
    
    model.zero_grad()
    output.backward(gradient=one_hot)

    forward_handle.remove()
    backward_handle.remove()

    if not activations or not gradients:
        raise RuntimeError("Grad-CAM hooks failed to capture activations/gradients")

    # [B, C_feat, H_feat, W_feat]
    activation = activations[0]
    gradient = gradients[0]
    
    # Global Average Pooling of gradients -> weights [B, C_feat, 1, 1]
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    
    # Weighted combination: sum(w * A) over channels
    # [B, C_feat, H_feat, W_feat] * [B, C_feat, 1, 1] -> sum dim 1
    heatmap = (weights * activation).sum(dim=1) # [B, H_feat, W_feat]
    
    # ReLU
    heatmap = F.relu(heatmap)

    # Upsample to input size [B, H, W] if needed
    # We return raw heatmap size here as per original impl? 
    # Original impl returns [B, H_feat, W_feat] but normalized.
    # Actually, original code does not upsample. Let's stick to feature map size.
    # Wait, cam_erase does "center_y = max_index // w". If w is feature map width, that's fine.
    
    # Normalize to [0, 1] per sample
    b, h, w = heatmap.shape
    heatmap_flat = heatmap.view(b, -1)
    heatmap_min = heatmap_flat.min(dim=1, keepdim=True)[0]
    heatmap_max = heatmap_flat.max(dim=1, keepdim=True)[0]
    
    # Avoid div by zero
    denominator = heatmap_max - heatmap_min
    denominator[denominator < 1e-8] = 1.0
    
    heatmap = (heatmap - heatmap_min.unsqueeze(-1)) / denominator.unsqueeze(-1)

    return heatmap

def cam_erase_batch(
    img_batch: torch.Tensor,
    model: nn.Module,
    sl: float = 0.02,
    sh: float = 0.4,
    r1: float = 0.3,
    r2: float = 3.3,
    fill_min: float = 0.0,
    fill_max: float = 1.0,
    target_class: torch.Tensor = None
) -> torch.Tensor:
    """Generate CAM-driven erasing variant for a batch of images.
    
    Args:
        img_batch: [B, C, H, W]
        model: Substitute model
        sl, sh, r1, r2, fill_min, fill_max: Erasing parameters
        target_class: Optional target classes for Grad-CAM
    
    Returns:
        erased_batch: [B, C, H, W]
    """
    B, C, H, W = img_batch.shape
    heatmap = generate_gradcam_heatmap_batch(model, img_batch, target_class) # [B, H_feat, W_feat]
    
    # Flatten heatmaps for sampling
    h_map, w_map = heatmap.shape[1], heatmap.shape[2]
    heatmap_flat = heatmap.view(B, -1)
    
    # Normalize probabilities
    heatmap_sum = heatmap_flat.sum(dim=1, keepdim=True)
    # Handle zero sums (uniform)
    mask_zero = (heatmap_sum <= 1e-8).squeeze(1)
    probs = heatmap_flat / (heatmap_sum + 1e-8)
    probs[mask_zero] = 1.0 / heatmap_flat.size(1)
    
    # Sample max_index for each image in batch
    # torch.multinomial works on batch input
    max_indices = torch.multinomial(probs, 1).squeeze(1) # [B]
    
    # Map back to feature coordinates
    center_y_feat = max_indices // w_map
    center_x_feat = max_indices % w_map
    
    # Scale coordinates to input image size
    # Assuming standard CNN structure where spatial dims are preserved or downsampled uniformly
    scale_h = H / h_map
    scale_w = W / w_map
    
    center_y = (center_y_feat.float() * scale_h).long()
    center_x = (center_x_feat.float() * scale_w).long()
    
    # Vectorized parameter sampling
    # We need B samples of area and aspect ratio
    area = float(H * W)
    
    # Random parameters on CPU/GPU as tensor
    device = img_batch.device
    target_areas = torch.empty(B, device=device).uniform_(sl, sh) * area
    aspect_ratios = torch.empty(B, device=device).uniform_(r1, r2)
    
    h_erase = torch.sqrt(target_areas * aspect_ratios).round().long()
    w_erase = torch.sqrt(target_areas / aspect_ratios).round().long()
    
    # Clip
    h_erase = torch.clamp(h_erase, min=1, max=H)
    w_erase = torch.clamp(w_erase, min=1, max=W)
    
    # Calculate bounding boxes
    y1 = torch.clamp(center_y - h_erase // 2, min=0)
    x1 = torch.clamp(center_x - w_erase // 2, min=0)
    
    # We can't easily vectorize the slice assignment for variable sized boxes in a pure tensor operation without a loop or mask
    # A mask-based approach is fully vectorized:
    
    # Grid [B, H, W]
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    # Expand to [B, H, W]
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
    
    y1 = y1.view(B, 1, 1)
    x1 = x1.view(B, 1, 1)
    h_e = h_erase.view(B, 1, 1)
    w_e = w_erase.view(B, 1, 1)
    
    # Mask [B, H, W] is True where we should erase
    mask = (grid_y >= y1) & (grid_y < y1 + h_e) & (grid_x >= x1) & (grid_x < x1 + w_e)
    # Expand mask to channels [B, C, H, W]
    mask = mask.unsqueeze(1).expand(-1, C, -1, -1)
    
    # Generate random fill values
    # "Uniform(0, 255) or Uniform(0, 1)"
    # We create a random tensor of same shape
    fill_values = torch.empty_like(img_batch).uniform_(fill_min, fill_max)
    
    erased_batch = img_batch.clone()
    erased_batch[mask] = fill_values[mask]
    
    return erased_batch

def random_erase_batch(
    img_batch: torch.Tensor,
    n: int = 10,
    sl: float = 0.02,
    sh: float = 0.4,
    r1: float = 0.3,
    r2: float = 3.3,
    fill_min: float = 0.0,
    fill_max: float = 1.0,
) -> torch.Tensor:
    """Generate N random erasing variants for a batch of images.
    
    Args:
        img_batch: [B, C, H, W]
        n: Number of variants per image
        
    Returns:
        erased_batch: [B * n, C, H, W]
    """
    B, C, H, W = img_batch.shape
    device = img_batch.device
    
    # Repeat images: [B, C, H, W] -> [B, N, C, H, W] -> [B*N, C, H, W]
    # We treat the B*N images as a single large batch for vectorized erasing
    imgs_repeated = img_batch.unsqueeze(1).repeat(1, n, 1, 1, 1).view(-1, C, H, W)
    total_imgs = B * n
    
    # Vectorized parameter sampling
    area = float(H * W)
    
    # Random centers
    center_y = torch.randint(0, H, (total_imgs,), device=device)
    center_x = torch.randint(0, W, (total_imgs,), device=device)
    
    # Random area and aspect ratio
    target_areas = torch.empty(total_imgs, device=device).uniform_(sl, sh) * area
    aspect_ratios = torch.empty(total_imgs, device=device).uniform_(r1, r2)
    
    h_erase = torch.sqrt(target_areas * aspect_ratios).round().long()
    w_erase = torch.sqrt(target_areas / aspect_ratios).round().long()
    
    # Clip sizes
    h_erase = torch.clamp(h_erase, min=1, max=H)
    w_erase = torch.clamp(w_erase, min=1, max=W)
    
    # Calculate bounding boxes
    y1 = torch.clamp(center_y - h_erase // 2, min=0)
    x1 = torch.clamp(center_x - w_erase // 2, min=0)
    
    # Create mask [B*N, H, W] using grid broadcasting
    # Note: Creating full grid for large batch might be memory intensive.
    # If OOM, we can iterate, but let's try full vectorization first.
    
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    
    # Expand dims for broadcasting against batch
    # grid: [1, H, W], box params: [B*N, 1, 1]
    grid_y = grid_y.unsqueeze(0)
    grid_x = grid_x.unsqueeze(0)
    
    y1 = y1.view(total_imgs, 1, 1)
    x1 = x1.view(total_imgs, 1, 1)
    h_e = h_erase.view(total_imgs, 1, 1)
    w_e = w_erase.view(total_imgs, 1, 1)
    
    mask = (grid_y >= y1) & (grid_y < y1 + h_e) & (grid_x >= x1) & (grid_x < x1 + w_e)
    
    # Expand mask to channels [B*N, C, H, W]
    mask = mask.unsqueeze(1).expand(-1, C, -1, -1)
    
    # Fill values
    fill_values = torch.empty_like(imgs_repeated).uniform_(fill_min, fill_max)
    
    erased_batch = imgs_repeated.clone()
    erased_batch[mask] = fill_values[mask]
    
    return erased_batch

