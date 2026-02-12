"""Black-box Dissector attack implementation."""

from typing import Dict, Any, List, Tuple, Optional
import copy
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import f1_score
from tqdm import tqdm

from mebench.attackers.runner import AttackRunner
from mebench.core.context import BenchmarkContext
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.data.loaders import create_dataloader, get_test_dataloader
from mebench.models.substitute_factory import create_substitute
from mebench.training import SubstituteTrainer, TrainRequest
from mebench.utils.dataloader import pool_loader_kwargs
from mebench.eval.metrics import evaluate_substitute


class _PseudoLabelDataset(Dataset):
    def __init__(
        self,
        indices: List[int],
        labels: Dict[int, torch.Tensor],
        pool: Dataset,
    ) -> None:
        self.indices = indices
        self.labels = labels
        self.pool = pool

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real_idx = self.indices[idx]
        x, _ = self.pool[real_idx]
        y = self.labels[real_idx]
        return x, y


class _LabeledTensorDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return int(self.x.size(0))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def generate_gradcam_heatmap(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int = None,
) -> torch.Tensor:
    """Generate Grad-CAM heatmap for attention region.

    Args:
        model: Substitute model (must have final conv + fc)
        x: Input image [B, C, H, W]
        target_class: Target class (if None, use predicted class)

    Returns:
        Heatmap [B, H, W] normalized to [0, 1]
    """
    model.eval()

    def _get_target_layer(net: nn.Module) -> nn.Module:
        if hasattr(net, "layer4"):
            return net.layer4
        if hasattr(net, "layer3"):
            return net.layer3
        if hasattr(net, "dense4"):
            return net.dense4
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
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    if hasattr(target_layer, "register_full_backward_hook"):
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
    else:
        backward_handle = target_layer.register_backward_hook(backward_hook)

    device = next(model.parameters()).device
    x = x.to(device)
    output = model(x)

    if target_class is None:
        target_class = output.argmax(dim=1)

    grad_output = torch.zeros_like(output)
    for i in range(output.shape[0]):
        grad_output[i, target_class[i]] = 1.0

    model.zero_grad()
    output.backward(gradient=grad_output)

    forward_handle.remove()
    backward_handle.remove()

    if not activations or not gradients:
        raise RuntimeError("Grad-CAM hooks failed to capture activations/gradients")

    activation = activations[0]
    gradient = gradients[0]

    weights = gradient.mean(dim=(2, 3), keepdim=True)
    heatmap = (weights * activation).sum(dim=1)
    heatmap = F.relu(heatmap)

    # Normalize to [0, 1]
    b, h, w = heatmap.shape
    heatmap_flat = heatmap.view(b, -1)
    heatmap_min = heatmap_flat.min(dim=1, keepdim=True)[0]
    heatmap_max = heatmap_flat.max(dim=1, keepdim=True)[0]
    heatmap = (heatmap - heatmap_min.unsqueeze(-1).view(b, 1, 1)) / (
        heatmap_max.unsqueeze(-1).view(b, 1, 1) - heatmap_min.unsqueeze(-1).view(b, 1, 1) + 1e-8
    )

    return heatmap


def random_erase(
    img: torch.Tensor,
    n: int = 10,
    sl: float = 0.02,
    sh: float = 0.4,
    r1: float = 0.3,
    r2: float = 3.3,
    fill_min: float = 0.0,
    fill_max: float = 1.0,
) -> List[torch.Tensor]:
    """Generate N random erasing variants.

    Args:
        img: Input image [C, H, W]
        n: Number of variants to generate
        sl: Lower bound of erasing area ratio
        sh: Upper bound of erasing area ratio
        r1: Lower bound of erasing aspect ratio
        r2: Upper bound of erasing aspect ratio
        fill_min: Minimum fill value
        fill_max: Maximum fill value

    Returns:
        List of erased images
    """
    variants = []
    c, h, w = img.shape

    for _ in range(n):
        erased = img.clone()

        # Paper Algorithm 1 (psi): sample area/aspect ratio and center by a prior.
        # For random erasing, the prior is uniform over pixels.
        center_y = int(np.random.randint(0, h))
        center_x = int(np.random.randint(0, w))

        area = float(h * w)
        target_area = float(np.random.uniform(sl, sh) * area)
        aspect_ratio = float(np.random.uniform(r1, r2))

        h_erase = int(round(np.sqrt(target_area * aspect_ratio)))
        w_erase = int(round(np.sqrt(target_area / aspect_ratio)))

        h_erase = min(h, max(1, h_erase))
        w_erase = min(w, max(1, w_erase))

        y1 = max(0, center_y - h_erase // 2)
        x1 = max(0, center_x - w_erase // 2)
        y2 = min(h, y1 + h_erase)
        x2 = min(w, x1 + w_erase)

        if y2 > y1 and x2 > x1:
            erased[:, y1:y2, x1:x2] = torch.empty(
                c,
                y2 - y1,
                x2 - x1,
                device=img.device,
                dtype=img.dtype,
            ).uniform_(float(fill_min), float(fill_max))

        variants.append(erased)

    return variants


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
    b, c, h, w = img_batch.shape
    device = img_batch.device

    imgs_repeated = img_batch.unsqueeze(1).repeat(1, n, 1, 1, 1).view(-1, c, h, w)
    total_imgs = b * n

    area = float(h * w)
    center_y = torch.randint(0, h, (total_imgs,), device=device)
    center_x = torch.randint(0, w, (total_imgs,), device=device)
    target_areas = torch.empty(total_imgs, device=device).uniform_(sl, sh) * area
    aspect_ratios = torch.empty(total_imgs, device=device).uniform_(r1, r2)

    h_erase = torch.sqrt(target_areas * aspect_ratios).round().long()
    w_erase = torch.sqrt(target_areas / aspect_ratios).round().long()

    h_erase = torch.clamp(h_erase, min=1, max=h)
    w_erase = torch.clamp(w_erase, min=1, max=w)

    y1 = torch.clamp(center_y - h_erase // 2, min=0)
    x1 = torch.clamp(center_x - w_erase // 2, min=0)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    grid_y = grid_y.unsqueeze(0)
    grid_x = grid_x.unsqueeze(0)

    y1 = y1.view(total_imgs, 1, 1)
    x1 = x1.view(total_imgs, 1, 1)
    h_e = h_erase.view(total_imgs, 1, 1)
    w_e = w_erase.view(total_imgs, 1, 1)

    mask = (grid_y >= y1) & (grid_y < y1 + h_e) & (grid_x >= x1) & (grid_x < x1 + w_e)
    mask = mask.unsqueeze(1).expand(-1, c, -1, -1)

    fill_values = torch.empty_like(imgs_repeated).uniform_(fill_min, fill_max)
    erased_batch = imgs_repeated.clone()
    erased_batch[mask] = fill_values[mask]

    return erased_batch


def cam_erase(
    img: torch.Tensor,
    model: nn.Module,
    sl: float = 0.02,
    sh: float = 0.4,
    r1: float = 0.3,
    r2: float = 3.3,
    fill_min: float = 0.0,
    fill_max: float = 1.0,
) -> torch.Tensor:
    """Generate CAM-driven erasing variant.

    Args:
        img: Input image [C, H, W]
        model: Substitute model for attention map
        sl: Lower bound of erasing area ratio
        sh: Upper bound of erasing area ratio
        r1: Lower bound of erasing aspect ratio
        r2: Upper bound of erasing aspect ratio
        fill_min: Minimum fill value
        fill_max: Maximum fill value

    Returns:
        Erased image
    """
    # Get attention heatmap
    x_batch = img.unsqueeze(0)
    heatmap = generate_gradcam_heatmap(model, x_batch)[0]  # [H, W]

    c, h, w = img.shape

    heatmap_flat = heatmap.view(-1)
    heatmap_sum = heatmap_flat.sum()
    if heatmap_sum <= 1e-8:
        # Fallback to uniform distribution if heatmap is zero/flat
        probs = torch.ones_like(heatmap_flat) / heatmap_flat.numel()
    else:
        probs = heatmap_flat / heatmap_sum
    
    max_index = torch.multinomial(probs, 1).item()
    center_y = max_index // w
    center_x = max_index % w

    # Paper Algorithm 1 (psi): sample area ratio and aspect ratio.
    area = float(h * w)
    target_area = float(np.random.uniform(sl, sh) * area)
    aspect_ratio = float(np.random.uniform(r1, r2))

    h_erase = int(round(np.sqrt(target_area * aspect_ratio)))
    w_erase = int(round(np.sqrt(target_area / aspect_ratio)))

    h_erase = min(h, max(1, h_erase))
    w_erase = min(w, max(1, w_erase))

    y1 = max(0, center_y - h_erase // 2)
    x1 = max(0, center_x - w_erase // 2)
    y2 = min(h, y1 + h_erase)
    x2 = min(w, x1 + w_erase)

    erased = img.clone()
    # Paper: fill erased region with Uniform(0, 255) (or Uniform(0, 1) for [0,1] tensors).
    if y2 > y1 and x2 > x1:
        erased[:, y1:y2, x1:x2] = torch.empty(
            c,
            y2 - y1,
            x2 - x1,
            device=img.device,
            dtype=img.dtype,
        ).uniform_(float(fill_min), float(fill_max))

    return erased


def cam_erase_batch(
    img_batch: torch.Tensor,
    model: nn.Module,
    sl: float = 0.02,
    sh: float = 0.4,
    r1: float = 0.3,
    r2: float = 3.3,
    fill_min: float = 0.0,
    fill_max: float = 1.0,
    target_class: torch.Tensor = None,
) -> torch.Tensor:
    """Generate CAM-driven erasing variants for a batch of images.

    Args:
        img_batch: [B, C, H, W]
        model: Substitute model
        sl, sh, r1, r2, fill_min, fill_max: Erasing parameters
        target_class: Optional target classes for Grad-CAM

    Returns:
        erased_batch: [B, C, H, W]
    """
    b, c, h, w = img_batch.shape
    heatmap = generate_gradcam_heatmap(model, img_batch, target_class)

    h_map, w_map = heatmap.shape[1], heatmap.shape[2]
    heatmap_flat = heatmap.view(b, -1)
    heatmap_sum = heatmap_flat.sum(dim=1, keepdim=True)
    mask_zero = (heatmap_sum <= 1e-8).squeeze(1)
    probs = heatmap_flat / (heatmap_sum + 1e-8)
    probs[mask_zero] = 1.0 / heatmap_flat.size(1)

    max_indices = torch.multinomial(probs, 1).squeeze(1)
    center_y_feat = max_indices // w_map
    center_x_feat = max_indices % w_map

    scale_h = h / h_map
    scale_w = w / w_map
    center_y = (center_y_feat.float() * scale_h).long()
    center_x = (center_x_feat.float() * scale_w).long()

    area = float(h * w)
    device = img_batch.device
    target_areas = torch.empty(b, device=device).uniform_(sl, sh) * area
    aspect_ratios = torch.empty(b, device=device).uniform_(r1, r2)

    h_erase = torch.sqrt(target_areas * aspect_ratios).round().long()
    w_erase = torch.sqrt(target_areas / aspect_ratios).round().long()

    h_erase = torch.clamp(h_erase, min=1, max=h)
    w_erase = torch.clamp(w_erase, min=1, max=w)

    y1 = torch.clamp(center_y - h_erase // 2, min=0)
    x1 = torch.clamp(center_x - w_erase // 2, min=0)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    grid_y = grid_y.unsqueeze(0).expand(b, -1, -1)
    grid_x = grid_x.unsqueeze(0).expand(b, -1, -1)

    y1 = y1.view(b, 1, 1)
    x1 = x1.view(b, 1, 1)
    h_e = h_erase.view(b, 1, 1)
    w_e = w_erase.view(b, 1, 1)

    mask = (grid_y >= y1) & (grid_y < y1 + h_e) & (grid_x >= x1) & (grid_x < x1 + w_e)
    mask = mask.unsqueeze(1).expand(-1, c, -1, -1)

    fill_values = torch.empty_like(img_batch).uniform_(fill_min, fill_max)
    erased_batch = img_batch.clone()
    erased_batch[mask] = fill_values[mask]

    return erased_batch

class BlackboxDissector(AttackRunner):
    """Black-box Dissector: CAM-driven erasing for hard-label victims.
    
    (Batch Optimized Implementation)
    """

    def __init__(self, config: dict, state: BenchmarkState):
        super().__init__(config, state)
        
        # Hyperparameters
        self.n_variants = int(config.get("n_variants", 10))

        # Algorithm 1 (psi) parameters
        self.sl = float(config.get("sl", 0.02))
        self.sh = float(config.get("sh", 0.4))
        self.r1 = float(config.get("r1", 0.3))
        self.r2 = float(config.get("r2", 3.3))
        self.fill_min = float(config.get("fill_min", 0.0))
        self.fill_max = float(config.get("fill_max", 1.0))

        # Training hyperparameters
        self.batch_size = int(config.get("batch_size", 128))
        paper_lr = 0.02 * (self.batch_size / 128.0)
        self.lr = float(config.get("lr", paper_lr))
        self.momentum = float(config.get("momentum", 0.9))
        # [P0 FIX] Paper mandates 200 epochs for BlackBox Dissector
        self.max_epochs = int(config.get("max_epochs", 200))
        self.patience = int(config.get("patience", 20))
        self.dropout = float(config.get("dropout", 0.1))
        self.l2_reg = float(config.get("l2_reg", 5e-4))
        
        # Selection batch size (for GPU efficiency)
        # [OPTIMIZATION] Increased to 512 for better 4090 utilization during Grad-CAM backprop
        self.selection_batch_size = int(config.get("selection_batch_size", 512))

        # Algorithm 2 (outer loop) iterative max-budget sequence.
        self.iterative_budgets = config.get("iterative_budgets")

        # Initialize attack state
        # Pool dataset (loaded during selection/init)
        self.pool_dataset = None
        self._initialize_state(state)

    def run(self, ctx: BenchmarkContext) -> None:
        # Ensure CUDA context is initialized to prevent cuBLAS warnings during GradCAM
        if torch.cuda.is_available():
            torch.cuda.init()
            # Trigger a real backward pass to fully establish context for autograd cuBLAS handles
            device = self.state.metadata.get("device", "cuda:0")
            dummy_x = torch.ones((1, 1), device=device, requires_grad=True)
            (dummy_x ** 2).sum().backward()

        self.victim = ctx.oracle.model

        pbar = tqdm(total=self.state.budget_remaining, desc="[BlackboxDissector] Extracting")
        while ctx.budget_remaining > 0:
            # [OPTIMIZATION] Set step_size to the delta until the next iteration milestone.
            # This follows the official implementation which queries in round-sized chunks
            # instead of small fixed batches, avoiding redundant heavy selection logic.
            target_q = int(self.state.attack_state.get("iter_target_q", 0))
            current_q = int(self.state.query_count)
            step_size = max(0, target_q - current_q)
            
            if step_size == 0:
                # If target is reached, use a small step to trigger the advance logic
                step_size = 1
            
            step_size = min(step_size, ctx.budget_remaining)
            
            # [FEATURE] Detailed logging for selection phase
            pbar.set_description(f"[BlackboxDissector] Selecting (k={step_size})")
            query_batch = self._select_query_batch(step_size, self.state)
            
            if int(query_batch.x.size(0)) == 0:
                if query_batch.meta.get("stage") == "noop":
                    pbar.set_description("[BlackboxDissector] Advancing Iteration")
                    self._advance_iteration_if_needed(self.state)
                continue
            
            pbar.set_description("[BlackboxDissector] Querying Oracle")
            oracle_output = ctx.query(query_batch.x, meta=query_batch.meta)
            
            pbar.set_description("[BlackboxDissector] Updating State")
            self._handle_oracle_output(query_batch, oracle_output, self.state)
            
            pbar.set_description("[BlackboxDissector] Extracting")
            pbar.update(query_batch.x.size(0))
        pbar.close()

        # Pool dataset
        self.pool_dataset = None

    def _advance_iteration_if_needed(self, state: BenchmarkState) -> None:
        target_q = int(state.attack_state.get("iter_target_q", 0))
        if target_q > 0 and int(state.query_count) >= target_q:
            teacher_model = state.attack_state.get("substitute")
            self._generate_pseudo_labels(state, teacher_model)
            self.train_substitute(state)

            ptr = int(state.attack_state.get("iter_ptr", 0))
            targets = state.attack_state.get("iter_targets", [])
            if ptr < len(targets) - 1:
                state.attack_state["iter_prev_q"] = target_q
                state.attack_state["iter_ptr"] = ptr + 1
                state.attack_state["iter_target_q"] = int(targets[ptr + 1])
                self._reset_iteration_stage_budgets(state)
            else:
                # If we reached the last milestone but budget remains,
                # extend the last milestone to consume all remaining budget
                max_b = int(state.metadata.get("max_budget", 0))
                if max_b > target_q:
                    state.attack_state["iter_prev_q"] = target_q
                    state.attack_state["iter_target_q"] = max_b
                    self._reset_iteration_stage_budgets(state)

    def _handle_oracle_output(
        self,
        query_batch: QueryBatch,
        oracle_output: OracleOutput,
        state: BenchmarkState,
    ) -> None:
        if oracle_output.kind != "hard_top1":
            raise ValueError("blackbox_dissector requires hard_top1")

        x_batch = query_batch.x
        y_batch = oracle_output.y

        stage = query_batch.meta.get("stage")
        indices = [int(x) for x in query_batch.meta.get("indices", [])]
        variant_types = query_batch.meta.get("variant_types", [])

        if stage == "A":
            victim_labels: dict[int, int] = state.attack_state.get("victim_labels", {})
            for i, idx in enumerate(indices):
                if idx < 0:
                    continue
                label_i = int(y_batch[i].item())
                victim_labels[idx] = label_i
            state.attack_state["victim_labels"] = victim_labels
            state.attack_state["D_T_x"].append(x_batch.detach().cpu())
            state.attack_state["D_T_y"].append(y_batch.detach().cpu().long())
        elif stage == "B":
            state.attack_state["D_E_x"].append(x_batch.detach().cpu())
            state.attack_state["D_E_y"].append(y_batch.detach().cpu().long())

        state.attack_state["query_data_x"].append(x_batch.detach().cpu())
        state.attack_state["query_data_y"].append(y_batch.detach().cpu())
        state.attack_state["query_data_indices"].append(torch.tensor(indices))

        self._advance_iteration_if_needed(state)

    def _initialize_state(self, state: BenchmarkState) -> None:
        """Initialize attack-specific state.

        Args:
            state: Global benchmark state to update
        """
        # Load pool dataset to get actual size
        if self.pool_dataset is None:
            dataset_config = state.metadata.get("dataset_config", {})
            if "data_mode" not in dataset_config:
                dataset_config = {"data_mode": "seed", **dataset_config}
            if "name" not in dataset_config:
                dataset_config = {"name": "CIFAR10", **dataset_config}
            self.pool_dataset = create_dataloader(
                dataset_config,
                batch_size=1,
                shuffle=False,
            ).dataset

        # If already initialized (e.g. resuming from checkpoint), do not reset
        if "labeled_indices" in state.attack_state:
            return

        pool_size = len(self.pool_dataset)
        state.attack_state["labeled_indices"] = []
        state.attack_state["unlabeled_indices"] = list(range(pool_size))
        state.attack_state["query_data_x"] = []
        state.attack_state["query_data_y"] = []
        state.attack_state["query_data_indices"] = []

        # Store pseudo-labels for Self-KD
        state.attack_state["pseudo_labels"] = {}  # idx -> averaged soft label

        # Victim top-1 labels for original images in transfer set (D_T)
        state.attack_state["victim_labels"] = {}  # idx -> int label

        # Algorithm 2 datasets
        state.attack_state["D_T_x"] = []  # list[tensor[B,C,H,W]]
        state.attack_state["D_T_y"] = []  # list[tensor[B]] hard labels
        state.attack_state["D_E_x"] = []
        state.attack_state["D_E_y"] = []
        state.attack_state["val_labeled_x"] = []
        state.attack_state["val_labeled_y"] = []

        # Store best variant per sample for selection
        state.attack_state["best_variant_idx"] = {}  # idx -> variant index
        state.attack_state["best_variant_img"] = {}  # idx -> tensor

        state.attack_state["substitute"] = None

        # Algorithm 2 schedule state
        state.attack_state["iter_targets"] = self._build_iter_targets(state)
        state.attack_state["iter_ptr"] = 0
        state.attack_state["iter_prev_q"] = 0
        state.attack_state["iter_target_q"] = state.attack_state["iter_targets"][0]
        state.attack_state["iter_stage"] = "A"  # 'A' (original) then 'B' (erased)
        state.attack_state["step1_trained"] = False
        self._reset_iteration_stage_budgets(state)

    def _build_iter_targets(self, state: BenchmarkState) -> List[int]:
        max_budget = int(state.metadata.get("max_budget", 0) or 0)
        if max_budget <= 0:
            max_budget = int(self.config.get("max_budget", 0) or 0)
        if max_budget <= 0:
            max_budget = 10000

        if isinstance(self.iterative_budgets, (list, tuple)) and len(self.iterative_budgets) > 0:
            raw = [int(x) for x in self.iterative_budgets]
        else:
            raw = [100, 200, 500, 800, 1000, 2000, 5000, 10000, 20000, 30000]

        targets = sorted({x for x in raw if 0 < x <= max_budget})
        if len(targets) == 0 or targets[-1] != max_budget:
            targets.append(max_budget)
        return targets

    def _reset_iteration_stage_budgets(self, state: BenchmarkState) -> None:
        prev_q = int(state.attack_state.get("iter_prev_q", 0))
        target_q = int(state.attack_state.get("iter_target_q", 0))
        delta = max(0, target_q - prev_q)
        # Two query stages per iteration with separate accounting.
        a = int(delta // 2)
        b = int(delta - a)
        state.attack_state["stage_a_remaining"] = a
        state.attack_state["stage_b_remaining"] = b
        state.attack_state["iter_stage"] = "A"
        state.attack_state["step1_trained"] = False

    def _select_query_batch(self, k: int, state: BenchmarkState) -> QueryBatch:
        """Algorithm 2 query proposal.

        This implementation preserves paper-faithful separation:
        - Stage A: query originals from D_U and add to D_T
        - Stage B: query erased samples selected from D_T and add to D_E

        Selection may return fewer than k samples to stop exactly at iteration
        milestones (paper max-budget sequence).
        """
        if k <= 0:
            raise ValueError("BlackboxDissector select(k) requires k>0")

        unlabeled_indices = state.attack_state["unlabeled_indices"]

        # Load pool dataset
        if self.pool_dataset is None:
            dataset_config = state.metadata.get("dataset_config", {})
            if "data_mode" not in dataset_config:
                dataset_config = {"data_mode": "seed", **dataset_config}
            if "name" not in dataset_config:
                dataset_config = {"name": "CIFAR10", **dataset_config}
            self.pool_dataset = create_dataloader(
                dataset_config,
                batch_size=1,
                shuffle=False,
            ).dataset

        substitute = state.attack_state.get("substitute")
        device = state.metadata.get("device", "cpu")

        if substitute is None:
            sub_config = state.metadata.get("substitute_config", {})
            num_classes = int(
                state.metadata.get("num_classes")
                or state.metadata.get("dataset_config", {}).get("num_classes", 10)
            )
            substitute = create_substitute(
                arch=sub_config.get("arch", "resnet18"),
                num_classes=num_classes,
                input_channels=state.metadata.get("input_shape", (3, 32, 32))[0],
            ).to(device)
            state.attack_state["substitute"] = substitute

        # Respect iteration milestone boundary.
        target_q = int(state.attack_state.get("iter_target_q", 0))
        remaining_to_target = max(0, target_q - int(state.query_count))
        if remaining_to_target == 0:
            # No-op until the handler advances the iteration.
            input_shape = state.metadata.get("input_shape", (3, 32, 32))
            x = torch.empty((0, *input_shape), device=device)
            return QueryBatch(x=x, meta={"stage": "noop", "indices": [], "variant_types": []})

        stage = str(state.attack_state.get("iter_stage", "A"))
        a_rem = int(state.attack_state.get("stage_a_remaining", 0))
        b_rem = int(state.attack_state.get("stage_b_remaining", 0))

        # If Stage A can't proceed, switch to B; if Stage B can't proceed, switch to A.
        if stage == "A" and a_rem > 0 and len(unlabeled_indices) == 0:
            stage = "B"
            state.attack_state["iter_stage"] = "B"
        if stage == "B" and b_rem > 0:
            victim_labels: dict[int, int] = state.attack_state.get("victim_labels", {})
            transfer_indices = [
                idx
                for idx in state.attack_state.get("labeled_indices", [])
                if idx in victim_labels
            ]
            if len(transfer_indices) == 0:
                stage = "A"
                state.attack_state["iter_stage"] = "A"

        if stage == "A":
            budget_rem = int(state.attack_state.get("stage_a_remaining", 0))
            k_eff = min(int(k), budget_rem, remaining_to_target)
            if k_eff <= 0:
                state.attack_state["iter_stage"] = "B"
                return self._select_query_batch(k, state)

            k_eff = min(k_eff, len(unlabeled_indices))
            if k_eff <= 0:
                # Can't sample from D_U; move to stage B.
                state.attack_state["iter_stage"] = "B"
                return self._select_query_batch(k, state)

            selected = np.random.choice(unlabeled_indices, k_eff, replace=False).tolist()
            for idx in selected:
                state.attack_state["unlabeled_indices"].remove(idx)
                state.attack_state["labeled_indices"].append(idx)

            x_list = [self.pool_dataset[idx][0] for idx in selected]
            x = torch.stack(x_list)
            state.attack_state["stage_a_remaining"] = budget_rem - k_eff
            if state.attack_state["stage_a_remaining"] == 0:
                state.attack_state["iter_stage"] = "B"

            return QueryBatch(
                x=x,
                meta={
                    "stage": "A",
                    "indices": selected,
                    "variant_types": ["original"] * k_eff,
                },
            )

        # Stage B
        if len(state.attack_state.get("D_T_x", [])) > 0 and not state.attack_state.get("step1_trained", False):
            teacher_model = state.attack_state.get("substitute")
            self._generate_pseudo_labels(state, teacher_model)
            self.train_substitute(state)
            state.attack_state["step1_trained"] = True
            substitute = state.attack_state.get("substitute")
        budget_rem = int(state.attack_state.get("stage_b_remaining", 0))
        k_eff = min(int(k), budget_rem, remaining_to_target)
        if k_eff <= 0:
            # No budget left for this stage; try other stage or fall back to consuming remaining budget.
            if int(state.attack_state.get("stage_a_remaining", 0)) > 0:
                state.attack_state["iter_stage"] = "A"
                return self._select_query_batch(k, state)

            if remaining_to_target > 0:
                raise ValueError(
                    f"Query pool exhausted for {self.__class__.__name__}. "
                    f"Cannot select more queries to reach target {target_q}."
                )

            input_shape = state.metadata.get("input_shape", (3, 32, 32))
            x = torch.rand((1, *input_shape), device=device)
            return QueryBatch(x=x, meta={"stage": "noise", "indices": [-1], "variant_types": ["noise"]})

        victim_labels = state.attack_state.get("victim_labels", {})
        transfer_indices = [
            idx for idx in state.attack_state.get("labeled_indices", []) if idx in victim_labels
        ]

        substitute.eval()
        scored: list[tuple[int, float]] = []
        
        # BATCH PROCESSING START
        candidate_subset = Subset(self.pool_dataset, transfer_indices)
        candidate_loader = DataLoader(
            candidate_subset, 
            batch_size=self.selection_batch_size, 
            shuffle=False, 
            **pool_loader_kwargs(device)
        )
        
        current_idx_ptr = 0
        
        # [FEATURE] Detailed sub-progress bar for CAM scoring
        sel_pbar = tqdm(
            total=len(transfer_indices), 
            desc=" > [CAM Selection]", 
            leave=False, 
            unit="img",
            disable=len(transfer_indices) < self.selection_batch_size
        )
        
        for imgs, _ in candidate_loader:
            batch_size = imgs.size(0)
            batch_indices = transfer_indices[current_idx_ptr : current_idx_ptr + batch_size]
            current_idx_ptr += batch_size
            
            imgs = imgs.to(device, non_blocking=str(device).startswith("cuda"))
            labels = torch.tensor([victim_labels[idx] for idx in batch_indices], device=device)
            
            imgs_repeated = imgs.unsqueeze(1).repeat(1, self.n_variants, 1, 1, 1).view(-1, *imgs.shape[1:])
            labels_repeated = labels.unsqueeze(1).repeat(1, self.n_variants).view(-1)
            
            variants = cam_erase_batch(
                imgs_repeated,
                substitute,
                sl=self.sl, sh=self.sh, r1=self.r1, r2=self.r2,
                fill_min=self.fill_min, fill_max=self.fill_max,
                target_class=None
            )
            
            with torch.no_grad():
                logits = substitute(variants)
                probs = F.softmax(logits, dim=1)
                
            p_y0 = probs.gather(1, labels_repeated.unsqueeze(1)).squeeze(1)
            p_y0 = p_y0.view(batch_size, self.n_variants)
            
            best_variant_indices = p_y0.argmin(dim=1)
            
            all_msp = probs.max(dim=1)[0].view(batch_size, self.n_variants)
            best_msps = all_msp.gather(1, best_variant_indices.unsqueeze(1)).squeeze(1)
            
            variants_reshaped = variants.view(batch_size, self.n_variants, *imgs.shape[1:])
            gather_idx = best_variant_indices.view(batch_size, 1, 1, 1, 1).expand(-1, 1, *imgs.shape[1:])
            best_imgs = variants_reshaped.gather(1, gather_idx).squeeze(1)
            
            for i, idx in enumerate(batch_indices):
                state.attack_state["best_variant_img"][idx] = best_imgs[i].cpu()
                scored.append((idx, float(best_msps[i].item())))
            
            sel_pbar.update(batch_size)
        
        sel_pbar.close()
        scored.sort(key=lambda x: x[1], reverse=True)
        # BATCH PROCESSING END
        k_eff = min(k_eff, len(scored))
        if k_eff <= 0:
            # Can't form erased queries; fall back to querying unlabeled originals.
            state.attack_state["iter_stage"] = "A"
            return self._select_query_batch(min(int(k), remaining_to_target), state)

        selected = [idx for idx, _ in scored[:k_eff]]
        x_list = [state.attack_state["best_variant_img"][idx] for idx in selected]
        x = torch.stack(x_list)

        state.attack_state["stage_b_remaining"] = budget_rem - k_eff

        return QueryBatch(
            x=x,
            meta={
                "stage": "B",
                "indices": selected,
                "variant_types": ["erased"] * k_eff,
            },
        )

    def _generate_pseudo_labels(self, state: BenchmarkState, teacher: Optional[nn.Module]) -> None:
        pseudo_labels: dict[int, torch.Tensor] = {}
        if teacher is None:
            state.attack_state["pseudo_labels"] = pseudo_labels
            return

        unlabeled_indices = state.attack_state.get("unlabeled_indices", [])
        if len(unlabeled_indices) == 0:
            state.attack_state["pseudo_labels"] = pseudo_labels
            return

        if self.pool_dataset is None:
            dataset_config = state.metadata.get("dataset_config", {})
            if "data_mode" not in dataset_config:
                dataset_config = {"data_mode": "seed", **dataset_config}
            if "name" not in dataset_config:
                dataset_config = {"name": "CIFAR10", **dataset_config}
            self.pool_dataset = create_dataloader(
                dataset_config,
                batch_size=1,
                shuffle=False,
            ).dataset

        device = state.metadata.get("device", "cpu")
        victim_config = state.metadata.get("victim_config", {})
        normalization = victim_config.get("normalization")
        if normalization is None:
            normalization = {"mean": [0.0], "std": [1.0]}
        norm_mean = torch.tensor(normalization["mean"]).view(1, -1, 1, 1).to(device)
        norm_std = torch.tensor(normalization["std"]).view(1, -1, 1, 1).to(device)

        teacher.eval()
        subset = Subset(self.pool_dataset, unlabeled_indices)
        loader = DataLoader(
            subset,
            batch_size=min(self.selection_batch_size, len(unlabeled_indices)),
            shuffle=False,
            **pool_loader_kwargs(device),
        )

        current_idx_ptr = 0
        # [FEATURE] Detailed sub-progress bar for Pseudo-label generation
        pseudo_pbar = tqdm(
            total=len(unlabeled_indices),
            desc=" > [Pseudo-Labeling]",
            leave=False,
            unit="img",
            disable=len(unlabeled_indices) < self.selection_batch_size
        )
        
        with torch.no_grad():
            for x_batch, _ in loader:
                batch_size = x_batch.size(0)
                batch_indices = unlabeled_indices[current_idx_ptr : current_idx_ptr + batch_size]
                current_idx_ptr += batch_size

                x_batch = x_batch.to(device, non_blocking=str(device).startswith("cuda"))
                x_variants = random_erase_batch(
                    x_batch,
                    n=self.n_variants,
                    sl=self.sl,
                    sh=self.sh,
                    r1=self.r1,
                    r2=self.r2,
                    fill_min=self.fill_min,
                    fill_max=self.fill_max,
                )
                x_variants_norm = (x_variants - norm_mean) / norm_std
                logits_variants = teacher(x_variants_norm)
                probs_variants = F.softmax(logits_variants, dim=1)
                probs_variants = probs_variants.view(batch_size, self.n_variants, -1)
                soft_targets = probs_variants.mean(dim=1)

                for i, idx in enumerate(batch_indices):
                    pseudo_labels[int(idx)] = soft_targets[i].detach().cpu()
                
                pseudo_pbar.update(batch_size)
        
        pseudo_pbar.close()
        state.attack_state["pseudo_labels"] = pseudo_labels

    def train_substitute(self, state: BenchmarkState) -> None:
        """Train substitute model with Self-KD on Unlabeled Data.

        Loss = CE(victim_labels) + alpha * Consistency(unlabeled_data)

        Args:
            state: Current benchmark state
        """
        device = state.metadata.get("device", "cpu")
        d_t_x = state.attack_state.get("D_T_x", [])
        d_t_y = state.attack_state.get("D_T_y", [])
        d_e_x = state.attack_state.get("D_E_x", [])
        d_e_y = state.attack_state.get("D_E_y", [])

        if len(d_t_x) == 0 and len(d_e_x) == 0:
            return

        x_labeled = []
        y_labeled = []
        if len(d_t_x) > 0:
            x_labeled.append(torch.cat(d_t_x, dim=0))
            y_labeled.append(torch.cat(d_t_y, dim=0))
        if len(d_e_x) > 0:
            x_labeled.append(torch.cat(d_e_x, dim=0))
            y_labeled.append(torch.cat(d_e_y, dim=0))

        x_all = torch.cat(x_labeled, dim=0)
        y_all = torch.cat(y_labeled, dim=0)

        if self.pool_dataset is None:
            # Should be initialized in selection, but safe check
            dataset_config = state.metadata.get("dataset_config", {})
            if "data_mode" not in dataset_config:
                dataset_config = {"data_mode": "seed", **dataset_config}
            if "name" not in dataset_config:
                dataset_config = {"name": "CIFAR10", **dataset_config}
            self.pool_dataset = create_dataloader(
                dataset_config,
                batch_size=1,
                shuffle=False,
            ).dataset

        val_labeled_x = state.attack_state.get("val_labeled_x", [])
        val_labeled_y = state.attack_state.get("val_labeled_y", [])

        if len(val_labeled_x) == 0 or len(val_labeled_y) == 0:
            _, val_target = self._resolve_seed_and_validation_targets(total_budget=state.budget_remaining)
            if val_target > 0 and int(x_all.size(0)) > 2:
                val_size = min(int(val_target), int(x_all.size(0)) - 2)
                if val_size > 0:
                    state.attack_state["val_labeled_x"] = [x_all[:val_size].detach().cpu()]
                    state.attack_state["val_labeled_y"] = [y_all[:val_size].detach().cpu()]
                    x_all = x_all[val_size:]
                    y_all = y_all[val_size:]

        val_labeled_x = state.attack_state.get("val_labeled_x", [])
        val_labeled_y = state.attack_state.get("val_labeled_y", [])

        train_size = int(x_all.size(0))

        device = state.metadata.get("device", "cpu")
        num_classes = int(
            state.metadata.get("num_classes")
            or state.metadata.get("dataset_config", {}).get("num_classes", 10)
        )

        if train_size < 2:
            return

        train_subset = _LabeledTensorDataset(x_all, y_all)
        if len(val_labeled_x) > 0 and len(val_labeled_y) > 0:
            x_val = torch.cat(val_labeled_x, dim=0)
            y_val = torch.cat(val_labeled_y, dim=0)
            val_subset = _LabeledTensorDataset(x_val, y_val)
        else:
            total_size = int(train_subset.__len__())
            val_size = max(1, int(0.2 * total_size))
            train_size = total_size - val_size
            if train_size < 2:
                return
            train_subset, val_subset = torch.utils.data.random_split(
                train_subset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )

        sub_config = state.metadata.get("substitute_config", {})
        train_batch_size = int(
            sub_config.get("batch_size")
            or sub_config.get("trackA", {}).get("batch_size", self.batch_size)
        )
        train_workers = int(
            sub_config.get(
                "train_num_workers",
                sub_config.get("num_workers", self.config.get("num_workers", 0)),
            )
        )
        val_workers = int(
            sub_config.get(
                "val_num_workers",
                sub_config.get("num_workers", train_workers),
            )
        )
        pseudo_workers = int(
            sub_config.get(
                "pseudo_num_workers",
                sub_config.get("num_workers", train_workers),
            )
        )
        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=train_batch_size,
            shuffle=True,
            **pool_loader_kwargs(device, {"num_workers": train_workers}),
            drop_last=False,
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=train_batch_size,
            shuffle=False,
            **pool_loader_kwargs(device, {"num_workers": val_workers}),
        )

        # Teacher model = frozen copy of previous substitute (Eq. 7)
        teacher_model = state.attack_state.get("substitute")
        teacher: Optional[nn.Module]
        if teacher_model is None:
            teacher = None
        else:
            teacher = copy.deepcopy(teacher_model)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)

        if teacher is not None and len(state.attack_state.get("pseudo_labels", {})) == 0:
            self._generate_pseudo_labels(state, teacher)

        pseudo_labels = state.attack_state.get("pseudo_labels", {})
        pseudo_loader = None
        pseudo_iter = None
        if len(pseudo_labels) > 0:
            pseudo_indices = list(pseudo_labels.keys())

            pseudo_loader = torch.utils.data.DataLoader(
                _PseudoLabelDataset(pseudo_indices, pseudo_labels, self.pool_dataset),
                batch_size=train_batch_size,
                shuffle=True,
                **pool_loader_kwargs(device, {"num_workers": pseudo_workers}),
                drop_last=False,
            )
            pseudo_iter = iter(pseudo_loader)

        # Initialize student model FROM SCRATCH each iteration
        width_mult = int(sub_config.get("width_mult", 1))
        dropout_prob = float(sub_config.get("dropout_prob", 0.0))
        model = create_substitute(
            arch=sub_config.get("arch", "resnet18"),
            num_classes=num_classes,
            input_channels=state.metadata.get("input_shape", (3, 32, 32))[0],
            width_mult=width_mult,
            dropout_prob=dropout_prob,
        ).to(device)

        def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
            log_probs = F.log_softmax(logits, dim=1)
            return -(soft_targets * log_probs).sum(dim=1)

        victim_config = state.metadata.get("victim_config", {})
        normalization = victim_config.get("normalization")
        if normalization is None:
            normalization = {"mean": [0.0], "std": [1.0]}
        
        norm_mean = torch.tensor(normalization["mean"]).view(1, -1, 1, 1).to(device)
        norm_std = torch.tensor(normalization["std"]).view(1, -1, 1, 1).to(device)

        def step_fn(model_local: nn.Module, x_batch: torch.Tensor, y_batch: torch.Tensor) -> torch.Tensor:
            nonlocal pseudo_iter
            x_norm = (x_batch - norm_mean) / norm_std
            outputs = model_local(x_norm)
            loss_sup = F.cross_entropy(outputs, y_batch.long())

            loss_kd = torch.tensor(0.0, device=device)
            if pseudo_loader is not None and pseudo_iter is not None:
                try:
                    x_pseudo, y_pseudo = next(pseudo_iter)
                except StopIteration:
                    pseudo_iter = iter(pseudo_loader)
                    x_pseudo, y_pseudo = next(pseudo_iter)

                x_pseudo = x_pseudo.to(device)
                y_pseudo = y_pseudo.to(device)
                x_pseudo_norm = (x_pseudo - norm_mean) / norm_std
                logits_pseudo = model_local(x_pseudo_norm)
                loss_kd = soft_cross_entropy(logits_pseudo, y_pseudo).mean()

            return loss_sup + loss_kd

        def eval_fn(model_local: nn.Module, loader: DataLoader) -> float:
            # [FIX] Unify validation metric to Validation Loss (CrossEntropy)
            # Previously used F1 Score. Standardizing to Val Loss for fair comparison.
            model_local.eval()
            total_loss = 0.0
            total_count = 0
            loss_func = nn.CrossEntropyLoss()
            
            # Use normalization if provided
            victim_config = state.metadata.get("victim_config", {})
            normalization = victim_config.get("normalization")
            if normalization is None:
                normalization = {"mean": [0.0], "std": [1.0]}
            norm_mean = torch.tensor(normalization["mean"]).view(1, -1, 1, 1).to(device)
            norm_std = torch.tensor(normalization["std"]).view(1, -1, 1, 1).to(device)

            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    x_norm = (x - norm_mean) / norm_std
                    outputs = model_local(x_norm)
                    loss = loss_func(outputs, y.long())
                    total_loss += loss.item() * x.size(0)
                    total_count += x.size(0)
            return total_loss / total_count if total_count > 0 else float('inf')

        train_config = dict(sub_config)
        optimizer_config = dict(train_config.get("optimizer", {}))
        optimizer_config.setdefault("name", "sgd")
        optimizer_config.setdefault("lr", self.lr)
        optimizer_config.setdefault("momentum", self.momentum)
        optimizer_config.setdefault("weight_decay", self.l2_reg)
        train_config["optimizer"] = optimizer_config
        train_config["max_epochs"] = int(sub_config.get("max_epochs", self.max_epochs))
        train_config["patience"] = int(sub_config.get("patience", self.patience))
        train_config["use_tqdm"] = True  # [FEATURE] Enable TQDM
        
        trainer = SubstituteTrainer(train_config, device=device, logger=self.logger)
        steps_per_epoch = max(1, int(math.ceil(train_size / max(1, train_batch_size))))
        request = TrainRequest(
            model=model,
            train_loader=train_loader,
            loss_fn=lambda outputs, targets: F.cross_entropy(outputs, targets.long()),
            step_fn=step_fn,
            val_loader=val_loader,
            eval_fn=eval_fn,
            early_stop_mode="min", # Minimizing Validation Loss
            load_best=True,
            max_steps=int(train_config["max_epochs"]) * steps_per_epoch,
            validate_every=steps_per_epoch,
            patience=int(train_config["patience"]) * steps_per_epoch,
        )
        result = trainer.train(request)

        # Store in state
        state.attack_state["substitute"] = model
        if result.best_value is not None:
            self.logger.info("Dissector substitute trained. Best Val Loss: %.4f", result.best_value)
        else:
            self.logger.info("Dissector substitute trained.")
        # Round Evaluation
        # self._evaluate_current_substitute(model, device)

    def _compute_f1(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: str,
        norm_mean: torch.Tensor,
        norm_std: torch.Tensor,
    ) -> float:
        """Compute F1 score on validation set.

        Args:
            model: Model to evaluate
            val_loader: Validation data loader
            device: Device to use

        Returns:
            F1 score (macro average)
        """
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                x_norm = (x_batch - norm_mean) / norm_std
                outputs = model(x_norm)

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)

                targets = y_batch.cpu().numpy()
                all_targets.extend(targets)

        return f1_score(all_targets, all_preds, average="macro")
