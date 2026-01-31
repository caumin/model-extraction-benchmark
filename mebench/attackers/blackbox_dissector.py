"""Black-box Dissector attack implementation."""

from typing import Dict, Any, List, Tuple, Optional
import copy
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score
from tqdm import tqdm

from mebench.attackers.runner import AttackRunner
from mebench.core.context import BenchmarkContext
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.data.loaders import create_dataloader, get_test_dataloader
from mebench.models.substitute_factory import create_substitute
from mebench.eval.metrics import evaluate_substitute


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


class BlackboxDissector(AttackRunner):
    """Black-box Dissector: CAM-driven erasing for hard-label victims.

    Algorithm loop:
    1. For each sample in pool, generate N erasing variants:
       - CAM-driven: Erase most attended region
       - Random: N random erasures
    2. Select best variant per sample:
       - Substitute's MSP change (original vs erased)
       - Maximize information gain
    3. Query victim on original + best erased (2x budget per sample)
    4. Train substitute:
       - CE loss on victim labels (original + erased)
       - CE loss on pseudo-labels (average of N random variants)
    5. Select top-k samples by substitute MSP (erased variants)
    6. Repeat until budget exhausted

    Hyperparameters:
    - n_variants: Number of erasing variants (10)
    - erasing_ratio: Ratio of area to erase (0.5)
    - batch_size: Training batch size (128)
    - lr: Learning rate (0.001, Adam)
    """

    def __init__(self, config: dict, state: BenchmarkState):
        """Initialize Blackbox Dissector attack.

        Args:
            config: Attack configuration
            state: Global benchmark state
        """
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
        self.lr = float(config.get("lr", 0.1))
        self.max_epochs = int(config.get("max_epochs", 1000))
        self.patience = int(config.get("patience", 100))
        self.dropout = float(config.get("dropout", 0.1))
        # [P0 FIX] Paper mandates 200 epochs for BlackBox Dissector
        self.max_epochs = int(config.get("max_epochs", 200))
        self.l2_reg = float(config.get("l2_reg", 5e-4))

        # Algorithm 2 (outer loop) iterative max-budget sequence.
        # Default: {0.1K, 0.2K, 0.5K, 0.8K, 1K, 2K, 5K, 10K, 20K, 30K}, capped to max_budget.
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
            step_size = self._default_step_size(ctx)
            query_batch = self._select_query_batch(step_size, self.state)
            if int(query_batch.x.size(0)) == 0:
                if query_batch.meta.get("stage") == "noop":
                    self._advance_iteration_if_needed(self.state)
                continue
            oracle_output = ctx.query(query_batch.x, meta=query_batch.meta)
            self._handle_oracle_output(query_batch, oracle_output, self.state)
            pbar.update(query_batch.x.size(0))
        pbar.close()

        # Pool dataset
        self.pool_dataset = None

    def _advance_iteration_if_needed(self, state: BenchmarkState) -> None:
        target_q = int(state.attack_state.get("iter_target_q", 0))
        if target_q > 0 and int(state.query_count) >= target_q:
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
        for idx in transfer_indices:
            img, _ = self.pool_dataset[idx]
            y0 = int(victim_labels[idx])

            best_variant: Optional[torch.Tensor] = None
            best_score: Optional[float] = None
            best_msp: Optional[float] = None
            for _ in range(self.n_variants):
                variant = cam_erase(
                    img,
                    substitute,
                    sl=self.sl,
                    sh=self.sh,
                    r1=self.r1,
                    r2=self.r2,
                    fill_min=self.fill_min,
                    fill_max=self.fill_max,
                )
                with torch.no_grad():
                    probs = F.softmax(
                        substitute(variant.unsqueeze(0).to(device)), dim=1
                    ).squeeze(0)
                p_y0 = float(probs[y0].item())
                msp = float(probs.max().item())
                # Eq.(4): choose erased variant minimizing prob of original label.
                if best_score is None or p_y0 < best_score:
                    best_score = p_y0
                    best_variant = variant
                    best_msp = msp

            if best_variant is not None and best_msp is not None:
                state.attack_state["best_variant_img"][idx] = best_variant
                scored.append((idx, best_msp))

        scored.sort(key=lambda x: x[1], reverse=True)
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

        class LabeledDataset(torch.utils.data.Dataset):
            def __init__(self, x: torch.Tensor, y: torch.Tensor):
                self.x = x
                self.y = y

            def __len__(self) -> int:
                return int(self.x.size(0))

            def __getitem__(self, idx: int):
                return self.x[idx], self.y[idx]

        labeled_dataset = LabeledDataset(x_all, y_all)

        # Create unlabeled dataset (D_U)
        unlabeled_indices = state.attack_state.get("unlabeled_indices", [])
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

        # Use Subset for unlabeled data
        # Note: pool_dataset returns (img, label), we ignore label
        unlabeled_dataset = Subset(self.pool_dataset, unlabeled_indices)

        # Use 20% validation split from Labeled Data
        total_size = len(labeled_dataset)
        val_size = max(1, int(0.2 * total_size))
        train_size = total_size - val_size

        device = state.metadata.get("device", "cpu")
        num_classes = int(
            state.metadata.get("num_classes")
            or state.metadata.get("dataset_config", {}).get("num_classes", 10)
        )

        if train_size < 2:
            return

        train_subset, val_subset = torch.utils.data.random_split(
            labeled_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Unlabeled loader (cycle it)
        if len(unlabeled_dataset) > 0:
            unlabeled_loader = torch.utils.data.DataLoader(
                unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True, # Drop last to avoid tiny batches
            )
            unlabeled_iter = iter(unlabeled_loader)
            unlabeled_desc = tqdm(total=len(train_loader), desc="[BlackboxDissector] Predicting Pool", leave=False)
        else:
            unlabeled_loader = None
            unlabeled_iter = None
            unlabeled_desc = None

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

        # Initialize student model FROM SCRATCH each iteration
        sub_config = state.metadata.get("substitute_config", {})
        opt_params = sub_config.get("optimizer", {})
        
        model = create_substitute(
            arch=sub_config.get("arch", "resnet18"),
            num_classes=num_classes,
            input_channels=state.metadata.get("input_shape", (3, 32, 32))[0],
        ).to(device)

        # Optimizer: SGD as in paper training process
        import torch.optim as optim
        optimizer = optim.SGD(
            model.parameters(),
            lr=float(opt_params.get("lr", self.lr)),
            momentum=0.9,
            weight_decay=5e-4,
        )

        def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
            log_probs = F.log_softmax(logits, dim=1)
            return -(soft_targets * log_probs).sum(dim=1)

        # Early stopping
        best_f1 = 0.0
        patience_counter = 0
        best_model_state = None

        # Training loop
        victim_config = state.metadata.get("victim_config", {})
        normalization = victim_config.get("normalization")
        if normalization is None:
            normalization = {"mean": [0.0], "std": [1.0]}
        
        norm_mean = torch.tensor(normalization["mean"]).view(1, -1, 1, 1).to(device)
        norm_std = torch.tensor(normalization["std"]).view(1, -1, 1, 1).to(device)

        epoch_pbar = tqdm(range(self.max_epochs), desc="[BlackboxDissector] Training Substitute", leave=False)
        for epoch in epoch_pbar:
            model.train()
            train_loss = 0.0

            for x_batch, y_batch in train_loader:
                if unlabeled_desc is not None:
                    unlabeled_desc.update(1)
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                # Supervised Loss on Labeled Data
                # Normalize images for substitute
                x_norm = (x_batch - norm_mean) / norm_std
                
                outputs = model(x_norm)
                y_labels = y_batch
                loss_sup = F.cross_entropy(outputs, y_labels.long())
                
                # Self-KD Loss on Unlabeled Data
                loss_kd = torch.tensor(0.0, device=device)
                
                if unlabeled_loader is not None and teacher is not None:
                    try:
                        x_unlab, _ = next(unlabeled_iter)
                    except StopIteration:
                        unlabeled_iter = iter(unlabeled_loader)
                        x_unlab, _ = next(unlabeled_iter)
                    
                    if len(x_unlab) > 1:
                        # 1. Generate pseudo-labels using erased variants
                        # For efficiency, we process variants in batches
                        # Target = Mean(Softmax(Model(Erased_Variants)))
                        
                        variants_list: list[torch.Tensor] = []
                        for img in x_unlab:
                            variants_list.extend(
                                random_erase(
                                    img,
                                    n=self.n_variants,
                                    sl=self.sl,
                                    sh=self.sh,
                                    r1=self.r1,
                                    r2=self.r2,
                                    fill_min=self.fill_min,
                                    fill_max=self.fill_max,
                                )
                            )
                        
                        # Stack all variants: [B_u * N, C, H, W]
                        x_variants = torch.stack(variants_list).to(device)
                        x_variants_norm = (x_variants - norm_mean) / norm_std
                        
                        with torch.no_grad():
                            logits_variants = teacher(x_variants_norm)
                            probs_variants = F.softmax(logits_variants, dim=1)
                            
                            # Reshape to [B_u, N, NumClasses]
                            probs_variants = probs_variants.view(len(x_unlab), self.n_variants, num_classes)
                            
                            # Average over variants to get pseudo-label
                            pseudo_labels = probs_variants.mean(dim=1)
                        
                        # 2. Consistency Loss
                        # Loss = CE(Model(Original), PseudoLabel)
                        x_unlab_norm = (x_unlab.to(device) - norm_mean) / norm_std
                        logits_unlab = model(x_unlab_norm)
                        
                        # Use Soft Cross Entropy
                        loss_kd = soft_cross_entropy(logits_unlab, pseudo_labels).mean()

                # Eq.(7) literal: supervised term + pseudo-label term (no extra weighting).
                loss = loss_sup + loss_kd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            val_f1 = self._compute_f1(model, val_loader, device, norm_mean, norm_std)
            epoch_pbar.set_postfix({"Loss": f"{train_loss/len(train_loader):.4f}", "F1": f"{val_f1:.4f}"})

            if unlabeled_desc is not None:
                unlabeled_desc.reset()

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            # if epoch % 10 == 0:
            #     print(
            #         f"Dissector Epoch {epoch}: Loss: {train_loss/len(train_loader):.4f}, Val F1: {val_f1:.4f}"
            #     )

            # Early stopping
            if patience_counter >= self.patience:
                self.logger.info(f"Dissector Early stopping at epoch {epoch}")
                break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Store in state
        state.attack_state["substitute"] = model
        self.logger.info(f"Dissector substitute trained. Best Val F1: {best_f1:.4f}")
        if unlabeled_desc is not None:
            unlabeled_desc.close()

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
