"""Black-box Dissector attack implementation."""

from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score

from mebench.attackers.base import BaseAttack
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.data.loaders import create_dataloader
from mebench.models.substitute_factory import create_substitute


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
    erase_ratio: float = 0.5,
) -> List[torch.Tensor]:
    """Generate N random erasing variants.

    Args:
        img: Input image [C, H, W]
        n: Number of variants to generate
        erase_ratio: Ratio of area to erase

    Returns:
        List of erased images
    """
    variants = []
    c, h, w = img.shape
    
    # Not used with random sampling of area/aspect ratio
    # erase_area = int(h * w * erase_ratio)

    for _ in range(n):
        # Random erasing
        erased = img.clone()

        # Random position
        y = np.random.randint(0, h)
        x = np.random.randint(0, w)

        # Paper: Sample random area and aspect ratio (consistent with cam_erase)
        area = h * w
        target_area = np.random.uniform(0.02, 0.4) * area
        aspect_ratio = np.random.uniform(0.3, 3.3)

        h_erase = int(round(np.sqrt(target_area * aspect_ratio)))
        w_erase = int(round(np.sqrt(target_area / aspect_ratio)))

        h_erase = min(h, max(1, h_erase))
        w_erase = min(w, max(1, w_erase))

        # Clip bounds
        y_end = min(y + h_erase, h)
        x_end = min(x + w_erase, w)
        h_erase = y_end - y
        w_erase = x_end - x

        # Erase with random noise
        erased[:, y:y_end, x:x_end] = torch.randn(c, h_erase, w_erase)

        variants.append(erased)

    return variants


def cam_erase(
    img: torch.Tensor,
    model: nn.Module,
    erase_ratio: float = 0.5,
) -> torch.Tensor:
    """Generate CAM-driven erasing variant.

    Args:
        img: Input image [C, H, W]
        model: Substitute model for attention map
        erase_ratio: Ratio of area to erase

    Returns:
        Erased image
    """
    # Get attention heatmap
    x_batch = img.unsqueeze(0)
    heatmap = generate_gradcam_heatmap(model, x_batch)[0]  # [H, W]

    c, h, w = img.shape

    erase_area = int(h * w * erase_ratio)
    
    # Not used with random sampling of area/aspect ratio
    # erase_size = max(1, int(np.sqrt(erase_area)))
    # h_erase = min(h, erase_size)
    # w_erase = min(w, max(1, erase_area // h_erase))

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

    # Paper: Sample random area and aspect ratio
    area = h * w
    target_area = np.random.uniform(0.02, 0.4) * area
    aspect_ratio = np.random.uniform(0.3, 3.3)

    h_erase = int(round(np.sqrt(target_area * aspect_ratio)))
    w_erase = int(round(np.sqrt(target_area / aspect_ratio)))

    h_erase = min(h, max(1, h_erase))
    w_erase = min(w, max(1, w_erase))

    y1 = max(0, center_y - h_erase // 2)
    x1 = max(0, center_x - w_erase // 2)
    y2 = min(h, y1 + h_erase)
    x2 = min(w, x1 + w_erase)

    erased = img.clone()
    # Paper (Prior-driven Erasing): fill erased region with random noise.
    erased[:, y1:y2, x1:x2] = torch.randn(
        c,
        y2 - y1,
        x2 - x1,
        device=img.device,
        dtype=img.dtype,
    )

    return erased


class BlackboxDissector(BaseAttack):
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
        self.n_variants = int(config.get("n_variants", 2))
        self.erasing_ratio = float(config.get("erasing_ratio", 0.25))
        self.erasing_ratio_schedule = config.get("erasing_ratio_schedule")

        # Training hyperparameters
        self.batch_size = int(config.get("batch_size", 128))
        self.lr = float(config.get("lr", 0.001))
        self.max_epochs = int(config.get("max_epochs", 1000))
        self.patience = int(config.get("patience", 100))
        self.dropout = float(config.get("dropout", 0.1))
        self.l2_reg = float(config.get("l2_reg", 0.001))

        # Initialize attack state
        self._initialize_state(state)

        # Pool dataset
        self.pool_dataset = None

    def _initialize_state(self, state: BenchmarkState) -> None:
        """Initialize attack-specific state.

        Args:
            state: Global benchmark state to update
        """
        dataset_config = state.metadata.get("dataset_config", {})
        config_dataset = self.config.get("dataset", {})
        if dataset_config:
            pool_config = dataset_config
        elif config_dataset:
            pool_config = config_dataset
        else:
            pool_config = {}

        pool_size = pool_config.get("pool_size")
        if pool_size is None:
            if "seed_size" in pool_config:
                pool_size = pool_config["seed_size"]
            else:
                pool_size = 10000

        state.attack_state["labeled_indices"] = []
        state.attack_state["unlabeled_indices"] = list(range(pool_size))
        state.attack_state["query_data_x"] = []
        state.attack_state["query_data_y"] = []
        state.attack_state["query_data_indices"] = []

        # Store pseudo-labels for Self-KD
        state.attack_state["pseudo_labels"] = {}  # idx -> averaged soft label

        # Victim top-1 labels for original images in transfer set (D_T)
        state.attack_state["victim_labels"] = {}  # idx -> int label

        # Store best variant per sample for selection
        state.attack_state["best_variant_idx"] = {}  # idx -> variant index
        state.attack_state["best_variant_img"] = {}  # idx -> tensor

        state.attack_state["substitute"] = None

    def propose(self, k: int, state: BenchmarkState) -> QueryBatch:
        """Propose k queries using CAM-driven selection.

        Contract: propose(k) returns exactly k images.

        Paper-aligned behavior:
        - Query original images from D_U to build/expand transfer set D_T.
        - For images in D_T (victim top-1 known), generate N CAM-erased variants
          and select the variant that minimizes prob of the original label (Eq. 4).
        - Select which erased variants to query by highest MSP of the selected
          erased image (Eq. 5).

        Args:
            k: Number of samples to select (actual queries = 2*k)
            state: Current benchmark state

        Returns:
            QueryBatch with 2*k queries
        """
        if k <= 0:
            raise ValueError("BlackboxDissector propose(k) requires k>0")

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

        substitute = state.attack_state["substitute"]
        device = state.metadata.get("device", "cpu")

        victim_labels: dict[int, int] = state.attack_state.get("victim_labels", {})
        transfer_indices = [idx for idx in state.attack_state.get("labeled_indices", []) if idx in victim_labels]

        n_erased = 0
        if substitute is not None and len(transfer_indices) > 0:
            n_erased = min(int(k) // 2, len(transfer_indices))

        n_orig = int(k) - n_erased
        if len(unlabeled_indices) < n_orig:
            short = n_orig - len(unlabeled_indices)
            n_orig = len(unlabeled_indices)
            if substitute is not None and len(transfer_indices) > n_erased:
                n_erased = min(n_erased + short, len(transfer_indices))

        if n_orig == 0 and n_erased == 0:
            input_shape = state.metadata.get("input_shape", (3, 32, 32))
            x = torch.randn(int(k), *input_shape)
            meta = {"indices": [], "variant_types": [], "pool_exhausted": True}
            return QueryBatch(x=x, meta=meta)

        # 1) Select originals from D_U (random)
        selected_orig: list[int] = []
        if n_orig > 0:
            selected_orig = np.random.choice(unlabeled_indices, n_orig, replace=False).tolist()

        for idx in selected_orig:
            state.attack_state["unlabeled_indices"].remove(idx)
            state.attack_state["labeled_indices"].append(idx)

        # 2) Select erased queries from D_T (Eq.4 then Eq.5)
        selected_erased: list[int] = []
        if n_erased > 0:
            substitute.eval()
            # Loop for erased variants selection
            # Note: We do NOT wrap this loop in torch.no_grad() because cam_erase requires gradients.
            erase_ratio = self._current_erasing_ratio(state)
            scored: list[tuple[int, float]] = []
            for idx in transfer_indices:
                img, _ = self.pool_dataset[idx]
                y0 = int(victim_labels[idx])

                best_variant = None
                best_ce = None
                best_msp = None
                for _ in range(self.n_variants):
                    variant = cam_erase(img, substitute, erase_ratio)
                    with torch.no_grad():
                        probs = F.softmax(substitute(variant.unsqueeze(0).to(device)), dim=1).squeeze(0)
                        p_y0 = float(probs[y0].item())
                        ce = -float(np.log(max(p_y0, 1e-12)))
                        if best_ce is None or ce > best_ce:
                            best_ce = ce
                            best_variant = variant
                            best_msp = float(probs.max().item())

                if best_variant is not None and best_msp is not None:
                    state.attack_state["best_variant_img"][idx] = best_variant
                    scored.append((idx, best_msp))

            scored.sort(key=lambda x: x[1], reverse=True)
            selected_erased = [idx for idx, _ in scored[:n_erased]]

        # Build query batch (exactly k images)
        x_list: list[torch.Tensor] = []
        indices_list: list[int] = []
        variant_types: list[str] = []

        for idx in selected_orig:
            img, _ = self.pool_dataset[idx]
            x_list.append(img)
            indices_list.append(idx)
            variant_types.append("original")

        for idx in selected_erased:
            img, _ = self.pool_dataset[idx]
            erased = state.attack_state["best_variant_img"].get(
                idx,
                cam_erase(img, substitute, self._current_erasing_ratio(state))
                if substitute is not None
                else random_erase(img, n=1, erase_ratio=self._current_erasing_ratio(state))[0],
            )
            x_list.append(erased)
            indices_list.append(idx)
            variant_types.append("erased")

        if len(x_list) < int(k):
            input_shape = state.metadata.get("input_shape", (3, 32, 32))
            for _ in range(int(k) - len(x_list)):
                x_list.append(torch.randn(*input_shape))
                indices_list.append(-1)
                variant_types.append("noise")

        x = torch.stack(x_list[: int(k)])
        meta = {
            "indices": indices_list[: int(k)],
            "variant_types": variant_types[: int(k)],
            "erased_indices": [i for i, t in enumerate(variant_types[: int(k)]) if t == "erased"],
        }

        return QueryBatch(x=x, meta=meta)

    def _current_erasing_ratio(self, state: BenchmarkState) -> float:
        if not self.erasing_ratio_schedule:
            return self.erasing_ratio
        if not isinstance(self.erasing_ratio_schedule, (list, tuple)):
            return self.erasing_ratio
        if len(self.erasing_ratio_schedule) != 2:
            return self.erasing_ratio

        start, end = self.erasing_ratio_schedule
        max_budget = state.metadata.get("max_budget", 1)
        progress = min(state.query_count / max_budget, 1.0) if max_budget else 0.0
        return float(start + (end - start) * progress)

    def observe(
        self,
        query_batch: QueryBatch,
        oracle_output: OracleOutput,
        state: BenchmarkState,
    ) -> None:
        """Observe oracle response and update internal state.

        Args:
            query_batch: The query batch that was sent
            oracle_output: Oracle response
            state: Current benchmark state
        """
        x_batch = query_batch.x
        y_batch = oracle_output.y

        # Record victim top-1 label for original images in transfer set D_T.
        indices = query_batch.meta.get("indices", [])
        variant_types = query_batch.meta.get("variant_types", [])
        victim_labels: dict[int, int] = state.attack_state.get("victim_labels", {})
        for i, idx in enumerate(indices):
            if i >= len(variant_types):
                break
            if variant_types[i] != "original":
                continue
            idx_i = int(idx)
            if idx_i < 0:
                continue
            if oracle_output.kind == "soft_prob":
                label_i = int(torch.argmax(y_batch[i]).item())
            else:
                label_i = int(y_batch[i].item())
            victim_labels[idx_i] = label_i
        state.attack_state["victim_labels"] = victim_labels

        # Store query data
        state.attack_state["query_data_x"].append(x_batch.cpu())
        state.attack_state["query_data_y"].append(y_batch.cpu())
        state.attack_state["query_data_indices"].append(
            torch.tensor(query_batch.meta.get("indices", []))
        )

        # Note: Pseudo-label generation logic for Self-KD has been moved 
        # to train_substitute to correctly use Unlabeled Data (D_U).
        # The previous logic here only used the query batch (D_T), which was incorrect.

        # Train substitute periodically
        labeled_count = len(state.attack_state["labeled_indices"])
        if labeled_count % 100 == 0 and labeled_count > 0:
            self.train_substitute(state)

    def train_substitute(self, state: BenchmarkState) -> None:
        """Train substitute model with Self-KD on Unlabeled Data.

        Loss = CE(victim_labels) + alpha * Consistency(unlabeled_data)

        Args:
            state: Current benchmark state
        """
        query_data_x = state.attack_state["query_data_x"]
        query_data_y = state.attack_state["query_data_y"]
        query_data_indices = state.attack_state["query_data_indices"]

        if len(query_data_x) == 0:
            return

        # Concatenate query data (Labeled Set D_T)
        x_all = torch.cat(query_data_x, dim=0)
        y_all = torch.cat(query_data_y, dim=0)
        idx_all = torch.cat(query_data_indices, dim=0)

        # Create labeled dataset
        class QueryDataset(torch.utils.data.Dataset):
            def __init__(self, x, y):
                self.x = x
                self.y = y
                self.indices = idx_all

            def __len__(self):
                return len(self.x)

            def __getitem__(self, idx):
                return self.x[idx], self.y[idx], self.indices[idx]

        labeled_dataset = QueryDataset(x_all, y_all)

        # Create unlabeled dataset (D_U)
        unlabeled_indices = state.attack_state.get("unlabeled_indices", [])
        if self.pool_dataset is None:
             # Should be initialized in propose, but safe check
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
            model = create_substitute(
                arch="resnet18",
                num_classes=num_classes,
                input_channels=state.metadata.get("input_shape", (3, 32, 32))[0],
            ).to(device)
            state.attack_state["substitute"] = model
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
        else:
            unlabeled_loader = None
            unlabeled_iter = None

        # Initialize model
        sub_config = state.metadata.get("substitute_config", {})
        opt_params = sub_config.get("optimizer", {})
        
        model = create_substitute(
            arch=sub_config.get("arch", "resnet18"),
            num_classes=num_classes,
            input_channels=state.metadata.get("input_shape", (3, 32, 32))[0],
        ).to(device)

        # Optimizer
        import torch.optim as optim
        optimizer = optim.Adam(
            model.parameters(),
            lr=float(opt_params.get("lr", self.lr)),
            weight_decay=float(opt_params.get("weight_decay", self.l2_reg)),
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

        erase_ratio_val = self._current_erasing_ratio(state)

        for epoch in range(self.max_epochs):
            model.train()
            train_loss = 0.0

            for x_batch, y_batch, _ in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                # Supervised Loss on Labeled Data
                # Normalize images for substitute
                x_norm = (x_batch - norm_mean) / norm_std
                
                outputs = model(x_norm)
                y_labels = y_batch.argmax(dim=1) if y_batch.ndim > 1 else y_batch
                
                loss_sup = F.cross_entropy(outputs, y_labels.long())
                
                # Self-KD Loss on Unlabeled Data
                loss_kd = torch.tensor(0.0, device=device)
                
                if unlabeled_loader is not None:
                    try:
                        x_unlab, _ = next(unlabeled_iter)
                    except StopIteration:
                        unlabeled_iter = iter(unlabeled_loader)
                        x_unlab, _ = next(unlabeled_iter)
                    
                    if len(x_unlab) > 1:
                        # 1. Generate pseudo-labels using erased variants
                        # For efficiency, we process variants in batches
                        # Target = Mean(Softmax(Model(Erased_Variants)))
                        
                        variants_list = []
                        # Generate N variants per image
                        for img in x_unlab:
                            # random_erase returns List[Tensor]
                            variants_list.extend(random_erase(img, n=self.n_variants, erase_ratio=erase_ratio_val))
                        
                        # Stack all variants: [B_u * N, C, H, W]
                        x_variants = torch.stack(variants_list).to(device)
                        x_variants_norm = (x_variants - norm_mean) / norm_std
                        
                        with torch.no_grad():
                            logits_variants = model(x_variants_norm)
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

                # Combined Loss
                # Paper typically uses alpha=1.0 or 0.5. Preserving previous 0.5 logic
                alpha = 0.5
                loss = (1.0 - alpha) * loss_sup + alpha * loss_kd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            val_f1 = self._compute_f1(model, val_loader, device)

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                print(
                    f"Dissector Epoch {epoch}: Loss: {train_loss/len(train_loader):.4f}, Val F1: {val_f1:.4f}"
                )

            # Early stopping
            if patience_counter >= self.patience:
                print(f"Dissector Early stopping at epoch {epoch}")
                break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Store in state
        state.attack_state["substitute"] = model
        print(f"Dissector substitute trained. Best F1: {best_f1:.4f}")

    def _compute_f1(self, model: nn.Module, val_loader: DataLoader, device: str) -> float:
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
            for x_batch, y_batch, _ in val_loader:
                x_batch = x_batch.to(device)
                outputs = model(x_batch)

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)

                if y_batch.ndim > 1:
                    targets = torch.argmax(y_batch, dim=1).cpu().numpy()
                else:
                    targets = y_batch.cpu().numpy()
                all_targets.extend(targets)

        return f1_score(all_targets, all_preds, average="macro")
