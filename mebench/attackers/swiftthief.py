"""SwiftThief attack implementation."""

from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score
import torchvision.transforms as transforms

from mebench.attackers.base import BaseAttack
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.data.loaders import create_dataloader
from mebench.models.substitute_factory import create_substitute


class SimSiamProjectionHead(nn.Module):
    """Projection head for SimSiam contrastive learning.

    Projects features to lower-dimensional space for contrastive learning.
    Architecture: Linear -> BN -> ReLU -> Linear -> BN -> ReLU -> Linear -> BN(affine=False)
    (3-layer MLP as per SimSiam paper and SwiftThief implementation)
    """

    def __init__(self, in_dim: int, proj_dim: int = 2048, hidden_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
            nn.BatchNorm1d(proj_dim, affine=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features [B, in_dim]

        Returns:
            Projected features [B, proj_dim]
        """
        return self.net(x)


class SimSiamPredictorHead(nn.Module):
    """Prediction head for SimSiam contrastive learning.

    Architecture: Linear -> BN -> ReLU -> Linear
    """

    def __init__(self, in_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features [B, in_dim]

        Returns:
            Predicted features [B, in_dim]
        """
        return self.net(x)


class SwiftThiefSubstitute(nn.Module):
    """Substitute model wrapper exposing a representation function."""

    def __init__(self, backbone: nn.Module, classifier: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def features(self, x: torch.Tensor) -> torch.Tensor:
        # Handle channel mismatch if SSL transform produces RGB but backbone expects 1-channel
        # For LeNet, the first layer is likely in a Sequential block if using create_substitute
        first_layer = None
        if hasattr(self.backbone, "conv1"):
            first_layer = self.backbone.conv1
        elif isinstance(self.backbone, nn.Sequential) and len(self.backbone) > 0:
             if isinstance(self.backbone[0], nn.Conv2d):
                 first_layer = self.backbone[0]
        
        if first_layer is not None:
            # Handle Sequential (LeNet features)
            if isinstance(first_layer, nn.Sequential) and len(first_layer) > 0:
                first_layer = first_layer[0]
            
            if hasattr(first_layer, 'in_channels') and x.shape[1] == 3 and first_layer.in_channels == 1:
                x = x.mean(dim=1, keepdim=True)
            
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        return self.classifier(feats)


class SwiftThief(BaseAttack):
    """SwiftThief: Contrastive learning for efficient model extraction.

    Algorithm loop:
    1. Split dataset: Queried set Q, Unqueried set U
    2. Initial sampling: Uniform random from U
    3. Train substitute with combined losses:
       - Self-supervised contrastive on U (SimSiam)
       - Soft-supervised contrastive on Q (entropy-weighted)
       - Matching loss: CE on Q vs victim labels
       - FGSM regularizer on Q
    4. Sampling strategy:
       - Default: Entropy-based selection from U
       - Switch to rare-class prioritization when budget low
    5. Repeat until budget exhausted

    Hyperparameters (from paper + questions doc):
    - I: Total rounds (default 10)
    - initial_seed_ratio: Initial random seed as ratio of budget (0.1)
    - lambda1: Self-supervised loss weight (1.0)
    - lambda2: FGSM regularizer weight (0.01)
    - lambda3: Matching loss weight (1.0)
    - fgsm_epsilon: FGSM perturbation size (0.01)
    - projection_dim: SimSiam projection head dim (2048)
    - batch_size: Training batch size (256)
    - lr: Learning rate (0.001, AdamW/Adam)
    """

    def __init__(self, config: dict, state: BenchmarkState):
        """Initialize SwiftThief attack.

        Args:
            config: Attack configuration
            state: Global benchmark state
        """
        super().__init__(config, state)

        # Hyperparameters
        self.I = int(config.get("I", 10))  # Total rounds
        self.initial_seed_ratio = float(config.get("initial_seed_ratio", 0.1))
        self.lambda1 = float(config.get("lambda1", 1.0))  # Self-supervised weight
        self.lambda2 = float(config.get("lambda2", 0.01))  # FGSM regularizer
        self.lambda3 = float(config.get("lambda3", 1.0))  # Matching loss weight
        self.fgsm_epsilon = float(config.get("fgsm_epsilon", 0.01))
        self.projection_dim = int(config.get("projection_dim", 2048))

        # Sampling hyperparameters (paper KDE-based rare-class prioritization)
        self.kde_sigma = float(config.get("kde_sigma", 1.0))
        self.max_pool_eval = int(config.get("max_pool_eval", 2000))

        # Training hyperparameters
        self.batch_size = int(config.get("batch_size", 256))
        self.lr = float(config.get("lr", 0.001))
        self.momentum = float(config.get("momentum", 0.9))
        self.weight_decay = float(config.get("weight_decay", 5e-4))
        self.max_epochs = int(config.get("max_epochs", 1000))
        self.patience = int(config.get("patience", 100))

        # Initialize attack state
        self._initialize_state(state)

        # Pool dataset (will be loaded in propose)
        self.pool_dataset = None

        # Feature extractor and projection head
        self.feature_extractor = None
        self.projection_head = None
        self.predictor_head = None

        self._ssl_transforms = None

    def _build_ssl_transforms(self, state: BenchmarkState) -> transforms.Compose:
        victim_config = state.metadata.get("victim_config", {})
        normalization = victim_config.get("normalization")
        if normalization is None:
            normalization = {"mean": [0.0], "std": [1.0]}
        mean = normalization["mean"]
        std = normalization["std"]
        
        # If mean is single value but we need RGB (SimSiam often expects 3-channel)
        if len(mean) == 1:
            mean = [mean[0]] * 3
            std = [std[0]] * 3
            
        return transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def _apply_two_crops(
        self, x_batch: torch.Tensor, device: str, state: BenchmarkState
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._ssl_transforms is None:
            self._ssl_transforms = self._build_ssl_transforms(state)

        to_pil = transforms.ToPILImage()
        view1_list = []
        view2_list = []

        for x in x_batch.cpu():
            img = to_pil(x.clamp(0, 1))
            # Convert grayscale to RGB for SSL transforms if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            view1_list.append(self._ssl_transforms(img))
            view2_list.append(self._ssl_transforms(img))

        view1 = torch.stack(view1_list).to(device)
        view2 = torch.stack(view2_list).to(device)
        return view1, view2

    def _neg_cosine_similarity(
        self, p: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        p = F.normalize(p, dim=1)
        z = F.normalize(z.detach(), dim=1)
        return -(p * z).sum(dim=1).mean()

    def _soft_sup_contrastive_loss(
        self, p: torch.Tensor, z: torch.Tensor, targets: torch.Tensor, num_classes: int
    ) -> torch.Tensor:
        p = F.normalize(p, dim=1)
        z = F.normalize(z.detach(), dim=1)

        dot_product = -torch.mm(p, z.t())

        targets = torch.clamp(targets, min=1e-8)
        entropy = -(targets * targets.log()).sum(dim=1)
        norm_entropy = entropy / torch.log(torch.tensor(float(num_classes), device=targets.device))
        reversed_norm_entropy = 1.0 - norm_entropy

        mask_similar_class1 = torch.outer(reversed_norm_entropy, reversed_norm_entropy)
        targets_norm = F.normalize(targets, dim=1)
        mask_similar_class2 = torch.mm(targets_norm, targets_norm.t())
        mask = mask_similar_class1 * mask_similar_class2

        mask.fill_diagonal_(0.0)
        denom = mask.sum().clamp_min(1.0)
        return (dot_product * mask).sum() / denom

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

        # Index tracking
        state.attack_state["labeled_indices"] = []
        state.attack_state["unlabeled_indices"] = list(range(pool_size))

        # Query data storage
        state.attack_state["query_data_x"] = []
        state.attack_state["query_data_y"] = []

        # Per-class counts for rare-class prioritization
        state.attack_state["class_counts"] = {}  # class_id -> count

        # Cached victim outputs for contrastive weight calculation
        state.attack_state["victim_outputs"] = {}  # idx -> softmax vector

        # Substitute model
        state.attack_state["substitute"] = None

        # Round tracking
        state.attack_state["round"] = 0

        # Sampling mode: "entropy" or "rare_class"
        state.attack_state["sampling_mode"] = "entropy"

    def propose(self, k: int, state: BenchmarkState) -> QueryBatch:
        """Propose k queries using sampling strategy.

        Args:
            k: Number of queries to propose
            state: Current benchmark state

        Returns:
            QueryBatch with k queries
        """
        labeled_indices = state.attack_state["labeled_indices"]
        unlabeled_indices = state.attack_state["unlabeled_indices"]

        # Handle empty pool
        if len(unlabeled_indices) == 0:
            input_shape = state.metadata.get("input_shape", (3, 32, 32))
            x = torch.randn(k, *input_shape)
            meta = {"indices": [], "pool_exhausted": True}
            return QueryBatch(x=x, meta=meta)

        # Calculate round-based budget
        total_budget = int(state.metadata.get("max_budget") or self.config.get("max_budget", 10000))
        initial_seed_size = int(self.initial_seed_ratio * total_budget)

        # Initial seed phase: random selection
        if len(labeled_indices) < initial_seed_size:
            n_seed = min(k, initial_seed_size - len(labeled_indices))
            n_seed = min(n_seed, len(unlabeled_indices))  # Clamp to available pool
            selected = (
                np.random.choice(unlabeled_indices, n_seed, replace=False).tolist() if n_seed > 0 else []
            )

            # If seed budget remaining is smaller than k, fill the rest randomly from D_U.
            if len(selected) < k:
                remaining = [idx for idx in unlabeled_indices if idx not in selected]
                n_extra = min(k - len(selected), len(remaining))
                if n_extra > 0:
                    selected.extend(np.random.choice(remaining, n_extra, replace=False).tolist())
        else:
            # Active sampling phase
            selected = self._select_samples(k, state)

        # Update state
        for idx in selected:
            if idx in state.attack_state["unlabeled_indices"]:
                state.attack_state["unlabeled_indices"].remove(idx)
                state.attack_state["labeled_indices"].append(idx)

        # Load actual images from pool dataset
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

        x_list: list[torch.Tensor] = []
        indices: list[int] = []
        for idx in selected:
            img, _ = self.pool_dataset[idx]
            x_list.append(img)
            indices.append(int(idx))

        if len(x_list) < k:
            input_shape = state.metadata.get("input_shape", (3, 32, 32))
            for _ in range(k - len(x_list)):
                x_list.append(torch.randn(*input_shape))
                indices.append(-1)

        x = torch.stack(x_list[:k])
        meta = {"indices": indices[:k], "sampling_mode": state.attack_state["sampling_mode"]}

        return QueryBatch(x=x, meta=meta)

    def _select_samples(self, k: int, state: BenchmarkState) -> List[int]:
        """Select k samples based on current sampling mode.

        Sampling strategies:
        - Entropy: Select samples with highest uncertainty
        - Rare class: Prioritize underrepresented classes

        Args:
            k: Number of samples to select
            state: Current benchmark state

        Returns:
            List of selected indices
        """
        # Check if we should switch to rare-class mode
        self._update_sampling_mode(state)

        unlabeled_indices = state.attack_state["unlabeled_indices"]
        substitute = state.attack_state["substitute"]

        if substitute is None:
            return np.random.choice(unlabeled_indices, k, replace=False).tolist()

        # Load pool dataset if needed
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

        device = next(substitute.parameters()).device
        sampling_mode = state.attack_state["sampling_mode"]

        if sampling_mode == "rare_class":
            return self._select_rare_class(k, state)
        else:  # entropy mode
            return self._select_entropy(k, state)

    def _update_sampling_mode(self, state: BenchmarkState) -> None:
        """Update sampling mode based on rare-class condition.

        Switch to rare-class mode when:
        B - |Q| <= N_R * (μ - μ_R)

        Where:
        - B: Total budget
        - |Q|: Number of queries made
        - N_R: Number of rare classes
        - μ: Mean samples per class
        - μ_R: Mean samples in rare classes

        Args:
            state: Current benchmark state
        """
        labeled_indices = state.attack_state["labeled_indices"]
        class_counts = state.attack_state["class_counts"]

        num_classes = int(
            state.metadata.get("num_classes")
            or self.config.get("num_classes")
            or 10
        )
        total_budget = int(
            state.metadata.get("max_budget")
            or self.config.get("max_budget", 10000)
        )

        if len(labeled_indices) == 0:
            return

        # Calculate statistics
        total_queries = len(labeled_indices)
        mean_per_class = total_queries / num_classes if num_classes > 0 else 0

        # Identify rare classes (samples below mean); include zero-count classes.
        rare_classes = [c for c in range(num_classes) if class_counts.get(c, 0) < mean_per_class]
        num_rare = len(rare_classes)

        if num_rare == 0:
            # No rare classes, stay in entropy mode
            state.attack_state["sampling_mode"] = "entropy"
            return

        # Calculate mean samples in rare classes
        if num_rare > 0:
            rare_sum = sum(class_counts.get(c, 0) for c in rare_classes)
            mean_rare = rare_sum / num_rare
        else:
            mean_rare = 0

        # Check switch condition
        remaining_budget = total_budget - total_queries
        threshold = num_rare * (mean_per_class - mean_rare)

        if remaining_budget <= threshold:
            state.attack_state["sampling_mode"] = "rare_class"
        else:
            state.attack_state["sampling_mode"] = "entropy"

    def _select_entropy(self, k: int, state: BenchmarkState) -> List[int]:
        """Select k samples with highest entropy.

        Args:
            k: Number of samples to select
            state: Current benchmark state

        Returns:
            List of selected indices
        """
        unlabeled_indices = state.attack_state["unlabeled_indices"]
        substitute = state.attack_state["substitute"]

        substitute.eval()
        device = next(substitute.parameters()).device
        entropy_scores = []

        with torch.no_grad():
            for idx in unlabeled_indices:
                img, _ = self.pool_dataset[idx]
                x = img.unsqueeze(0).to(device)
                probs = F.softmax(substitute(x), dim=1).squeeze(0)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                entropy_scores.append((idx, entropy.item()))

        # Sort by entropy (highest first)
        entropy_scores.sort(key=lambda x: x[1], reverse=True)
        selected = [idx for idx, _ in entropy_scores[:k]]

        return selected

    def _extract_features_for_indices(
        self,
        indices: List[int],
        substitute: nn.Module,
        device: torch.device,
        batch_size: int = 64,
    ) -> torch.Tensor:
        """Extract representation features for a list of pool indices."""
        if not hasattr(substitute, "features"):
            raise AttributeError("substitute must expose a .features(x) method for SwiftThief")

        feats: list[torch.Tensor] = []
        substitute.eval()
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                chunk = indices[start : start + batch_size]
                x = torch.stack([self.pool_dataset[idx][0] for idx in chunk]).to(device)
                f = substitute.features(x)
                feats.append(f.detach().cpu())
        return torch.cat(feats, dim=0) if feats else torch.empty(0)

    def _select_rare_class(self, k: int, state: BenchmarkState) -> List[int]:
        """Select k samples prioritizing rare classes.

        Args:
            k: Number of samples to select
            state: Current benchmark state

        Returns:
            List of selected indices
        """
        unlabeled_indices = state.attack_state["unlabeled_indices"]
        labeled_indices = state.attack_state["labeled_indices"]
        class_counts = state.attack_state["class_counts"]
        victim_outputs = state.attack_state.get("victim_outputs", {})

        substitute = state.attack_state.get("substitute")
        if substitute is None or not hasattr(substitute, "features"):
            return self._select_entropy(k, state)

        num_classes = int(
            state.metadata.get("num_classes")
            or self.config.get("num_classes")
            or 10
        )
        if num_classes <= 0:
            return self._select_entropy(k, state)

        total_q = sum(class_counts.values())
        mean_per_class = total_q / num_classes
        rare_classes = [c for c in range(num_classes) if class_counts.get(c, 0) < mean_per_class]
        if not rare_classes:
            return self._select_entropy(k, state)

        # Paper: pick the most rarely-queried class y_n and use KDE in representation space.
        y_n = min(rare_classes, key=lambda c: class_counts.get(c, 0))
        q_y = [
            idx
            for idx in labeled_indices
            if idx in victim_outputs and int(victim_outputs[idx].argmax().item()) == y_n
        ]
        if len(q_y) == 0:
            return self._select_entropy(k, state)

        candidate_count = min(len(unlabeled_indices), int(self.max_pool_eval))
        candidates = (
            np.random.choice(unlabeled_indices, candidate_count, replace=False).tolist()
            if candidate_count > 0
            else []
        )
        if len(candidates) == 0:
            return []

        device = next(substitute.parameters()).device
        rare_feats = self._extract_features_for_indices(q_y, substitute, device)
        pool_feats = self._extract_features_for_indices(candidates, substitute, device)
        if rare_feats.numel() == 0 or pool_feats.numel() == 0:
            return self._select_entropy(k, state)

        # KDE score: s_j = sum_{x_i in Q_y} exp(-||f(x_j)-f(x_i)||^2 / (2*sigma^2))
        dists_sq = torch.cdist(pool_feats, rare_feats).pow(2)
        kernel = torch.exp(-dists_sq / (2.0 * (self.kde_sigma ** 2)))
        scores = kernel.sum(dim=1)

        topk = min(int(k), scores.numel())
        _, top_idx = torch.topk(scores, k=topk, largest=True)
        selected = [candidates[i] for i in top_idx.tolist()]

        if len(selected) < int(k):
            extra = int(k) - len(selected)
            remaining = [idx for idx in unlabeled_indices if idx not in selected]
            if len(remaining) > 0:
                n_extra = min(extra, len(remaining))
                selected.extend(np.random.choice(remaining, n_extra, replace=False).tolist())

        return selected

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

        # Store query data
        state.attack_state["query_data_x"].append(x_batch.cpu())
        state.attack_state["query_data_y"].append(y_batch.cpu())

        # Store victim outputs for contrastive weights
        if oracle_output.kind == "soft_prob":
            for i, idx in enumerate(query_batch.meta.get("indices", [])):
                if int(idx) < 0:
                    continue
                state.attack_state["victim_outputs"][idx] = y_batch[i].cpu()
        else:
            # Hard labels: convert to one-hot for victim_outputs
            num_classes = int(
                state.metadata.get("num_classes")
                or state.metadata.get("dataset_config", {}).get("num_classes", 10)
            )
            for i, idx in enumerate(query_batch.meta.get("indices", [])):
                if int(idx) < 0:
                    continue
                label = y_batch[i].item() if y_batch[i].ndim == 0 else y_batch[i].argmax().item()
                one_hot = torch.zeros(num_classes)
                one_hot[label] = 1.0
                state.attack_state["victim_outputs"][idx] = one_hot

        # Update class counts
        if oracle_output.kind == "soft_prob":
            labels = [y.argmax().item() for y in y_batch]
        else:
            labels = [y.item() if y.ndim == 0 else y.argmax().item() for y in y_batch]

        for label in labels:
            state.attack_state["class_counts"][label] = state.attack_state["class_counts"].get(label, 0) + 1

        # Train substitute periodically
        labeled_count = len(state.attack_state["labeled_indices"])
        total_budget = int(state.metadata.get("max_budget") or self.config.get("max_budget", 10000))
        round_size = max(1, total_budget // max(self.I, 1))
        if labeled_count % round_size == 0 and labeled_count > 0:
            self.train_substitute(state)

    def train_substitute(self, state: BenchmarkState) -> None:
        """Train substitute model with contrastive losses.

        Args:
            state: Current benchmark state
        """
        query_data_x = state.attack_state["query_data_x"]
        query_data_y = state.attack_state["query_data_y"]

        if len(query_data_x) == 0:
            return

        # Concatenate query data
        x_all = torch.cat(query_data_x, dim=0)
        y_all = torch.cat(query_data_y, dim=0)

        # Create dataset
        class QueryDataset(torch.utils.data.Dataset):
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def __len__(self):
                return len(self.x)

            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]

        dataset = QueryDataset(x_all, y_all)

        # Use 20% validation split
        total_size = len(dataset)
        val_size = max(1, int(0.2 * total_size))
        train_size = total_size - val_size

        if train_size < 2:
            device = state.metadata.get("device", "cpu")
            num_classes = int(
                state.metadata.get("num_classes")
                or state.metadata.get("dataset_config", {}).get("num_classes", 10)
            )
            model = create_substitute(
                arch="resnet18",
                num_classes=num_classes,
                input_channels=state.metadata.get("input_shape", (3, 32, 32))[0],
            ).to(device)
            state.attack_state["substitute"] = model
            return

        train_subset, val_subset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        # Create dataloader for unqueried pool U for L_self computation (critical fix)
        unlabeled_indices = state.attack_state["unlabeled_indices"]
        if len(unlabeled_indices) > 0:
            class PoolDataset(torch.utils.data.Dataset):
                def __init__(self, indices, pool_dataset):
                    self.indices = indices
                    self.pool_dataset = pool_dataset
                
                def __len__(self):
                    return len(self.indices)
                
                def __getitem__(self, idx):
                    pool_idx = self.indices[idx]
                    img, _ = self.pool_dataset[pool_idx]
                    return img
            
            pool_dataset = PoolDataset(unlabeled_indices, self.pool_dataset)
            # Use smaller batch size for pool to manage memory
            pool_batch_size = min(self.batch_size, 32)
            pool_loader = torch.utils.data.DataLoader(
                pool_dataset, batch_size=pool_batch_size, shuffle=True, drop_last=True
            )
        else:
            pool_loader = None

        device = state.metadata.get("device", "cpu")
        num_classes = int(
            state.metadata.get("num_classes")
            or state.metadata.get("dataset_config", {}).get("num_classes", 10)
        )

        def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
            for p in module.parameters():
                p.requires_grad = enabled

        # Initialize or reuse substitute (paper: warm-start across rounds)
        substitute = state.attack_state.get("substitute")
        if isinstance(substitute, SwiftThiefSubstitute):
            substitute = substitute.to(device)
        else:
            base = create_substitute(
                arch="resnet18",
                num_classes=num_classes,
                input_channels=state.metadata.get("input_shape", (3, 32, 32))[0],
            ).to(device)
            if not hasattr(base, "fc"):
                raise ValueError("SwiftThief expects a ResNet-style substitute with .fc")
            classifier = base.fc
            feature_dim = classifier.in_features
            base.fc = nn.Identity()
            substitute = SwiftThiefSubstitute(base, classifier).to(device)

        # Projection/predictor heads (w_h, w_p)
        feature_dim = substitute.classifier.in_features
        if self.projection_head is None:
            self.projection_head = SimSiamProjectionHead(feature_dim, self.projection_dim).to(device)
        if self.predictor_head is None:
            self.predictor_head = SimSiamPredictorHead(self.projection_dim).to(device)

        sub_config = state.metadata.get("substitute_config", {})
        opt_params = sub_config.get("optimizer", {})
        
        import torch.optim as optim
        optimizer = optim.AdamW(
            list(substitute.parameters())
            + list(self.projection_head.parameters())
            + list(self.predictor_head.parameters()),
            lr=float(opt_params.get("lr", self.lr)),
            weight_decay=float(opt_params.get("weight_decay", self.weight_decay)),
        )

        # Early stopping
        best_f1 = 0.0
        patience_counter = 0
        best_state = None

        for epoch in range(self.max_epochs):
            substitute.train()
            self.projection_head.train()
            self.predictor_head.train()
            train_loss = 0.0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                victim_config = state.metadata.get("victim_config", {})
                normalization = victim_config.get("normalization")
                if normalization is None:
                    normalization = {"mean": [0.0], "std": [1.0]}
                norm_mean = torch.tensor(normalization["mean"]).view(1, -1, 1, 1).to(device)
                norm_std = torch.tensor(normalization["std"]).view(1, -1, 1, 1).to(device)
                
                def _norm(img):
                    return (img - norm_mean) / norm_std

                # Class-balanced weights (used for matching + regularizer)
                y_labels = y_batch.argmax(dim=1) if y_batch.ndim > 1 else y_batch
                logits = substitute(_norm(x_batch))
                ce_losses = F.cross_entropy(logits, y_labels.long(), reduction="none")

                class_counts = state.attack_state.get("class_counts", {})
                max_count = max(class_counts.values()) if class_counts else 1
                weights = torch.ones_like(ce_losses)
                for i, label in enumerate(y_labels.tolist()):
                    count = class_counts.get(label, 1)
                    weights[i] = float(max_count) / float(max(count, 1))

                # Use KL divergence for soft probabilities, CE for hard labels (paper parity)
                if y_batch.ndim > 1:  # soft probabilities
                    # KL divergence: p * log(p/q) where p=victim, q=substitute
                    victim_probs = y_batch
                    substitute_probs = F.softmax(logits, dim=1)
                    kl_losses = F.kl_div(substitute_probs.log(), victim_probs, reduction='none')
                    loss_match = (kl_losses * weights).mean()
                else:  # hard labels
                    loss_match = (ce_losses * weights).mean()

                # Build perturbation for sharpness regularizer (FGSM)
                x_adv = x_batch.detach().requires_grad_(True)
                logits_adv = substitute(x_adv)
                ce_adv = F.cross_entropy(logits_adv, y_labels.long(), reduction="none")
                loss_adv = (ce_adv * weights).mean()
                grad = torch.autograd.grad(loss_adv, x_adv, retain_graph=False, create_graph=False)[0]
                x_pert = (x_adv + self.fgsm_epsilon * torch.sign(grad.detach())).detach()

# L_self: Contrastive learning on UNQUERIED POOL U (critical fix for paper parity)
                if pool_loader is not None:
                    # Sample from unqueried pool for L_self computation
                    try:
                        x_pool_batch = next(iter(pool_loader))
                    except StopIteration:
                        # Recreate pool loader if exhausted
                        pool_loader = torch.utils.data.DataLoader(
                            pool_dataset, batch_size=pool_batch_size, shuffle=True, drop_last=True
                        )
                        x_pool_batch = next(iter(pool_loader))
                    
                    x_pool_batch = x_pool_batch.to(device)
                    
                    # Two views from pool data for self-supervised learning
                    x_pool_view1, x_pool_view2 = self._apply_two_crops(x_pool_batch, device, state)
                    f1_pool = substitute.features(x_pool_view1)
                    f2_pool = substitute.features(x_pool_view2)
                    z1_pool = self.projection_head(f1_pool)
                    z2_pool = self.projection_head(f2_pool)
                    p1_pool = self.predictor_head(z1_pool)
                    p2_pool = self.predictor_head(z2_pool)
                    
                    loss_self = 0.5 * (
                        self._neg_cosine_similarity(p1_pool, z2_pool) + self._neg_cosine_similarity(p2_pool, z1_pool)
                    )
                else:
                    # Fallback: use queried data (not paper-faithful but prevents crash)
                    x_view1, x_view2 = self._apply_two_crops(x_batch.detach(), device, state)
                    f1 = substitute.features(x_view1)
                    f2 = substitute.features(x_view2)
                    z1 = self.projection_head(f1)
                    z2 = self.projection_head(f2)
                    p1 = self.predictor_head(z1)
                    p2 = self.predictor_head(z2)
                    
                    loss_self = 0.5 * (
                        self._neg_cosine_similarity(p1, z2) + self._neg_cosine_similarity(p2, z1)
                    )

                loss_soft = torch.tensor(0.0, device=device)
                if y_batch.ndim > 1:
                    targets = torch.cat([y_batch, y_batch], dim=0)
                    p_cat = torch.cat([p1, p2], dim=0)
                    z_cat = torch.cat([z1, z2], dim=0)
                    loss_soft = self._soft_sup_contrastive_loss(p_cat, z_cat, targets, num_classes)

                # Sharpness regularizer in projection space
                f_clean = substitute.features(x_batch)
                f_pert = substitute.features(x_pert)
                z_clean = F.normalize(self.projection_head(f_clean), dim=1)
                z_pert = F.normalize(self.projection_head(f_pert), dim=1)
                reg_per = (z_clean - z_pert).pow(2).mean(dim=1)
                loss_reg = (reg_per * weights).mean()

                # Alternating optimization (paper): first optimize contrastive objective L_c,
                # then optimize matching objective L_m, warm-starting shared w_r.

                # Step 1: contrastive update (freeze classifier)
                _set_requires_grad(substitute.classifier, False)
                _set_requires_grad(self.projection_head, True)
                _set_requires_grad(self.predictor_head, True)
                optimizer.zero_grad()
                loss_c = loss_self + self.lambda1 * loss_soft + self.lambda2 * loss_reg
                loss_c.backward()
                optimizer.step()

                # Step 2: matching update (freeze SSL heads)
                _set_requires_grad(substitute.classifier, True)
                _set_requires_grad(self.projection_head, False)
                _set_requires_grad(self.predictor_head, False)
                optimizer.zero_grad()
                logits2 = substitute(x_batch.detach())
                ce2 = F.cross_entropy(logits2, y_labels.long(), reduction="none")
                loss_m = (ce2 * weights).mean() * self.lambda3
                loss_m.backward()
                optimizer.step()

                # Restore grads for next batch
                _set_requires_grad(self.projection_head, True)
                _set_requires_grad(self.predictor_head, True)

                train_loss += float((loss_c.detach() + loss_m.detach()).item())

            val_f1 = self._compute_f1(substitute, val_loader, device)
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                best_state = {
                    "substitute": {k: v.cpu().clone() for k, v in substitute.state_dict().items()},
                    "projection": {k: v.cpu().clone() for k, v in self.projection_head.state_dict().items()},
                    "predictor": {k: v.cpu().clone() for k, v in self.predictor_head.state_dict().items()},
                }
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                print(
                    f"SwiftThief Epoch {epoch}: Loss: {train_loss/len(train_loader):.4f}, Val F1: {val_f1:.4f}"
                )

            if patience_counter >= self.patience:
                print(f"SwiftThief Early stopping at epoch {epoch}")
                break

        if best_state is not None:
            substitute.load_state_dict(best_state["substitute"])
            self.projection_head.load_state_dict(best_state["projection"])
            self.predictor_head.load_state_dict(best_state["predictor"])

        state.attack_state["substitute"] = substitute
        print(f"SwiftThief substitute trained. Best F1: {best_f1:.4f}")

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
            for x_batch, y_batch in val_loader:
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
