"""SwiftThief attack implementation (mebench plugin, LeNet/MNIST compatible, repo-style CL + KD).

This file:
- Supports substitute architectures: resnet18 / resnet18-8x / lenet / lenet_mnist (via create_substitute)
- Supports datasets with input_shape from state.metadata["input_shape"] (e.g., MNIST 28x28x1, CIFAR 32x32x3)
- Uses repo-aligned Contrastive Learning (CL) stage:
    loss = loss1 (SimSiam on U) + loss2 (SoftSupSimSiamLossV17 on Q) + lambda2 * loss3 (CL_FGSM + cost-sensitive)
- Adds KD/SL stage after CL (hardcoded):
    kd_epochs = 10, kd_lr = 1e-2
  matching oracle outputs on Q via KL (soft_prob) or CE (hard_top1)
- Fixes common pitfalls:
  * unlabeled_indices uses real pool length (no guessed pool_size)
  * LeNet classifier is nn.Sequential -> feat_dim inferred via dummy forward
  * drop_last=False and batch_size clamped to avoid 0-step training in small seeds
  * normalization always applied consistently for substitute/classifier evaluation and sampling
"""

from typing import Dict, Any, List, Tuple, Optional
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import torchvision.transforms as transforms
from tqdm import tqdm

from mebench.attackers.runner import AttackRunner
from mebench.core.context import BenchmarkContext
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.data.loaders import create_dataloader
from mebench.models.substitute_factory import create_substitute


class QueryDataset(torch.utils.data.Dataset):
    """Simple dataset for query data that can be pickled."""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class PoolUDataset(torch.utils.data.Dataset):
    def __init__(self, indices, pool):
        self.indices = indices
        self.pool = pool

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        pool_idx = self.indices[idx]
        img, _ = self.pool[pool_idx]
        return img


# ============================================================
# Repo: normalize.py
# ============================================================

class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float())

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = self.mean[None, :, None, None]
        std = self.std[None, :, None, None]
        return tensor.sub(mean).div(std)


class NormalizeByChannelMeanStdSimSiam(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float())

    def forward(self, im_aug1: torch.Tensor, im_aug2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean[None, :, None, None]
        std = self.std[None, :, None, None]
        return im_aug1.sub(mean).div(std), im_aug2.sub(mean).div(std)


# ============================================================
# Repo: criterion.py
# ============================================================

class SimSiamLoss(nn.Module):
    def __init__(self, version='simplified'):
        super().__init__()
        self.ver = version

    def asymmetric_loss(self, p, z):
        if self.ver == 'original':
            z = z.detach()
            p = F.normalize(p, dim=1)
            z = F.normalize(z, dim=1)
            return -(p * z).sum(dim=1).mean()
        elif self.ver == 'simplified':
            z = z.detach()
            return -F.cosine_similarity(p, z, dim=-1).mean()

    def forward(self, z1, z2, p1, p2):
        loss1 = self.asymmetric_loss(p1, z2)
        loss2 = self.asymmetric_loss(p2, z1)
        return 0.5 * loss1 + 0.5 * loss2


class SoftSupSimSiamLossV17(nn.Module):
    def __init__(self, device, num_classes):
        super().__init__()
        self.device = device
        self.num_classes = num_classes

    def forward(self, p, z, targets):
        """Paper-faithful soft-supervised contrastive loss (SwiftThief Eq.2-3).

        Given view-level tensors:
        - z: projection outputs for 2q views
        - p: predictor outputs for 2q views
        - targets: victim soft labels for 2q views (each query sample duplicated)

        Eq.(2): - sum_{i} sum_{j} eta_ij ( z_i^T z'_j + z_j^T z'_i )
        where z'_j corresponds to predictor output p_j.
        Eq.(3): eta_ij = 1[i!=j] (1 + H(y_i)/logK)(1 + H(y_j)/logK) cos(y_i, y_j)
        """
        if p.numel() == 0 or z.numel() == 0:
            return torch.zeros((), device=self.device)

        # Normalize and stop-gradient on z as in SimSiam.
        z = z.detach()
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)

        # Ensure targets are valid probability vectors.
        targets = targets.to(self.device).float()
        targets = targets.clamp_min(1e-12)
        targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-12)

        # Entropy term: H(y) / logK
        entr = -(targets * targets.log()).sum(dim=1)
        log_k = torch.log(torch.tensor(float(self.num_classes), device=self.device))
        norm_entr = entr / log_k
        a = 1.0 + norm_entr

        # cos angle between soft labels
        targets_norm = F.normalize(targets, dim=1)
        cos_y = targets_norm @ targets_norm.T

        # eta_ij with zero diagonal
        eta = (a.unsqueeze(1) * a.unsqueeze(0)) * cos_y
        eta.fill_diagonal_(0.0)

        # Similarity matrix for sym term (z_i^T p_j + z_j^T p_i)
        sim = z @ p.T
        sym = sim + sim.T

        # Use mean over off-diagonal pairs to keep scale stable across batch sizes.
        n = int(sym.shape[0])
        denom = float(max(1, n * (n - 1)))
        return -((eta * sym).sum() / denom)


class CL_FGSM(nn.Module):
    def __init__(self, model, eps, device):
        super().__init__()
        self.device = device
        self.model = model
        self.eps = eps

    def asymmetric_loss(self, p, z):
        z = z.detach()
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p * z).sum(dim=1).mean()

    def forward(self, x1, x2):
        self.model.eval()
        x1.requires_grad = True

        outs = self.model(im_aug1=x1, im_aug2=x2)
        loss1 = self.asymmetric_loss(outs['p1'], outs['z2'])
        loss2 = self.asymmetric_loss(outs['p2'], outs['z1'])
        loss = 0.5 * loss1 + 0.5 * loss2

        loss.backward()
        adv_x1 = x1 + self.eps * x1.grad.sign()
        return adv_x1.detach()


class SimSiamLoss_cost_sensitive(nn.Module):
    def __init__(self, costs: torch.Tensor):
        super().__init__()
        self.costs = costs

    def asymmetric_loss(self, p, z, targets):
        z = z.detach()
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -((p * z).sum(dim=1) * self.costs[targets]).mean()

    def forward(self, z1, z2, p1, p2, targets):
        loss1 = self.asymmetric_loss(p1, z2, targets)
        loss2 = self.asymmetric_loss(p2, z1, targets)
        return 0.5 * loss1 + 0.5 * loss2


# ============================================================
# Heads (keep user's format)
# ============================================================

class SimSiamProjectionHead(nn.Module):
    """3-layer MLP projector."""
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
        return self.net(x)


class SimSiamPredictorHead(nn.Module):
    """Predictor head."""
    def __init__(self, in_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Substitute wrapper
# ============================================================

class SwiftThiefSubstitute(nn.Module):
    """Backbone returns features (or exposes .features module); classifier consumes flattened features."""
    def __init__(self, backbone: nn.Module, classifier: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def features(self, x: torch.Tensor) -> torch.Tensor:
        feat_attr = getattr(self.backbone, "features", None)
        if feat_attr is not None:
            if callable(feat_attr):
                return feat_attr(x)
            if isinstance(feat_attr, nn.Module):
                return feat_attr(x)
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        if feats.ndim > 2:
            feats = feats.view(feats.size(0), -1)
        return self.classifier(feats)


# ============================================================
# SimSiamWrapper (for CL_FGSM)
# ============================================================

class _SimSiamWrapper(nn.Module):
    """Returns {'z1','z2','p1','p2'} for two-view input. Flattens features before projector."""
    def __init__(self, substitute: SwiftThiefSubstitute, projector: nn.Module, predictor: nn.Module):
        super().__init__()
        self.substitute = substitute
        self.projector = projector
        self.predictor = predictor

    @staticmethod
    def _feat2vec(feats: torch.Tensor) -> torch.Tensor:
        if feats.ndim > 2:
            feats = feats.view(feats.size(0), -1)
        return feats

    def forward(self, im_aug1, im_aug2=None):
        if im_aug2 is None:
            f1 = self._feat2vec(self.substitute.features(im_aug1))
            z1 = self.projector(f1)
            p1 = self.predictor(z1)
            return p1

        f1 = self._feat2vec(self.substitute.features(im_aug1))
        f2 = self._feat2vec(self.substitute.features(im_aug2))
        z1 = self.projector(f1)
        z2 = self.projector(f2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return {'z1': z1, 'z2': z2, 'p1': p1, 'p2': p2}


class TwoCropTransform:
    """Apply transform twice to create two views."""
    def __init__(self, transform, to_pil_first=True, input_channels=3):
        self.transform = transform
        self.to_pil_first = to_pil_first
        self.to_pil = transforms.ToPILImage() if self.to_pil_first else None
        self.input_channels = input_channels

    def __call__(self, x):
        # x is Tensor [C, H, W]
        if self.to_pil_first:
            img = self.to_pil(x.clamp(0, 1))
            # Handle channel mismatch if needed (e.g. 1-channel PIL)
            if self.input_channels == 1 and img.mode != "L":
                img = img.convert("L")
            elif self.input_channels == 3 and img.mode != "RGB":
                img = img.convert("RGB")
        else:
            img = x
            if img.dtype == torch.uint8:
                img = img.float().div(255.0)
            else:
                img = img.float()
            img = img.clamp(0, 1)
        
        v1 = self.transform(img)
        v2 = self.transform(img)
        return v1, v2

class SimSiamDataset(torch.utils.data.Dataset):
    """Dataset wrapper that applies TwoCropTransform."""
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Base dataset returns (img, label) or just img depending on implementation
        # But here we pass specific datasets.
        # Check what base_dataset returns.
        item = self.base_dataset[idx]
        if isinstance(item, tuple):
            img = item[0]
            target = item[1]
            v1, v2 = self.transform(img)
            return v1, v2, target
        else:
            img = item
            v1, v2 = self.transform(img)
            return v1, v2

# ============================================================
# SwiftThief Attack
# ============================================================

class SwiftThief(AttackRunner):
    def __init__(self, config: dict, state: BenchmarkState):
        super().__init__(config, state)

        # Core hyperparameters
        self.I = int(config.get("I", 10))
        self.initial_seed_ratio = float(config.get("initial_seed_ratio", 0.1))
        # SwiftThief paper (IJCAI'24): lambda1=1.0, lambda2=0.01
        self.lambda1 = float(config.get("lambda1", 1.0))
        self.lambda2 = float(config.get("lambda2", 0.01))  # weight for loss3
        self.fgsm_epsilon = float(config.get("fgsm_epsilon", 0.01))
        # Effective-number beta for cost-sensitive regularization (paper defines beta, but
        # does not pin a single value in the setup section).
        self.effective_beta = float(config.get("effective_beta", 0.99))
        self.projection_dim = int(config.get("projection_dim", 2048))

        # Sampling
        self.kde_sigma = float(config.get("sigma", 1.0))

        # Training (CL)
        self.batch_size = int(config.get("batch_size", 256))
        self.num_workers = int(config.get("num_workers", 4))
        self.prefetch_factor = int(config.get("prefetch_factor", 2))
        self.use_pil_transforms = bool(config.get("use_pil_transforms", False))
        self.lr = float(config.get("lr", 0.06))
        self.momentum = float(config.get("momentum", 0.9))
        self.weight_decay = float(config.get("weight_decay", 5e-4))
        self.cl_epochs = int(config.get("cl_epochs", 40))
        self.final_cl_epochs = int(config.get("final_cl_epochs", 100))
        self.patience = int(config.get("patience", 50))
        self.unlabeled_ssl_size = int(config.get("unlabeled_ssl_size", 50000))

        # KD defaults aligned to paper
        self.kd_epochs = int(config.get("kd_epochs", 40))
        self.kd_lr = float(config.get("kd_lr", self.lr))

        # internal
        self.pool_dataset = None
        self.projection_head: Optional[nn.Module] = None
        self.predictor_head: Optional[nn.Module] = None
        self.normalize: Optional[nn.Module] = None
        self.normalize_pair: Optional[nn.Module] = None
        self._ssl_transforms = None

        self._initialize_state(state)

    # -------------------------
    # State init
    # -------------------------

    def _initialize_state(self, state: BenchmarkState) -> None:
        st = state.attack_state
        st["labeled_indices"] = []
        st["unlabeled_indices"] = []  # filled after pool load
        st["query_data_x"] = []
        st["query_data_y"] = []
        st["val_query_data_x"] = []
        st["val_query_data_y"] = []
        st["class_counts"] = {}
        st["victim_outputs"] = {}
        st["substitute"] = None
        st["sampling_mode"] = "entropy"
        st["last_train_labeled_count"] = 0

    def run(self, ctx: BenchmarkContext) -> None:
        self.victim = ctx.oracle.model
        self._ensure_pool_dataset(self.state)

        total_budget = int(
            self.state.metadata.get("max_budget")
            or self.config.get("max_budget", ctx.budget_remaining)
        )
        round_size = max(1, int(math.ceil(total_budget / max(self.I, 1))))

        pbar = tqdm(total=total_budget, desc="[SwiftThief] Extracting")
        while ctx.budget_remaining > 0:
            step_size = min(round_size, ctx.budget_remaining)
            query_batch = self._select_query_batch(step_size, self.state)
            
            # [FIX] Handle pool exhaustion
            if query_batch.x.size(0) == 0:
                self.logger.warning("SwiftThief query selection returned empty batch. Stopping attack.")
                break
                
            oracle_output = ctx.query(query_batch.x, meta=query_batch.meta)
            self._handle_oracle_output(query_batch, oracle_output, self.state)
            pbar.update(query_batch.x.size(0))

        labeled_count = len(self.state.attack_state["labeled_indices"])
        last_train = int(self.state.attack_state.get("last_train_labeled_count", 0))
        if labeled_count > last_train:
            self.train_substitute(self.state, cl_epochs_override=self.final_cl_epochs)
            self.state.attack_state["last_train_labeled_count"] = labeled_count

        pbar.close()

    # -------------------------
    # Dataset + Normalizer
    # -------------------------

    def _ensure_pool_dataset(self, state: BenchmarkState) -> None:
        if self.pool_dataset is not None:
            return

        dataset_config = state.metadata.get("dataset_config", {})
        if "data_mode" not in dataset_config:
            dataset_config = {"data_mode": "seed", **dataset_config}
        if "name" not in dataset_config:
            dataset_config = {"name": "CIFAR10", **dataset_config}

        self.pool_dataset = create_dataloader(dataset_config, batch_size=1, shuffle=False).dataset

        N = len(self.pool_dataset)
        if not state.attack_state["unlabeled_indices"]:
            state.attack_state["unlabeled_indices"] = list(range(N))
        else:
            state.attack_state["unlabeled_indices"] = [i for i in state.attack_state["unlabeled_indices"] if 0 <= i < N]

    def _ensure_normalizers(self, state: BenchmarkState, device: torch.device) -> None:
        if self.normalize is not None and getattr(self.normalize, "mean", None) is not None:
            if self.normalize.mean.device == device:
                return

        victim_config = state.metadata.get("victim_config", {})
        normalization = victim_config.get("normalization") or {"mean": [0.0], "std": [1.0]}
        mean = list(normalization["mean"])
        std = list(normalization["std"])

        C = state.metadata.get("input_shape", (3, 32, 32))[0]
        if len(mean) != C:
            if len(mean) == 1 and C == 3:
                mean = [mean[0]] * 3
                std = [std[0]] * 3
            elif len(mean) == 3 and C == 1:
                mean = [mean[0]]
                std = [std[0]]

        mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
        std_t = torch.tensor(std, dtype=torch.float32, device=device)
        self.normalize = NormalizeByChannelMeanStd(mean_t, std_t).to(device)
        self.normalize_pair = NormalizeByChannelMeanStdSimSiam(mean_t, std_t).to(device)

    # -------------------------
    # SSL transforms (raw -> aug -> raw, then normalize_pair)
    # -------------------------

    def _build_ssl_transforms(self, state: BenchmarkState, use_pil: bool) -> transforms.Compose:
        C, H, W = state.metadata.get("input_shape", (3, 32, 32))

        if C == 1:
            cj = transforms.ColorJitter(0.4, 0.4, 0.0, 0.0)
        else:
            cj = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)

        ops = [
            transforms.RandomResizedCrop((H, W), scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([cj], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]
        if use_pil:
            ops.append(transforms.ToTensor())
        return transforms.Compose(ops)

    def _apply_two_crops(self, x_batch: torch.Tensor, device: torch.device, state: BenchmarkState) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._ssl_transforms is None:
            self._ssl_transforms = self._build_ssl_transforms(state, self.use_pil_transforms)

        C = state.metadata.get("input_shape", (3, 32, 32))[0]
        two_crop = TwoCropTransform(
            self._ssl_transforms,
            to_pil_first=self.use_pil_transforms,
            input_channels=C,
        )

        v1_list, v2_list = [], []
        for x in x_batch.detach().cpu():
            v1, v2 = two_crop(x)
            v1_list.append(v1)
            v2_list.append(v2)

        return torch.stack(v1_list).to(device), torch.stack(v2_list).to(device)

    # -------------------------
    # Propose + sampling
    # -------------------------

    def _select_query_batch(self, k: int, state: BenchmarkState) -> QueryBatch:
        self._ensure_pool_dataset(state)

        labeled = state.attack_state["labeled_indices"]
        unlabeled = state.attack_state["unlabeled_indices"]

        if len(unlabeled) == 0:
            raise ValueError(
                f"Query pool exhausted for {self.__class__.__name__}. "
                f"Cannot select {k} more queries."
            )
        if k > len(unlabeled):
            raise ValueError(
                f"Requested step_size={k} exceeds remaining pool size={len(unlabeled)} "
                f"for {self.__class__.__name__}."
            )

        total_budget = int(state.metadata.get("max_budget") or self.config.get("max_budget", 10000))
        initial_seed_size = int(self.initial_seed_ratio * total_budget)

        if len(labeled) < initial_seed_size:
            n_seed = min(k, initial_seed_size - len(labeled), len(unlabeled))
            selected = np.random.choice(unlabeled, n_seed, replace=False).tolist() if n_seed > 0 else []
            if len(selected) < k:
                remaining = [i for i in unlabeled if i not in selected]
                n_extra = min(k - len(selected), len(remaining))
                if n_extra > 0:
                    selected.extend(np.random.choice(remaining, n_extra, replace=False).tolist())
        else:
            selected = self._select_samples(k, state)

        for idx in selected:
            if idx in state.attack_state["unlabeled_indices"]:
                state.attack_state["unlabeled_indices"].remove(idx)
                state.attack_state["labeled_indices"].append(idx)

        x_list, indices = [], []
        for idx in selected:
            img, _ = self.pool_dataset[idx]
            x_list.append(img)
            indices.append(int(idx))

        x = torch.stack(x_list)
        # Slicing x_list[:k] is redundant if len < k, but safe.
        # However, indices was appended in loop.
        return QueryBatch(x=x, meta={"indices": indices, "sampling_mode": state.attack_state["sampling_mode"]})

    def _update_sampling_mode(self, state: BenchmarkState) -> None:
        labeled = state.attack_state["labeled_indices"]
        if len(labeled) == 0:
            return

        class_counts = state.attack_state["class_counts"]
        num_classes = int(state.metadata.get("num_classes") or self.config.get("num_classes") or 10)
        total_budget = int(state.metadata.get("max_budget") or self.config.get("max_budget", 10000))

        total_q = len(labeled)
        mean_per_class = total_q / num_classes

        rare_classes = [c for c in range(num_classes) if class_counts.get(c, 0) < mean_per_class]
        if not rare_classes:
            state.attack_state["sampling_mode"] = "entropy"
            return

        total_budget = int(state.metadata.get("max_budget", 10000))
        total_q = sum(class_counts.values())
        mu = total_q / num_classes if num_classes > 0 else 0.0
        mu_rare = (
            sum(class_counts.get(c, 0) for c in rare_classes) / len(rare_classes)
            if len(rare_classes) > 0
            else 0.0
        )
        remaining = total_budget - total_q
        threshold = len(rare_classes) * (mu - mu_rare)

        state.attack_state["sampling_mode"] = "rare_class" if remaining <= threshold else "entropy"

    def _select_samples(self, k: int, state: BenchmarkState) -> List[int]:
        self._update_sampling_mode(state)
        sub = state.attack_state["substitute"]
        if sub is None:
            unlabeled = state.attack_state["unlabeled_indices"]
            return np.random.choice(unlabeled, min(k, len(unlabeled)), replace=False).tolist()

        if state.attack_state["sampling_mode"] == "rare_class":
            return self._select_rare_class(k, state)
        return self._select_entropy(k, state)

    def _select_entropy(self, k: int, state: BenchmarkState) -> List[int]:
        unlabeled = state.attack_state["unlabeled_indices"]
        substitute = state.attack_state["substitute"]
        if substitute is None:
            return np.random.choice(unlabeled, min(k, len(unlabeled)), replace=False).tolist()

        substitute.eval()
        device = next(substitute.parameters()).device
        self._ensure_normalizers(state, device)

        # Paper (Algorithm 1): entropy-based sampling (pure Shannon entropy top-k).
        candidates = unlabeled
        if not candidates:
            return []

        scores: list[tuple[int, float]] = []
        bs = min(self.batch_size, len(candidates))
        with torch.no_grad():
            for start in tqdm(range(0, len(candidates), bs), desc="[SwiftThief] Scoring Pool", leave=False):
                chunk = candidates[start : start + bs]
                x_raw = torch.stack([self.pool_dataset[i][0] for i in chunk]).to(device)
                x = self.normalize(x_raw)
                probs = F.softmax(substitute(x), dim=1)
                entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
                for i, idx in enumerate(chunk):
                    scores.append((int(idx), float(entropy[i].item())))

        scores.sort(key=lambda t: t[1], reverse=True)
        return [idx for idx, _ in scores[: min(int(k), len(scores))]]

    def _extract_features_for_indices(
        self,
        indices: List[int],
        substitute: SwiftThiefSubstitute,
        device: torch.device,
        state: BenchmarkState,
        batch_size: int = 256
    ) -> torch.Tensor:
        self._ensure_normalizers(state, device)
        feats = []
        substitute.eval()
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                chunk = indices[start:start + batch_size]
                x_raw = torch.stack([self.pool_dataset[i][0] for i in chunk]).to(device)
                x = self.normalize(x_raw)
                f = substitute.features(x)
                if f.ndim > 2:
                    f = f.view(f.size(0), -1)
                feats.append(f.detach())
        return torch.cat(feats, dim=0) if feats else torch.empty(0, device=device)

    def _select_rare_class(self, k: int, state: BenchmarkState) -> List[int]:
        unlabeled = state.attack_state["unlabeled_indices"]
        labeled = state.attack_state["labeled_indices"]
        class_counts = state.attack_state["class_counts"]
        victim_outputs = state.attack_state.get("victim_outputs", {})

        substitute = state.attack_state.get("substitute")
        if substitute is None or not hasattr(substitute, "features"):
            return self._select_entropy(k, state)

        num_classes = int(state.metadata.get("num_classes") or self.config.get("num_classes") or 10)

        total_q = sum(class_counts.values())
        mean_per_class = total_q / num_classes if num_classes > 0 else 0
        rare_classes = [c for c in range(num_classes) if class_counts.get(c, 0) < mean_per_class]
        if not rare_classes:
            return self._select_entropy(k, state)

        y_n = min(rare_classes, key=lambda c: class_counts.get(c, 0))
        q_y = [idx for idx in labeled if idx in victim_outputs and int(victim_outputs[idx].argmax().item()) == y_n]
        if not q_y:
            return self._select_entropy(k, state)

        # Use the ENTIRE unlabeled pool (Strict Protocol)
        candidates = unlabeled
        if not candidates:
            return []

        device = next(substitute.parameters()).device
        rare_feats = self._extract_features_for_indices(
            q_y, substitute, device, state, batch_size=self.batch_size
        )
        pool_feats = self._extract_features_for_indices(
            candidates, substitute, device, state, batch_size=self.batch_size
        )
        if rare_feats.numel() == 0 or pool_feats.numel() == 0:
            return self._select_entropy(k, state)

        dists_sq = torch.cdist(pool_feats, rare_feats).pow(2)
        kernel = torch.exp(-dists_sq / (2.0 * (self.kde_sigma ** 2)))
        scores = kernel.sum(dim=1)

        topk = min(int(k), scores.numel())
        _, top_idx = torch.topk(scores, k=topk, largest=True)
        return [candidates[i] for i in top_idx.tolist()]

    # -------------------------
    # Observe
    # -------------------------

    def _handle_oracle_output(
        self, query_batch: QueryBatch, oracle_output: OracleOutput, state: BenchmarkState
    ) -> None:
        x = query_batch.x
        y = oracle_output.y

        state.attack_state["query_data_x"].append(x.detach().cpu())

        indices = query_batch.meta.get("indices", [])
        if oracle_output.kind == "soft_prob":
            y_for_training = y.detach().cpu()
            for i, idx in enumerate(indices):
                if int(idx) >= 0:
                    state.attack_state["victim_outputs"][int(idx)] = y[i].detach().cpu()
            labels = [int(t.argmax().item()) for t in y]
        else:
            num_classes = int(state.metadata.get("num_classes") or 10)
            y_for_training = F.one_hot(y.long(), num_classes=num_classes).float().detach().cpu()
            for i, idx in enumerate(indices):
                if int(idx) >= 0:
                    lab = int(y[i].item()) if y[i].ndim == 0 else int(y[i].argmax().item())
                    one_hot = torch.zeros(num_classes)
                    one_hot[lab] = 1.0
                    state.attack_state["victim_outputs"][int(idx)] = one_hot
            labels = [int(t.item()) if t.ndim == 0 else int(t.argmax().item()) for t in y]

        state.attack_state["query_data_y"].append(y_for_training)

        for lab in labels:
            state.attack_state["class_counts"][lab] = state.attack_state["class_counts"].get(lab, 0) + 1

        labeled_count = len(state.attack_state["labeled_indices"])
        total_budget = int(state.metadata.get("max_budget") or self.config.get("max_budget", 10000))
        round_size = max(1, int(math.ceil(total_budget / max(self.I, 1))))
        last_train = int(state.attack_state.get("last_train_labeled_count", 0))
        if labeled_count > 0 and (labeled_count - last_train) >= round_size:
            is_final_round = labeled_count >= total_budget
            cl_epochs = self.final_cl_epochs if is_final_round else self.cl_epochs
            self.train_substitute(state, cl_epochs_override=cl_epochs)
            state.attack_state["last_train_labeled_count"] = labeled_count

    # -------------------------
    # KD stage (hardcoded)
    # -------------------------

    def _train_kd_epoch(
        self,
        substitute: SwiftThiefSubstitute,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        state: BenchmarkState,
    ) -> float:
        substitute.train()
        self._ensure_normalizers(state, device)

        total_loss = 0.0
        n_batches = 0

        for x_raw, y in loader:
            x_raw = x_raw.to(device)
            x = self.normalize(x_raw)
            logits = substitute(x)

            if y.ndim > 1 and y.shape[1] > 1:
                y = y.to(device).float()
                y = y.clamp_min(1e-12)
                y = y / y.sum(dim=1, keepdim=True).clamp_min(1e-12)
                loss = F.kl_div(F.log_softmax(logits, dim=1), y, reduction="batchmean")
            else:
                y = y.to(device).long()
                loss = F.cross_entropy(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        return total_loss / max(1, n_batches)

    # -------------------------
    # Train (CL + KD)
    # -------------------------

    def train_substitute(self, state: BenchmarkState, cl_epochs_override: Optional[int] = None) -> None:
        self._ensure_fixed_validation_holdout(state)
        self._ensure_pool_dataset(state)

        qx = state.attack_state["query_data_x"]
        qy = state.attack_state["query_data_y"]
        vqx = state.attack_state.get("val_query_data_x", [])
        vqy = state.attack_state.get("val_query_data_y", [])
        if len(qx) == 0:
            return

        x_all = torch.cat(qx, dim=0)
        y_all = torch.cat(qy, dim=0)

        dataset_q = QueryDataset(x_all, y_all)

        if len(vqx) > 0 and len(vqy) > 0:
            x_val = torch.cat(vqx, dim=0)
            y_val = torch.cat(vqy, dim=0)
            train_q = dataset_q
            val_q = QueryDataset(x_val, y_val)
            train_size = len(train_q)
        else:
            total_size = len(dataset_q)
            val_size = max(1, int(0.2 * total_size))
            train_size = total_size - val_size
            if train_size < 2:
                return

            train_q, val_q = torch.utils.data.random_split(
                dataset_q, [train_size, val_size], generator=torch.Generator().manual_seed(42)
            )

        # [OPTIMIZATION] Pre-build transforms and wrap datasets for multi-worker loading
        ssl_transform = self._build_ssl_transforms(state, self.use_pil_transforms)
        input_channels = state.metadata.get("input_shape", (3, 32, 32))[0]
        two_crop = TwoCropTransform(
            ssl_transform,
            to_pil_first=self.use_pil_transforms,
            input_channels=input_channels,
        )

        # Wrap labeled training set
        train_q_ssl = SimSiamDataset(train_q, two_crop)

        bs_q = min(self.batch_size, len(train_q))
        if bs_q <= 0:
            return

        loader_kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": True,
            "drop_last": False,
        }
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
             
        labeled_loader = torch.utils.data.DataLoader(
            train_q_ssl,
            batch_size=bs_q,
            shuffle=True,
            **loader_kwargs,
        )
        
        # Validation loader (standard)
        val_loader = torch.utils.data.DataLoader(
            val_q,
            batch_size=min(self.batch_size, len(val_q)),
            shuffle=False,
            **loader_kwargs,
        )

        # U loader
        unlabeled_indices = state.attack_state["unlabeled_indices"]
        if len(unlabeled_indices) == 0:
            unlabeled_loader = None
        else:
            # SwiftThief paper uses a fixed-size random subset from U for SSL.
            if len(unlabeled_indices) > self.unlabeled_ssl_size:
                u_indices = np.random.choice(
                    unlabeled_indices, self.unlabeled_ssl_size, replace=False
                ).tolist()
            else:
                u_indices = unlabeled_indices
            if len(u_indices) == 0:
                unlabeled_loader = None
            else:
                dataset_u = PoolUDataset(u_indices, self.pool_dataset)
                # Wrap unlabeled set
                dataset_u_ssl = SimSiamDataset(dataset_u, two_crop)
                
                bs_u = min(self.batch_size, len(dataset_u))
                unlabeled_loader = torch.utils.data.DataLoader(
                    dataset_u_ssl,
                    batch_size=bs_u,
                    shuffle=True,
                    **loader_kwargs,
                )

        device = torch.device(state.metadata.get("device", "cpu"))
        num_classes = int(state.metadata.get("num_classes") or 10)

        # init / warm-start substitute
        substitute_config = state.metadata.get("substitute_config") or self.config.get("substitute") or {}
        arch = substitute_config.get("arch", "resnet18")
        width_mult = int(substitute_config.get("width_mult", 1))
        dropout_prob = float(substitute_config.get("dropout_prob", 0.0))

        substitute = state.attack_state.get("substitute")
        if not isinstance(substitute, SwiftThiefSubstitute):
            base = create_substitute(
                arch=arch,
                num_classes=num_classes,
                input_channels=state.metadata.get("input_shape", (3, 32, 32))[0],
                width_mult=width_mult,
                dropout_prob=dropout_prob,
            ).to(device)

            if hasattr(base, "fc"):
                classifier = base.fc
                base.fc = nn.Identity()
            elif hasattr(base, "classifier"):
                classifier = base.classifier
                base.classifier = nn.Identity()
            else:
                raise ValueError(f"SwiftThief: Could not find classifier head (.fc or .classifier) in {arch}")

            substitute = SwiftThiefSubstitute(base, classifier).to(device)
        else:
            substitute = substitute.to(device)

        # feat_dim via dummy forward (LeNet-safe)
        C, H, W = state.metadata.get("input_shape", (1, 28, 28))
        dummy = torch.zeros(2, C, H, W, device=device)
        with torch.no_grad():
            f = substitute.features(dummy)
            if f.ndim > 2:
                f = f.view(f.size(0), -1)
        feat_dim = int(f.shape[1])

        if self.projection_head is None:
            self.projection_head = SimSiamProjectionHead(feat_dim, self.projection_dim).to(device)
        if self.predictor_head is None:
            self.predictor_head = SimSiamPredictorHead(self.projection_dim).to(device)

        self._ensure_normalizers(state, device)

        # costs (effective number) from histogram (paper Eq.4)
        # c_y = (1 - beta) / (1 - beta^{N_y}), larger for rarer classes.
        cnt = torch.zeros(num_classes, device=device)
        for c in range(num_classes):
            cnt[c] = float(state.attack_state["class_counts"].get(c, 0))
        beta = float(self.effective_beta)
        beta_t = torch.tensor(beta, device=device)
        # Avoid division by zero when cnt==0 by clamping exponent.
        denom = 1.0 - torch.pow(beta_t, cnt.clamp_min(1.0))
        costs = (1.0 - beta_t) / denom.clamp_min(1e-12)

        criterion = SimSiamLoss('original').to(device)
        soft_criterion = SoftSupSimSiamLossV17(device, num_classes).to(device)
        cost_sensitive_criterion = SimSiamLoss_cost_sensitive(costs).to(device)

        fgsm_model = _SimSiamWrapper(substitute, self.projection_head, self.predictor_head).to(device)
        # FGSM is implemented inline in the CL loop (paper Eq.4).

        opt_config = substitute_config.get("optimizer", {})
        optimizer_cl = self._build_optimizer(
            list(substitute.parameters())
            + list(self.projection_head.parameters())
            + list(self.predictor_head.parameters()),
            opt_config,
        )

        best_f1 = 0.0
        patience_counter = 0
        best_state = None

        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader) if unlabeled_loader is not None else None

        # ---------------- CL stage ----------------
        cl_epochs = int(cl_epochs_override) if cl_epochs_override is not None else self.cl_epochs
        cl_pbar = tqdm(range(cl_epochs), desc="[SwiftThief] Training (CL)", leave=False)
        for epoch in cl_pbar:
            substitute.train()
            self.projection_head.train()
            self.predictor_head.train()

            steps = len(unlabeled_loader) if unlabeled_loader is not None else len(labeled_loader)
            if steps == 0:
                break

            epoch_loss = 0.0
            for _ in range(steps):
                # U batch
                if unlabeled_loader is not None:
                    try:
                        u1_raw, u2_raw = next(unlabeled_iter)
                    except StopIteration:
                        unlabeled_iter = iter(unlabeled_loader)
                        u1_raw, u2_raw = next(unlabeled_iter)

                    u1_raw = u1_raw.to(device, non_blocking=True)
                    u2_raw = u2_raw.to(device, non_blocking=True)
                    
                    # Normalization on GPU
                    u1, u2 = self.normalize_pair(u1_raw, u2_raw)
                    
                    outs_u = fgsm_model(im_aug1=u1, im_aug2=u2)
                    loss1 = criterion(outs_u['z1'], outs_u['z2'], outs_u['p1'], outs_u['p2'])
                else:
                    loss1 = torch.zeros((), device=device)

                # Q batch
                try:
                    x1_raw, x2_raw, y = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(labeled_loader)
                    x1_raw, x2_raw, y = next(labeled_iter)

                x1_raw = x1_raw.to(device, non_blocking=True)
                x2_raw = x2_raw.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                x1, x2 = self.normalize_pair(x1_raw, x2_raw)

                # Minority class regularization (paper Eq.4): FGSM inner-maximization in pixel space.
                # Determine sample class indices for cost-sensitive weights.
                if y.ndim > 1 and y.shape[1] > 1:
                    targets_probs = y.clamp_min(1e-8)
                    targets_probs = targets_probs / targets_probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
                    y_idx = targets_probs.argmax(dim=1)
                else:
                    targets_probs = None
                    y_idx = y.long()

                # Compute adversarial perturbation on the first view only (x1_raw).
                x1_raw_adv = x1_raw.detach().clone().requires_grad_(True)
                x1_adv_n, x2_n = self.normalize_pair(x1_raw_adv, x2_raw)
                outs_advgen = fgsm_model(im_aug1=x1_adv_n, im_aug2=x2_n)
                adv_loss = cost_sensitive_criterion(
                    outs_advgen["z1"], outs_advgen["z2"], outs_advgen["p1"], outs_advgen["p2"], y_idx
                )
                optimizer_cl.zero_grad(set_to_none=True)
                adv_loss.backward()
                with torch.no_grad():
                    grad = x1_raw_adv.grad
                    if grad is None:
                        x1_raw_adv = x1_raw
                    else:
                        x1_raw_adv = (x1_raw_adv + self.fgsm_epsilon * grad.sign()).clamp(0.0, 1.0)
                x1_raw_adv = x1_raw_adv.detach()

                outs_l = fgsm_model(im_aug1=x1, im_aug2=x2)

                if targets_probs is not None:
                    # SwiftThief Eq.(2)-(3)
                    loss2 = soft_criterion(
                        p=torch.cat([outs_l["p1"], outs_l["p2"]], dim=0),
                        z=torch.cat([outs_l["z1"], outs_l["z2"]], dim=0),
                        targets=torch.cat([targets_probs, targets_probs], dim=0),
                    )
                else:
                    # Hard-label mode: soft contrastive loss is undefined in the paper.
                    loss2 = torch.zeros((), device=device)

                x1_adv, x2_again = self.normalize_pair(x1_raw_adv, x2_raw)
                outs_adv = fgsm_model(im_aug1=x1_adv, im_aug2=x2_again)
                loss3 = cost_sensitive_criterion(
                    outs_adv["z1"], outs_adv["z2"], outs_adv["p1"], outs_adv["p2"], y_idx
                )

                loss = loss1 + (self.lambda1 * loss2) + (self.lambda2 * loss3)

                optimizer_cl.zero_grad(set_to_none=True)
                loss.backward()
                optimizer_cl.step()
                epoch_loss += loss.item()

            val_f1 = self._compute_f1(substitute, val_loader, device, state)
            cl_pbar.set_postfix({"Loss": f"{epoch_loss/steps:.4f}", "F1": f"{val_f1:.4f}"})
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                best_state = {
                    "sub": {k: v.detach().cpu().clone() for k, v in substitute.state_dict().items()},
                    "proj": {k: v.detach().cpu().clone() for k, v in self.projection_head.state_dict().items()},
                    "pred": {k: v.detach().cpu().clone() for k, v in self.predictor_head.state_dict().items()},
                }
            else:
                patience_counter += 1

            # if epoch % 10 == 0:
            #     print(f"[SwiftThief-CL] epoch={epoch} val_f1={val_f1:.4f}")

            if patience_counter >= self.patience:
                break

        if best_state is not None:
            substitute.load_state_dict(best_state["sub"])
            self.projection_head.load_state_dict(best_state["proj"])
            self.predictor_head.load_state_dict(best_state["pred"])

        self.logger.info(f"SwiftThief substitute trained (CL stage). Best Val F1: {best_f1:.4f}")

        # ---------------- KD stage (hardcoded) ----------------
        # Freeze projection/predictor for KD (optional but recommended)
        for p in self.projection_head.parameters():
            p.requires_grad = False
        for p in self.predictor_head.parameters():
            p.requires_grad = False
            
        # Re-create loader for KD (standard, no TwoCrop)
        # We can reuse train_q but need standard loader
        kd_loader = torch.utils.data.DataLoader(
            train_q,  # Standard dataset, not wrapped
            batch_size=bs_q,
            shuffle=True,
            **loader_kwargs,
        )

        kd_opt_config = dict(opt_config)
        kd_opt_config["lr"] = self.kd_lr
        optimizer_kd = self._build_optimizer(substitute.parameters(), kd_opt_config)

        kd_pbar = tqdm(range(self.kd_epochs), desc="[SwiftThief] Training (KD)", leave=False)
        for e in kd_pbar:
            kd_loss = self._train_kd_epoch(substitute, kd_loader, optimizer_kd, device, state)
            kd_pbar.set_postfix({"Loss": f"{kd_loss:.4f}"})

        final_val_f1 = self._compute_f1(substitute, val_loader, device, state)
        self.logger.info(f"[SwiftThief-KD] done. Val F1: {final_val_f1:.4f}")

        # Restore requires_grad for next CL call (if any)
        for p in self.projection_head.parameters():
            p.requires_grad = True
        for p in self.predictor_head.parameters():
            p.requires_grad = True

        state.attack_state["substitute"] = substitute

    # -------------------------
    # Validation
    # -------------------------

    def _compute_f1(self, model: nn.Module, val_loader: DataLoader, device: torch.device, state: BenchmarkState) -> float:
        model.eval()
        self._ensure_normalizers(state, device)

        all_preds: List[int] = []
        all_targets: List[int] = []

        with torch.no_grad():
            for x_raw, y in val_loader:
                x_raw = x_raw.to(device)
                x = self.normalize(x_raw)
                logits = model(x)

                preds = logits.argmax(dim=1).cpu().numpy().tolist()
                all_preds.extend(preds)

                if y.ndim > 1:
                    targets = y.argmax(dim=1).cpu().numpy().tolist()
                else:
                    targets = y.cpu().numpy().tolist()
                all_targets.extend(targets)

        return f1_score(all_targets, all_preds, average="macro")
