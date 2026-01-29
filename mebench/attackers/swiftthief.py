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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import torchvision.transforms as transforms

from mebench.attackers.base import BaseAttack
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.data.loaders import create_dataloader
from mebench.models.substitute_factory import create_substitute


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
        z = z.detach()
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)

        dot_product = -torch.mm(p, z.T)

        entr = -(targets * targets.log()).sum(dim=1)
        entr[torch.isnan(entr)] = 0.
        norm_entr = entr / torch.log(torch.tensor(self.num_classes, device=self.device))
        reversed_norm_entr = 1 - norm_entr
        mask_similar_class1 = torch.outer(reversed_norm_entr, reversed_norm_entr)

        mask_similar_class2 = F.cosine_similarity(
            targets.T.repeat(len(targets), 1, 1),
            targets.unsqueeze(2)
        ).to(self.device)

        mask_anchor_out = (1 - torch.eye(dot_product.shape[0], device=self.device))
        mask_combined = mask_similar_class1 * mask_similar_class2 * mask_anchor_out

        dot_product_selected = dot_product * mask_combined
        selected = dot_product_selected[dot_product_selected.nonzero(as_tuple=True)]
        if selected.numel() == 0:
            return torch.zeros((), device=self.device)
        return selected.mean()


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


# ============================================================
# SwiftThief Attack
# ============================================================

class SwiftThief(BaseAttack):
    def __init__(self, config: dict, state: BenchmarkState):
        super().__init__(config, state)

        # Core hyperparameters
        self.I = int(config.get("I", 10))
        self.initial_seed_ratio = float(config.get("initial_seed_ratio", 0.1))
        self.lambda2 = float(config.get("lambda2", 0.01))  # weight for loss3
        self.fgsm_epsilon = float(config.get("fgsm_epsilon", 0.01))
        self.projection_dim = int(config.get("projection_dim", 2048))

        # Sampling
        self.kde_sigma = float(config.get("kde_sigma", 1.0))
        self.max_pool_eval = int(config.get("max_pool_eval", 2000))

        # Training (CL)
        self.batch_size = int(config.get("batch_size", 256))
        self.lr = float(config.get("lr", 0.06))
        self.momentum = float(config.get("momentum", 0.9))
        self.weight_decay = float(config.get("weight_decay", 5e-4))
        self.cl_epochs = int(config.get("cl_epochs", 40))
        self.patience = int(config.get("patience", 50))

        # KD hardcoded (requested)
        self.kd_epochs = 10
        self.kd_lr = 1e-2

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
        st["class_counts"] = {}
        st["victim_outputs"] = {}
        st["substitute"] = None
        st["sampling_mode"] = "entropy"

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

    def _build_ssl_transforms(self, state: BenchmarkState) -> transforms.Compose:
        C, H, W = state.metadata.get("input_shape", (3, 32, 32))

        if C == 1:
            cj = transforms.ColorJitter(0.4, 0.4, 0.0, 0.0)
        else:
            cj = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)

        return transforms.Compose(
            [
                transforms.RandomResizedCrop((H, W), scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([cj], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
            ]
        )

    def _apply_two_crops(self, x_batch: torch.Tensor, device: torch.device, state: BenchmarkState) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._ssl_transforms is None:
            self._ssl_transforms = self._build_ssl_transforms(state)

        C = state.metadata.get("input_shape", (3, 32, 32))[0]

        to_pil = transforms.ToPILImage()
        v1_list, v2_list = [], []

        for x in x_batch.detach().cpu():
            img = to_pil(x.clamp(0, 1))
            if C == 1:
                if img.mode != "L":
                    img = img.convert("L")
            else:
                if img.mode != "RGB":
                    img = img.convert("RGB")

            v1_list.append(self._ssl_transforms(img))
            v2_list.append(self._ssl_transforms(img))

        return torch.stack(v1_list).to(device), torch.stack(v2_list).to(device)

    # -------------------------
    # Propose + sampling
    # -------------------------

    def propose(self, k: int, state: BenchmarkState) -> QueryBatch:
        self._ensure_pool_dataset(state)

        labeled = state.attack_state["labeled_indices"]
        unlabeled = state.attack_state["unlabeled_indices"]

        if len(unlabeled) == 0:
            C, H, W = state.metadata.get("input_shape", (3, 32, 32))
            x = torch.randn(k, C, H, W)
            return QueryBatch(x=x, meta={"indices": [], "pool_exhausted": True})

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

        if len(x_list) < k:
            C, H, W = state.metadata.get("input_shape", (3, 32, 32))
            for _ in range(k - len(x_list)):
                x_list.append(torch.randn(C, H, W))
                indices.append(-1)

        x = torch.stack(x_list[:k])
        return QueryBatch(x=x, meta={"indices": indices[:k], "sampling_mode": state.attack_state["sampling_mode"]})

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

        rare_sum = sum(class_counts.get(c, 0) for c in rare_classes)
        mean_rare = rare_sum / len(rare_classes)
        remaining = total_budget - total_q
        threshold = len(rare_classes) * (mean_per_class - mean_rare)

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

        cand_n = min(len(unlabeled), self.max_pool_eval)
        candidates = np.random.choice(unlabeled, cand_n, replace=False).tolist() if cand_n > 0 else []
        if not candidates:
            return []

        entropy_scores = []
        bs = 128
        with torch.no_grad():
            for start in range(0, len(candidates), bs):
                chunk = candidates[start:start + bs]
                x_raw = torch.stack([self.pool_dataset[i][0] for i in chunk]).to(device)
                x = self.normalize(x_raw)
                probs = F.softmax(substitute(x), dim=1)
                ent = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                entropy_scores.extend([(chunk[i], float(ent[i].item())) for i in range(len(chunk))])

        entropy_scores.sort(key=lambda t: t[1], reverse=True)
        return [idx for idx, _ in entropy_scores[:min(k, len(entropy_scores))]]

    def _extract_features_for_indices(
        self,
        indices: List[int],
        substitute: SwiftThiefSubstitute,
        device: torch.device,
        state: BenchmarkState,
        batch_size: int = 128
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

        cand_n = min(len(unlabeled), self.max_pool_eval)
        candidates = np.random.choice(unlabeled, cand_n, replace=False).tolist() if cand_n > 0 else []
        if not candidates:
            return []

        device = next(substitute.parameters()).device
        rare_feats = self._extract_features_for_indices(q_y, substitute, device, state)
        pool_feats = self._extract_features_for_indices(candidates, substitute, device, state)
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

    def observe(self, query_batch: QueryBatch, oracle_output: OracleOutput, state: BenchmarkState) -> None:
        x = query_batch.x
        y = oracle_output.y

        state.attack_state["query_data_x"].append(x.detach().cpu())
        state.attack_state["query_data_y"].append(y.detach().cpu())

        indices = query_batch.meta.get("indices", [])
        if oracle_output.kind == "soft_prob":
            for i, idx in enumerate(indices):
                if int(idx) >= 0:
                    state.attack_state["victim_outputs"][int(idx)] = y[i].detach().cpu()
            labels = [int(t.argmax().item()) for t in y]
        else:
            num_classes = int(state.metadata.get("num_classes") or 10)
            for i, idx in enumerate(indices):
                if int(idx) >= 0:
                    lab = int(y[i].item()) if y[i].ndim == 0 else int(y[i].argmax().item())
                    one_hot = torch.zeros(num_classes)
                    one_hot[lab] = 1.0
                    state.attack_state["victim_outputs"][int(idx)] = one_hot
            labels = [int(t.item()) if t.ndim == 0 else int(t.argmax().item()) for t in y]

        for lab in labels:
            state.attack_state["class_counts"][lab] = state.attack_state["class_counts"].get(lab, 0) + 1

        labeled_count = len(state.attack_state["labeled_indices"])
        total_budget = int(state.metadata.get("max_budget") or self.config.get("max_budget", 10000))
        round_size = max(1, total_budget // max(self.I, 1))
        if labeled_count % round_size == 0 and labeled_count > 0:
            self.train_substitute(state)

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

    def train_substitute(self, state: BenchmarkState) -> None:
        self._ensure_pool_dataset(state)

        qx = state.attack_state["query_data_x"]
        qy = state.attack_state["query_data_y"]
        if len(qx) == 0:
            return

        x_all = torch.cat(qx, dim=0)
        y_all = torch.cat(qy, dim=0)

        class QueryDataset(torch.utils.data.Dataset):
            def __init__(self, x, y):
                self.x = x
                self.y = y
            def __len__(self): return len(self.x)
            def __getitem__(self, idx): return self.x[idx], self.y[idx]

        dataset_q = QueryDataset(x_all, y_all)

        total_size = len(dataset_q)
        val_size = max(1, int(0.2 * total_size))
        train_size = total_size - val_size
        if train_size < 2:
            return

        train_q, val_q = torch.utils.data.random_split(
            dataset_q, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

        bs_q = min(256, len(train_q))
        if bs_q <= 0:
            return
        labeled_loader = torch.utils.data.DataLoader(train_q, batch_size=bs_q, shuffle=True, num_workers=0, drop_last=False)
        val_loader = torch.utils.data.DataLoader(val_q, batch_size=min(256, len(val_q)), shuffle=False, num_workers=0, drop_last=False)

        # U loader
        unlabeled_indices = state.attack_state["unlabeled_indices"]
        if len(unlabeled_indices) == 0:
            unlabeled_loader = None
        else:
            class PoolU(torch.utils.data.Dataset):
                def __init__(self, indices, pool):
                    self.indices = indices
                    self.pool = pool
                def __len__(self): return len(self.indices)
                def __getitem__(self, i):
                    idx = self.indices[i]
                    img, _ = self.pool[idx]
                    return img

            cap = min(len(unlabeled_indices), self.max_pool_eval)
            u_indices = np.random.choice(unlabeled_indices, cap, replace=False).tolist() if cap > 0 else []
            if len(u_indices) == 0:
                unlabeled_loader = None
            else:
                dataset_u = PoolU(u_indices, self.pool_dataset)
                bs_u = min(self.batch_size, 128, len(dataset_u))
                unlabeled_loader = torch.utils.data.DataLoader(dataset_u, batch_size=bs_u, shuffle=True, num_workers=0, drop_last=False)

        device = torch.device(state.metadata.get("device", "cpu"))
        num_classes = int(state.metadata.get("num_classes") or 10)

        # init / warm-start substitute
        substitute_config = state.metadata.get("substitute_config") or self.config.get("substitute") or {}
        arch = substitute_config.get("arch", "resnet18")

        substitute = state.attack_state.get("substitute")
        if not isinstance(substitute, SwiftThiefSubstitute):
            base = create_substitute(
                arch=arch,
                num_classes=num_classes,
                input_channels=state.metadata.get("input_shape", (3, 32, 32))[0],
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

        # costs (effective number) from histogram
        cnt = torch.zeros(num_classes, device=device)
        for c in range(num_classes):
            cnt[c] = float(state.attack_state["class_counts"].get(c, 0))
        beta = 0.99
        costs = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta, device=device), cnt + 1.0))
        costs = costs / costs.sum().clamp_min(1e-12)

        criterion = SimSiamLoss('simplified').to(device)
        soft_criterion = SoftSupSimSiamLossV17(device, num_classes).to(device)
        cost_sensitive_criterion = SimSiamLoss_cost_sensitive(costs).to(device)

        fgsm_model = _SimSiamWrapper(substitute, self.projection_head, self.predictor_head).to(device)
        reg_adversary = CL_FGSM(fgsm_model, self.fgsm_epsilon, device)

        optimizer_cl = torch.optim.SGD(
            list(substitute.parameters()) + list(self.projection_head.parameters()) + list(self.predictor_head.parameters()),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

        best_f1 = 0.0
        patience_counter = 0
        best_state = None

        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader) if unlabeled_loader is not None else None

        # ---------------- CL stage ----------------
        for epoch in range(self.cl_epochs):
            substitute.train()
            self.projection_head.train()
            self.predictor_head.train()

            steps = len(unlabeled_loader) if unlabeled_loader is not None else len(labeled_loader)
            if steps == 0:
                break

            for _ in range(steps):
                # U batch
                if unlabeled_loader is not None:
                    try:
                        u_raw = next(unlabeled_iter)
                    except StopIteration:
                        unlabeled_iter = iter(unlabeled_loader)
                        u_raw = next(unlabeled_iter)

                    u_raw = u_raw.to(device)
                    u1_raw, u2_raw = self._apply_two_crops(u_raw, device, state)
                    u1, u2 = self.normalize_pair(u1_raw, u2_raw)
                    outs_u = fgsm_model(im_aug1=u1, im_aug2=u2)
                    loss1 = criterion(outs_u['z1'], outs_u['z2'], outs_u['p1'], outs_u['p2'])
                else:
                    loss1 = torch.zeros((), device=device)

                # Q batch
                try:
                    x_raw, y = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(labeled_loader)
                    x_raw, y = next(labeled_iter)

                x_raw = x_raw.to(device)
                y = y.to(device)

                x1_raw, x2_raw = self._apply_two_crops(x_raw, device, state)
                x1, x2 = self.normalize_pair(x1_raw, x2_raw)

                adv_x1 = reg_adversary(x1, x2)

                outs_l = fgsm_model(im_aug1=x1, im_aug2=x2)

                if y.ndim > 1 and y.shape[1] > 1:
                    targets = y.clamp_min(1e-8)
                    targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-12)
                    loss2 = soft_criterion(
                        p=torch.cat([outs_l['p1'], outs_l['p2']], dim=0),
                        z=torch.cat([outs_l['z1'], outs_l['z2']], dim=0),
                        targets=torch.cat([targets, targets], dim=0)
                    )
                    y_idx = targets.argmax(dim=1)
                else:
                    loss2 = torch.zeros((), device=device)
                    y_idx = y.long()

                outs_adv = fgsm_model(im_aug1=adv_x1, im_aug2=x2)
                loss3 = cost_sensitive_criterion(
                    outs_adv['z1'], outs_adv['z2'],
                    outs_adv['p1'], outs_adv['p2'],
                    y_idx
                )

                loss = loss1 + loss2 + self.lambda2 * loss3

                optimizer_cl.zero_grad(set_to_none=True)
                loss.backward()
                optimizer_cl.step()

            val_f1 = self._compute_f1(substitute, val_loader, device, state)
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

            if epoch % 10 == 0:
                print(f"[SwiftThief-CL] epoch={epoch} val_f1={val_f1:.4f}")

            if patience_counter >= self.patience:
                break

        if best_state is not None:
            substitute.load_state_dict(best_state["sub"])
            self.projection_head.load_state_dict(best_state["proj"])
            self.predictor_head.load_state_dict(best_state["pred"])

        print(f"SwiftThief substitute trained (CL stage). Best F1: {best_f1:.4f}")

        # ---------------- KD stage (hardcoded) ----------------
        # Freeze projection/predictor for KD (optional but recommended)
        for p in self.projection_head.parameters():
            p.requires_grad = False
        for p in self.predictor_head.parameters():
            p.requires_grad = False

        optimizer_kd = torch.optim.SGD(
            substitute.parameters(),
            lr=self.kd_lr,
            momentum=0.9,
            weight_decay=self.weight_decay
        )

        for e in range(self.kd_epochs):
            kd_loss = self._train_kd_epoch(substitute, labeled_loader, optimizer_kd, device, state)
            if e % 5 == 0:
                val_f1_kd = self._compute_f1(substitute, val_loader, device, state)
                print(f"[SwiftThief-KD] epoch={e} kd_loss={kd_loss:.4f} val_f1={val_f1_kd:.4f}")

        final_val_f1 = self._compute_f1(substitute, val_loader, device, state)
        print(f"[SwiftThief-KD] done. val_f1={final_val_f1:.4f}")

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

