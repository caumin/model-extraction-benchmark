"""SwiftThief attack implementation (mebench plugin).

Reference: SwiftThief (Lee, Han, Lee. IJCAI'24)
  - Paper:  https://www.ijcai.org/proceedings/2024/47
  - Repo:   https://github.com/ku-air/SwiftThief
            (committed copy at repro/SwiftThief_official/)

This module mirrors the official algorithm closely. Where mebench's common
benchmarking contract diverges from the reference code we annotate the spot
with one of these markers:

  [논문 일치]    : matches the paper / official repo exactly.
  [프로젝트 적응] : mebench-side adjustment (common contract takes precedence).

Common (mebench-wide) policies that are NOT modified here:
  - U pool definition:      U = (surrogate pool) \\ labeled
  - normalization:          dataset-specific mean/std from victim_config
  - output_mode handling:   soft_prob / hard_top1 wired through OracleOutput
  - early-stop / patience:  shared substitute training policy
  - query budget accounting: only ctx.query() consumes budget

SwiftThief-specific algorithmic components implemented per paper/repo:
  - SimSiam projector (2-layer, hidden=in_dim, last BN affine=False)
  - SimSiam predictor (in_dim -> in_dim/4 -> in_dim)
  - Loss: loss1 (SimSiam on U) + lambda1 * loss2 (SoftSupSimSiamLossV17 on Q)
                                + lambda2 * loss3 (CL_FGSM + cost-sensitive)
  - Outer KD-per-epoch loop driven by sl_epoch / sl_aug_interval
  - CL stage SGD(lr=0.06, momentum=0.9, wd=5e-4) + cosine schedule (T=800)
  - Round-wise unlabeled SSL subset ramp-up (5000 -> +5000 per round)
  - imbalance_kde sampling via sklearn KernelDensity(bandwidth=0.5)
  - Two-view augmentation pipeline used inside the CL stage only
"""

from typing import Dict, Any, List, Tuple, Optional
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score
import torchvision.transforms as transforms
from tqdm import tqdm

from mebench.attackers.runner import AttackRunner
from mebench.core.context import BenchmarkContext
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState


def _build_stage_optimizer(spec: Dict[str, Any], params, default_lr: float) -> torch.optim.Optimizer:
    """Build a per-stage optimizer for SwiftThief from a config spec.

    Defaults to paper SGD (lr=default_lr, momentum=0.9, wd=5e-4) so existing
    SGD configs reproduce identically. Override via:
        substitute.cl_optimizer: { name: adamw, lr: 1e-3, weight_decay: 0.01,
                                   betas: [0.9, 0.999] }
        substitute.kd_optimizer: { name: adamw, lr: 5e-4, ... }
    """
    name = str(spec.get("name", "sgd")).lower()
    lr = float(spec.get("lr", default_lr))
    if name == "adamw":
        wd = float(spec.get("weight_decay", 0.01))
        betas = tuple(spec.get("betas", [0.9, 0.999]))
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas)
    # SGD default (paper-matching behaviour)
    wd = float(spec.get("weight_decay", 5e-4))
    momentum = float(spec.get("momentum", 0.9))
    return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
from mebench.data.loaders import create_dataloader
from mebench.models.substitute_factory import create_substitute
from mebench.utils.binary import (
    binary_bce_loss,
    binary_distribution_from_labels,
    binary_distribution_from_positive_probs,
    binary_entropy_from_positive_probs,
    binary_hard_labels_from_positive_probs,
    binary_positive_probs_from_logits,
    is_single_logit_binary_num_classes,
)
from mebench.utils.dataloader import (
    pool_loader_kwargs,
    resolve_pool_num_workers,
    resolve_train_num_workers,
    resolve_val_num_workers,
)


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
        """Soft-supervised contrastive loss as in the official SwiftThief repo.

        Reference: https://github.com/ku-air/SwiftThief
        File: contrastive_learning/simsiam/criterion.py (SoftSupSimSiamLossV17)
        """
        if p.numel() == 0 or z.numel() == 0:
            return torch.zeros((), device=self.device)

        z = z.detach()  # stop gradient

        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)

        dot_product = -torch.mm(p, z.T)

        targets = targets.to(self.device).float().clamp_min(1e-12)
        targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-12)

        entr = -(targets * targets.log()).sum(dim=1)
        entr[torch.isnan(entr)] = 0.0
        denom = torch.log(torch.tensor(float(max(2, int(self.num_classes))), device=self.device))
        norm_entr = entr / denom.clamp_min(1e-12)
        reversed_norm_entr = 1.0 - norm_entr
        mask_similar_class1 = torch.outer(reversed_norm_entr, reversed_norm_entr)

        mask_similar_class2 = F.cosine_similarity(
            targets.T.repeat(len(targets), 1, 1),
            targets.unsqueeze(2),
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
    """[논문 일치] 2-layer MLP projector (official model_factory.projection_MLP, num_layers=2).

    layer1: Linear(in -> hidden=in) -> BN -> ReLU
    layer3: Linear(hidden -> proj)  -> BN(affine=False)
    (the official `num_layers=2` path skips layer2)
    """
    def __init__(self, in_dim: int, proj_dim: int = 2048):
        super().__init__()
        hidden_dim = in_dim  # [논문 일치] official sets hidden_dim = out_dim of backbone
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
            nn.BatchNorm1d(proj_dim, affine=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimSiamPredictorHead(nn.Module):
    """[논문 일치] Predictor: Linear(in -> in/4) -> BN -> ReLU -> Linear(in/4 -> in)."""
    def __init__(self, in_dim: int):
        super().__init__()
        hidden_dim = max(1, int(in_dim) // 4)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Generic last-Linear head extraction
# ============================================================
# [프로젝트 적응] Official repo always wraps backbone with a fixed `.fc`. mebench
# uses a unified create_substitute(arch=...) factory whose archs may expose the
# classifier as `.fc`, `.classifier`, `.fc3`, etc. We locate the *last* nn.Linear
# (in module insertion order) and replace it with nn.Identity to get a backbone
# that returns features. Behaviour matches the official `backbone.fc = Identity`
# pattern.

def _detach_last_linear_head(model: nn.Module) -> nn.Module:
    """[프로젝트 적응] Detach the classification head from a backbone.

    Priority:
      1. If `model.fc` is a non-Identity nn.Module, return it as the head and
         replace it with Identity. (Matches official `backbone.fc = Identity`.)
      2. Else if `model.classifier` is a non-Identity nn.Module (e.g. Xie2019's
         multi-layer Sequential), return it whole and replace with Identity.
         This preserves the feature/classifier split when the backbone exposes
         features via `.features` and the head is a multi-layer MLP whose input
         dim equals the conv output (not the last Linear's input).
      3. Fallback: replace the last nn.Linear in module order with Identity and
         return that Linear (used by LeNet5MNIST etc. which have fc1/fc2/fc3).
    """
    fc = getattr(model, "fc", None)
    if isinstance(fc, nn.Module) and not isinstance(fc, nn.Identity):
        model.fc = nn.Identity()
        return fc

    cls = getattr(model, "classifier", None)
    if isinstance(cls, nn.Module) and not isinstance(cls, nn.Identity):
        model.classifier = nn.Identity()
        return cls

    last_name = None
    last_module = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_name = name
            last_module = module
    if last_name is None or last_module is None:
        raise ValueError("SwiftThief: could not find any nn.Linear head in substitute backbone")

    parent = model
    parts = last_name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], nn.Identity())
    return last_module


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
                out = feat_attr(x)
            elif isinstance(feat_attr, nn.Module):
                out = feat_attr(x)
            else:
                return self.backbone(x)
            # Apply avgpool if backbone has one (e.g. Xie2019)
            avgpool = getattr(self.backbone, "avgpool", None)
            if avgpool is not None and isinstance(avgpool, nn.Module):
                out = avgpool(out)
            return out
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
    """[논문 일치] Mirrors the official sl_transforms pipeline:

        de-normalize -> RandomResizedCrop / HFlip / ColorJitter / Grayscale (in [0,1])
        -> re-normalize

    Input  : a tensor [C,H,W] already normalized by the pool loader (mebench common
             policy: pool returns normalized tensors).
    Output : two normalized augmented views (v1, v2) — the same channel/scale as
             the input, ready to be fed to the substitute without further
             normalization.

    If `mean`/`std` are None, input is treated as raw [0,1] and no de/re-norm is done.
    """

    def __init__(self, aug_transform, mean=None, std=None, input_channels: int = 3):
        self.aug = aug_transform
        self.input_channels = int(input_channels)
        if mean is None or std is None:
            self.mean = None
            self.std = None
        else:
            mean_t = torch.tensor(list(mean), dtype=torch.float32).view(-1, 1, 1)
            std_t = torch.tensor(list(std), dtype=torch.float32).view(-1, 1, 1)
            self.mean = mean_t
            self.std = std_t

    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            return x
        return (x * self.std.to(x.device) + self.mean.to(x.device)).clamp(0.0, 1.0)

    def _renorm(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            return x
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def _one_view(self, x: torch.Tensor) -> torch.Tensor:
        x_raw = self._denorm(x.float())
        x_aug = self.aug(x_raw)
        return self._renorm(x_aug)

    def __call__(self, x):
        return self._one_view(x), self._one_view(x)

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

        # Training (CL) — [논문 일치] official defaults: SGD lr=0.06, momentum=0.9, wd=5e-4.
        self.batch_size = int(config.get("batch_size", 256))
        # [논문 일치] official: U batch=512, Q batch=256, KD batch=100.
        self.cl_u_batch_size = int(config.get("cl_u_batch_size", 512))
        self.cl_q_batch_size = int(config.get("cl_q_batch_size", 256))
        self.kd_batch_size = int(config.get("kd_batch_size", 100))
        # Pool scanning/scoring workers. Keep `num_workers` as a backwards-compatible alias.
        self.pool_num_workers = int(config.get("pool_num_workers", config.get("num_workers", 4)))
        self.prefetch_factor = int(config.get("prefetch_factor", 2))
        self.use_pil_transforms = bool(config.get("use_pil_transforms", False))
        self.cl_init_lr = float(config.get("cl_init_lr", config.get("lr", 0.06)))
        self.cl_epochs = int(config.get("cl_epochs", 40))
        self.final_cl_epochs = int(config.get("final_cl_epochs", 100))
        # [프로젝트 적응] common-policy patience-based early-stop within a CL stage.
        self.patience = int(config.get("patience", 50))

        # [논문 일치] U SSL subset ramp-up (5000 -> +5000 per CL stage).
        self.unlabeled_ssl_init = int(config.get("unlabeled_ssl_init", 5000))
        self.unlabeled_ssl_step = int(config.get("unlabeled_ssl_step", 5000))
        # Hard cap on the SSL subset size (defaults to a large value; bounded by U pool).
        self.unlabeled_ssl_size = int(config.get("unlabeled_ssl_size", 1_000_000))

        # [논문 일치] query schedule from official swiftthief.py.
        self.query_fraction_per_round = float(config.get("query_fraction_per_round", 0.1))
        self.imbalance_kde_splits = int(config.get("imbalance_kde_splits", 5))
        self.sl_epoch = int(config.get("sl_epoch", 500))
        self.sl_aug_interval = int(config.get("sl_aug_interval", 50))

        # KD optimizer config. Paper-default: SGD(lr=1e-2, momentum=0.9, wd=5e-4).
        # Override via substitute.kd_optimizer.{name,lr,weight_decay,betas,momentum}
        # if present in the config; otherwise the paper SGD default is used so
        # legacy SGD results remain reproducible.
        self.kd_lr = float(config.get("kd_lr", 1e-2))
        self._kd_optimizer_spec = dict(config.get("kd_optimizer") or {})
        # CL optimizer config. Paper-default: SGD(lr=0.06, momentum=0.9, wd=5e-4).
        self._cl_optimizer_spec = dict(config.get("cl_optimizer") or {})

        # internal
        self.pool_dataset = None
        self.projection_head: Optional[nn.Module] = None
        self.predictor_head: Optional[nn.Module] = None
        self.normalize: Optional[nn.Module] = None
        self.normalize_pair: Optional[nn.Module] = None
        self._ssl_transforms = None
        self.output_dim = int(state.metadata.get("num_classes") or config.get("num_classes") or 10)
        self.is_single_logit_binary = is_single_logit_binary_num_classes(self.output_dim)
        self.semantic_num_classes = 2 if self.is_single_logit_binary else self.output_dim

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

        # [논문 일치] CL global epoch counter for cosine schedule lr*0.5*(1+cos(pi*e/800)).
        st["cl_global_epoch"] = 0
        # [논문 일치] U SSL subset starts at 5000 and grows by +5000 per CL stage call.
        st["cl_round_idx"] = 0
        st["kd_optimizer_state"] = None  # persistent KD optimizer state across rounds

    def run(self, ctx: BenchmarkContext) -> None:
        """[논문 일치] Mirrors official swiftthief.py main():

            1. Initial 10% random seed -> query oracle.
            2. CL stage (40 epochs).
            3. for epoch in range(sl_epoch=500):
                 if epoch % sl_aug_interval == 0 and epoch != 0:
                     sampling (entropy single split, or imbalance_kde x 5)
                     -> query oracle for ~10% budget
                     update costs from labeled bincount
                     CL stage (40 epochs; 100 if epoch == 450)
                 KD 1 epoch on labeled set (output matching)
        """
        self.victim = ctx.oracle.model
        self._ensure_pool_dataset(self.state)
        self._ensure_fixed_validation_holdout(self.state)

        total_budget = int(self.state.metadata.get("max_budget") or ctx.budget_remaining)
        round_quota = max(1, int(total_budget * float(self.query_fraction_per_round)))

        pbar = tqdm(total=total_budget, desc="[SwiftThief] Extracting")

        # ---- Step 1: initial 10% random seed query ---------------------------------
        if ctx.budget_remaining > 0:
            queried = self._execute_query_round(
                ctx,
                query_count=min(round_quota, int(ctx.budget_remaining)),
                force_random=True,
            )
            pbar.update(int(queried))

        # ---- Initialize substitute, heads, optimizers ------------------------------
        self._init_training_objects(self.state)

        # ---- Step 2: first CL stage (40 epochs) ------------------------------------
        if len(self.state.attack_state["labeled_indices"]) > 0:
            self._cl_stage(self.state, epochs=int(self.cl_epochs))

        # ---- Step 3: outer KD loop with sampling triggers --------------------------
        # [논문 일치] sl_epoch=500, sl_aug_interval=50, KD per epoch.
        for epoch in range(int(self.sl_epoch)):
            # sampling + CL at every aug interval (skip epoch=0; that was the init seed).
            if epoch != 0 and (epoch % int(self.sl_aug_interval) == 0):
                if ctx.budget_remaining > 0:
                    self._update_sampling_mode(self.state)
                    mode = str(self.state.attack_state.get("sampling_mode", "entropy"))
                    splits = 1 if mode == "entropy" else max(1, int(self.imbalance_kde_splits))
                    per_split = max(1, int(round_quota // splits))
                    for _ in range(splits):
                        if ctx.budget_remaining <= 0:
                            break
                        q = min(per_split, int(ctx.budget_remaining))
                        queried = self._execute_query_round(ctx, query_count=q, force_random=False)
                        pbar.update(int(queried))
                        if queried <= 0:
                            break

                # [논문 일치] epoch == 450 -> CL 100 epochs; else 40.
                cl_ep = int(self.final_cl_epochs) if epoch == 450 else int(self.cl_epochs)
                if len(self.state.attack_state["labeled_indices"]) > 0:
                    self._cl_stage(self.state, epochs=cl_ep)

            # KD 1 epoch (always)
            if len(self.state.attack_state["labeled_indices"]) > 0:
                self._kd_one_epoch(self.state)

        # final state
        self.state.attack_state["last_train_labeled_count"] = len(
            self.state.attack_state["labeled_indices"]
        )
        pbar.close()

    def _execute_query_round(
        self,
        ctx: BenchmarkContext,
        *,
        query_count: int,
        force_random: bool,
    ) -> int:
        if query_count <= 0 or ctx.budget_remaining <= 0:
            return 0
        query_batch = self._select_query_batch(query_count, self.state, force_random=force_random)
        if query_batch.x.size(0) == 0:
            return 0
        oracle_output = ctx.query(query_batch.x, meta=query_batch.meta)
        self._handle_oracle_output(query_batch, oracle_output, self.state)
        return int(query_batch.x.size(0))

    # -------------------------
    # Dataset + Normalizer
    # -------------------------

    def _ensure_pool_dataset(self, state: BenchmarkState) -> None:
        if self.pool_dataset is None:
            dataset_config = state.metadata.get("dataset_config", {})
            if "data_mode" not in dataset_config:
                dataset_config = {"data_mode": "seed", **dataset_config}
            if "name" not in dataset_config:
                dataset_config = {"name": "CIFAR10", **dataset_config}

            self.pool_dataset = create_dataloader(dataset_config, batch_size=1, shuffle=False).dataset

        # Always ensure indices are populated, even if pool_dataset was injected by tests.
        N = len(self.pool_dataset)
        unlabeled = state.attack_state.get("unlabeled_indices") or []
        if len(unlabeled) == 0:
            state.attack_state["unlabeled_indices"] = list(range(N))
        else:
            state.attack_state["unlabeled_indices"] = [i for i in unlabeled if 0 <= int(i) < N]

    def observe(self, query_batch: QueryBatch, oracle_output: OracleOutput, state: BenchmarkState) -> None:
        self._handle_oracle_output(query_batch, oracle_output, state)

    def _ensure_normalizers(self, state: BenchmarkState, device: torch.device) -> None:
        """[프로젝트 적응] No-op kept for backwards compat.

        mebench common policy: pool dataset already returns normalized tensors,
        so SwiftThief should NOT apply further normalization on forward. The CL
        stage's TwoCropTransform handles de-norm/re-norm internally for aug.
        """
        return None

    def _get_pool_norm_stats(
        self, state: BenchmarkState
    ) -> Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
        """Recover the (mean, std) the pool loader applied. Used by TwoCropTransform
        to de-normalize for [0,1]-domain augmentation, then re-normalize back.
        Returns None if the pool returns raw [0,1] tensors (no de-norm needed).
        """
        from mebench.data.loaders import (
            _resolve_surrogate_normalization,
            get_surrogate_standard_normalization,
        )

        dataset_config = state.metadata.get("dataset_config", {}) or {}
        data_mode = str(dataset_config.get("data_mode", "seed"))
        channels = int(state.metadata.get("input_shape", (3, 32, 32))[0])

        try:
            if data_mode == "surrogate":
                stats = _resolve_surrogate_normalization(
                    surrogate_name=str(dataset_config.get("surrogate_name", "ImageNet")),
                    output_channels=channels,
                    surrogate_normalization=dataset_config.get("surrogate_normalization", "standard"),
                    surrogate_norm_mean=dataset_config.get("surrogate_norm_mean"),
                    surrogate_norm_std=dataset_config.get("surrogate_norm_std"),
                )
            else:
                stats = get_surrogate_standard_normalization(
                    str(dataset_config.get("name", "CIFAR10")),
                    channels=channels,
                )
        except Exception:
            stats = None

        if stats is None:
            return None
        mean, std = stats
        return tuple(float(m) for m in mean), tuple(float(s) for s in std)

    # -------------------------
    # SSL augmentation (operates in [0,1] tensor domain)
    # -------------------------

    def _build_ssl_transforms(self, state: BenchmarkState, use_pil: bool = False) -> transforms.Compose:
        """[논문 일치] official ssl/sl transform spatial+color ops (no Normalize step
        here — TwoCropTransform handles de-norm/re-norm around these ops).

        Operates directly on [C,H,W] tensors; modern torchvision transforms
        support tensor inputs natively. `use_pil` is kept for backward compat
        but is intentionally a no-op now.
        """
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
        return transforms.Compose(ops)

    def _apply_two_crops(self, x_batch: torch.Tensor, device: torch.device, state: BenchmarkState) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._ssl_transforms is None:
            self._ssl_transforms = self._build_ssl_transforms(state)

        C = state.metadata.get("input_shape", (3, 32, 32))[0]
        norm_stats = self._get_pool_norm_stats(state)
        mean, std = (norm_stats if norm_stats is not None else (None, None))
        two_crop = TwoCropTransform(self._ssl_transforms, mean=mean, std=std, input_channels=C)

        v1_list, v2_list = [], []
        for x in x_batch.detach().cpu():
            v1, v2 = two_crop(x)
            v1_list.append(v1)
            v2_list.append(v2)

        return torch.stack(v1_list).to(device), torch.stack(v2_list).to(device)

    # -------------------------
    # Propose + sampling
    # -------------------------

    def _select_query_batch(self, k: int, state: BenchmarkState, force_random: bool = False) -> QueryBatch:
        self._ensure_pool_dataset(state)

        labeled = state.attack_state["labeled_indices"]
        unlabeled = state.attack_state["unlabeled_indices"]

        if len(unlabeled) == 0:
            input_shape = state.metadata.get("input_shape", (3, 32, 32))
            x_empty = torch.empty((0, *input_shape))
            return QueryBatch(x=x_empty, meta={"indices": [], "sampling_mode": "exhausted"})

        if k > len(unlabeled):
            k = int(len(unlabeled))

        total_budget = int(state.metadata.get("max_budget") or state.budget_remaining or 10000)
        initial_seed_size = int(self.initial_seed_ratio * total_budget)

        if force_random or len(labeled) < initial_seed_size:
            n_seed = min(k, initial_seed_size - len(labeled), len(unlabeled))
            if force_random:
                n_seed = min(k, len(unlabeled))
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
        total_budget = int(state.metadata.get("max_budget") or state.budget_remaining or 10000)

        total_q = len(labeled)
        mean_per_class = total_q / num_classes

        rare_classes = [c for c in range(num_classes) if class_counts.get(c, 0) < mean_per_class]
        if not rare_classes:
            state.attack_state["sampling_mode"] = "entropy"
            return

        total_budget = int(state.metadata.get("max_budget") or state.budget_remaining or 10000)
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

        pool_workers = resolve_pool_num_workers(
            self.config,
            state.metadata.get("dataset_config", {}),
            default=self.pool_num_workers,
        )
        loader_kwargs = (
            pool_loader_kwargs(str(device), {"num_workers": int(pool_workers)})
            if pool_workers is not None
            else pool_loader_kwargs(str(device))
        )

        subset = Subset(self.pool_dataset, candidates)
        loader = DataLoader(
            subset,
            batch_size=bs,
            shuffle=False,
            **loader_kwargs,
        )

        ptr = 0
        with torch.no_grad():
            pbar = tqdm(loader, total=len(loader), desc="[SwiftThief] Scoring Pool", leave=False)
            for x_raw, _ in pbar:
                batch_len = int(x_raw.size(0))
                batch_indices = candidates[ptr : ptr + batch_len]
                ptr += batch_len

                # [프로젝트 적응] pool tensors are already normalized; no extra normalize.
                x = x_raw.to(device, non_blocking=str(device).startswith("cuda"))
                logits = substitute(x)
                if self.is_single_logit_binary:
                    probs = binary_positive_probs_from_logits(logits)
                    entropy = binary_entropy_from_positive_probs(probs)
                else:
                    probs = F.softmax(logits, dim=1)
                    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
                entropy_list = entropy.detach().cpu().tolist()
                for idx, ent in zip(batch_indices, entropy_list):
                    scores.append((int(idx), float(ent)))

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
        pool_workers = resolve_pool_num_workers(
            self.config,
            state.metadata.get("dataset_config", {}),
            default=self.pool_num_workers,
        )
        loader_kwargs = (
            pool_loader_kwargs(str(device), {"num_workers": int(pool_workers)})
            if pool_workers is not None
            else pool_loader_kwargs(str(device))
        )

        subset = Subset(self.pool_dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=int(batch_size),
            shuffle=False,
            **loader_kwargs,
        )

        feats = []
        substitute.eval()
        with torch.no_grad():
            for x_raw, _ in loader:
                # [프로젝트 적응] pool tensors are already normalized.
                x = x_raw.to(device, non_blocking=str(device).startswith("cuda"))
                f = substitute.features(x)
                if f.ndim > 2:
                    f = f.view(f.size(0), -1)
                feats.append(f.detach())
        return torch.cat(feats, dim=0) if feats else torch.empty(0, device=device)

    def _select_rare_class(self, k: int, state: BenchmarkState) -> List[int]:
        """[논문 일치] imbalance_aware_kde from official attack_tools/sampling.py.

        - rare class = nonzero argmin of victim-prediction bincount over labeled set
        - sklearn KernelDensity(kernel='gaussian', bandwidth=0.5) fitted on rare-class
          labeled features; score = log-density on unlabeled features; top-k descending.
        - The official code uses backbone features directly. We use substitute.features(x)
          flattened, equivalent to model.backbone(x) in SimSiam wrapper.
        """
        from sklearn.neighbors import KernelDensity  # local import to keep startup lean

        unlabeled = state.attack_state["unlabeled_indices"]
        labeled = state.attack_state["labeled_indices"]
        class_counts = state.attack_state["class_counts"]
        victim_outputs = state.attack_state.get("victim_outputs", {})

        substitute = state.attack_state.get("substitute")
        if substitute is None or not hasattr(substitute, "features"):
            return self._select_entropy(k, state)

        num_classes = int(self.semantic_num_classes)

        # [논문 일치] bincount over victim predictions on labeled set; rare = nonzero argmin.
        if self.is_single_logit_binary:
            preds = [
                int(binary_hard_labels_from_positive_probs(victim_outputs[idx].view(1, 1))[0].item())
                for idx in labeled if idx in victim_outputs
            ]
        else:
            preds = [
                int(victim_outputs[idx].argmax().item())
                for idx in labeled if idx in victim_outputs
            ]
        if not preds:
            return self._select_entropy(k, state)

        bincount = np.bincount(np.asarray(preds, dtype=np.int64), minlength=num_classes).astype(np.float64)
        nonzero = np.nonzero(bincount)[0]
        if nonzero.size == 0:
            return self._select_entropy(k, state)
        y_n = int(nonzero[int(np.argmin(bincount[nonzero]))])

        if self.is_single_logit_binary:
            q_y = [
                idx for idx in labeled
                if idx in victim_outputs and int(binary_hard_labels_from_positive_probs(victim_outputs[idx].view(1, 1))[0].item()) == y_n
            ]
        else:
            q_y = [idx for idx in labeled if idx in victim_outputs and int(victim_outputs[idx].argmax().item()) == y_n]
        if not q_y:
            return self._select_entropy(k, state)

        # [프로젝트 적응] candidates = full mebench-side U pool (= unlabeled_indices).
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

        rare_np = rare_feats.detach().cpu().numpy().astype(np.float64)
        pool_np = pool_feats.detach().cpu().numpy().astype(np.float64)

        # [논문 일치] official: KernelDensity(kernel='gaussian', bandwidth=0.5,
        # atol=0.0005, rtol=0.01).fit(labeled_features[pred==ur_class])
        kde = KernelDensity(kernel="gaussian", bandwidth=0.5, atol=0.0005, rtol=0.01).fit(rare_np)

        # Score in chunks to avoid blowing memory on big U pools.
        chunk = 1000
        scores = np.empty(pool_np.shape[0], dtype=np.float64)
        for s in range(0, pool_np.shape[0], chunk):
            scores[s : s + chunk] = kde.score_samples(pool_np[s : s + chunk])

        topk = min(int(k), scores.shape[0])
        order = np.argsort(-scores)[:topk]
        return [candidates[int(i)] for i in order.tolist()]

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
            if self.is_single_logit_binary:
                labels = [int(binary_hard_labels_from_positive_probs(t.view(1, 1))[0].item()) for t in y]
            else:
                labels = [int(t.argmax().item()) for t in y]
        else:
            num_classes = int(self.semantic_num_classes)
            if self.is_single_logit_binary:
                y_for_training = y.float().view(-1, 1).detach().cpu()
            else:
                y_for_training = F.one_hot(y.long(), num_classes=num_classes).float().detach().cpu()
            for i, idx in enumerate(indices):
                if int(idx) >= 0:
                    lab = int(y[i].item()) if y[i].ndim == 0 else int(y[i].argmax().item())
                    if self.is_single_logit_binary:
                        state.attack_state["victim_outputs"][int(idx)] = torch.tensor([float(lab)])
                    else:
                        one_hot = torch.zeros(num_classes)
                        one_hot[lab] = 1.0
                        state.attack_state["victim_outputs"][int(idx)] = one_hot
            labels = [int(t.item()) if t.ndim == 0 else int(t.argmax().item()) for t in y]

        state.attack_state["query_data_y"].append(y_for_training)

        for lab in labels:
            state.attack_state["class_counts"][lab] = state.attack_state["class_counts"].get(lab, 0) + 1

        # Query handling only; training schedule is orchestrated in run().

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

        total_loss = 0.0
        n_batches = 0

        for x_raw, y in loader:
            # [프로젝트 적응] pool/labeled tensors are already normalized.
            x = x_raw.to(device)
            logits = substitute(x)

            if self.is_single_logit_binary:
                y = y.to(device).float()
                loss = binary_bce_loss(logits, y)
            elif y.ndim > 1 and y.shape[1] > 1:
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

    # ============================================================
    # Training plumbing (shared across CL stages and KD epochs)
    # ============================================================

    def _build_q_split(self, state: BenchmarkState):
        """Returns (train_q, val_q, train_size). [프로젝트 적응] uses fixed validation
        holdout (common policy) when present; falls back to 80/20 split otherwise."""
        qx = state.attack_state["query_data_x"]
        qy = state.attack_state["query_data_y"]
        vqx = state.attack_state.get("val_query_data_x", [])
        vqy = state.attack_state.get("val_query_data_y", [])
        if len(qx) == 0:
            return None, None, 0

        x_all = torch.cat(qx, dim=0)
        y_all = torch.cat(qy, dim=0)
        dataset_q = QueryDataset(x_all, y_all)

        if len(vqx) > 0 and len(vqy) > 0:
            x_val = torch.cat(vqx, dim=0)
            y_val = torch.cat(vqy, dim=0)
            train_q = dataset_q
            val_q = QueryDataset(x_val, y_val)
            return train_q, val_q, len(train_q)

        total = len(dataset_q)
        val_size = max(1, int(0.2 * total))
        train_size = total - val_size
        if train_size < 2:
            return None, None, 0
        train_q, val_q = torch.utils.data.random_split(
            dataset_q, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
        return train_q, val_q, train_size

    def _make_loader_kwargs(self, state: BenchmarkState):
        substitute_config = state.metadata.get("substitute_config") or self.config.get("substitute") or {}
        device_str = str(state.metadata.get("device", "cpu"))
        pin_memory = device_str.startswith("cuda")
        train_workers = resolve_train_num_workers(substitute_config, self.config, default=0)
        val_workers = resolve_val_num_workers(substitute_config, self.config, default=train_workers)
        train_kwargs = {
            "num_workers": int(train_workers),
            "pin_memory": bool(pin_memory),
            "drop_last": False,
        }
        val_kwargs = {
            "num_workers": int(val_workers),
            "pin_memory": bool(pin_memory),
            "drop_last": False,
        }
        if int(train_workers) > 0 and bool(self.config.get("persistent_workers", False)):
            train_kwargs["persistent_workers"] = True
            train_kwargs["prefetch_factor"] = self.prefetch_factor
        if int(val_workers) > 0 and bool(self.config.get("persistent_workers", False)):
            val_kwargs["persistent_workers"] = True
            val_kwargs["prefetch_factor"] = self.prefetch_factor
        return train_kwargs, val_kwargs, substitute_config, device_str

    def _build_costs(self, state: BenchmarkState, device: torch.device) -> torch.Tensor:
        """[논문 일치] effective-number costs from per-class histogram."""
        semantic_num_classes = int(self.semantic_num_classes)
        cnt = torch.zeros(semantic_num_classes, device=device)
        for c in range(semantic_num_classes):
            cnt[c] = float(state.attack_state["class_counts"].get(c, 0))
        beta_t = torch.tensor(float(self.effective_beta), device=device)
        denom = 1.0 - torch.pow(beta_t, cnt + 1.0)
        costs = (1.0 - beta_t) / denom.clamp_min(1e-12)
        return costs / costs.sum().clamp_min(1e-12)

    def _init_training_objects(self, state: BenchmarkState) -> None:
        """Create substitute, SimSiam projector/predictor, criteria, and the
        persistent KD optimizer. Called once at the top of run()."""
        train_kwargs, val_kwargs, substitute_config, device_str = self._make_loader_kwargs(state)
        device = torch.device(device_str)

        num_classes = int(self.output_dim)
        arch = substitute_config.get("arch", "resnet18")
        width_mult = int(substitute_config.get("width_mult", 1))
        dropout_prob = float(substitute_config.get("dropout_prob", 0.0))

        base = create_substitute(
            arch=arch,
            num_classes=num_classes,
            input_channels=state.metadata.get("input_shape", (3, 32, 32))[0],
            width_mult=width_mult,
            dropout_prob=dropout_prob,
        ).to(device)
        # [프로젝트 적응] generic last-Linear head extraction (handles .fc, .classifier,
        # .fc3 [LeNet5MNIST], etc.). Mirrors official `backbone.fc = Identity`.
        classifier = _detach_last_linear_head(base).to(device)
        substitute = SwiftThiefSubstitute(base, classifier).to(device)

        # feat_dim via dummy forward (size/channel-aware)
        C, H, W = state.metadata.get("input_shape", (3, 32, 32))
        with torch.no_grad():
            f = substitute.features(torch.zeros(2, C, H, W, device=device))
            if f.ndim > 2:
                f = f.view(f.size(0), -1)
        feat_dim = int(f.shape[1])

        self.projection_head = SimSiamProjectionHead(feat_dim, self.projection_dim).to(device)
        self.predictor_head = SimSiamPredictorHead(self.projection_dim).to(device)

        self._ensure_normalizers(state, device)

        # KD optimizer. Default SGD(lr=sl_lr=1e-2, momentum=0.9, wd=5e-4); paper-matching.
        # Override via substitute.kd_optimizer (see _build_stage_optimizer).
        self._kd_optimizer = _build_stage_optimizer(
            self._kd_optimizer_spec,
            substitute.parameters(),
            default_lr=float(self.kd_lr),
        )

        state.attack_state["substitute"] = substitute
        state.attack_state["cl_global_epoch"] = 0
        state.attack_state["cl_round_idx"] = 0

    @staticmethod
    def _cosine_lr(base_lr: float, global_epoch: int, total: int = 800) -> float:
        """[논문 일치] official adjust_learning_rate: lr*0.5*(1+cos(pi*epoch/800))."""
        return float(base_lr) * 0.5 * (1.0 + math.cos(math.pi * float(global_epoch) / float(total)))

    def _build_cl_loaders(self, state: BenchmarkState):
        """Rebuild labeled/unlabeled SSL loaders for a CL stage.

        [논문 일치] U SSL subset size = 5000 + 5000 * cl_round_idx (capped by U size).
        """
        train_q, val_q, train_size = self._build_q_split(state)
        if train_q is None or train_size < 2:
            return None, None, None

        ssl_transform = self._build_ssl_transforms(state)
        input_channels = state.metadata.get("input_shape", (3, 32, 32))[0]
        # [논문 일치] use the same stats the pool loader applied so we can
        # de-normalize for [0,1]-domain aug then re-normalize back.
        norm_stats = self._get_pool_norm_stats(state)
        mean, std = (norm_stats if norm_stats is not None else (None, None))
        two_crop = TwoCropTransform(
            ssl_transform, mean=mean, std=std, input_channels=input_channels,
        )

        train_kwargs, val_kwargs, _, _ = self._make_loader_kwargs(state)

        # [논문 일치] CL Q batch size = 256 (official labeled_data_loader batch_size).
        bs_q = min(self.cl_q_batch_size, len(train_q))
        labeled_loader = torch.utils.data.DataLoader(
            SimSiamDataset(train_q, two_crop), batch_size=bs_q, shuffle=True, **train_kwargs
        )
        val_loader = torch.utils.data.DataLoader(
            val_q, batch_size=min(self.cl_q_batch_size, len(val_q)), shuffle=False, **val_kwargs
        )

        unlabeled_indices = state.attack_state["unlabeled_indices"]
        round_idx = int(state.attack_state.get("cl_round_idx", 0))
        # [논문 일치] num_unlabeled = init + step * round_idx; bounded by U size.
        target_n = int(self.unlabeled_ssl_init) + int(self.unlabeled_ssl_step) * round_idx
        target_n = max(1, min(target_n, int(self.unlabeled_ssl_size)))

        unlabeled_loader = None
        if len(unlabeled_indices) > 0:
            if len(unlabeled_indices) > target_n:
                u_indices = np.random.choice(unlabeled_indices, target_n, replace=False).tolist()
            else:
                u_indices = list(unlabeled_indices)
            if len(u_indices) > 0:
                # [논문 일치] CL U batch size = 512 (official unlabeled_data_loader batch_size).
                bs_u = min(self.cl_u_batch_size, len(u_indices))
                unlabeled_loader = torch.utils.data.DataLoader(
                    SimSiamDataset(PoolUDataset(u_indices, self.pool_dataset), two_crop),
                    batch_size=bs_u,
                    shuffle=True,
                    drop_last=True,  # [논문 일치] official unlabeled drop_last=True
                    **{k: v for k, v in train_kwargs.items() if k != "drop_last"},
                )
        return labeled_loader, unlabeled_loader, val_loader

    def _cl_stage(self, state: BenchmarkState, epochs: int) -> None:
        """[논문 일치] One contrastive-learning stage: `epochs` epochs over U-loader length,
        each step combining loss1 (SimSiam on U) + lambda1 * loss2 (SoftSup on Q)
        + lambda2 * loss3 (CL_FGSM cost-sensitive on Q). Cosine LR over T=800 using
        the *global* CL epoch counter (state.attack_state['cl_global_epoch']).
        """
        device_str = str(state.metadata.get("device", "cpu"))
        device = torch.device(device_str)
        self._ensure_normalizers(state, device)

        labeled_loader, unlabeled_loader, val_loader = self._build_cl_loaders(state)
        if labeled_loader is None:
            return

        substitute: SwiftThiefSubstitute = state.attack_state["substitute"].to(device)
        semantic_num_classes = int(self.semantic_num_classes)
        costs = self._build_costs(state, device)
        criterion = SimSiamLoss("simplified").to(device)
        soft_criterion = SoftSupSimSiamLossV17(device, semantic_num_classes).to(device)
        cost_sensitive_criterion = SimSiamLoss_cost_sensitive(costs).to(device)
        fgsm_model = _SimSiamWrapper(substitute, self.projection_head, self.predictor_head).to(device)
        reg_adversary = CL_FGSM(fgsm_model, float(self.fgsm_epsilon), str(device)).to(device)

        # CL optimizer. Default SGD(lr=0.06, momentum=0.9, wd=5e-4); paper-matching.
        # Override via substitute.cl_optimizer (see _build_stage_optimizer).
        # Cosine schedule (T=800) below adjusts lr in-place via param_groups, so
        # this works identically for SGD and AdamW.
        optimizer_cl = _build_stage_optimizer(
            self._cl_optimizer_spec,
            list(substitute.parameters())
            + list(self.projection_head.parameters())
            + list(self.predictor_head.parameters()),
            default_lr=float(self.cl_init_lr),
        )
        # Effective CL base lr (after spec override) drives the cosine schedule.
        cl_base_lr = float(optimizer_cl.param_groups[0]["lr"])

        # [논문 일치] No patience / best-state restoration within a CL stage.
        # Rationale: CL losses (loss1+loss2+loss3) only update backbone+projector+
        # predictor (no gradient reaches `fc`). `_compute_f1` evaluates substitute
        # = fc(backbone(.)), which depends on the still-untrained `fc`. As a
        # result val F1 is dominated by random fc fluctuation and is meaningless
        # as a CL-stage early-stop signal. Official code keeps the final epoch's
        # weights — we follow that. Logging F1 only for diagnostic purposes.
        last_f1 = float("nan")

        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader) if unlabeled_loader is not None else None

        cl_pbar = tqdm(range(int(epochs)), desc="[SwiftThief] CL", leave=False)
        for local_e in cl_pbar:
            # cosine LR by global epoch count
            global_e = int(state.attack_state.get("cl_global_epoch", 0)) + local_e
            lr_now = self._cosine_lr(cl_base_lr, global_e, total=800)
            for g in optimizer_cl.param_groups:
                g["lr"] = lr_now

            substitute.train()
            self.projection_head.train()
            self.predictor_head.train()

            steps = len(unlabeled_loader) if unlabeled_loader is not None else len(labeled_loader)
            if steps == 0:
                break

            epoch_loss = 0.0
            for _ in range(steps):
                # ---- fetch U and Q batches first ------------------------------------
                # [프로젝트 적응] TwoCropTransform already returns normalized two-views.
                if unlabeled_loader is not None:
                    try:
                        u1, u2 = next(unlabeled_iter)
                    except StopIteration:
                        unlabeled_iter = iter(unlabeled_loader)
                        u1, u2 = next(unlabeled_iter)
                    u1 = u1.to(device, non_blocking=True)
                    u2 = u2.to(device, non_blocking=True)

                try:
                    x1, x2, y = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(labeled_loader)
                    x1, x2, y = next(labeled_iter)
                x1 = x1.to(device, non_blocking=True)
                x2 = x2.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                # [프로젝트 적응] mebench oracle output is already a probability vector
                # (or hard one-hot / single-logit). Official feeds raw victim logits
                # then applies softmax inside the train loop. We normalize here to a
                # valid probability distribution to keep SoftSupSimSiamLossV17 well-defined.
                if self.is_single_logit_binary:
                    targets_probs = binary_distribution_from_positive_probs(y.float())
                    y_idx = binary_hard_labels_from_positive_probs(y.float())
                elif y.ndim > 1 and y.shape[1] > 1:
                    targets_probs = y.clamp_min(1e-8)
                    targets_probs = targets_probs / targets_probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
                    y_idx = targets_probs.argmax(dim=1)
                else:
                    y_idx = y.long()
                    targets_probs = F.one_hot(y_idx, num_classes=semantic_num_classes).float()

                # ---- FGSM adversarial view (puts substitute in eval) ----------------
                adv_x1 = reg_adversary(x1, x2)

                # [논문 일치] CL_FGSM.forward sets self.model.eval(); restore train mode
                # before main forward passes so BN uses batch stats (matches
                # main_sup_soft.py line 186 in the official repo).
                substitute.train()
                self.projection_head.train()
                self.predictor_head.train()

                # ---- Main forward passes (all in TRAIN mode) ------------------------
                if unlabeled_loader is not None:
                    outs_u = fgsm_model(im_aug1=u1, im_aug2=u2)
                    loss1 = criterion(outs_u["z1"], outs_u["z2"], outs_u["p1"], outs_u["p2"])
                else:
                    loss1 = torch.zeros((), device=device)

                outs_l = fgsm_model(im_aug1=x1, im_aug2=x2)
                loss2 = soft_criterion(
                    p=torch.cat([outs_l["p1"], outs_l["p2"]], dim=0),
                    z=torch.cat([outs_l["z1"], outs_l["z2"]], dim=0),
                    targets=torch.cat([targets_probs, targets_probs], dim=0),
                )
                outs_adv = fgsm_model(im_aug1=adv_x1, im_aug2=x2)
                loss3 = cost_sensitive_criterion(
                    outs_adv["z1"], outs_adv["z2"], outs_adv["p1"], outs_adv["p2"], y_idx
                )

                loss = loss1 + (self.lambda1 * loss2) + (self.lambda2 * loss3)
                optimizer_cl.zero_grad(set_to_none=True)
                loss.backward()
                optimizer_cl.step()
                epoch_loss += float(loss.item())

            # [프로젝트 적응] val F1 logged for diagnostics only; not used for early-stop
            # nor best-state restoration (see rationale above).
            last_f1 = self._compute_f1(substitute, val_loader, device, state)
            cl_pbar.set_postfix({"lr": f"{lr_now:.4f}", "Loss": f"{epoch_loss/max(1,steps):.4f}", "F1": f"{last_f1:.4f}"})

        # advance global cl-epoch counter and round counter for next CL stage
        state.attack_state["cl_global_epoch"] = int(state.attack_state.get("cl_global_epoch", 0)) + int(epochs)
        state.attack_state["cl_round_idx"] = int(state.attack_state.get("cl_round_idx", 0)) + 1
        state.attack_state["substitute"] = substitute
        self.logger.info(f"SwiftThief CL stage done. Final Val F1: {last_f1:.4f} (diagnostic only)")

    def _kd_one_epoch(self, state: BenchmarkState) -> None:
        """[논문 일치] One KD epoch (output matching) on the labeled set.

        Runs once per outer SL epoch (sl_epoch=500). Optimizer is the persistent
        SGD(lr=sl_lr=1e-2) created in _init_training_objects.
        """
        device_str = str(state.metadata.get("device", "cpu"))
        device = torch.device(device_str)

        train_q, _val_q, _train_size = self._build_q_split(state)
        if train_q is None:
            return

        train_kwargs, _val_kwargs, _, _ = self._make_loader_kwargs(state)
        # [논문 일치] KD batch size = 100 (official attack_train_loader batch_size).
        bs_kd = min(self.kd_batch_size, len(train_q))
        kd_loader = torch.utils.data.DataLoader(
            train_q, batch_size=bs_kd, shuffle=True, **train_kwargs
        )
        substitute: SwiftThiefSubstitute = state.attack_state["substitute"].to(device)

        # Freeze SimSiam heads during KD (paper does not back-prop through them here).
        for p in self.projection_head.parameters():
            p.requires_grad = False
        for p in self.predictor_head.parameters():
            p.requires_grad = False
        try:
            self._train_kd_epoch(substitute, kd_loader, self._kd_optimizer, device, state)
        finally:
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
                # [프로젝트 적응] val tensors are already normalized.
                x = x_raw.to(device)
                logits = model(x)

                if self.is_single_logit_binary:
                    preds = binary_hard_labels_from_positive_probs(binary_positive_probs_from_logits(logits)).cpu().numpy().tolist()
                else:
                    preds = logits.argmax(dim=1).cpu().numpy().tolist()
                all_preds.extend(preds)

                if self.is_single_logit_binary:
                    targets = binary_hard_labels_from_positive_probs(y.float()).cpu().numpy().tolist()
                elif y.ndim > 1:
                    targets = y.argmax(dim=1).cpu().numpy().tolist()
                else:
                    targets = y.cpu().numpy().tolist()
                all_targets.extend(targets)

        return f1_score(all_targets, all_preds, average="macro")
