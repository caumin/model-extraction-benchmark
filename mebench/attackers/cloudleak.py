"""CloudLeak attack implementation."""

from typing import Dict, Any, List, Optional, Tuple
from collections import OrderedDict
import gc
import bisect
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from mebench.attackers.runner import AttackRunner
from mebench.core.context import BenchmarkContext
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.data.loaders import create_dataloader
from mebench.utils.dataloader import (
    pool_loader_kwargs,
    resolve_pool_num_workers,
    resolve_train_num_workers,
    resolve_val_num_workers,
)
from mebench.models.substitute_factory import create_substitute
from mebench.training import SubstituteTrainer, TrainRequest
from mebench.utils.binary import (
    binary_bce_loss,
    binary_hard_labels_from_logits,
    binary_hard_targets,
    binary_positive_probs_from_logits,
    is_single_logit_binary_num_classes,
)
from mebench.utils.config_aliases import resolve_iterations


class FeatureFool:
    def __init__(
        self,
        model: nn.Module,
        margin_m: float = 0.5,
        lambda_adv: float = 0.001,
        objective: str = "euclidean",
        max_iters: int = 10,
        epsilon: float = 10.0 / 255.0,
        factr: float = 1e7,
        pgtol: float = 1e-5,
        device: str = "cpu",
        feature_layer: Optional[str] = None,
    ) -> None:
        self.model = model.eval()
        self.margin_m = margin_m
        self.lambda_adv = lambda_adv
        self.objective = str(objective)
        self.max_iters = max_iters
        self.epsilon = float(epsilon)
        self.factr = factr
        self.pgtol = pgtol
        self.device = device
        self.feature_layer_name = feature_layer
        self._feature_layer = self._get_feature_layer(model)

        if self.epsilon < 0:
            raise ValueError(f"FeatureFool epsilon must be >= 0, got {self.epsilon}")

        for param in self.model.parameters():
            param.requires_grad = False

    def _get_feature_layer(self, model: nn.Module) -> nn.Module:
        if self.feature_layer_name:
            for name, module in model.named_modules():
                if name == self.feature_layer_name:
                    return module

        if hasattr(model, "avgpool"):
            return model.avgpool

        for name, module in model.named_modules():
            if name.endswith("avgpool"):
                return module

        if hasattr(model, "classifier"):
            classifier = model.classifier
            if isinstance(classifier, nn.Sequential):
                linears = [m for m in classifier if isinstance(m, nn.Linear)]
                if len(linears) >= 2:
                    return linears[-2]
                if len(linears) == 1:
                    return linears[0]

        linears: List[nn.Linear] = []
        for module in model.modules():
            if isinstance(module, nn.Linear):
                linears.append(module)
        if not linears:
            raise ValueError("FeatureFool requires a Linear layer for feature extraction")

        # Prefer the penultimate Linear (representation) when available.
        if len(linears) >= 2:
            return linears[-2]
        return linears[-1]

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        activations: List[torch.Tensor] = []

        def forward_hook(_module, _inputs, output):
            activations.append(output)

        handle = self._feature_layer.register_forward_hook(forward_hook)
        _ = self.model(x)
        handle.remove()

        if not activations:
            raise RuntimeError("Feature layer hook failed to capture features")
        feats = activations[0]
        if feats.ndim > 2:
            feats = feats.view(feats.size(0), -1)
        return feats

    def generate_batch(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor,
        margin_m: Optional[torch.Tensor] = None,
        *,
        to_cpu: bool = True,
    ) -> torch.Tensor:
        if x_source.numel() == 0:
            return x_source.detach().cpu()

        B = x_source.size(0)
        self.model.eval()

        if margin_m is None:
            margin_m = torch.full((B,), self.margin_m, device=self.device)
        else:
            margin_m = margin_m.to(self.device)

        activations: List[torch.Tensor] = []

        def forward_hook(_module, _inputs, output):
            activations.append(output)

        hook_handle = self._feature_layer.register_forward_hook(forward_hook)

        try:
            x_source_dev = x_source.to(self.device)
            x_target_dev = x_target.to(self.device)

            with torch.no_grad():
                activations.clear()
                _ = self.model(x_target_dev)
                phi_t = activations.pop(0).detach().view(B, -1)

            objective = str(self.objective).lower().strip()
            if objective not in {"euclidean", "triplet"}:
                raise ValueError(f"Unknown FeatureFool objective: {self.objective}")

            phi_s = None
            if objective == "triplet":
                with torch.no_grad():
                    activations.clear()
                    _ = self.model(x_source_dev)
                    phi_s = activations.pop(0).detach().view(B, -1)

            # Official implementation uses explicit box bounds around the base image.
            # Implement per-pixel bounds in [0,1] via a reparameterization that always
            # stays within the intersection of [0,1] and [x_source - eps, x_source + eps].
            x_base = x_source_dev.detach().clamp(0.0, 1.0)
            eps = float(self.epsilon)
            lb = (x_base - eps).clamp(0.0, 1.0)
            ub = (x_base + eps).clamp(0.0, 1.0)
            span = ub - lb

            # Initialize w so that x_adv starts (approximately) at x_base.
            safe_span = span.clone()
            safe_span[safe_span == 0] = 1.0
            t_init = ((x_base - lb) / safe_span).clamp(1e-6, 1.0 - 1e-6)
            w = torch.log(t_init / (1.0 - t_init)).clone().detach().requires_grad_(True)
            optimizer = torch.optim.LBFGS(
                [w],
                lr=1.0,
                max_iter=self.max_iters,
                history_size=10,
                line_search_fn="strong_wolfe",
                tolerance_grad=float(self.pgtol),
            )

            if margin_m is not None:
                margin_m = margin_m.view(B)

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                x_adv = lb + span * torch.sigmoid(w)

                activations.clear()
                _ = self.model(x_adv)
                phi_adv = activations.pop(0).view(B, -1)

                if objective == "euclidean":
                    diff = phi_adv - phi_t
                    loss = torch.mean(torch.sum(diff * diff, dim=1))
                else:
                    if phi_s is None or margin_m is None:
                        raise RuntimeError("Triplet objective requires phi_s and margin_m")
                    dist_t = torch.norm(phi_adv - phi_t, p=2, dim=1)
                    dist_s = torch.norm(phi_adv - phi_s, p=2, dim=1)
                    triplet = torch.clamp(dist_t - dist_s + margin_m, min=0.0)
                    visual_loss = torch.sum((x_adv - x_source_dev) ** 2, dim=(1, 2, 3))
                    loss = torch.mean(visual_loss + self.lambda_adv * triplet)
                loss.backward()
                return loss

            optimizer.step(closure)

            with torch.no_grad():
                x_adv = lb + span * torch.sigmoid(w)
            out = x_adv.detach()
            if to_cpu:
                out = out.cpu()
            return out
        finally:
            hook_handle.remove()


class _QueryListDataset(torch.utils.data.Dataset):
    def __init__(self, x_batches: List[torch.Tensor], y_batches: List[torch.Tensor]) -> None:
        self.x_batches = x_batches
        self.y_batches = y_batches
        self._cum_sizes: List[int] = []
        total = 0
        for xb in self.x_batches:
            total += int(xb.size(0))
            self._cum_sizes.append(total)
        self._total = total

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_idx = bisect.bisect_right(self._cum_sizes, idx)
        prev = 0 if batch_idx == 0 else self._cum_sizes[batch_idx - 1]
        offset = idx - prev
        return self.x_batches[batch_idx][offset], self.y_batches[batch_idx][offset]


class _PairedIndexDataset(torch.utils.data.Dataset):
    """Dataset that returns (source_img, source_label, target_img) by index lists.

    This allows DataLoader workers to load BOTH source and target images in parallel
    without invoking random target selection in worker processes.
    """

    def __init__(
        self,
        pool_dataset: torch.utils.data.Dataset,
        source_indices: List[int],
        target_indices: List[int],
    ) -> None:
        if len(source_indices) != len(target_indices):
            raise ValueError("source_indices and target_indices must have same length")
        self.pool_dataset = pool_dataset
        self.source_indices = source_indices
        self.target_indices = target_indices

    def __len__(self) -> int:
        return len(self.source_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        s_idx = int(self.source_indices[idx])
        t_idx = int(self.target_indices[idx])
        s_img, s_label = self.pool_dataset[s_idx]
        t_img, _ = self.pool_dataset[t_idx]
        return s_img, int(s_label), t_img


class CloudLeakVGGDeepID(nn.Module):
    """VGG19-based substitute with a DeepID layer (CloudLeak Section III.C).

    The paper removes FC6 and adds a DeepID layer. For benchmark integration we
    implement a VGG19 feature extractor + AdaptiveAvgPool2d(7,7) and a DeepID
    projection (dim=480 by default), followed by a linear classifier.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        input_channels: int,
        deepid_dim: int = 480,
        weights: str = "IMAGENET1K_V1",
    ) -> None:
        super().__init__()

        try:
            from torchvision.models import vgg19, VGG19_Weights
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"torchvision is required for CloudLeakVGGDeepID: {exc}")

        if weights == "IMAGENET1K_V1":
            base = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        else:
            base = vgg19(weights=None)

        # VGG expects 3 channels. If grayscale, we repeat via a 1x1 conv.
        self.pre = None
        if int(input_channels) == 1:
            self.pre = nn.Conv2d(1, 3, kernel_size=1, bias=False)

        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        feat_dim = 512 * 7 * 7

        # DeepID layer (paper: dim=480)
        self.deepid = nn.Linear(feat_dim, int(deepid_dim))
        self.classifier = nn.Linear(int(deepid_dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre is not None:
            x = self.pre(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.deepid(x)
        x = F.relu(x, inplace=True)
        return self.classifier(x)


class CloudLeak(AttackRunner):
    def __init__(self, config: dict, state: BenchmarkState) -> None:
        super().__init__(config, state)

        # Paper uses box-constrained L-BFGS (Section IV.A.3). Use enough iters to converge.
        self.lbfgs_iters = int(config.get("lbfgs_iters", 10))
        self.lbfgs_factr = float(config.get("lbfgs_factr", 1e7))
        self.lbfgs_pgtol = float(config.get("lbfgs_pgtol", 1e-5))

        self.margin_m = float(config.get("margin_m", 0.5))
        self.lambda_adv = float(config.get("lambda_adv", 0.001))
        # Official reference code (optimize.py) uses a per-pixel bound `max_thres` around
        # the base image. In our [0,1] input contract, interpret it as `max_thres/255`.
        max_thres = config.get("max_thres")
        if max_thres is None:
            max_thres = config.get("epsilon")
        if max_thres is None:
            max_thres = 10.0 / 255.0
        self.epsilon = float(max_thres)

        self.num_rounds = resolve_iterations(
            config,
            default=10,
            context="cloudleak",
            allow_num_rounds=True,
        )
        initial_seed_size = config.get("initial_seed_size")
        if initial_seed_size is None:
            initial_seed_size = config.get("initial_pool_size")
        self.initial_seed_size = int(initial_seed_size) if initial_seed_size is not None else None
        # Paper selection scores the available pool; for benchmark fairness and
        # determinism, we always score the full remaining pool rather than an
        # arbitrary candidate subset.

        self.batch_size = int(config.get("batch_size", 64))
        self.gen_batch_size = int(config.get("gen_batch_size", 64))
        self.lr = float(config.get("lr", 0.01))
        self.momentum = float(config.get("momentum", 0.9))
        self.weight_decay = float(config.get("weight_decay", 5e-4))
        self.max_epochs = int(config.get("max_epochs", 200))
        self.patience = int(config.get("patience", 20))

        # Paper Section III.C uses VGG19 modified with a DeepID layer.
        # For benchmark fairness across attacks, pretrained substitutes are disabled by default.
        self.use_pretrained = bool(config.get("use_pretrained", False))

        # If pretrained is enabled but no explicit pretrained arch is provided, default to the
        # configured substitute architecture (keeps behavior consistent with other attacks).
        pretrained_arch = config.get("pretrained_arch")
        if pretrained_arch is None:
            pretrained_arch = (state.metadata.get("substitute_config", {}) or {}).get("arch")
        if pretrained_arch is None:
            pretrained_arch = "resnet18"
        self.pretrained_arch = str(pretrained_arch)
        self.deepid_dim = int(config.get("deepid_dim", 480))
        self.feature_layer = config.get("feature_layer")
        self.featurefool_objective = str(config.get("featurefool_objective", "euclidean"))

        self.pool_dataset = None
        self.featurefool: Optional[FeatureFool] = None
        # Cache per-class computed margins (float) to avoid repeated full-pool scans.
        self._class_feature_cache: Dict[int, float] = {}
        self._class_indices_cache: Dict[int, List[int]] = {}
        num_classes = int(
            state.metadata.get("num_classes")
            or state.metadata.get("dataset_config", {}).get("num_classes", 10)
        )
        self.is_single_logit_binary = is_single_logit_binary_num_classes(num_classes)

        self._initialize_state(state)

    def run(self, ctx: BenchmarkContext) -> None:
        self.victim = ctx.oracle.model

        self._ensure_pool_dataset(self.state)

        self._bootstrap_seed_and_validation_sets(ctx, self.state)

        active_budget = int(ctx.budget_remaining)
        if active_budget <= 0:
            return

        round_size = max(1, int(math.ceil(active_budget / max(self.num_rounds, 1))))

        pbar = self._create_progress_bar(active_budget, "[CloudLeak] Extracting")
        while ctx.budget_remaining > 0:
            pool_indices = self.state.attack_state.get("pool_indices", [])
            if not pool_indices:
                break

            step_size = min(round_size, ctx.budget_remaining, len(pool_indices))

            if step_size <= 0:
                break

            pbar.set_postfix_str("selecting")
            query_batch = self._select_query_batch(step_size, self.state)
            if query_batch.x.size(0) == 0:
                self.logger.warning("CloudLeak query selection returned empty batch. Stopping attack.")
                break

            pbar.set_postfix_str("querying")
            oracle_output = ctx.query(query_batch.x, meta=query_batch.meta)
            pbar.set_postfix_str("training")
            self.observe(query_batch, oracle_output, self.state)
            pbar.update(query_batch.x.size(0))
        pbar.close()

        self.pool_dataset = None
        self.featurefool = None
        self._class_feature_cache = {}
        self._class_indices_cache = {}

    def observe(
        self,
        query_batch: QueryBatch,
        oracle_output: OracleOutput,
        state: BenchmarkState,
    ) -> None:
        self._handle_oracle_output(query_batch, oracle_output, state)

    def _initialize_state(self, state: BenchmarkState) -> None:
        state.attack_state.setdefault("pool_indices", [])
        state.attack_state.setdefault("query_data_x", [])
        state.attack_state.setdefault("query_data_y", [])
        state.attack_state.setdefault("val_query_data_x", [])
        state.attack_state.setdefault("val_query_data_y", [])
        state.attack_state.setdefault("seed_indices", [])
        state.attack_state.setdefault("val_indices", [])
        state.attack_state.setdefault("initial_seed_queried", False)
        state.attack_state.setdefault("validation_built", False)
        state.attack_state.setdefault("synthetic_indices", [])
        state.attack_state.setdefault("substitute", None)
        state.attack_state.setdefault("round", 0)

    def _ensure_pool_dataset(self, state: BenchmarkState) -> None:
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

        pool_indices = state.attack_state.get("pool_indices", [])
        if not pool_indices:
            state.attack_state["pool_indices"] = list(range(len(self.pool_dataset)))
        else:
            state.attack_state["pool_indices"] = [
                i for i in pool_indices if 0 <= i < len(self.pool_dataset)
            ]

    def _bootstrap_seed_and_validation_sets(self, ctx: BenchmarkContext, state: BenchmarkState) -> None:
        total_budget = int(
            state.metadata.get("max_budget")
            or ctx.budget_remaining
        )
        seed_target, val_target = self._resolve_seed_and_validation_targets(
            total_budget=total_budget,
            default_seed_ratio=0.1,
            default_validation_ratio=0.2,
        )

        if self.initial_seed_size is not None:
            seed_target = int(self.initial_seed_size)
        else:
            self.initial_seed_size = int(seed_target)

        device = state.metadata.get("device", "cpu")
        pool_workers = resolve_pool_num_workers(self.config, state.metadata.get("dataset_config", {}))
        loader_kwargs = (
            pool_loader_kwargs(device, {"num_workers": int(pool_workers)})
            if pool_workers is not None
            else pool_loader_kwargs(device)
        )

        if not bool(state.attack_state.get("validation_built", False)) and int(val_target) > 0:
            pool_indices = state.attack_state.get("pool_indices", [])
            val_k = min(int(val_target), int(ctx.budget_remaining), len(pool_indices))
            if val_k > 0:
                val_indices = np.random.choice(pool_indices, size=val_k, replace=False).tolist()
                selected_set = set(int(i) for i in val_indices)
                state.attack_state["pool_indices"] = [i for i in pool_indices if i not in selected_set]
                state.attack_state["val_indices"] = [int(i) for i in val_indices]

                subset = Subset(self.pool_dataset, state.attack_state["val_indices"])
                loader = DataLoader(
                    subset,
                    batch_size=min(self.batch_size, len(state.attack_state["val_indices"])),
                    shuffle=False,
                    **loader_kwargs,
                )
                ptr = 0
                for x_batch, _ in loader:
                    batch_indices = state.attack_state["val_indices"][ptr : ptr + int(x_batch.size(0))]
                    ptr += int(x_batch.size(0))
                    query_batch = QueryBatch(
                        x=x_batch,
                        meta={
                            "indices": batch_indices,
                            "synthetic": False,
                            "is_validation": True,
                            "defer_train": True,
                        },
                    )
                    oracle_output = ctx.query(query_batch.x, meta=query_batch.meta)
                    self.observe(query_batch, oracle_output, state)
            state.attack_state["validation_built"] = True

        queried_seed = False
        if not bool(state.attack_state.get("initial_seed_queried", False)) and int(seed_target) > 0:
            pool_indices = state.attack_state.get("pool_indices", [])
            seed_k = min(int(seed_target), int(ctx.budget_remaining), len(pool_indices))
            if seed_k > 0:
                seed_indices = np.random.choice(pool_indices, size=seed_k, replace=False).tolist()
                selected_set = set(int(i) for i in seed_indices)
                state.attack_state["pool_indices"] = [i for i in pool_indices if i not in selected_set]
                state.attack_state["seed_indices"] = [int(i) for i in seed_indices]

                subset = Subset(self.pool_dataset, state.attack_state["seed_indices"])
                loader = DataLoader(
                    subset,
                    batch_size=min(self.batch_size, len(state.attack_state["seed_indices"])),
                    shuffle=False,
                    **loader_kwargs,
                )
                ptr = 0
                for x_batch, _ in loader:
                    batch_indices = state.attack_state["seed_indices"][ptr : ptr + int(x_batch.size(0))]
                    ptr += int(x_batch.size(0))
                    query_batch = QueryBatch(
                        x=x_batch,
                        meta={
                            "indices": batch_indices,
                            "synthetic": False,
                            "is_validation": False,
                            "is_seed": True,
                            "defer_train": True,
                        },
                    )
                    oracle_output = ctx.query(query_batch.x, meta=query_batch.meta)
                    self.observe(query_batch, oracle_output, state)
                    queried_seed = True
            state.attack_state["initial_seed_queried"] = True

        if queried_seed:
            self.train_substitute(state)

    def _ensure_substitute(self, state: BenchmarkState) -> nn.Module:
        substitute = state.attack_state.get("substitute")
        if substitute is not None:
            device = state.metadata.get("device", "cpu")
            return substitute.to(device)

        device = state.metadata.get("device", "cpu")
        num_classes = int(
            state.metadata.get("num_classes")
            or state.metadata.get("dataset_config", {}).get("num_classes", 10)
        )
        input_shape = state.metadata.get("input_shape", (3, 32, 32))
        input_channels = int(input_shape[0])

        substitute = None
        if self.use_pretrained:
            substitute = self._build_pretrained_substitute(
                num_classes=num_classes,
                input_channels=input_channels,
                input_size=input_shape[1:],
                device=device,
            )
            if substitute is None:
                self.logger.warning("Falling back to non-pretrained substitute for CloudLeak.")

        if substitute is None:
            sub_config = state.metadata.get("substitute_config", {})
            substitute = create_substitute(
                arch=sub_config.get("arch", "resnet18"),
                num_classes=num_classes,
                input_channels=input_channels,
                width_mult=int(sub_config.get("width_mult", 1)),
                dropout_prob=float(sub_config.get("dropout_prob", 0.0)),
            ).to(device)

        state.attack_state["substitute"] = substitute
        return substitute

    def _build_pretrained_substitute(
        self,
        num_classes: int,
        input_channels: int,
        input_size: Tuple[int, int],
        device: str,
    ) -> Optional[nn.Module]:
        if input_channels not in (1, 3):
            return None

        if self.pretrained_arch == "resnet18":
            try:
                from torchvision.models import resnet18, ResNet18_Weights
            except Exception as exc:
                self.logger.warning("torchvision unavailable for pretrained CloudLeak: %s", exc)
                return None

            base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            base.fc = nn.Linear(base.fc.in_features, num_classes)

            pre_layers: List[nn.Module] = []

            named_pre_layers: list[tuple[str, nn.Module]] = []
            if input_channels == 1:
                # Trainable input adapter for grayscale -> RGB.
                named_pre_layers.append(("pre", nn.Conv2d(1, 3, kernel_size=1, bias=False)))

            if input_size != (224, 224):
                named_pre_layers.append(
                    ("resize", nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False))
                )

            if named_pre_layers:
                named_pre_layers.append(("backbone", base))
                model = nn.Sequential(OrderedDict(named_pre_layers))
            else:
                model = base

            return model.to(device)

        if self.pretrained_arch == "vgg19_deepid":
            # DeepID substitute defined in CloudLeak Section III.C.
            model = CloudLeakVGGDeepID(
                num_classes=num_classes,
                input_channels=input_channels,
                deepid_dim=self.deepid_dim,
            ).to(device)

            # VGG19 weights are trained at 224x224; resize inputs accordingly.
            pre_layers: List[nn.Module] = []
            if input_size != (224, 224):
                pre_layers.append(
                    nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False)
                )
            if pre_layers:
                return nn.Sequential(*pre_layers, model).to(device)
            return model

        return None

    def _ensure_featurefool(self, substitute: nn.Module, device: str) -> None:
        if self.featurefool is None or getattr(self.featurefool, "model", None) is not substitute:
            if self.feature_layer is None and self.pretrained_arch == "vgg19_deepid":
                self.feature_layer = "deepid"
            self.featurefool = FeatureFool(
                substitute,
                margin_m=self.margin_m,
                lambda_adv=self.lambda_adv,
                objective=self.featurefool_objective,
                max_iters=self.lbfgs_iters,
                epsilon=self.epsilon,
                factr=self.lbfgs_factr,
                pgtol=self.lbfgs_pgtol,
                device=device,
                feature_layer=self.feature_layer,
            )

    def _select_query_batch(self, k: int, state: BenchmarkState) -> QueryBatch:
        self._ensure_pool_dataset(state)
        pool_indices = state.attack_state.get("pool_indices", [])

        if k <= 0 or not pool_indices:
            return QueryBatch(
                x=torch.empty(0, *self.state.metadata.get("input_shape", (3, 32, 32))),
                meta={"indices": [], "status": "exhausted"},
            )

        k = min(k, len(pool_indices))
        substitute = self._ensure_substitute(state)
        device = state.metadata.get("device", "cpu")

        self._ensure_featurefool(substitute, device)
        self._build_class_indices()

        # Always score the full remaining pool.
        candidate_indices = list(pool_indices)

        scored = self._generate_and_score(candidate_indices, substitute, device)
        if not scored:
            return self._fallback_random_selection(k, pool_indices, state)

        scored.sort(key=lambda t: t[0], reverse=True)
        top_scored = scored[:k]

        selected_indices = [int(idx) for _, idx, _ in top_scored]
        x_list = [img for _, _, img in top_scored]

        selected_set = set(selected_indices)
        state.attack_state["pool_indices"] = [i for i in pool_indices if i not in selected_set]
        state.attack_state["round"] = int(state.attack_state.get("round", 0)) + 1

        x_batch = torch.stack(x_list)
        meta = {"indices": selected_indices, "synthetic": True}
        return QueryBatch(x=x_batch, meta=meta)

    def _fallback_random_selection(
        self,
        k: int,
        pool_indices: List[int],
        state: BenchmarkState,
    ) -> QueryBatch:
        if not pool_indices or k <= 0:
            return QueryBatch(
                x=torch.empty(0, *self.state.metadata.get("input_shape", (3, 32, 32))),
                meta={"indices": [], "status": "exhausted"},
            )

        selected = np.random.choice(pool_indices, min(k, len(pool_indices)), replace=False).tolist()
        x_list = [self.pool_dataset[idx][0] for idx in selected]
        selected_set = set(int(i) for i in selected)
        state.attack_state["pool_indices"] = [i for i in pool_indices if i not in selected_set]
        state.attack_state["round"] = int(state.attack_state.get("round", 0)) + 1

        x_batch = torch.stack(x_list)
        meta = {"indices": selected, "synthetic": False}
        return QueryBatch(x=x_batch, meta=meta)

    def _build_class_indices(self) -> None:
        if self._class_indices_cache or self.pool_dataset is None:
            return

        targets = self._try_get_pool_targets()
        if targets is not None:
            for idx, y in enumerate(targets):
                self._class_indices_cache.setdefault(int(y), []).append(int(idx))
            return

        # Fallback: scan via DataLoader (may decode images for some datasets).
        device = self.state.metadata.get("device", "cpu")
        pool_workers = resolve_pool_num_workers(self.config, self.state.metadata.get("dataset_config", {}))
        loader_kwargs = (
            pool_loader_kwargs(device, {"num_workers": int(pool_workers)})
            if pool_workers is not None
            else pool_loader_kwargs(device)
        )
        loader = DataLoader(
            self.pool_dataset,
            batch_size=256,
            shuffle=False,
            **loader_kwargs,
        )
        ptr = 0
        for _, y_batch in loader:
            y_batch = y_batch.view(-1)
            for y in y_batch:
                label = int(y.item())
                self._class_indices_cache.setdefault(label, []).append(ptr)
                ptr += 1

    def _try_get_pool_targets(self) -> Optional[List[int]]:
        """Best-effort label extraction without decoding images.

        For ImageNet surrogate (ImageFolder) and most torchvision datasets, labels
        are accessible via `.targets` / `.samples` on the underlying dataset.
        """

        if self.pool_dataset is None:
            return None

        ds: Any = self.pool_dataset
        if hasattr(ds, "dataset"):
            ds = ds.dataset

        # Unwrap Subset
        indices = None
        if isinstance(ds, Subset):
            indices = list(ds.indices)
            ds = ds.dataset

        # torchvision datasets commonly expose `.targets`
        if hasattr(ds, "targets"):
            base_targets = ds.targets
            try:
                base_targets_list = list(base_targets)
            except TypeError:
                # Some dataset-like objects (e.g., mocks) may define `.targets` but
                # are not actually iterable.
                return None
            if indices is None:
                return [int(t) for t in base_targets_list]
            return [int(base_targets_list[i]) for i in indices]

        # ImageFolder exposes `.samples` (list[(path, class_index)])
        if hasattr(ds, "samples"):
            samples = ds.samples
            try:
                samples_list = list(samples)
            except TypeError:
                return None
            if indices is None:
                return [int(s[1]) for s in samples_list]
            return [int(samples_list[i][1]) for i in indices]

        return None

    def _generate_and_score(
        self,
        candidate_indices: List[int],
        substitute: nn.Module,
        device: str,
    ) -> List[Tuple[float, int, torch.Tensor]]:
        if not candidate_indices:
            return []

        pool_workers = resolve_pool_num_workers(self.config, self.state.metadata.get("dataset_config", {}))
        loader_kwargs = (
            pool_loader_kwargs(device, {"num_workers": int(pool_workers)})
            if pool_workers is not None
            else pool_loader_kwargs(device)
        )
        # Precompute per-candidate target indices in the SAME order as candidates to
        # preserve RNG consumption while allowing workers to load both images.
        targets = self._try_get_pool_targets()
        if targets is None:
            # Fallback: will decode source images later; keep legacy behavior.
            targets = []

        target_indices: List[int] = []
        for s_idx in candidate_indices:
            if targets:
                source_label = int(targets[int(s_idx)])
            else:
                # Worst-case fallback: decode label.
                _, y = self.pool_dataset[int(s_idx)]
                source_label = int(y)
            target_indices.append(self._select_target_index(source_label, int(s_idx)))

        paired = _PairedIndexDataset(self.pool_dataset, candidate_indices, target_indices)
        loader = DataLoader(
            paired,
            batch_size=self.gen_batch_size,
            shuffle=False,
            **loader_kwargs,
        )

        scored: List[Tuple[float, int, torch.Tensor]] = []
        ptr = 0
        score_pbar = tqdm(
            total=len(candidate_indices),
            desc="[CloudLeak] Scoring Candidates",
            leave=False,
        )

        substitute.eval()
        non_blocking = str(device).startswith("cuda")
        try:
            for s_imgs, s_labels, t_imgs in loader:
                batch_len = int(s_imgs.size(0))
                curr_indices = candidate_indices[ptr : ptr + batch_len]
                ptr += batch_len

                # Extract labels once to avoid per-element `.item()` overhead.
                labels_list = [int(v) for v in s_labels.view(-1).detach().cpu().tolist()]

                margin_m = None
                if self.featurefool_objective.lower().strip() == "triplet":
                    # Triplet objective requires per-sample margins.
                    unique_labels = sorted(set(labels_list))
                    margin_by_label = {
                        lab: self._compute_margin_m(lab, device) for lab in unique_labels
                    }
                    margin_m = torch.tensor(
                        [margin_by_label[lab] for lab in labels_list], device=device
                    )

                s_imgs_adv = self.featurefool.generate_batch(
                    s_imgs.to(device, non_blocking=non_blocking),
                    t_imgs.to(device, non_blocking=non_blocking),
                    margin_m=margin_m,
                    to_cpu=False,
                )

                with torch.inference_mode():
                    logits = substitute(s_imgs_adv)
                    if self.is_single_logit_binary:
                        prob_pos = binary_positive_probs_from_logits(logits)[:, 0]
                        scores = 1.0 - torch.maximum(prob_pos, 1.0 - prob_pos)
                    else:
                        probs = F.softmax(logits, dim=1)
                        max_prob, _ = probs.max(dim=1)
                        scores = 1.0 - max_prob

                # Avoid per-element CUDA sync (`.item()`) and per-element D2H copies.
                scores_list = scores.detach().cpu().tolist()
                adv_cpu = s_imgs_adv.detach().cpu()
                for j, idx in enumerate(curr_indices):
                    scored.append((float(scores_list[j]), int(idx), adv_cpu[j]))

                score_pbar.update(batch_len)
        finally:
            score_pbar.close()

        return scored

    def _select_target_index(self, source_label: int, source_idx: int) -> int:
        if not self._class_indices_cache:
            if len(self.pool_dataset) > 1:
                choices = list(range(len(self.pool_dataset)))
                return int(np.random.choice([i for i in choices if i != source_idx]))
            return source_idx

        labels = list(self._class_indices_cache.keys())
        if len(labels) <= 1:
            choices = self._class_indices_cache.get(labels[0], []) if labels else []
            if len(choices) > 1:
                return int(np.random.choice([i for i in choices if i != source_idx]))
            return source_idx

        other_labels = [l for l in labels if l != source_label]
        target_label = int(np.random.choice(other_labels))
        target_indices = self._class_indices_cache.get(target_label, [])
        if not target_indices:
            return source_idx
        return int(np.random.choice(target_indices))

    def _compute_margin_m(self, class_id: int, device: str) -> float:
        if class_id in self._class_feature_cache:
            return float(self._class_feature_cache[class_id])

        if self.pool_dataset is None or self.featurefool is None:
            return float(self.margin_m)

        class_indices = self._class_indices_cache.get(class_id, [])
        if len(class_indices) < 2:
            return float(self.margin_m)

        subset = Subset(self.pool_dataset, class_indices)
        pool_workers = resolve_pool_num_workers(self.config, self.state.metadata.get("dataset_config", {}))
        loader_kwargs = (
            pool_loader_kwargs(device, {"num_workers": int(pool_workers)})
            if pool_workers is not None
            else pool_loader_kwargs(device)
        )
        loader = DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=False,
            **loader_kwargs,
        )

        sum_feats: Optional[torch.Tensor] = None
        sum_norms_t = torch.zeros((), device=device)
        count = 0

        with torch.no_grad():
            for x_batch, _ in loader:
                feats = self.featurefool._extract_features(
                    x_batch.to(device, non_blocking=str(device).startswith("cuda"))
                )
                if sum_feats is None:
                    sum_feats = feats.sum(dim=0)
                else:
                    sum_feats = sum_feats + feats.sum(dim=0)
                sum_norms_t = sum_norms_t + (feats ** 2).sum()
                count += int(feats.size(0))

        if sum_feats is None or count < 2:
            return float(self.margin_m)

        sum_feats_norm = (sum_feats ** 2).sum()
        mean_sq_dist = (2.0 * count * sum_norms_t - 2.0 * sum_feats_norm) / (count * (count - 1))
        alpha = 0.5
        margin_m_t = alpha - mean_sq_dist
        margin_m = float(margin_m_t.detach().cpu().item())
        self._class_feature_cache[class_id] = margin_m
        return margin_m

    def _handle_oracle_output(
        self,
        query_batch: QueryBatch,
        oracle_output: OracleOutput,
        state: BenchmarkState,
    ) -> None:
        meta = query_batch.meta or {}
        x_cpu = query_batch.x.detach().cpu()
        y_cpu = oracle_output.y.detach().cpu()

        if bool(meta.get("is_validation", False)):
            state.attack_state["val_query_data_x"].append(x_cpu)
            state.attack_state["val_query_data_y"].append(y_cpu)
            return

        state.attack_state["query_data_x"].append(x_cpu)
        state.attack_state["query_data_y"].append(y_cpu)

        if meta.get("synthetic", False):
            indices = meta.get("indices", [])
            state.attack_state["synthetic_indices"].extend(indices)

        if bool(meta.get("defer_train", False)):
            return

        self.train_substitute(state)

    def train_substitute(self, state: BenchmarkState) -> None:
        query_data_x = state.attack_state.get("query_data_x", [])
        query_data_y = state.attack_state.get("query_data_y", [])
        val_query_x = state.attack_state.get("val_query_data_x", [])
        val_query_y = state.attack_state.get("val_query_data_y", [])

        if len(query_data_x) == 0:
            return

        dataset = _QueryListDataset(query_data_x, query_data_y)
        if len(dataset) < 2:
            return

        sub_config = state.metadata.get("substitute_config", {})
        train_batch_size = int(
            sub_config.get("batch_size")
            or self.batch_size
        )

        if len(val_query_x) > 0 and len(val_query_y) > 0:
            x_val = torch.cat(val_query_x, dim=0)
            y_val = torch.cat(val_query_y, dim=0)
            train_subset = dataset
            val_subset = _QueryListDataset([x_val], [y_val])
            train_size = len(train_subset)
        else:
            total_size = len(dataset)
            val_size = max(1, int(0.2 * total_size))
            train_size = total_size - val_size
            if train_size < 1:
                return

            train_subset, val_subset = torch.utils.data.random_split(
                dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )

        device = state.metadata.get("device", "cpu")
        train_workers = resolve_train_num_workers(sub_config, self.config, default=0)
        val_workers = resolve_val_num_workers(sub_config, self.config, default=train_workers)

        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=train_batch_size,
            shuffle=True,
            **pool_loader_kwargs(device, {"num_workers": int(train_workers)}),
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=train_batch_size,
            shuffle=False,
            **pool_loader_kwargs(device, {"num_workers": int(val_workers)}),
        )

        substitute = self._ensure_substitute(state)
        substitute = substitute.to(device)

        if self.use_pretrained:
            self._freeze_backbone(substitute)
        else:
            for param in substitute.parameters():
                param.requires_grad = True

        # [FIX] Unify validation metric to Validation Loss (CrossEntropy or KL-Div)
        # Previously used F1 Score, but we now standardize across all attacks.
        def eval_fn(model: nn.Module, loader: DataLoader) -> float:
            model.eval()
            total_loss = 0.0
            total_count = 0
            
            # Determine loss function based on targets shape/type
            # CloudLeak targets can be soft (KL) or hard (CE)
            # We inspect the first batch to decide
            first_y = next(iter(loader))[1]
            if first_y.ndim == 2:
                loss_func = nn.KLDivLoss(reduction="batchmean")
            else:
                loss_func = nn.CrossEntropyLoss()
                
            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)

                    if self.is_single_logit_binary:
                        loss = binary_bce_loss(outputs, y)
                    elif first_y.ndim == 2: # Soft labels
                        log_probs = F.log_softmax(outputs, dim=1)
                        loss = loss_func(log_probs, y)
                    else: # Hard labels
                        loss = loss_func(outputs, y.long())
                        
                    total_loss += loss.item() * x.size(0)
                    total_count += x.size(0)
            return total_loss / total_count if total_count > 0 else float('inf')

        batch_size = max(1, int(train_batch_size))
        steps_per_epoch = max(1, int(math.ceil(train_size / batch_size)))
        max_epochs = int(sub_config.get("max_epochs", self.max_epochs))
        patience_epochs = int(sub_config.get("patience", self.patience))

        trainer = SubstituteTrainer(dict(sub_config), device=device)
        request = TrainRequest(
            model=substitute,
            train_loader=train_loader,
            val_loader=val_loader,
            eval_fn=eval_fn,
            loss_fn=self._compute_training_loss,
            max_steps=max_epochs * steps_per_epoch,
            validate_every=steps_per_epoch,
            patience=patience_epochs * steps_per_epoch,
            early_stop_mode="min",     # Minimizing Validation Loss
            load_best=True,
        )
        trainer.train(request)

        state.attack_state["substitute"] = substitute
        self._evaluate_current_substitute(substitute, device)

        substitute = substitute.cpu()
        state.attack_state["substitute"] = substitute
        self._class_feature_cache = {}
        self.featurefool = None
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    def _freeze_backbone(self, model: nn.Module) -> None:
        # For pretrained substitutes, keep the backbone frozen but allow training for:
        # - classifier heads (fc/classifier)
        # - CloudLeak-specific projection layer (deepid)
        # - input adapters (pre) inserted for grayscale compatibility
        head_keywords = ["fc", "classifier", "last_linear", "deepid", "pre"]
        for name, param in model.named_parameters():
            if not any(k in name for k in head_keywords):
                param.requires_grad = False
            else:
                param.requires_grad = True

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            all_params = list(model.parameters())
            if all_params:
                all_params[-1].requires_grad = True
                if len(all_params) > 1:
                    all_params[-2].requires_grad = True

    def _compute_training_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.is_single_logit_binary:
            return binary_bce_loss(outputs, targets)
        if targets.ndim == 2:
            log_probs = F.log_softmax(outputs, dim=1)
            return F.kl_div(log_probs, targets, reduction="batchmean")

        return F.cross_entropy(outputs, targets.long())

    def _compute_f1(self, model: nn.Module, val_loader: DataLoader, device: str) -> float:
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                outputs = model(x_batch)
                if self.is_single_logit_binary:
                    preds = binary_hard_labels_from_logits(outputs).cpu().numpy()
                else:
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)

                if self.is_single_logit_binary:
                    targets = binary_hard_targets(y_batch).cpu().numpy()
                elif y_batch.ndim > 1:
                    targets = torch.argmax(y_batch, dim=1).cpu().numpy()
                else:
                    targets = y_batch.cpu().numpy()
                all_targets.extend(targets)

        if not all_targets:
            return 0.0
        try:
            from sklearn.metrics import f1_score
        except Exception:
            return 0.0

        return float(f1_score(all_targets, all_preds, average="macro"))
