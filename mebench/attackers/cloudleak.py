"""CloudLeak attack implementation."""

from typing import Dict, Any, List, Optional, Tuple
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
from mebench.models.substitute_factory import create_substitute


class FeatureFool:
    def __init__(
        self,
        model: nn.Module,
        margin_m: float = 0.5,
        lambda_adv: float = 0.001,
        max_iters: int = 10,
        epsilon: float = 8.0 / 255.0,
        factr: float = 1e7,
        pgtol: float = 1e-5,
        device: str = "cpu",
        feature_layer: Optional[str] = None,
    ) -> None:
        self.model = model.eval()
        self.margin_m = margin_m
        self.lambda_adv = lambda_adv
        self.max_iters = max_iters
        self.epsilon = epsilon
        self.factr = factr
        self.pgtol = pgtol
        self.device = device
        self.feature_layer_name = feature_layer
        self._feature_layer = self._get_feature_layer(model)

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

        last_linear = None
        for module in model.modules():
            if isinstance(module, nn.Linear):
                last_linear = module
        if last_linear is None:
            raise ValueError("FeatureFool requires a Linear layer for feature extraction")
        return last_linear

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
                _ = self.model(x_source_dev)
                phi_s = activations.pop(0).detach().view(B, -1)

                activations.clear()
                _ = self.model(x_target_dev)
                phi_t = activations.pop(0).detach().view(B, -1)

            delta = torch.zeros_like(x_source_dev, requires_grad=True)
            optimizer = torch.optim.LBFGS(
                [delta],
                lr=1.0,
                max_iter=self.max_iters,
                history_size=10,
                line_search_fn="strong_wolfe",
            )

            margin_m = margin_m.view(B, 1)

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                delta_clamped = torch.clamp(delta, -self.epsilon, self.epsilon)
                x_adv = torch.clamp(x_source_dev + delta_clamped, 0.0, 1.0)

                activations.clear()
                _ = self.model(x_adv)
                phi_adv = activations.pop(0).view(B, -1)

                dist_t = torch.norm(phi_adv - phi_t, p=2, dim=1).view(B, 1)
                dist_s = torch.norm(phi_adv - phi_s, p=2, dim=1).view(B, 1)
                triplet = torch.clamp(dist_t - dist_s + margin_m, min=0.0)
                visual_loss = torch.sum(delta_clamped ** 2, dim=(1, 2, 3)).view(B, 1)
                loss = torch.mean(visual_loss + self.lambda_adv * triplet)
                loss.backward()
                return loss

            optimizer.step(closure)

            with torch.no_grad():
                delta_final = torch.clamp(delta, -self.epsilon, self.epsilon)
                x_adv = torch.clamp(x_source_dev + delta_final, 0.0, 1.0)
            return x_adv.detach().cpu()
        finally:
            hook_handle.remove()


class CloudLeak(AttackRunner):
    def __init__(self, config: dict, state: BenchmarkState) -> None:
        super().__init__(config, state)

        self.lbfgs_iters = int(config.get("lbfgs_iters", 10))
        self.lbfgs_factr = float(config.get("lbfgs_factr", 1e7))
        self.lbfgs_pgtol = float(config.get("lbfgs_pgtol", 1e-5))

        self.margin_m = float(config.get("margin_m", 0.5))
        self.lambda_adv = float(config.get("lambda_adv", 0.001))
        self.epsilon = float(config.get("epsilon", 8.0 / 255.0))

        self.num_rounds = int(config.get("num_rounds", 10))
        self.initial_pool_size = int(config.get("initial_pool_size", 0))
        self.candidate_ratio = float(config.get("candidate_ratio", 5.0))
        self.use_full_pool = bool(config.get("use_full_pool", False))

        self.batch_size = int(config.get("batch_size", 64))
        self.gen_batch_size = int(config.get("gen_batch_size", 64))
        self.lr = float(config.get("lr", 0.01))
        self.momentum = float(config.get("momentum", 0.9))
        self.weight_decay = float(config.get("weight_decay", 5e-4))
        self.max_epochs = int(config.get("max_epochs", 1000))
        self.patience = int(config.get("patience", 100))

        self.use_pretrained = bool(config.get("use_pretrained", False))
        self.pretrained_arch = str(config.get("pretrained_arch", "resnet18"))
        self.feature_layer = config.get("feature_layer")

        self.pool_dataset = None
        self.featurefool: Optional[FeatureFool] = None
        self._class_feature_cache: Dict[int, torch.Tensor] = {}
        self._class_indices_cache: Dict[int, List[int]] = {}

        self._initialize_state(state)

    def run(self, ctx: BenchmarkContext) -> None:
        self.victim = ctx.oracle.model
        device = self.state.metadata.get("device", "cpu")

        self._ensure_pool_dataset(self.state)

        total_budget = int(
            self.state.metadata.get("max_budget")
            or self.config.get("max_budget", ctx.budget_remaining)
        )
        round_size = max(1, int(math.ceil(total_budget / max(self.num_rounds, 1))))
        if self.initial_pool_size <= 0:
            self.initial_pool_size = max(1, int(0.1 * total_budget))

        pbar = self._create_progress_bar(total_budget, "[CloudLeak] Extracting")
        while ctx.budget_remaining > 0:
            pool_indices = self.state.attack_state.get("pool_indices", [])
            if not pool_indices:
                break

            current_queries = total_budget - ctx.budget_remaining
            if current_queries == 0:
                step_size = min(self.initial_pool_size, ctx.budget_remaining, len(pool_indices))
            else:
                step_size = min(round_size, ctx.budget_remaining, len(pool_indices))

            if step_size <= 0:
                break

            query_batch = self._select_query_batch(step_size, self.state)
            if query_batch.x.size(0) == 0:
                self.logger.warning("CloudLeak query selection returned empty batch. Stopping attack.")
                break

            oracle_output = ctx.query(query_batch.x, meta=query_batch.meta)
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

    def _ensure_substitute(self, state: BenchmarkState) -> nn.Module:
        substitute = state.attack_state.get("substitute")
        if substitute is not None:
            return substitute

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

        if self.pretrained_arch != "resnet18":
            return None

        try:
            from torchvision.models import resnet18, ResNet18_Weights
        except Exception as exc:
            self.logger.warning("torchvision unavailable for pretrained CloudLeak: %s", exc)
            return None

        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        base.fc = nn.Linear(base.fc.in_features, num_classes)

        pre_layers: List[nn.Module] = []
        if input_channels == 1:
            pre_layers.append(nn.Conv2d(1, 3, kernel_size=1, bias=False))

        if input_size != (224, 224):
            pre_layers.append(nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False))

        if pre_layers:
            model = nn.Sequential(*pre_layers, base)
        else:
            model = base

        return model.to(device)

    def _ensure_featurefool(self, substitute: nn.Module, device: str) -> None:
        if self.featurefool is None:
            self.featurefool = FeatureFool(
                substitute,
                margin_m=self.margin_m,
                lambda_adv=self.lambda_adv,
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

        if self.use_full_pool:
            candidate_indices = list(pool_indices)
        else:
            candidate_size = min(len(pool_indices), max(k, int(self.candidate_ratio * k)))
            candidate_indices = np.random.choice(pool_indices, candidate_size, replace=False).tolist()

        scored = self._generate_and_score(candidate_indices, substitute, device)
        if not scored:
            return self._fallback_random_selection(k, pool_indices, state)

        scored.sort(key=lambda t: t[0], reverse=True)
        top_scored = scored[:k]

        selected_indices = [int(idx) for _, idx, _ in top_scored]
        x_list = [img for _, _, img in top_scored]

        state.attack_state["pool_indices"] = [i for i in pool_indices if i not in selected_indices]
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
        state.attack_state["pool_indices"] = [i for i in pool_indices if i not in selected]
        state.attack_state["round"] = int(state.attack_state.get("round", 0)) + 1

        x_batch = torch.stack(x_list)
        meta = {"indices": selected, "synthetic": False}
        return QueryBatch(x=x_batch, meta=meta)

    def _build_class_indices(self) -> None:
        if self._class_indices_cache or self.pool_dataset is None:
            return

        loader = DataLoader(self.pool_dataset, batch_size=256, shuffle=False, num_workers=0)
        ptr = 0
        for _, y_batch in loader:
            y_batch = y_batch.view(-1)
            for y in y_batch:
                label = int(y.item())
                self._class_indices_cache.setdefault(label, []).append(ptr)
                ptr += 1

    def _generate_and_score(
        self,
        candidate_indices: List[int],
        substitute: nn.Module,
        device: str,
    ) -> List[Tuple[float, int, torch.Tensor]]:
        if not candidate_indices:
            return []

        subset = Subset(self.pool_dataset, candidate_indices)
        loader = DataLoader(subset, batch_size=self.gen_batch_size, shuffle=False, num_workers=0)

        scored: List[Tuple[float, int, torch.Tensor]] = []
        ptr = 0

        substitute.eval()
        for s_imgs, s_labels in loader:
            batch_len = s_imgs.size(0)
            curr_indices = candidate_indices[ptr : ptr + batch_len]
            ptr += batch_len

            t_imgs_list = []
            for k_idx, s_idx in enumerate(curr_indices):
                source_label = int(s_labels[k_idx].item())
                target_idx = self._select_target_index(source_label, s_idx)
                t_img, _ = self.pool_dataset[target_idx]
                t_imgs_list.append(t_img)

            t_imgs = torch.stack(t_imgs_list)
            margins = [self._compute_margin_m(int(s_labels[i].item()), device) for i in range(batch_len)]
            margin_m = torch.tensor(margins, device=device)

            s_imgs_adv = self.featurefool.generate_batch(
                s_imgs.to(device),
                t_imgs.to(device),
                margin_m=margin_m,
            )

            with torch.no_grad():
                logits = substitute(s_imgs_adv.to(device))
                probs = F.softmax(logits, dim=1)
                max_prob, _ = probs.max(dim=1)
                scores = 1.0 - max_prob

            for j in range(s_imgs_adv.size(0)):
                scored.append((float(scores[j].item()), int(curr_indices[j]), s_imgs_adv[j].detach().cpu()))

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
        loader = DataLoader(subset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        sum_feats = None
        sum_norms = 0.0
        count = 0

        with torch.no_grad():
            for x_batch, _ in loader:
                feats = self.featurefool._extract_features(x_batch.to(device))
                if sum_feats is None:
                    sum_feats = feats.sum(dim=0)
                else:
                    sum_feats = sum_feats + feats.sum(dim=0)
                sum_norms += float((feats ** 2).sum().item())
                count += feats.size(0)

        if sum_feats is None or count < 2:
            return float(self.margin_m)

        sum_feats_norm = float((sum_feats ** 2).sum().item())
        mean_sq_dist = (2.0 * count * sum_norms - 2.0 * sum_feats_norm) / (count * (count - 1))
        alpha = 0.5
        margin_m = alpha - mean_sq_dist
        self._class_feature_cache[class_id] = torch.tensor(margin_m)
        return float(margin_m)

    def _handle_oracle_output(
        self,
        query_batch: QueryBatch,
        oracle_output: OracleOutput,
        state: BenchmarkState,
    ) -> None:
        state.attack_state["query_data_x"].append(query_batch.x.cpu())
        state.attack_state["query_data_y"].append(oracle_output.y.cpu())

        if query_batch.meta.get("synthetic", False):
            indices = query_batch.meta.get("indices", [])
            state.attack_state["synthetic_indices"].extend(indices)

        self.train_substitute(state)

    def train_substitute(self, state: BenchmarkState) -> None:
        query_data_x = state.attack_state.get("query_data_x", [])
        query_data_y = state.attack_state.get("query_data_y", [])

        if len(query_data_x) == 0:
            return

        class QueryListDataset(torch.utils.data.Dataset):
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

            def __getitem__(self, idx: int):
                batch_idx = bisect.bisect_right(self._cum_sizes, idx)
                prev = 0 if batch_idx == 0 else self._cum_sizes[batch_idx - 1]
                offset = idx - prev
                return self.x_batches[batch_idx][offset], self.y_batches[batch_idx][offset]

        dataset = QueryListDataset(query_data_x, query_data_y)
        if len(dataset) < 2:
            return

        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        device = state.metadata.get("device", "cpu")
        substitute = self._ensure_substitute(state)
        substitute = substitute.to(device)

        if self.use_pretrained:
            self._freeze_backbone(substitute)
        else:
            for param in substitute.parameters():
                param.requires_grad = True

        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, substitute.parameters()),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        for _ in tqdm(range(self.max_epochs), desc="[CloudLeak] Training Substitute", leave=False):
            substitute.train()
            epoch_loss = 0.0
            batch_count = 0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                outputs = substitute(x_batch)

                loss = self._compute_training_loss(outputs, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                batch_count += 1

            if batch_count == 0:
                break

            avg_loss = epoch_loss / batch_count
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_state = {k: v.cpu() for k, v in substitute.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        if best_state is not None:
            substitute.load_state_dict(best_state)

        state.attack_state["substitute"] = substitute
        self._evaluate_current_substitute(substitute, device)

        self._class_feature_cache = {}
        self.featurefool = None

    def _freeze_backbone(self, model: nn.Module) -> None:
        head_keywords = ["fc", "classifier", "last_linear"]
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
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)

                if y_batch.ndim > 1:
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
