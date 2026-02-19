"""InverseNet attack implementation."""

from typing import Dict, Any, List, Tuple, Optional
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
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
from mebench.models.inversion import InversionGenerator
from mebench.utils.adversarial import deepfool_distance_vectorized


class InverseNet(AttackRunner):
    """InverseNet with minimal inversion and retraining pipeline."""

    def __init__(self, config: dict, state: BenchmarkState):
        super().__init__(config, state)

        self.batch_size = int(config.get("batch_size", 128))
        self.num_classes = int(
            state.metadata.get("num_classes")
            or config.get("num_classes")
            or state.metadata.get("dataset_config", {}).get("num_classes", 10)
        )
        self.inversion_lr = float(config.get("inversion_lr", 1e-3))
        self.substitute_lr = float(config.get("substitute_lr", 0.01))
        self.inversion_epochs = int(config.get("inversion_epochs", 5))
        self.substitute_epochs = int(config.get("substitute_epochs", 5))
        self.phase_ratios = config.get("phase_ratios", [0.45, 0.45, 0.1])
        # Paper: train twice only (end of phase 1, end of phase 3)
        self.train_phase_1 = False
        self.train_phase_3 = False
        
        # Paper commonly uses top-1 truncation.
        self.truncation_k = int(config.get("truncation_k", 1))
        self.coreset_seed = int(config.get("coreset_seed", 20))
        self.hcss_xi = float(config.get("hcss_xi", 0.02))
        self.hcss_max_iter = int(config.get("hcss_max_iter", 20))

        self.pool_dataset = None
        self.pool_data: torch.Tensor | None = None
        self.inversion_model: nn.Module | None = None
        self.inversion_optimizer: torch.optim.Optimizer | None = None
        self.substitute: nn.Module | None = None
        self.substitute_optimizer: torch.optim.Optimizer | None = None

        self._initialize_state(state)

        # Runtime caches (protocol-preserving)
        self._phase3_templates: Dict[str, torch.Tensor] = {}
        self._inversion_aug = None

    def run(self, ctx: BenchmarkContext) -> None:
        self.victim = ctx.oracle.model
        device = self.state.metadata.get("device", "cpu")
        total_budget = self.state.budget_remaining
        pbar = tqdm(total=total_budget, desc="[InverseNet] Extracting")
        
        while ctx.budget_remaining > 0:
            step_size = self._default_step_size(ctx)
            query_batch = self._select_query_batch(step_size, self.state)
            if int(query_batch.x.size(0)) == 0:
                break
            oracle_output = ctx.query(query_batch.x, meta=query_batch.meta)
            self._handle_oracle_output(query_batch.x, query_batch.meta, oracle_output, self.state)
            pbar.update(query_batch.x.size(0))
        pbar.close()

        # Paper-faithful training schedule:
        # - Train the initial substitute once after Phase 1 (K1).
        # - Retrain the substitute once after Phase 3 (K3).
        if not self.train_phase_1 and len(self.state.attack_state.get("query_data_x", [])) > 0:
            self._train_substitute_from_queries(self.state)
            if self.substitute is not None:
                self.train_phase_1 = True
                self.state.attack_state["substitute"] = self.substitute
                self._evaluate_current_substitute(self.substitute, device)

        if not self.train_phase_3 and len(self.state.attack_state.get("retrain_x", [])) > 0:
            self._retrain_substitute_from_inverse_queries(self.state)
            if self.substitute is not None:
                self.train_phase_3 = True
                self.state.attack_state["substitute"] = self.substitute
                self._evaluate_current_substitute(self.substitute, device)

    def _select_query_batch(self, k: int, state: BenchmarkState) -> QueryBatch:
        if self.pool_dataset is None and self.pool_data is None:
            self._load_pool(state)

        total_budget = self._resolve_total_budget(state)
        phase = self._phase_for_query_count(state.query_count, total_budget)
        remaining_phase = self._remaining_in_phase(phase, state.query_count, total_budget)
        k = min(int(k), int(remaining_phase))
        if k <= 0:
            input_shape = state.metadata.get("input_shape", (3, 32, 32))
            x_empty = torch.empty((0, *input_shape))
            return QueryBatch(x=x_empty, meta={"indices": [], "phase": phase, "status": "phase_exhausted"})

        self._update_phase(state)

        # Paper: build an initial substitute model after Phase 1 (K1) and use it
        # to score Phase 2 (K2) samples by confidence (HCSS).
        if phase >= 2 and not self.train_phase_1:
            self._train_substitute_from_queries(state)
            if self.substitute is not None:
                self.train_phase_1 = True
                state.attack_state["substitute"] = self.substitute

        if phase == 3:
            self._ensure_inversion_trained_for_phase3(state)

        if phase == 3 and self.inversion_model is not None:
            device = state.metadata.get("device", "cpu")

            templates = self._get_phase3_templates(device)
            idx = torch.randint(0, templates.size(0), (k,), device=device)
            y_sample = templates.index_select(0, idx)

            with torch.no_grad():
                x = self.inversion_model(y_sample)

            x = self._augment_inversion(x, y_sample)
            meta = {"phase": phase, "synthetic": True, "augmented": True}
            return QueryBatch(x=x, meta=meta)

        if self._pool_size() == 0:
            raise ValueError(
                f"Query pool exhausted for {self.__class__.__name__}. "
                f"Cannot select {k} more queries."
            )

        indices = self._select_phase_indices(k, state, phase)

        used_set = state.attack_state.get("used_pool_set")
        if not isinstance(used_set, set):
            used_set = set(state.attack_state.get("used_pool_indices", []))
            state.attack_state["used_pool_set"] = used_set
        for idx in indices:
            used_set.add(int(idx))
        # Keep list for compatibility/serialization (avoid sorting every step).
        state.attack_state["used_pool_indices"] = list(used_set)

        if len(indices) == 0:
            input_shape = state.metadata.get("input_shape", (3, 32, 32))
            x_empty = torch.empty((0, *input_shape))
            return QueryBatch(x=x_empty, meta={"indices": [], "phase": phase, "status": "exhausted"})

        device = state.metadata.get("device", "cpu")
        x = self._gather_pool_x(indices, device=str(device))
        meta = {"indices": indices, "phase": phase}
        return QueryBatch(x=x, meta=meta)

    def _pool_size(self) -> int:
        if isinstance(self.pool_data, torch.Tensor):
            return int(self.pool_data.size(0))
        if self.pool_dataset is not None:
            return int(len(self.pool_dataset))
        return 0

    def _gather_pool_x(self, indices: List[int], *, device: str) -> torch.Tensor:
        if len(indices) == 0:
            input_shape = self.state.metadata.get("input_shape", (3, 32, 32))
            return torch.empty((0, *input_shape))

        if isinstance(self.pool_data, torch.Tensor):
            pool = self.pool_data
            idx = torch.as_tensor(indices, dtype=torch.long, device=pool.device)
            x = pool.index_select(0, idx)
            if x.device != torch.device(device):
                x = x.to(device)
            return x

        if self.pool_dataset is None:
            raise RuntimeError("Pool dataset not loaded")

        # Fallback: avoid per-step worker spawn overhead (especially on Windows).
        loader_kwargs = pool_loader_kwargs(device, {"num_workers": 0})
        subset = Subset(self.pool_dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=min(len(indices), int(self.batch_size)),
            shuffle=False,
            **loader_kwargs,
        )
        x_batches: List[torch.Tensor] = []
        for x_b, _y_b in loader:
            x_batches.append(x_b)
        return torch.cat(x_batches, dim=0)

    def observe(self, query_batch: QueryBatch, oracle_output: OracleOutput, state: BenchmarkState) -> None:
        self._handle_oracle_output(query_batch.x, query_batch.meta, oracle_output, state)

    def _handle_oracle_output(
        self,
        x_query: torch.Tensor,
        meta: dict,
        oracle_output: OracleOutput,
        state: BenchmarkState,
    ) -> None:
        meta_phase = meta.get("phase") if isinstance(meta, dict) else None
        phase = int(meta_phase) if meta_phase is not None else int(state.attack_state.get("phase", 1))

        if oracle_output.kind != "hard_top1":
            raise ValueError("InverseNet requires hard_top1 oracle outputs.")

        victim_labels = oracle_output.y.detach().cpu().long()
        victim_probs = F.one_hot(victim_labels, num_classes=self.num_classes).float()
        query_targets = victim_labels

        # Phase 1 (K1): collect labeled coreset samples for initial substitute.
        if phase == 1:
            state.attack_state["query_data_x"].append(x_query.detach().cpu())
            state.attack_state["query_data_y"].append(query_targets)

        # Phase 2 (K2): collect (x, trunc1(FV(x))) pairs for training inversion model.
        if phase == 2:
            state.attack_state["inversion_x"].append(x_query.detach().cpu())
            trunc = self._truncate_logits(victim_probs)
            state.attack_state["inversion_y"].append(trunc)

        # Phase 3 (K3): query victim on inversed/augmented samples, then retrain once at the end.
        if phase == 3:
            state.attack_state["retrain_x"].append(x_query.detach().cpu())
            state.attack_state["retrain_y"].append(query_targets)

        # Keep phase metadata up to date for selection/caching purposes.
        self._update_phase(state)

    def _initialize_state(self, state: BenchmarkState) -> None:
        state.attack_state["query_data_x"] = []
        state.attack_state["query_data_y"] = []
        state.attack_state["val_query_data_x"] = []
        state.attack_state["val_query_data_y"] = []
        state.attack_state["inversion_x"] = []
        state.attack_state["inversion_y"] = []
        # Phase-3 (K3) labeled inverse queries for final retraining.
        state.attack_state["retrain_x"] = []
        state.attack_state["retrain_y"] = []
        state.attack_state["phase"] = 1
        state.attack_state["substitute"] = None
        state.attack_state["coreset_centers"] = []
        state.attack_state["inversion_trained"] = False
        state.attack_state["used_pool_indices"] = []
        state.attack_state["phase1_pending"] = []

        # Phase-2-only HCSS cache (DeepFool scoring over full pool is expensive).
        state.attack_state["hcss_cache"] = None
        state.attack_state["hcss_cache_cursor"] = 0
        state.attack_state["hcss_cache_sub_id"] = None

    def _resolve_total_budget(self, state: BenchmarkState) -> int:
        total_budget = int(
            state.metadata.get("max_budget")
            or self.config.get("max_budget")
            or self.config.get("total_budget")
            or 0
        )
        if total_budget <= 0:
            return 10_000
        return total_budget

    def _phase_boundaries(self, total_budget: int) -> tuple[int, int, int]:
        """Return (phase1_end, phase2_end, total_budget).

        Phase budgets follow the paper split K1:K2:K3 = 0.45:0.45:0.1.
        We implement this with integer budgets where Phase 3 receives the remainder
        to ensure K1+K2+K3 == total_budget.
        """

        ratios = list(self.phase_ratios) if isinstance(self.phase_ratios, list) else [0.45, 0.45, 0.1]
        while len(ratios) < 3:
            ratios.append(0.0)

        p1 = max(0, int(float(ratios[0]) * int(total_budget)))
        p2 = max(0, int(float(ratios[1]) * int(total_budget)))
        if p1 + p2 > int(total_budget):
            p2 = max(0, int(total_budget) - p1)

        phase1_end = p1
        phase2_end = p1 + p2
        return int(phase1_end), int(phase2_end), int(total_budget)

    def _phase_for_query_count(self, query_count: int, total_budget: int) -> int:
        phase1_end, phase2_end, _ = self._phase_boundaries(total_budget)
        q = int(query_count)
        if q < int(phase1_end):
            return 1
        if q < int(phase2_end):
            return 2
        return 3

    def _remaining_in_phase(self, phase: int, query_count: int, total_budget: int) -> int:
        phase1_end, phase2_end, total = self._phase_boundaries(total_budget)
        q = int(query_count)
        if int(phase) == 1:
            return max(0, int(phase1_end) - q)
        if int(phase) == 2:
            return max(0, int(phase2_end) - q)
        return max(0, int(total) - q)

    def _get_dataset_config(self, state: BenchmarkState) -> dict:
        dataset_config = self.config.get("attack", {}).get("dataset")
        if not dataset_config:
            dataset_config = state.metadata.get("dataset_config", {})
        if not dataset_config:
            dataset_config = self.config.get("dataset", {})
        if "data_mode" not in dataset_config:
            dataset_config = {"data_mode": "seed", **dataset_config}
        if dataset_config.get("data_mode") == "seed" and "name" not in dataset_config:
            dataset_config = {"name": "CIFAR10", **dataset_config}
        if dataset_config.get("data_mode") == "surrogate" and "surrogate_name" not in dataset_config:
            dataset_config = {"surrogate_name": "SVHN", **dataset_config}
        return dataset_config

    def _load_pool(self, state: BenchmarkState) -> None:
        dataset_config = self._get_dataset_config(state)
        self.pool_dataset = create_dataloader(
            dataset_config,
            batch_size=1,
            shuffle=False,
        ).dataset

        # Protocol-preserving runtime cache: preload the pool into a single tensor to
        # avoid per-step DataLoader construction/spawn overhead (especially on Windows).
        if bool(self.config.get("cache_pool_to_memory", True)):
            device = str(state.metadata.get("device", "cpu"))
            cache_batch_size = int(self.config.get("pool_cache_batch_size", 512))
            cache_max_samples = int(self.config.get("pool_cache_max_samples", 0))
            cache_max_samples = max(0, cache_max_samples)

            pool_workers = resolve_pool_num_workers(self.config, state.metadata.get("dataset_config", {}))
            loader_kwargs = (
                pool_loader_kwargs(device, {"num_workers": int(pool_workers)})
                if pool_workers is not None
                else pool_loader_kwargs(device)
            )
            loader = DataLoader(
                self.pool_dataset,
                batch_size=max(1, cache_batch_size),
                shuffle=False,
                **loader_kwargs,
            )

            batches: List[torch.Tensor] = []
            total = 0
            for x_b, _y_b in tqdm(loader, desc="[InverseNet] Caching pool data", leave=False):
                batches.append(x_b)
                total += int(x_b.size(0))
                if cache_max_samples and total >= cache_max_samples:
                    break

            if batches:
                pool_x = torch.cat(batches, dim=0)
                if cache_max_samples and int(pool_x.size(0)) > cache_max_samples:
                    pool_x = pool_x[:cache_max_samples]

                if device.startswith("cuda"):
                    try:
                        pool_x = pool_x.to(device)
                    except RuntimeError:
                        pool_x = pool_x.cpu()

                self.pool_data = pool_x
                self.logger.info(
                    "[InverseNet] Pool cache ready: shape=%s device=%s",
                    tuple(int(x) for x in pool_x.shape),
                    str(pool_x.device),
                )

        pool_len = self._pool_size()

        # Cache pool index list once.
        state.attack_state["all_pool_indices"] = list(range(int(pool_len)))
        # Track used indices as a set for fast membership checks.
        state.attack_state["used_pool_set"] = set()

    def _update_phase(self, state: BenchmarkState) -> None:
        prev_phase = state.attack_state.get("phase")
        total_budget = self._resolve_total_budget(state)
        phase1_end, phase2_end, _total = self._phase_boundaries(total_budget)

        q = int(state.query_count)
        if q < int(phase1_end):
            state.attack_state["phase"] = 1
        elif q < int(phase2_end):
            state.attack_state["phase"] = 2
        else:
            state.attack_state["phase"] = 3

        new_phase = state.attack_state.get("phase")
        if prev_phase != new_phase:
            # HCSS cache is only valid within Phase 2.
            state.attack_state["hcss_cache"] = None
            state.attack_state["hcss_cache_cursor"] = 0
            state.attack_state["hcss_cache_sub_id"] = None

    def _get_phase3_templates(self, device: str) -> torch.Tensor:
        key = str(device)
        cached = self._phase3_templates.get(key)
        if cached is not None:
            return cached

        confidences = [1.0, 0.9, 0.8]
        templates = []
        for c in range(self.num_classes):
            for conf in confidences:
                y = torch.zeros(self.num_classes, device=device)
                y[c] = float(conf)
                templates.append(y)

        out = torch.stack(templates)
        self._phase3_templates[key] = out
        return out

    def _train_inversion(self, state: BenchmarkState) -> None:
        if self.inversion_model is None:
            device = state.metadata.get("device", "cpu")
            self.inversion_model = InversionGenerator(
                num_classes=self.num_classes,
                output_channels=state.metadata.get("input_shape", (3, 32, 32))[0],
                base_channels=int(self.config.get("base_channels", 64)),
                output_activation=self.config.get("output_activation", "sigmoid"),
                output_size=state.metadata.get("input_shape", (3, 32, 32))[1],
            ).to(device)
            self.inversion_optimizer = torch.optim.Adam(
                self.inversion_model.parameters(), lr=self.inversion_lr
            )

        x_list = state.attack_state["inversion_x"]
        y_list = state.attack_state["inversion_y"]
        if len(x_list) == 0:
            return

        x_all = torch.cat(x_list, dim=0)
        y_all = torch.cat(y_list, dim=0)
        dataset = torch.utils.data.TensorDataset(x_all, y_all)
        sub_config = state.metadata.get("substitute_config", {})
        train_batch_size = int(
            sub_config.get("batch_size")
            or sub_config.get("trackA", {}).get("batch_size", self.batch_size)
        )
        loader = DataLoader(
            dataset,
            batch_size=train_batch_size,
            shuffle=True,
            **pool_loader_kwargs(
                state.metadata.get("device", "cpu"),
                {
                    "num_workers": int(
                        sub_config.get(
                            "inversion_num_workers",
                            sub_config.get("num_workers", self.config.get("num_workers", 0)),
                        )
                    )
                },
            ),
        )

        device = state.metadata.get("device", "cpu")
        self.inversion_model.train()
        epochs = max(1, int(self.inversion_epochs))
        inv_pbar = tqdm(range(epochs), desc="[InverseNet] Training Inversion", leave=False)
        for _ in inv_pbar:
            epoch_loss = 0.0
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                self.inversion_optimizer.zero_grad()
                recon = self.inversion_model(y_batch)
                loss = F.mse_loss(recon, x_batch)
                loss.backward()
                self.inversion_optimizer.step()
                epoch_loss += loss.item()
            inv_pbar.set_postfix({"Loss": f"{epoch_loss/len(loader):.4f}"})

        state.attack_state["inversion_trained"] = True

    def _ensure_inversion_trained_for_phase3(self, state: BenchmarkState) -> None:
        if state.attack_state.get("inversion_trained") is True:
            return

        # Train once (paper) using accumulated Phase 2 data.
        if len(state.attack_state.get("inversion_x", [])) == 0:
            return
        if len(state.attack_state.get("inversion_y", [])) == 0:
            return

        self._train_inversion(state)

    def _train_substitute_on_batch(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        state: BenchmarkState,
    ) -> None:
        """Phase 3: train substitute directly on victim re-query outputs.

        Paper Phase 3 requires: synthetic sample -> (augment) -> victim re-query -> retrain substitute
        using the victim outputs. We keep the existing optimizer/LR/epochs configuration.
        """
        sub_config = state.metadata.get("substitute_config", {})
        if self.substitute is None:
            device = state.metadata.get("device", "cpu")
            self.substitute = create_substitute(
                arch=self.config.get("substitute_arch", "resnet18"),
                num_classes=self.num_classes,
                input_channels=state.metadata.get("input_shape", (3, 32, 32))[0],
            ).to(device)
            
            # [UNIFIED] Use runner's build_optimizer
            opt_config = sub_config.get("optimizer", {})
            if "lr" not in opt_config:
                opt_config["lr"] = self.substitute_lr
                
            self.substitute_optimizer = self._build_optimizer(self.substitute.parameters(), opt_config)

        device = state.metadata.get("device", "cpu")
        if x_batch.size(0) == 0:
            return

        train_workers = resolve_train_num_workers(sub_config, self.config, default=0)

        loader = DataLoader(
            torch.utils.data.TensorDataset(x_batch.detach().cpu(), y_batch.detach().cpu()),
            batch_size=self.batch_size,
            shuffle=True,
            **pool_loader_kwargs(device, {"num_workers": int(train_workers)}),
        )

        self.substitute.train()
        epochs = max(1, int(self.substitute_epochs))
        for _ in range(epochs):
            for x_replay, y_replay in loader:
                x_replay = x_replay.to(device)
                y_replay = y_replay.to(device)
                self.substitute_optimizer.zero_grad(set_to_none=True)
                logits = self.substitute(x_replay)
                if y_replay.ndim == 1 or (y_replay.ndim == 2 and y_replay.size(1) == 1):
                    loss = F.cross_entropy(logits, y_replay.long().view(-1))
                else:
                    log_probs = F.log_softmax(logits, dim=1)
                    loss = F.kl_div(log_probs, y_replay, reduction="batchmean")
                loss.backward()
                self.substitute_optimizer.step()

        state.attack_state["substitute"] = self.substitute

    def _truncate_logits(self, probs: torch.Tensor) -> torch.Tensor:
        if probs.ndim == 1:
            probs = probs.unsqueeze(0)
        k = min(self.truncation_k, probs.size(1))
        topk = torch.topk(probs, k=k, dim=1)
        mask = torch.zeros_like(probs)
        mask.scatter_(1, topk.indices, 1.0)
        truncated = probs * mask
        return truncated

    def _augment_inversion(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Cache augmentation pipeline to avoid per-call object construction.
        if self._inversion_aug is None:
            # Custom Gaussian Noise transform
            class GaussianNoise:
                def __init__(self, mean: float = 0.0, std: float = 0.1) -> None:
                    self.mean = float(mean)
                    self.std = float(std)

                def __call__(self, img: torch.Tensor) -> torch.Tensor:
                    noise = torch.randn_like(img) * self.std + self.mean
                    return torch.clamp(img + noise, 0.0, 1.0)

            self._inversion_aug = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        x.shape[-2:],
                        scale=(0.8, 1.0),
                        ratio=(0.9, 1.1),
                    ),
                    transforms.RandomRotation(degrees=15),
                    transforms.RandomAffine(
                        degrees=0,
                        shear=10,
                        scale=(0.9, 1.1),
                    ),
                    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
                    transforms.RandomApply([GaussianNoise(mean=0.0, std=0.05)], p=0.3),
                ]
            )

        x_aug = self._inversion_aug(x)

        if labels is None or x_aug.size(0) < 2:
            return x_aug

        if labels.ndim > 1:
            class_ids = labels.argmax(dim=1)
        else:
            class_ids = labels

        class_ids = class_ids.to(x_aug.device)
        x_mix = x_aug.clone()
        for class_id in class_ids.unique():
            idxs = (class_ids == class_id).nonzero(as_tuple=False).view(-1)
            if idxs.numel() < 2:
                continue
            perm = idxs[torch.randperm(idxs.numel())]
            lam = torch.rand(idxs.numel(), 1, 1, 1, device=x_aug.device)
            x_mix[idxs] = lam * x_aug[idxs] + (1.0 - lam) * x_aug[perm]

        return x_mix

    def _train_substitute_from_queries(self, state: BenchmarkState) -> None:
        query_x = state.attack_state["query_data_x"]
        query_y = state.attack_state["query_data_y"]
        if len(query_x) == 0:
            return

        x_all = torch.cat(query_x, dim=0)
        y_all = torch.cat(query_y, dim=0)

        sub_config = state.metadata.get("substitute_config", {})
        dataset = torch.utils.data.TensorDataset(x_all, y_all)
        train_batch_size = int(
            sub_config.get("batch_size")
            or sub_config.get("trackA", {}).get("batch_size", self.batch_size)
        )

        train_dataset = dataset
        val_dataset = dataset
        train_size = len(train_dataset)
        if train_size < 2:
            return

        device = state.metadata.get("device", "cpu")
        train_workers = resolve_train_num_workers(sub_config, self.config, default=0)
        val_workers = resolve_val_num_workers(sub_config, self.config, default=train_workers)

        loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            **pool_loader_kwargs(device, {"num_workers": int(train_workers)}),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_batch_size,
            shuffle=False,
            **pool_loader_kwargs(device, {"num_workers": int(val_workers)}),
        )

        sub_config = state.metadata.get("substitute_config", {})
        if self.substitute is None:
            arch = sub_config.get("arch") or self.config.get("substitute_arch", "resnet18")
            width_mult = int(sub_config.get("width_mult", 1))
            dropout_prob = float(sub_config.get("dropout_prob", 0.0))
            self.substitute = create_substitute(
                arch=arch,
                num_classes=self.num_classes,
                input_channels=state.metadata.get("input_shape", (3, 32, 32))[0],
                width_mult=width_mult,
                dropout_prob=dropout_prob,
            ).to(device)
            # [UNIFIED] Use runner's build_optimizer to respect config (LR, optimizer type)
            opt_config = sub_config.get("optimizer", {})
            if "lr" not in opt_config:
                opt_config["lr"] = self.substitute_lr
            
            self.substitute_optimizer = self._build_optimizer(self.substitute.parameters(), opt_config)
        epochs = max(1, int(self.substitute_epochs))

        def loss_fn(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            if targets.ndim == 1 or (targets.ndim == 2 and targets.size(1) == 1):
                return F.cross_entropy(outputs, targets.long().view(-1))
            log_probs = F.log_softmax(outputs, dim=1)
            return F.kl_div(log_probs, targets, reduction="batchmean")

        def eval_fn(model_local: nn.Module, loader_local: DataLoader) -> float:
            model_local.eval()
            total_loss = 0.0
            total_count = 0
            with torch.no_grad():
                for x_val_b, y_val_b in loader_local:
                    x_val_b = x_val_b.to(device)
                    y_val_b = y_val_b.to(device)
                    outputs = model_local(x_val_b)
                    if y_val_b.ndim == 1 or (y_val_b.ndim == 2 and y_val_b.size(1) == 1):
                        loss = F.cross_entropy(outputs, y_val_b.long().view(-1))
                    else:
                        loss = F.kl_div(F.log_softmax(outputs, dim=1), y_val_b, reduction="batchmean")
                    total_loss += float(loss.item()) * int(x_val_b.size(0))
                    total_count += int(x_val_b.size(0))
            return total_loss / max(1, total_count)

        train_config = dict(sub_config)
        train_config["max_epochs"] = int(sub_config.get("max_epochs", epochs))
        train_config["patience"] = int(sub_config.get("patience", 20))
        trainer = SubstituteTrainer(train_config, device=device, logger=self.logger)
        steps_per_epoch = max(1, int(math.ceil(train_size / max(1, train_batch_size))))
        request = TrainRequest(
            model=self.substitute,
            train_loader=loader,
            val_loader=val_loader,
            eval_fn=eval_fn,
            loss_fn=loss_fn,
            max_steps=int(train_config["max_epochs"]) * steps_per_epoch,
            validate_every=steps_per_epoch,
            patience=int(train_config["patience"]) * steps_per_epoch,
            early_stop_mode="min",
            load_best=True,
        )
        trainer.train(request)

        state.attack_state["substitute"] = self.substitute
        self.logger.info("InverseNet substitute trained from queries.")
        self._evaluate_current_substitute(self.substitute, device)

    def _retrain_substitute_from_inverse_queries(self, state: BenchmarkState) -> None:
        """Paper Phase 4: retrain the initial substitute with inversed data queries.

        InverseNet queries the victim in three phases (K1, K2, K3). Phase 3 (K3)
        uses inversed/augmented samples; their labeled responses are then used to
        retrain the initial substitute.

        This implementation follows the paper's "train twice" schedule:
        - one training pass after Phase 1 (K1)
        - one retraining pass after Phase 3 (K3)
        """

        inv_x = state.attack_state.get("retrain_x", [])
        inv_y = state.attack_state.get("retrain_y", [])
        if len(inv_x) == 0 or len(inv_y) == 0:
            return

        # Ensure an initial substitute exists.
        if self.substitute is None:
            self._train_substitute_from_queries(state)
        if self.substitute is None:
            return

        x_all = torch.cat([x.detach().cpu() for x in inv_x], dim=0)
        y_all = torch.cat([y.detach().cpu() for y in inv_y], dim=0)
        dataset = torch.utils.data.TensorDataset(x_all, y_all)

        train_size = int(len(dataset))
        if train_size < 2:
            return
        train_dataset = dataset
        val_dataset = dataset

        sub_config = state.metadata.get("substitute_config", {})
        train_batch_size = int(
            sub_config.get("batch_size")
            or sub_config.get("trackA", {}).get("batch_size", self.batch_size)
        )
        device = state.metadata.get("device", "cpu")
        train_workers = resolve_train_num_workers(sub_config, self.config, default=0)
        val_workers = resolve_val_num_workers(sub_config, self.config, default=train_workers)

        loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            **pool_loader_kwargs(device, {"num_workers": int(train_workers)}),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_batch_size,
            shuffle=False,
            **pool_loader_kwargs(device, {"num_workers": int(val_workers)}),
        )

        def loss_fn(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            if targets.ndim == 1 or (targets.ndim == 2 and targets.size(1) == 1):
                return F.cross_entropy(outputs, targets.long().view(-1))
            log_probs = F.log_softmax(outputs, dim=1)
            return F.kl_div(log_probs, targets, reduction="batchmean")

        def eval_fn(model_local: nn.Module, loader_local: DataLoader) -> float:
            model_local.eval()
            total_loss = 0.0
            total_count = 0
            with torch.no_grad():
                for x_val_b, y_val_b in loader_local:
                    x_val_b = x_val_b.to(device)
                    y_val_b = y_val_b.to(device)
                    outputs = model_local(x_val_b)
                    if y_val_b.ndim == 1 or (y_val_b.ndim == 2 and y_val_b.size(1) == 1):
                        loss = F.cross_entropy(outputs, y_val_b.long().view(-1))
                    else:
                        loss = F.kl_div(
                            F.log_softmax(outputs, dim=1),
                            y_val_b,
                            reduction="batchmean",
                        )
                    total_loss += float(loss.item()) * int(x_val_b.size(0))
                    total_count += int(x_val_b.size(0))
            return total_loss / max(1, total_count)

        train_config = dict(sub_config)
        max_epochs = int(sub_config.get("max_epochs", self.substitute_epochs))
        patience_epochs = int(sub_config.get("patience", 20))
        train_config["max_epochs"] = max_epochs
        train_config["patience"] = patience_epochs
        trainer = SubstituteTrainer(train_config, device=device, logger=self.logger)
        steps_per_epoch = max(1, int(math.ceil(train_size / max(1, train_batch_size))))
        request = TrainRequest(
            model=self.substitute,
            train_loader=loader,
            val_loader=val_loader,
            eval_fn=eval_fn,
            loss_fn=loss_fn,
            max_steps=int(max_epochs) * steps_per_epoch,
            validate_every=steps_per_epoch,
            patience=int(patience_epochs) * steps_per_epoch,
            early_stop_mode="min",
            load_best=True,
        )
        trainer.train(request)

        state.attack_state["substitute"] = self.substitute
        self.logger.info("InverseNet substitute retrained with inversed dataset.")

    def _select_phase_indices(self, k: int, state: BenchmarkState, phase: int) -> List[int]:
        pool_len = self._pool_size()
        if pool_len == 0:
            return []

        used = state.attack_state.get("used_pool_set")
        if not isinstance(used, set):
            used = set(state.attack_state.get("used_pool_indices", []))
            state.attack_state["used_pool_set"] = used

        all_indices = state.attack_state.get("all_pool_indices")
        if not isinstance(all_indices, list) or len(all_indices) != int(pool_len):
            all_indices = list(range(int(pool_len)))
            state.attack_state["all_pool_indices"] = all_indices

        unused = [i for i in all_indices if i not in used]
        if not unused:
            return []

        # Phase 2 (HCSS): Remove coreset overlap as per paper
        if phase == 2:
            coreset_centers = state.attack_state.get("coreset_centers", [])

            if len(coreset_centers) > 0:
                coreset_set = set(int(i) for i in coreset_centers)
                available = [i for i in unused if i not in coreset_set]
            else:
                available = unused
                
            # Use ENTIRE available pool (Strict Protocol)
            candidates = available
            if not candidates:
                return []

            substitute = state.attack_state.get("substitute")
            if substitute is None:
                return candidates[: min(k, len(candidates))]
            return self._hcss_select(k, candidates, substitute)

        # Phase 1 (Coreset) or others: Select from full pool (Strict Protocol)
        candidates = unused

        if phase == 1:
            return self._coreset_select(k, candidates, state)
            
        return candidates[: min(k, len(candidates))]

    def _coreset_select(self, k: int, candidates: List[int], state: BenchmarkState) -> List[int]:
        pending = list(state.attack_state.get("phase1_pending", []))
        if pending:
            take = min(int(k), len(pending))
            out = pending[:take]
            state.attack_state["phase1_pending"] = pending[take:]
            return out

        centers = list(state.attack_state.get("coreset_centers", []))
        if len(centers) == 0:
            seed_count = min(self.coreset_seed, len(candidates))
            centers = np.random.choice(candidates, seed_count, replace=False).tolist()
            state.attack_state["coreset_centers"] = centers
            state.attack_state["phase1_pending"] = list(centers)
            return self._coreset_select(k, candidates, state)

        remaining = [idx for idx in candidates if idx not in centers]
        if len(remaining) == 0:
            return centers[:k]

        selected = []
        
        # [OPTIMIZATION] Batched feature extraction & distance computation
        # 1. Extract features (flattened images) for all centers and remaining candidates ONCE
        # Note: InverseNet uses raw image L1 distance as "feature"
        
        device = str(state.metadata.get("device", "cpu"))

        if isinstance(self.pool_data, torch.Tensor):
            pool = self.pool_data
            pool_flat = pool.view(int(pool.size(0)), -1)

            rem_idx = torch.as_tensor(remaining, dtype=torch.long, device=pool.device)
            candidates_matrix = pool_flat.index_select(0, rem_idx)

            ctr_idx = torch.as_tensor(centers, dtype=torch.long, device=pool.device)
            centers_matrix = pool_flat.index_select(0, ctr_idx)
        else:
            # Fallback path: load from dataset if pool cache is disabled.
            subset = torch.utils.data.Subset(self.pool_dataset, remaining)
            pool_workers = resolve_pool_num_workers(self.config, state.metadata.get("dataset_config", {}))
            loader_kwargs = (
                pool_loader_kwargs(device, {"num_workers": int(pool_workers)})
                if pool_workers is not None
                else pool_loader_kwargs(device)
            )
            loader = DataLoader(
                subset,
                batch_size=256,
                shuffle=False,
                **loader_kwargs,
            )

            candidates_matrix_list: List[torch.Tensor] = []
            for x_batch, _ in loader:
                candidates_matrix_list.append(x_batch.view(x_batch.size(0), -1))

            if not candidates_matrix_list:
                return centers[:k]

            candidates_matrix = torch.cat(candidates_matrix_list, dim=0)

            center_subset = torch.utils.data.Subset(self.pool_dataset, centers)
            center_loader = DataLoader(
                center_subset,
                batch_size=256,
                shuffle=False,
                **loader_kwargs,
            )

            centers_matrix_list: List[torch.Tensor] = []
            for x_batch, _ in center_loader:
                centers_matrix_list.append(x_batch.view(x_batch.size(0), -1))

            centers_matrix = torch.cat(centers_matrix_list, dim=0)

        if candidates_matrix.device != torch.device(device):
            candidates_matrix = candidates_matrix.to(device)
        if centers_matrix.device != torch.device(device):
            centers_matrix = centers_matrix.to(device)
        
        # Initialize min_distances with current centers
        # dists: [N_remaining, N_centers]
        # We need min_dist for each candidate: min_{c in centers} ||x - c||_1
        # L1 distance: sum(|x - y|)
        
        # Memory optimization: Compute initial min_dists in chunks if needed
        # But for typical pool sizes (e.g. 50k), we can do it iteratively to save memory
        
        # Initialize with infinity
        min_dists = torch.full((candidates_matrix.size(0),), float('inf'), device=device)
        
        # Update min_dists against existing centers.
        # Use torch.cdist(p=1) in chunks to avoid Python loops.
        chunk_size = int(self.config.get("coreset_cdist_chunk", 128))
        chunk_size = max(1, chunk_size)
        for i in range(0, centers_matrix.size(0), chunk_size):
            c_chunk = centers_matrix[i : i + chunk_size]  # [C, D]
            # dists: [N_rem, C]
            dists = torch.cdist(candidates_matrix, c_chunk, p=1)
            min_dists = torch.minimum(min_dists, dists.min(dim=1).values)

        # Greedy selection loop
        for _ in range(min(k, len(remaining))):
            # Find candidate with MAX min_dist
            max_val, max_idx_in_matrix = torch.max(min_dists, dim=0)
            max_idx = max_idx_in_matrix.item()
            
            best_real_idx = remaining[max_idx]
            selected.append(best_real_idx)
            
            # Update centers and remove from consideration
            new_center_vec = candidates_matrix[max_idx].unsqueeze(0) # [1, D]
            
            # Update min_dists for all candidates using the new center
            # New min_dist = min(old_min_dist, dist(x, new_center))
            new_dists = torch.norm(candidates_matrix - new_center_vec, p=1, dim=1)
            min_dists = torch.minimum(min_dists, new_dists)
            
            # Effectively remove the selected one by setting its dist to -1
            min_dists[max_idx] = -1.0
            
            # Update state (strictly speaking we should append to centers list, but we reconstruct at end)
        
        # Reconstruct centers list
        final_centers = centers + selected
        state.attack_state["coreset_centers"] = final_centers
        return selected

    def _hcss_select(
        self, k: int, candidates: List[int], substitute: nn.Module
    ) -> List[int]:
        device = next(substitute.parameters()).device
        substitute.eval()
        cache = self.state.attack_state.get("hcss_cache")
        cache_sub_id = self.state.attack_state.get("hcss_cache_sub_id")
        if cache is not None and cache_sub_id == id(substitute):
            cursor = int(self.state.attack_state.get("hcss_cache_cursor", 0))
            candidate_set = set(int(i) for i in candidates)
            selected: List[int] = []
            for pos in range(cursor, len(cache)):
                idx = int(cache[pos])
                if idx in candidate_set:
                    selected.append(idx)
                    if len(selected) >= min(int(k), len(candidates)):
                        self.state.attack_state["hcss_cache_cursor"] = pos + 1
                        return selected
            self.state.attack_state["hcss_cache_cursor"] = len(cache)
            return selected

        scores: List[Tuple[int, float]] = []

        # DeepFool scoring is expensive; use a moderate batch size for GPU efficiency.
        # Tunable via attack config: hcss_batch_size.
        batch_size = min(self.batch_size, int(self.config.get("hcss_batch_size", 64)))

        if isinstance(self.pool_data, torch.Tensor):
            pool = self.pool_data
            n = len(candidates)
            for start in range(0, n, int(batch_size)):
                end = min(start + int(batch_size), n)
                batch_indices = candidates[start:end]
                if not batch_indices:
                    continue

                idx = torch.as_tensor(batch_indices, dtype=torch.long, device=pool.device)
                x_batch = pool.index_select(0, idx)
                if x_batch.device != device:
                    x_batch = x_batch.to(device)

                distances = self._hcss_noise_distance_batch(substitute, x_batch)
                batch_scores = ((1.0 + self.hcss_xi) * distances).detach().cpu().tolist()
                for i, idx_val in enumerate(batch_indices):
                    scores.append((int(idx_val), float(batch_scores[i])))
        else:
            pool_workers = resolve_pool_num_workers(self.config, self.state.metadata.get("dataset_config", {}))
            loader_kwargs = (
                pool_loader_kwargs(str(device), {"num_workers": int(pool_workers)})
                if pool_workers is not None
                else pool_loader_kwargs(str(device))
            )

            # Fallback: DataLoader retrieval from the underlying dataset.
            subset = Subset(self.pool_dataset, candidates)
            loader = DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=False,
                **loader_kwargs,
            )

            current_idx_ptr = 0
            for x_batch, _ in loader:
                batch_len = x_batch.size(0)
                batch_indices = candidates[current_idx_ptr : current_idx_ptr + batch_len]
                current_idx_ptr += batch_len

                x_batch = x_batch.to(device, non_blocking=str(device).startswith("cuda"))
                distances = self._hcss_noise_distance_batch(substitute, x_batch)

                batch_scores = ((1.0 + self.hcss_xi) * distances).detach().cpu().tolist()
                for i, idx_val in enumerate(batch_indices):
                    scores.append((int(idx_val), float(batch_scores[i])))

        scores.sort(key=lambda x: x[1], reverse=True)

        sorted_indices = [idx for idx, _ in scores]
        self.state.attack_state["hcss_cache"] = sorted_indices
        self.state.attack_state["hcss_cache_cursor"] = min(int(k), len(sorted_indices))
        self.state.attack_state["hcss_cache_sub_id"] = id(substitute)
        return sorted_indices[: min(int(k), len(sorted_indices))]

    def _hcss_noise_distance_batch(
        self,
        model: nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        device = x.device
        if x.shape[0] == 0:
            return torch.empty(0, device=device)

        with torch.enable_grad():
            return deepfool_distance_vectorized(
                model,
                x,
                max_iter=self.hcss_max_iter,
                batch_size=min(self.batch_size, x.shape[0]),
            )
