"""Random selection baseline attack."""

import logging
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from mebench.models.substitute_factory import create_substitute
from mebench.training import SubstituteTrainer, TrainRequest
from mebench.utils.dataloader import pool_loader_kwargs
from mebench.attackers.runner import AttackRunner
from mebench.core.context import BenchmarkContext
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.data.loaders import create_dataloader


class RandomBaseline(AttackRunner):
    """Attack that randomly samples from surrogate or seed pool."""

    def __init__(self, config: dict, state: BenchmarkState):
        super().__init__(config, state)
        self.dataloader = None
        self.iterator = None
        self._initial_batch_size = None
        self.pool_dataset = None
        self._initialize_state(state)

    def _initialize_state(self, state: BenchmarkState) -> None:
        state.attack_state.setdefault("queried_indices", [])
        state.attack_state.setdefault("unqueried_indices", [])
        state.attack_state.setdefault("query_data_x", [])
        state.attack_state.setdefault("query_data_y", [])
        state.attack_state.setdefault("val_query_data_x", [])
        state.attack_state.setdefault("val_query_data_y", [])
        state.attack_state.setdefault("substitute", None)

    def _ensure_pool_dataset(self, state: BenchmarkState) -> None:
        if self.pool_dataset is None:
            self.dataloader = create_dataloader(
                state.metadata.get("dataset_config", {}),
                batch_size=1,
                shuffle=False,
            )
            self.pool_dataset = self.dataloader.dataset

        if not state.attack_state["unqueried_indices"] and self.pool_dataset is not None:
            dataset_config = state.metadata.get("dataset_config", {})
            seed_size = dataset_config.get("seed_size")
            if seed_size is None and isinstance(dataset_config.get("dataset"), dict):
                seed_size = dataset_config["dataset"].get("seed_size")
            pool_size = len(self.pool_dataset)
            if seed_size is not None:
                pool_size = min(pool_size, int(seed_size))
            state.attack_state["unqueried_indices"] = list(range(pool_size))

    def run(self, ctx: BenchmarkContext) -> None:
        self.victim = ctx.oracle.model
        device = self.state.metadata.get("device", "cpu")

        total_budget = ctx.budget_remaining
        indices, pool_exhausted = self._sample_indices(total_budget, self.state)
        step_size = int(self.config.get("batch_size", self._default_step_size(ctx)))

        pbar = self._create_progress_bar(total_budget, f"[{self.__class__.__name__}] Extracting")
        offset = 0
        while offset < total_budget:
            k = min(step_size, total_budget - offset)
            batch_indices = indices[offset : offset + k]
            x_list = [self.pool_dataset[i][0] for i in batch_indices]
            x_batch = torch.stack(x_list)

            meta = {"synthetic": False}
            if pool_exhausted:
                meta["pool_exhausted"] = True

            query_batch = QueryBatch(x=x_batch, meta=meta)
            oracle_output = ctx.query(query_batch.x, meta=getattr(query_batch, "meta", None))
            self._handle_oracle_output(query_batch, oracle_output, self.state)
            pbar.update(query_batch.x.size(0))
            offset += k
        pbar.close()
        # Train once on the collected labeled set.
        self._train_substitute(self.state)
        # Final evaluation is handled by the engine.

    def _sample_indices(self, k: int, state: BenchmarkState) -> tuple[list[int], bool]:
        self._ensure_pool_dataset(state)

        if self.pool_dataset is None or len(self.pool_dataset) == 0:
            raise ValueError("RandomBaseline requires a non-empty pool dataset.")

        indices: list[int] = []
        pool_exhausted = len(state.attack_state["unqueried_indices"]) == 0

        if not pool_exhausted:
            available = state.attack_state["unqueried_indices"]
            n_take = min(int(k), len(available))
            indices = np.random.choice(available, n_take, replace=False).tolist()
            if n_take < int(k):
                remainder = int(k) - n_take
                extra = np.random.choice(len(self.pool_dataset), remainder, replace=True).tolist()
                indices.extend(extra)
                pool_exhausted = True
        else:
            indices = np.random.choice(len(self.pool_dataset), int(k), replace=True).tolist()

        for idx in indices:
            if idx in state.attack_state["unqueried_indices"]:
                state.attack_state["unqueried_indices"].remove(idx)
            state.attack_state["queried_indices"].append(int(idx))

        return indices, pool_exhausted

    def _select_query_batch(self, k: int, state: BenchmarkState) -> QueryBatch:
        indices, pool_exhausted = self._sample_indices(k, state)
        x_list = [self.pool_dataset[i][0] for i in indices]
        x_batch = torch.stack(x_list)

        meta = {"synthetic": False}
        if pool_exhausted:
            meta["pool_exhausted"] = True

        return QueryBatch(x=x_batch[: int(k)], meta=meta)

    def _handle_oracle_output(
        self,
        query_batch: QueryBatch,
        oracle_output: OracleOutput,
        state: BenchmarkState,
    ) -> None:
        state.attack_state["query_data_x"].append(query_batch.x.detach().cpu())
        state.attack_state["query_data_y"].append(oracle_output.y.detach().cpu())

    def _train_substitute(self, state: BenchmarkState) -> None:
        """Train substitute model with fixed validation holdout when available."""
        self._ensure_fixed_validation_holdout(state)
        query_x = state.attack_state.get("query_data_x", [])
        query_y = state.attack_state.get("query_data_y", [])
        val_query_x = state.attack_state.get("val_query_data_x", [])
        val_query_y = state.attack_state.get("val_query_data_y", [])
        if len(query_x) == 0:
            return

        x_all = torch.cat(query_x, dim=0)
        y_all = torch.cat(query_y, dim=0)
        
        # Ensure we have enough data for split
        if x_all.size(0) < 10:
             # Too few samples, fallback to simple training without split
             self._train_substitute_simple(state, x_all, y_all)
             return

        if len(val_query_x) > 0 and len(val_query_y) > 0:
            x_val = torch.cat(val_query_x, dim=0)
            y_val = torch.cat(val_query_y, dim=0)
            train_dataset = torch.utils.data.TensorDataset(x_all, y_all)
            val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
            train_size = int(x_all.size(0))
        else:
            total_size = x_all.size(0)
            val_size = max(1, int(0.2 * total_size))
            train_size = total_size - val_size

            full_dataset = torch.utils.data.TensorDataset(x_all, y_all)
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )

        sub_config = state.metadata.get("substitute_config", {})
        device = state.metadata.get("device", "cpu")
        
        # Create fresh substitute
        num_classes = int(
            state.metadata.get("num_classes")
            or state.metadata.get("victim_config", {}).get("num_classes")
            or state.metadata.get("dataset_config", {}).get("num_classes", 10)
        )
        input_channels = int(state.metadata.get("input_shape", (3, 32, 32))[0])
        width_mult = int(sub_config.get("width_mult", 1))
        dropout_prob = float(sub_config.get("dropout_prob", 0.0))
        substitute = create_substitute(
            arch=sub_config.get("arch", "resnet18"),
            num_classes=num_classes,
            input_channels=input_channels,
            width_mult=width_mult,
            dropout_prob=dropout_prob,
        ).to(device)

        train_batch_size = int(
            sub_config.get("batch_size")
            or sub_config.get("trackA", {}).get("batch_size")
            or int(self.config.get("batch_size", 128))
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
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            **pool_loader_kwargs(device, {"num_workers": train_workers}),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_batch_size,
            shuffle=False,
            **pool_loader_kwargs(device, {"num_workers": val_workers}),
        )

        output_mode = str(self.config.get("output_mode", "soft_prob"))
        
        # [FIX] Use Loss for Early Stopping (minimizing val loss)
        def eval_fn(model: nn.Module, loader: DataLoader) -> float:
            model.eval()
            total_loss = 0.0
            total_count = 0
            loss_func = nn.KLDivLoss(reduction="batchmean") if output_mode == "soft_prob" else nn.CrossEntropyLoss()
            
            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)
                    if output_mode == "soft_prob":
                        y = y.clamp_min(1e-10)
                        y = y / y.sum(dim=1, keepdim=True).clamp_min(1e-12)
                        log_probs = torch.log_softmax(outputs, dim=1)
                        loss = loss_func(log_probs, y)
                    else:
                        loss = loss_func(outputs, y.long())
                    total_loss += loss.item() * x.size(0)
                    total_count += x.size(0)
            return total_loss / total_count if total_count > 0 else float('inf')

        def train_loss_fn(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            if output_mode == "soft_prob":
                targets = targets.clamp_min(1e-10)
                targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-12)
                log_probs = torch.log_softmax(outputs, dim=1)
                return nn.KLDivLoss(reduction="batchmean")(log_probs, targets)
            return nn.CrossEntropyLoss()(outputs, targets.long())

        batch_size = max(1, int(train_batch_size))
        steps_per_epoch = max(1, int(math.ceil(train_size / batch_size)))
        max_epochs = int(sub_config.get("max_epochs", 200))
        patience_epochs = int(sub_config.get("patience", 20))

        trainer = SubstituteTrainer(dict(sub_config), device=device)
        request = TrainRequest(
            model=substitute,
            train_loader=train_loader,
            val_loader=val_loader,     # [ADDED]
            eval_fn=eval_fn,           # [ADDED]
            loss_fn=train_loss_fn,
            max_steps=max_epochs * steps_per_epoch,
            validate_every=steps_per_epoch,
            patience=patience_epochs * steps_per_epoch,
            early_stop_mode="min",     # Minimize Val Loss
            load_best=True,
        )
        trainer.train(request)

        state.attack_state["substitute"] = substitute

    def _train_substitute_simple(self, state, x_all, y_all):
        """Fallback for very small datasets."""
        sub_config = state.metadata.get("substitute_config", {})
        device = state.metadata.get("device", "cpu")
        num_classes = int(state.metadata.get("num_classes") or 10)
        input_channels = int(state.metadata.get("input_shape", (3, 32, 32))[0])
        
        substitute = create_substitute(
            arch=sub_config.get("arch", "resnet18"),
            num_classes=num_classes,
            input_channels=input_channels,
        ).to(device)
        
        loader = DataLoader(
            torch.utils.data.TensorDataset(x_all, y_all),
            batch_size=32,
            shuffle=True,
            **pool_loader_kwargs(
                device,
                {
                    "num_workers": int(
                        sub_config.get(
                            "train_num_workers",
                            sub_config.get("num_workers", self.config.get("num_workers", 0)),
                        )
                    )
                },
            ),
        )
        
        output_mode = str(self.config.get("output_mode", "soft_prob"))
        def loss_fn(outputs, targets):
             if output_mode == "soft_prob":
                targets = targets.clamp_min(1e-10)
                targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-12)
                return nn.KLDivLoss(reduction="batchmean")(torch.log_softmax(outputs, dim=1), targets)
             return nn.CrossEntropyLoss()(outputs, targets.long())

        trainer = SubstituteTrainer(dict(sub_config), device=device, logger=self.logger)
        request = TrainRequest(
            model=substitute,
            train_loader=loader,
            loss_fn=loss_fn,
            max_steps=100, # Minimal steps
        )
        trainer.train(request)
        state.attack_state["substitute"] = substitute
