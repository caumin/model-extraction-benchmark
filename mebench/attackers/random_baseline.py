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
        query_x = state.attack_state.get("query_data_x", [])
        query_y = state.attack_state.get("query_data_y", [])
        if len(query_x) == 0:
            return

        x_all = torch.cat(query_x, dim=0)
        y_all = torch.cat(query_y, dim=0)
        if x_all.size(0) < 2:
            return

        sub_config = state.metadata.get("substitute_config", {})
        device = state.metadata.get("device", "cpu")

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
        loader = DataLoader(
            torch.utils.data.TensorDataset(x_all, y_all),
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=0,
        )

        output_mode = str(self.config.get("output_mode", "soft_prob"))

        def loss_fn(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            if output_mode == "soft_prob":
                targets = targets.clamp_min(1e-10)
                targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-12)
                log_probs = torch.log_softmax(outputs, dim=1)
                return nn.KLDivLoss(reduction="batchmean")(log_probs, targets)
            return nn.CrossEntropyLoss()(outputs, targets.long())

        # Use Track-A style step budget by default (keeps runtime bounded).
        steps_coeff_c = float(sub_config.get("trackA", {}).get("steps_coeff_c", 0.2))
        max_steps = int(math.ceil(steps_coeff_c * x_all.size(0)))
        max_steps = max(1, max_steps)

        trainer = SubstituteTrainer(dict(sub_config), device=device, logger=self.logger)
        request = TrainRequest(
            model=substitute,
            train_loader=loader,
            loss_fn=loss_fn,
            max_steps=max_steps,
            load_best=True,
        )
        trainer.train(request)

        state.attack_state["substitute"] = substitute
