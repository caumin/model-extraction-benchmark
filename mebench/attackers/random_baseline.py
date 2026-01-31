"""Random selection baseline attack."""

import logging
import numpy as np
import torch
from tqdm import tqdm
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
        
        pbar = tqdm(total=self.state.budget_remaining, desc=f"[{self.__class__.__name__}] Extracting")
        while ctx.budget_remaining > 0:
            step_size = self._default_step_size(ctx)
            query_batch = self._select_query_batch(step_size, self.state)
            oracle_output = ctx.query(query_batch.x, meta=getattr(query_batch, "meta", None))
            self._handle_oracle_output(query_batch, oracle_output, self.state)
            pbar.update(query_batch.x.size(0))
        pbar.close()
        # Final Evaluation (handled by engine)

    def _select_query_batch(self, k: int, state: BenchmarkState) -> QueryBatch:
        self._ensure_pool_dataset(state)

        pool_exhausted = len(state.attack_state["unqueried_indices"]) == 0
        if pool_exhausted:
            if self.pool_dataset is None or len(self.pool_dataset) == 0:
                raise ValueError("RandomBaseline requires a non-empty pool dataset.")
            indices = np.random.choice(len(self.pool_dataset), int(k), replace=True).tolist()
        else:
            available = state.attack_state["unqueried_indices"]
            n_take = min(int(k), len(available))
            indices = np.random.choice(available, n_take, replace=False).tolist()
            if n_take < int(k):
                remainder = int(k) - n_take
                extra = np.random.choice(len(self.pool_dataset), remainder, replace=True).tolist()
                indices.extend(extra)
                pool_exhausted = True

        x_list = [self.pool_dataset[i][0] for i in indices]
        x_batch = torch.stack(x_list)

        for idx in indices:
            if idx in state.attack_state["unqueried_indices"]:
                state.attack_state["unqueried_indices"].remove(idx)
            state.attack_state["queried_indices"].append(int(idx))

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
        # Random baseline doesn't update state based on output
        return
