"""Benchmark execution context (Track B first)."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch

from mebench.core.state import BenchmarkState
from mebench.core.types import OracleOutput
from mebench.core.query_storage import QueryStorage
from mebench.core.logging import ArtifactLogger
from mebench.oracles.oracle import Oracle


class BenchmarkContext:
    """Execution context provided to attacks (IOC)."""

    def __init__(
        self,
        state: BenchmarkState,
        oracle: Oracle,
        logger: Optional[ArtifactLogger] = None,
        config: Optional[Dict[str, Any]] = None,
        query_storage: Optional[QueryStorage] = None,
        record_queries: bool = False,
    ) -> None:
        self.state = state
        self.oracle = oracle
        self.logger = logger
        self.config = config or {}
        self.query_storage = query_storage
        self.record_queries = record_queries

        checkpoints = self.config.get("budget", {}).get("checkpoints", [])
        self.checkpoints = sorted(int(c) for c in checkpoints)
        self._checkpoint_reached = set(self.state.attack_state.get("checkpoint_reached", []))

    @property
    def budget_remaining(self) -> int:
        return int(self.state.budget_remaining)

    @property
    def query_count(self) -> int:
        return int(self.state.query_count)

    def query(self, x: torch.Tensor, meta: Optional[Dict[str, Any]] = None) -> OracleOutput:
        if x is None:
            raise ValueError("BenchmarkContext.query requires a tensor.")

        batch_size = int(x.size(0))
        if batch_size <= 0:
            raise ValueError("BenchmarkContext.query called with empty batch.")

        if batch_size > self.state.budget_remaining:
            raise ValueError(
                f"Query batch size {batch_size} exceeds remaining budget {self.state.budget_remaining}."
            )

        oracle_output = self.oracle.query(x)
        # Print progress every ~1000 queries (handles batch jumps)
        if self.state.query_count % 1000 < batch_size:
            print(f"[Query Progress] Used: {self.state.query_count} / Remaining: {self.state.budget_remaining}")

        if self.record_queries:
            if self.query_storage is None:
                raise RuntimeError("record_queries=True requires query_storage.")
            self.query_storage.add_batch(x, oracle_output.y)

        self._maybe_checkpoint()
        return oracle_output

    def log_event(self, name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if self.logger is None:
            return

        safe_payload: Dict[str, Any] = {}
        if payload:
            for key, value in payload.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    safe_payload[key] = value
                else:
                    safe_payload[key] = str(value)

        self.logger.log_history(self.state.query_count, {"event": name, **safe_payload})

    def on_checkpoint(self, query_count: int) -> None:
        self.log_event("checkpoint_reached", {"checkpoint": int(query_count)})

    def _maybe_checkpoint(self) -> None:
        if not self.checkpoints:
            return

        for checkpoint in self.checkpoints:
            if self.state.query_count >= checkpoint and checkpoint not in self._checkpoint_reached:
                self._checkpoint_reached.add(checkpoint)
                reached = sorted(self._checkpoint_reached)
                self.state.attack_state["checkpoint_reached"] = reached
                self.on_checkpoint(checkpoint)
