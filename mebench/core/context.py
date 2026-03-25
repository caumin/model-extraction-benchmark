"""Benchmark execution context (Track B first)."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

import torch

from mebench.core.state import BenchmarkState
from mebench.core.types import OracleOutput
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
        checkpoint_callback: Optional[Callable[[int], None]] = None,
    ) -> None:
        self._logger = logging.getLogger(__name__)
        self.state = state
        self.oracle = oracle
        self.logger = logger
        self.config = config or {}
        self._checkpoint_callback = checkpoint_callback

        checkpoints = self.config.get("budget", {}).get("checkpoints", [])
        self.checkpoints = sorted(int(c) for c in checkpoints)
        self._checkpoint_reached = set(self.state.attack_state.get("checkpoint_reached", []))

        attack_config = self.config.get("attack", {})
        interval = attack_config.get("log_query_interval")
        if interval is not None:
            interval = int(interval)
            self._progress_interval = interval if interval > 0 else None
        else:
            total_budget = (
                self.config.get("budget", {}).get("max_budget")
                or self.state.metadata.get("max_budget")
                or 0
            )
            if total_budget > 0:
                self._progress_interval = max(1000, int(total_budget // 100))
            else:
                self._progress_interval = 1000

        self._query_batch_size = self._resolve_query_batch_size()

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

        prev_queries = int(self.state.query_count)
        query_batch_size = self._query_batch_size
        if query_batch_size is None or batch_size <= query_batch_size:
            oracle_output = self.oracle.query(x)
        else:
            outputs = []
            output_kind: Optional[str] = None
            for start in range(0, batch_size, query_batch_size):
                x_chunk = x[start : start + query_batch_size]
                chunk_output = self.oracle.query(x_chunk)
                if output_kind is None:
                    output_kind = chunk_output.kind
                elif chunk_output.kind != output_kind:
                    raise ValueError(
                        "Oracle returned inconsistent output kinds across query chunks: "
                        f"{output_kind} vs {chunk_output.kind}."
                    )
                outputs.append(chunk_output.y)

            if output_kind is None:
                raise ValueError("BenchmarkContext.query failed to produce oracle outputs.")
            oracle_output = OracleOutput(kind=output_kind, y=torch.cat(outputs, dim=0))

        # Print progress at coarse intervals (handles batch jumps).
        # Always print once on the first successful query so users can confirm
        # that oracle traffic has started, even if tqdm rendering is suppressed.
        if self._progress_interval:
            if prev_queries == 0:
                self._logger.info(
                    "[Query Progress] Used: %d / Remaining: %d",
                    self.state.query_count,
                    self.state.budget_remaining,
                )
            elif self.state.query_count % self._progress_interval < batch_size:
                self._logger.info(
                    "[Query Progress] Used: %d / Remaining: %d",
                    self.state.query_count,
                    self.state.budget_remaining,
                )

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

        self.logger.log_event(self.state.query_count, name, safe_payload)

    def log_resource_snapshot(
        self,
        phase: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        reset_peak: bool = False,
    ) -> None:
        if self.logger is None:
            return
        self.logger.log_resource_snapshot(
            self.state.query_count,
            phase,
            device=str(self.state.metadata.get("device", "cpu")),
            payload=payload,
            reset_peak=reset_peak,
        )

    def on_checkpoint(self, query_count: int) -> None:
        self.log_event("checkpoint_reached", {"checkpoint": int(query_count)})
        if self._checkpoint_callback is not None:
            self._checkpoint_callback(int(query_count))

    def _maybe_checkpoint(self) -> None:
        if not self.checkpoints:
            return

        for checkpoint in self.checkpoints:
            if self.state.query_count >= checkpoint and checkpoint not in self._checkpoint_reached:
                self._checkpoint_reached.add(checkpoint)
                reached = sorted(self._checkpoint_reached)
                self.state.attack_state["checkpoint_reached"] = reached
                self.on_checkpoint(checkpoint)

    def _resolve_query_batch_size(self) -> Optional[int]:
        attack_cfg = self.config.get("attack", {}) or {}
        raw = attack_cfg.get("query_batch_size") or attack_cfg.get("batch_size")
        if raw is None:
            return None
        resolved = int(raw)
        return max(1, resolved)
