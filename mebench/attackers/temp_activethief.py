"""Internal ActiveThief ablation runner with dual query paths."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from mebench.attackers.activethief import ActiveThief
from mebench.attackers.runner import AttackRunner
from mebench.core.context import BenchmarkContext
from mebench.core.state import BenchmarkState
from mebench.core.types import OracleOutput
from mebench.data.preprocessing import apply_official_preprocess_batch


@dataclass
class _BranchSpec:
    name: str
    label: str
    budget: int
    query_transform: Callable[[torch.Tensor], torch.Tensor]


class _BranchContext:
    def __init__(
        self,
        parent_ctx: BenchmarkContext,
        branch_state: BenchmarkState,
        branch_budget: int,
        query_transform: Callable[[torch.Tensor], torch.Tensor],
        on_query: Optional[Callable[[int], None]] = None,
    ) -> None:
        self._parent_ctx = parent_ctx
        self.state = branch_state
        self.oracle = parent_ctx.oracle
        self.logger = parent_ctx.logger
        self._branch_budget_remaining = int(max(0, branch_budget))
        self._query_transform = query_transform
        self._on_query = on_query

    @property
    def budget_remaining(self) -> int:
        return int(min(self._branch_budget_remaining, self._parent_ctx.budget_remaining))

    @property
    def branch_budget_remaining(self) -> int:
        return int(self._branch_budget_remaining)

    @property
    def branch_consumed(self) -> int:
        return int(max(0, self.state.metadata.get("max_budget", 0) - self._branch_budget_remaining))

    def query(self, x_batch: torch.Tensor, meta: Optional[dict] = None) -> OracleOutput:
        if int(x_batch.size(0)) > int(self.budget_remaining):
            raise ValueError(
                f"Branch budget exceeded: requested={int(x_batch.size(0))}, "
                f"remaining={int(self.budget_remaining)}"
            )
        x_query = self._query_transform(x_batch)
        oracle_out = self._parent_ctx.query(x_query, meta=meta)
        self._branch_budget_remaining -= int(x_batch.size(0))
        self.state.query_count += int(x_batch.size(0))
        self.state.budget_remaining = int(self._branch_budget_remaining)
        if self._on_query is not None:
            self._on_query(int(x_batch.size(0)))
        return oracle_out


class TempActiveThief(AttackRunner):
    """Compare two ActiveThief variants in one run.

    - raw_query_raw_train: query victim with raw pool images, train on raw queried images
    - norm_query_raw_train: apply victim-like normalization only at query boundary,
      but keep substitute training on raw queried images
    """

    def __init__(self, config: dict, state: BenchmarkState):
        super().__init__(config, state)
        self._warned_missing_norm = False
        self._norm_cache: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def _identity(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _apply_victim_like_normalization(self, x_unit: torch.Tensor) -> torch.Tensor:
        victim_cfg = self.state.metadata.get("victim_config", {}) or {}
        profile = victim_cfg.get("official_preprocess_profile")
        if profile:
            return apply_official_preprocess_batch(x_unit, str(profile))

        normalization = victim_cfg.get("normalization")
        if isinstance(normalization, dict):
            mean = normalization.get("mean")
            std = normalization.get("std")
            if isinstance(mean, (list, tuple)) and isinstance(std, (list, tuple)):
                channels = int(x_unit.size(1))
                if len(mean) == channels and len(std) == channels:
                    key = (str(x_unit.device), channels)
                    cached = self._norm_cache.get(key)
                    if cached is None:
                        mean_t = torch.tensor(mean, dtype=x_unit.dtype, device=x_unit.device).view(
                            1, channels, 1, 1
                        )
                        std_t = torch.tensor(std, dtype=x_unit.dtype, device=x_unit.device).view(
                            1, channels, 1, 1
                        )
                        self._norm_cache[key] = (mean_t, std_t)
                        cached = (mean_t, std_t)
                    mean_t, std_t = cached
                    return (x_unit - mean_t) / std_t.clamp_min(1e-8)

        if not self._warned_missing_norm:
            self.logger.warning(
                "[TempActiveThief] No victim normalization configured; "
                "norm-query branch uses identity over raw [0,1] images."
            )
            self._warned_missing_norm = True
        return x_unit

    def _make_branch_state(self, branch_budget: int) -> BenchmarkState:
        metadata = copy.deepcopy(self.state.metadata)
        metadata["max_budget"] = int(branch_budget)
        return BenchmarkState(
            query_count=0,
            budget_remaining=int(branch_budget),
            checkpoint_reached=False,
            attack_state={},
            metadata=metadata,
        )

    def _run_branch(
        self,
        parent_ctx: BenchmarkContext,
        spec: _BranchSpec,
        device: str,
        pbar,
    ) -> tuple[Optional[torch.nn.Module], int]:
        if int(spec.budget) <= 0:
            return None, 0

        branch_state = self._make_branch_state(int(spec.budget))
        branch_attack = ActiveThief(copy.deepcopy(self.config), branch_state)
        branch_attack.ctx = self.ctx
        branch_attack.victim = self.victim

        def _branch_eval(substitute, eval_device, *, track="track_b", query_count=None):
            return AttackRunner._evaluate_current_substitute(
                branch_attack,
                substitute,
                eval_device,
                track=f"track_b_{spec.name}",
                query_count=query_count,
            )

        branch_attack._evaluate_current_substitute = _branch_eval  # type: ignore[method-assign]

        branch_ctx = _BranchContext(
            parent_ctx=parent_ctx,
            branch_state=branch_state,
            branch_budget=int(spec.budget),
            query_transform=spec.query_transform,
            on_query=lambda n: pbar.update(int(n)),
        )

        self.logger.info(
            "[TempActiveThief][%s] start (budget=%d, mode=%s)",
            spec.name,
            int(spec.budget),
            spec.label,
        )
        branch_attack.run(branch_ctx)

        model = branch_state.attack_state.get("substitute")
        if model is None:
            model = getattr(branch_attack, "substitute", None)
        consumed = int(spec.budget - branch_ctx.branch_budget_remaining)

        self.state.attack_state[f"substitute_{spec.name}"] = model
        self.state.attack_state.setdefault("ablation_query_counts", {})[spec.name] = consumed
        self.logger.info(
            "[TempActiveThief][%s] done (consumed=%d)",
            spec.name,
            consumed,
        )
        return model, consumed

    def run(self, ctx: BenchmarkContext) -> None:
        self.victim = ctx.oracle.model
        device = str(self.state.metadata.get("device", "cpu"))
        total_budget = int(ctx.budget_remaining)
        raw_budget = int(total_budget // 2)
        norm_budget = int(total_budget - raw_budget)

        specs = [
            _BranchSpec(
                name="raw_query_raw_train",
                label="query=raw, train=raw",
                budget=raw_budget,
                query_transform=self._identity,
            ),
            _BranchSpec(
                name="norm_query_raw_train",
                label="query=victim_norm(raw), train=raw",
                budget=norm_budget,
                query_transform=self._apply_victim_like_normalization,
            ),
        ]

        self.logger.info(
            "[TempActiveThief] Variants: raw_query_raw_train(query=raw,train=raw), "
            "norm_query_raw_train(query=victim_norm(raw),train=raw)"
        )

        pbar = self._create_progress_bar(total_budget, "[TempActiveThief] Ablation")
        for spec in specs:
            if ctx.budget_remaining <= 0:
                break
            self._run_branch(ctx, spec, device, pbar)
        pbar.close()

        self.state.attack_state["substitute"] = None
