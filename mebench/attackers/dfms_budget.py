"""Budget planning utilities for DFMS-HL stages."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DFMSBudgetPlan:
    """Planned query allocation across DFMS-HL stages."""

    mode: str
    total_budget: int
    stage2_target_queries: int
    stage4_target_queries: int
    stage5_target_queries: int
    proxy_subset_size: int
    stage5_planned_epochs: int


def planned_stage5_epochs(stage5_target_queries: int, proxy_subset_size: int) -> int:
    """Return minimum alternate epochs needed for a target budget."""

    target = max(0, int(stage5_target_queries))
    subset = max(1, int(proxy_subset_size))
    if target <= 0:
        return 0
    return int(math.ceil(float(target) / float(subset)))
