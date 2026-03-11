"""Core data structures for the benchmark."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import torch


@dataclass
class QueryBatch:
    """A batch of queries to send to the oracle.

    Attributes:
        x: Tensor of images to query, shape [B, C, H, W]
        meta: Per-image metadata (origin, pool index, generator step, scores, etc.)
    """

    x: torch.Tensor
    meta: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return number of queries in batch."""
        return self.x.shape[0]


@dataclass
class OracleOutput:
    """Oracle response to a query batch.

    Attributes:
        kind: Output mode - "soft_prob" or "hard_top1"
        y: Soft probabilities [B, K] for multiclass soft_prob, positive-class
           probabilities [B, 1] for single-logit binary soft_prob, class labels
           [B] if hard_top1
    """

    kind: str  # "soft_prob" or "hard_top1"
    y: torch.Tensor

    def __post_init__(self) -> None:
        """Validate output shape matches kind."""
        if self.kind == "soft_prob":
            if self.y.ndim != 2:
                raise ValueError(f"soft_prob requires 2D tensor, got {self.y.ndim}D")
        elif self.kind == "hard_top1":
            if self.y.ndim != 1:
                raise ValueError(f"hard_top1 requires 1D tensor, got {self.y.ndim}D")
        else:
            raise ValueError(f"Unknown output mode: {self.kind}")
