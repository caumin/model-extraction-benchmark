"""BenchmarkContext budget accounting tests."""

import pytest
import torch
import torch.nn as nn

from mebench.core.context import BenchmarkContext
from mebench.core.state import BenchmarkState
from mebench.oracles.oracle import Oracle


def _make_context(budget: int) -> BenchmarkContext:
    state = BenchmarkState(budget_remaining=budget, metadata={"input_shape": (3, 4, 4)})
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 10))
    victim_config = {
        "output_mode": "soft_prob",
        "temperature": 1.0,
        "input_size": [4, 4],
        "channels": 3,
    }
    oracle = Oracle(model, victim_config, state)
    config = {"budget": {"checkpoints": [2]}}
    return BenchmarkContext(state=state, oracle=oracle, config=config)


def test_context_query_updates_budget_and_checkpoint() -> None:
    ctx = _make_context(5)
    x = torch.zeros(2, 3, 4, 4)

    _ = ctx.query(x)

    assert ctx.query_count == 2
    assert ctx.budget_remaining == 3
    assert ctx.state.attack_state.get("checkpoint_reached") == [2]


def test_context_query_rejects_over_budget() -> None:
    ctx = _make_context(1)
    x = torch.zeros(2, 3, 4, 4)

    with pytest.raises(ValueError):
        _ = ctx.query(x)


def test_context_query_rejects_empty_batch() -> None:
    ctx = _make_context(1)
    x = torch.zeros(0, 3, 4, 4)

    with pytest.raises(ValueError):
        _ = ctx.query(x)


def test_context_query_chunks_oracle_calls_by_query_batch_size() -> None:
    ctx = _make_context(5)
    ctx.config.setdefault("attack", {})["query_batch_size"] = 2
    ctx._query_batch_size = ctx._resolve_query_batch_size()
    x = torch.zeros(5, 3, 4, 4)

    chunk_sizes = []
    original_query = ctx.oracle.query

    def wrapped_query(x_batch: torch.Tensor):
        chunk_sizes.append(int(x_batch.size(0)))
        return original_query(x_batch)

    ctx.oracle.query = wrapped_query  # type: ignore[method-assign]

    out = ctx.query(x)

    assert chunk_sizes == [2, 2, 1]
    assert out.y.shape == (5, 10)
    assert ctx.query_count == 5
    assert ctx.budget_remaining == 0
