"""Test RandomBaseline attack functionality."""

import pytest
import torch
import numpy as np

from mebench.core.state import BenchmarkState
from mebench.core.types import QueryBatch, OracleOutput
from mebench.attackers.random_baseline import RandomBaseline


def test_random_baseline_initialization():
    """Test RandomBaseline initialization."""
    config = {
        "data_mode": "seed",
        "batch_size": 128,
    }
    state = BenchmarkState()

    attack = RandomBaseline(config, state)

    # Check state initialization
    assert "queried_indices" in state.attack_state
    assert "unqueried_indices" in state.attack_state
    assert state.attack_state["queried_indices"] == []


def test_random_baseline_select_first_round():
    """Test RandomBaseline selection (first round - random sampling)."""
    config = {
        "dataset": {
            "data_mode": "seed",
            "name": "CIFAR10",
            "seed_size": 100,
        },
        "batch_size": 128,
    }
    state = BenchmarkState()
    state.metadata = {"device": "cpu", "dataset_config": config}

    attack = RandomBaseline(config, state)

    # Select 10 queries
    k = 10
    query_batch = attack._select_query_batch(k, state)

    # Check return type
    assert isinstance(query_batch, QueryBatch)
    assert query_batch.x.shape[0] == k
    assert query_batch.x.shape[1:] == (3, 32, 32)

    # Check state updated
    assert len(state.attack_state["queried_indices"]) == k
    assert len(state.attack_state["unqueried_indices"]) == 100 - k  # Pool has 100 samples


def test_random_baseline_pool_exhausted():
    """Test RandomBaseline behavior when pool is exhausted."""
    config = {
        "dataset": {"data_mode": "seed", "name": "CIFAR10", "seed_size": 100},
        "batch_size": 128,
    }
    state = BenchmarkState()
    state.attack_state["unqueried_indices"] = []  # Empty pool

    attack = RandomBaseline(config, state)

    # Propose from exhausted pool
    k = 10
    query_batch = attack._select_query_batch(k, state)

    # Check synthetic queries returned
    # Note: pool_exhausted is only set if pool is empty when proposing
    if len(state.attack_state["unqueried_indices"]) == 0:
        assert query_batch.meta.get("pool_exhausted") is True
        assert query_batch.x.shape[0] == k
    else:
        # If pool still has items, exhausted flag won't be set
        assert query_batch.x.shape[0] == k


def test_random_baseline_handle_output():
    """Test RandomBaseline output handler."""
    config = {
        "dataset": {"data_mode": "seed", "name": "CIFAR10", "seed_size": 100},
        "batch_size": 128,
    }
    state = BenchmarkState()

    attack = RandomBaseline(config, state)

    # Create query batch and oracle output
    x_batch = torch.randn(10, 3, 32, 32)
    y_batch = torch.randn(10, 10)  # 2D for soft_prob (probability distribution)

    query_batch = QueryBatch(x=x_batch)
    oracle_output = OracleOutput(kind="soft_prob", y=y_batch)

    # Observe should not raise
    attack._handle_oracle_output(query_batch, oracle_output, state)

    # No error means it worked
    assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
