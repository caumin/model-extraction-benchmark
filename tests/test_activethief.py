"""Test ActiveThief selection strategies."""

import pytest
import torch
import torch.nn.functional as F
from mebench.core.state import BenchmarkState
from mebench.core.types import QueryBatch, OracleOutput
from mebench.attackers.activethief import ActiveThief


def test_activethief_initialization():
    """Test ActiveThief initialization."""
    config = {
        "strategy": "uncertainty",
        "batch_size": 32,
        "max_epochs": 2,
        "patience": 10,
        "l2_reg": 0.001,
        "initial_seed_size": 10,
    }
    state = BenchmarkState()

    attack = ActiveThief(config, state)

    # Check state initialization (no _initialize_state method)
    assert "labeled_indices" in state.attack_state
    assert "unlabeled_indices" in state.attack_state
    assert state.attack_state["labeled_indices"] == []
    assert len(state.attack_state["unlabeled_indices"]) == 10000

    print("ActiveThief initialized successfully")


def test_activethief_select_first_round():
    """Test ActiveThief selection (first round - random seed)."""
    config = {
        "dataset": {
            "data_mode": "seed",
            "name": "CIFAR10",
            "seed_size": 100,
        },
        "strategy": "uncertainty",
        "batch_size": 32,
        "max_epochs": 2,
        "patience": 10,
        "l2_reg": 0.001,
        "initial_seed_size": 10,
    }
    state = BenchmarkState()
    state.metadata = {"device": "cpu", "dataset_config": config["dataset"]}

    attack = ActiveThief(config, state)

    # Select 5 queries (less than initial seed size)
    k = 5
    query_batch = attack._select_query_batch(k, state)

    # Check return type
    assert isinstance(query_batch, QueryBatch)
    assert query_batch.x.shape[0] == k

    # Should be random selection (no substitute trained yet)
    assert "strategy" in query_batch.meta
    assert len(state.attack_state["labeled_indices"]) == k

    print(f"ActiveThief first round: {k} queries selected")


def test_activethief_uncertainty_strategy():
    """Test uncertainty (entropy-based) selection."""
    config = {
        "strategy": "uncertainty",
        "batch_size": 32,
        "max_epochs": 2,
        "patience": 10,
        "l2_reg": 0.001,
        "initial_seed_size": 10,
    }
    state = BenchmarkState()
    state.metadata = {"device": "cpu", "dataset_config": {"name": "CIFAR10", "seed_size": 100}}

    attack = ActiveThief(config, state)

    # First, collect initial seed (10 queries)
    for _ in range(2):  # 2 rounds to get 10 queries
        attack._select_query_batch(5, state)
        # Simulate oracle response
        x = attack.pool_dataset[0][0].unsqueeze(0)  # Get first sample
        y = torch.randn(1, 10)  # Soft labels
        attack._handle_oracle_output(QueryBatch(x=x), OracleOutput(kind="soft_prob", y=y), state)

    # Now train substitute
    attack.train_substitute(state)

    # Propose with uncertainty strategy (substitute now trained)
    k = 5
    query_batch = attack._select_query_batch(k, state)

    assert query_batch.x.shape[0] == k
    assert len(state.attack_state["labeled_indices"]) == 15  # 10 + 5

    print("ActiveThief uncertainty strategy passed")


def test_activethief_kcenter_strategy():
    """Test K-center (diverse selection) strategy."""
    config = {
        "strategy": "k_center",
        "batch_size": 32,
        "max_epochs": 2,
        "patience": 10,
        "l2_reg": 0.001,
        "initial_seed_size": 10,
    }
    state = BenchmarkState()
    state.metadata = {"device": "cpu", "dataset_config": {"name": "CIFAR10", "seed_size": 100}}

    attack = ActiveThief(config, state)

    # Collect initial seed
    for _ in range(2):
        attack._select_query_batch(5, state)
        x = attack.pool_dataset[0][0].unsqueeze(0)
        y = torch.randn(1, 10)
        attack._handle_oracle_output(QueryBatch(x=x), OracleOutput(kind="soft_prob", y=y), state)

    # Train and select with k_center
    attack.train_substitute(state)
    query_batch = attack._select_query_batch(5, state)

    assert query_batch.x.shape[0] == 5
    assert len(state.attack_state["labeled_indices"]) == 15

    print("ActiveThief K-center strategy passed")


def test_activethief_dfal_strategy():
    """Test DFAL (decision boundary) selection strategy."""
    config = {
        "strategy": "dfal",
        "batch_size": 32,
        "max_epochs": 2,
        "patience": 10,
        "l2_reg": 0.001,
        "initial_seed_size": 10,
    }
    state = BenchmarkState()
    state.metadata = {"device": "cpu", "dataset_config": {"name": "CIFAR10", "seed_size": 100}}

    attack = ActiveThief(config, state)

    # Collect initial seed
    for _ in range(2):
        attack._select_query_batch(5, state)
        x = attack.pool_dataset[0][0].unsqueeze(0)
        y = torch.randn(1, 10)
        attack._handle_oracle_output(QueryBatch(x=x), OracleOutput(kind="soft_prob", y=y), state)

    # Train and select with dfal
    attack.train_substitute(state)
    query_batch = attack._select_query_batch(5, state)

    assert query_batch.x.shape[0] == 5
    assert len(state.attack_state["labeled_indices"]) == 15

    print("ActiveThief DFAL strategy passed")


def test_activethief_pool_exhausted():
    """Test ActiveThief behavior when pool is exhausted."""
    config = {
        "strategy": "uncertainty",
        "initial_seed_size": 10,
    }
    state = BenchmarkState()
    attack = ActiveThief(config, state)
    state.attack_state["unlabeled_indices"] = []  # Force empty pool

    # Propose from exhausted pool should now raise ValueError
    with pytest.raises(ValueError, match="Query pool exhausted"):
        attack._select_query_batch(10, state)

    print("ActiveThief pool exhausted test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
