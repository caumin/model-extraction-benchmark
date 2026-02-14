"""Test ActiveThief selection strategies."""

import pytest
import torch
import torch.nn.functional as F
from typing import Optional
from torch.utils.data import TensorDataset
from mebench.core.state import BenchmarkState
from mebench.core.types import QueryBatch, OracleOutput
from mebench.attackers.activethief import ActiveThief


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


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


def test_activethief_rounds_fallback_from_iterations():
    """ActiveThief uses `iterations` as fallback when `rounds` is unset."""

    config = {"iterations": 7}
    state = BenchmarkState()
    state.metadata = {"max_budget": 7000}

    attack = ActiveThief(config, state)

    assert attack.rounds == 7




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
    state.metadata = {"device": DEVICE, "dataset_config": config["dataset"]}

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
    state.metadata = {"device": DEVICE, "dataset_config": {"name": "CIFAR10", "seed_size": 100}}

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
    state.metadata = {"device": DEVICE, "dataset_config": {"name": "CIFAR10", "seed_size": 100}}

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
    import types

    config = {
        "strategy": "dfal",
        "batch_size": 32,
        "max_epochs": 2,
        "patience": 10,
        "l2_reg": 0.001,
        "initial_seed_size": 10,
    }
    state = BenchmarkState()
    state.metadata = {"device": DEVICE, "dataset_config": {"name": "CIFAR10", "seed_size": 100}}

    attack = ActiveThief(config, state)

    # Collect initial seed
    for _ in range(2):
        attack._select_query_batch(5, state)
        x = attack.pool_dataset[0][0].unsqueeze(0)
        y = torch.randn(1, 10)
        attack._handle_oracle_output(QueryBatch(x=x), OracleOutput(kind="soft_prob", y=y), state)

    # Train and select with dfal
    attack.train_substitute(state)

    # DFAL selection uses DeepFool-style scoring which is expensive on CPU.
    # For this unit test, mock the DFAL selector to keep runtime bounded.
    def fake_select_dfal(self, state_local, k_local):
        _ = state_local
        return self.unlabeled_indices[: int(k_local)]

    attack._select_dfal = types.MethodType(fake_select_dfal, attack)
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
    # Force empty pool (bypass dataset setup)
    state.attack_state["initialized"] = True
    attack.unlabeled_indices = []

    query_batch = attack._select_query_batch(10, state)
    assert query_batch.x.numel() == 0
    assert query_batch.meta.get("status") == "exhausted"

    print("ActiveThief pool exhausted test passed")


def test_activethief_default_step_size_uses_training_budget_split():
    class DummyContext:
        def __init__(self, state: BenchmarkState):
            self.state = state
            self.oracle = type("_DummyOracle", (), {"model": None})()
            self.query_sizes = []
            self.reserved_budget = {"seed": 0, "val": 0}

        @property
        def budget_remaining(self) -> int:
            return int(self.state.budget_remaining)

        def query(self, x: torch.Tensor, meta: Optional[dict] = None) -> OracleOutput:
            batch_size = int(x.shape[0])
            assert batch_size <= int(self.state.budget_remaining)
            self.state.budget_remaining -= batch_size
            self.state.query_count += batch_size
            self.query_sizes.append(batch_size)
            # Emit a valid probability output for ActiveThief observer path.
            return OracleOutput(kind="soft_prob", y=torch.full((batch_size, 10), 1.0 / 10))

    budget = 1000
    pool_size = 1000
    state = BenchmarkState()
    state.budget_remaining = budget
    state.query_count = 0
    state.metadata = {
        "device": "cpu",
        "num_classes": 10,
        "max_budget": budget,
        "dataset_config": {"name": "CIFAR10", "seed_size": pool_size},
    }

    # Use a fixed seed budget so the step-size formula should follow:
    # B=1000, B_seed=100, B_val=200 => B_train=700, N=10 => k=70.
    attack = ActiveThief(
        {"strategy": "random", "batch_size": 32, "initial_seed_size": 100, "rounds": 10},
        state,
    )

    # Full dataset is synthetic and already prepared so we can run the lightweight loop path.
    attack.pool_dataset = TensorDataset(
        torch.zeros(pool_size, 3, 32, 32),
        torch.zeros(pool_size, dtype=torch.long),
    )
    attack.unlabeled_indices = list(range(pool_size))
    state.attack_state["initialized"] = True

    # Replace expensive internals with no-op stubs for this budget accounting test.
    attack._setup_datasets = lambda state: None
    attack._train_substitute = lambda state: None
    attack._evaluate_current_substitute = lambda *args, **kwargs: None

    def bootstrap_stub(ctx: DummyContext, state: BenchmarkState) -> None:
        total_budget = int(state.metadata.get("max_budget") or ctx.state.budget_remaining)
        seed_target, val_target = attack._resolve_seed_and_validation_targets(
            total_budget=total_budget,
            default_seed_ratio=0.1,
            default_validation_ratio=0.2,
        )
        ctx.reserved_budget["seed"] = min(int(seed_target), int(ctx.state.budget_remaining), len(attack.unlabeled_indices))
        for _ in range(ctx.reserved_budget["seed"]):
            if attack.unlabeled_indices:
                attack.unlabeled_indices.pop()
        ctx.state.budget_remaining -= ctx.reserved_budget["seed"]

        ctx.reserved_budget["val"] = min(int(val_target), int(ctx.state.budget_remaining), len(attack.unlabeled_indices))
        for _ in range(ctx.reserved_budget["val"]):
            if attack.unlabeled_indices:
                attack.unlabeled_indices.pop()
        ctx.state.budget_remaining -= ctx.reserved_budget["val"]

    attack._bootstrap_seed_and_validation_sets = bootstrap_stub

    ctx = DummyContext(state)
    attack.run(ctx)

    assert sum(ctx.query_sizes) == 700
    assert ctx.query_sizes[0] == 70
    assert len(ctx.query_sizes) == 10
    assert ctx.reserved_budget == {"seed": 100, "val": 200}



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
