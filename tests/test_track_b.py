"""Test Track B evaluation with DFME."""

import pytest
import torch
from mebench.core.state import BenchmarkState
from mebench.attackers.dfme import DFME
from mebench.core.context import BenchmarkContext
from mebench.oracles.oracle import Oracle


def test_dfme_track_b_evaluation():
    """Test Track B evaluation loop with DFME."""
    budget = 64
    num_classes = 10

    config = {
        "run": {"name": "test", "seeds": [0], "device": "cpu"},
        "victim": {
            "victim_id": "test_victim",
            "arch": "toy",
            "channels": 3,
            "num_classes": num_classes,
            "input_size": [32, 32],
            "normalization": None,
            "output_mode": "soft_prob",
            "temperature": 1.0,
        },
        "substitute": {
            "arch": "resnet18",
            "optimizer": {"name": "sgd", "lr": 0.01, "momentum": 0.9, "weight_decay": 5e-4},
            "patience": 5,
        },
        "attack": {
            "name": "dfme",
            "output_mode": "soft_prob",
            "batch_size": 8,
            "n_g_steps": 1,
            "n_s_steps": 1,
            "grad_approx_m": 1,
            "grad_approx_epsilon": 1e-3,
        },
        "budget": {"max_budget": budget, "checkpoints": [budget]},
        "dataset": {"data_mode": "data_free"},
    }

    state = BenchmarkState(budget_remaining=budget)
    state.metadata = {
        "device": "cpu",
        "input_shape": (3, 32, 32),
        "num_classes": num_classes,
        "substitute_config": config["substitute"],
        "victim_config": config["victim"],
        "dataset_config": config.get("dataset", {}),
        "max_budget": budget,
    }

    victim = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 32 * 32, num_classes),
    )
    oracle = Oracle(victim, config["victim"], state)
    ctx = BenchmarkContext(state=state, oracle=oracle, logger=None, config=config)

    attack = DFME(config["attack"], state)
    attack.run(ctx)

    assert state.query_count == budget
    assert state.budget_remaining == 0
    assert state.attack_state.get("substitute") is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
