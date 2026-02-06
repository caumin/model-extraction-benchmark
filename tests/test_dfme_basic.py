"""Basic DFME functionality test."""

import pytest
import torch
import torch.nn as nn

from mebench.attackers.dfme import DFME
from mebench.core.context import BenchmarkContext
from mebench.core.state import BenchmarkState
from mebench.oracles.oracle import Oracle


def test_dfme_basic_flow() -> None:
    """DFME should consume budget via ctx.query and produce a substitute."""

    budget = 4
    num_classes = 10

    attack_config = {
        "batch_size": 16,
        "n_g_steps": 1,
        "n_s_steps": 1,
        "grad_approx_m": 1,
        "grad_approx_epsilon": 1e-3,
    }

    full_config = {
        "run": {"device": "cpu"},
        "victim": {
            "output_mode": "soft_prob",
            "temperature": 1.0,
            "channels": 3,
            "num_classes": num_classes,
            "input_size": [32, 32],
        },
        "attack": {"name": "dfme", "output_mode": "soft_prob"},
        "budget": {"max_budget": budget, "checkpoints": [budget]},
        "dataset": {"data_mode": "data_free"},
        "substitute": {
            "arch": "resnet18",
            "optimizer": {"name": "sgd", "lr": 0.01, "momentum": 0.9, "weight_decay": 5e-4},
        },
    }

    class RangeCheckedVictim(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, num_classes))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            assert x.min().item() >= -1e-6
            assert x.max().item() <= 1.0 + 1e-6
            return self.net(x)

    state = BenchmarkState(budget_remaining=budget)
    state.metadata = {
        "device": "cpu",
        "input_shape": (3, 32, 32),
        "num_classes": num_classes,
        "max_budget": budget,
        "victim_config": full_config["victim"],
        "substitute_config": full_config["substitute"],
        "dataset_config": full_config.get("dataset", {}),
    }

    victim = RangeCheckedVictim()
    oracle = Oracle(victim, full_config["victim"], state)
    ctx = BenchmarkContext(state=state, oracle=oracle, logger=None, config=full_config)

    attack = DFME(attack_config, state)
    attack.run(ctx)

    assert state.query_count == budget
    assert state.budget_remaining == 0
    assert state.attack_state.get("substitute") is not None
    assert state.attack_state.get("checkpoint_reached") == [budget]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
