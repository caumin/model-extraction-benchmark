"""Basic MAZE attack tests."""

import torch
import torch.nn as nn

from mebench.attackers.maze import MAZE
from mebench.core.context import BenchmarkContext
from mebench.core.state import BenchmarkState
from mebench.oracles.oracle import Oracle


def test_maze_basic_flow() -> None:
    budget = 10
    num_classes = 10

    attack_config = {
        "batch_size": 2,
        "noise_dim": 16,
        "num_classes": num_classes,
        "n_g_steps": 1,
        "n_c_steps": 1,
        "n_r_steps": 0,
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
        "attack": {"name": "maze", "output_mode": "soft_prob"},
        "budget": {"max_budget": budget, "checkpoints": [5, 10]},
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

    attack = MAZE(attack_config, state)
    attack.run(ctx)

    assert state.query_count == budget
    assert state.budget_remaining == 0
    assert state.attack_state.get("substitute") is not None
    assert state.attack_state.get("checkpoint_reached") == [5, 10]
