"""Basic GAME attack tests."""

import torch
from mebench.core.state import BenchmarkState
from mebench.core.types import OracleOutput
from mebench.attackers.game import GAME


def test_game_basic_flow() -> None:
    config = {
        "batch_size": 2,
        "noise_dim": 16,
        "num_classes": 10,
        "acs_strategy": "uncertainty",
        "proxy_dataset": {
            "name": "CIFAR10",
            "data_mode": "seed",
            "seed_size": 10,
        },
    }
    state = BenchmarkState()
    state.metadata = {"device": "cpu", "input_shape": (3, 32, 32)}

    attack = GAME(config, state)
    x_query, meta = attack._select_query_batch(2, state)
    assert "y_g" in meta
    assert meta["y_g"].shape[0] == 2
    probs = torch.softmax(torch.randn(2, 10), dim=1)
    oracle_output = OracleOutput(kind="soft_prob", y=probs)
    attack._handle_oracle_output(x_query, meta, oracle_output, state)

    assert state.attack_state["substitute"] is not None
