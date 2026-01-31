"""Basic MAZE attack tests."""

import torch
from mebench.core.state import BenchmarkState
from mebench.core.types import QueryBatch, OracleOutput
from mebench.attackers.maze import MAZE


def test_maze_basic_flow() -> None:
    config = {
        "batch_size": 2,
        "noise_dim": 16,
        "num_classes": 10,
    }
    state = BenchmarkState()
    state.metadata = {"device": "cpu", "input_shape": (3, 32, 32)}

    attack = MAZE(config, state)
    # Use k large enough so MAZE packs >=2 base samples (avoids BN batch=1).
    query_batch = attack._select_query_batch(12, state)

    assert isinstance(query_batch, QueryBatch)
    # Selection returns exactly k images.
    assert query_batch.x.shape[0] == 12

    probs = torch.softmax(torch.randn(query_batch.x.shape[0], 10), dim=1)
    oracle_output = OracleOutput(kind="soft_prob", y=probs)
    attack._handle_oracle_output(query_batch, oracle_output, state)

    assert state.attack_state["substitute"] is not None
