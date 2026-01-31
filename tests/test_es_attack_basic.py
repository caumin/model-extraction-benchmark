"""Basic ES Attack tests."""

import torch
from mebench.core.state import BenchmarkState
from mebench.core.types import OracleOutput
from mebench.attackers.es_attack import ESAttack


def test_es_attack_dnn_syn_flow() -> None:
    config = {
        "batch_size": 2,
        "noise_dim": 16,
        "num_classes": 10,
        "synthesis_mode": "dnn_syn",
    }
    state = BenchmarkState()
    state.metadata = {"device": "cpu", "input_shape": (3, 32, 32)}

    attack = ESAttack(config, state)
    x_query, meta = attack._select_query_batch(2, state)
    probs = torch.softmax(torch.randn(2, 10), dim=1)
    oracle_output = OracleOutput(kind="soft_prob", y=probs)
    attack._handle_oracle_output(x_query, meta, oracle_output, state)

    assert state.attack_state["substitute"] is not None


def test_es_attack_opt_syn_flow() -> None:
    config = {
        "batch_size": 2,
        "noise_dim": 16,
        "num_classes": 10,
        "synthesis_mode": "opt_syn",
        "opt_steps": 2,
    }
    state = BenchmarkState()
    state.metadata = {"device": "cpu", "input_shape": (3, 32, 32)}

    attack = ESAttack(config, state)
    x_query, meta = attack._select_query_batch(2, state)
    probs = torch.softmax(torch.randn(2, 10), dim=1)
    oracle_output = OracleOutput(kind="soft_prob", y=probs)
    attack._handle_oracle_output(x_query, meta, oracle_output, state)

    assert state.attack_state["substitute"] is not None
