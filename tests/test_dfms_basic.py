"""Basic DFMS-HL attack tests."""

import torch
from mebench.core.state import BenchmarkState
from mebench.core.types import OracleOutput
from mebench.attackers.dfms import DFMSHL


def test_dfms_basic_flow(tmp_path) -> None:
    config = {
        "batch_size": 2,
        "noise_dim": 16,
        "num_classes": 10,
        "clone_arch": "lenet",  # Use small model for fast CPU testing
        "proxy_dataset": {
            "name": "CIFAR10",
            "data_mode": "seed",
            "seed_size": 10,
            "num_workers": 0,  # Avoid multiprocessing hang on Windows
        },
    }
    state = BenchmarkState()
    state.metadata = {"device": "cpu", "input_shape": (3, 32, 32)}

    attack = DFMSHL(config, state)
    query_batch = attack.propose(2, state)
    labels = torch.randint(0, 10, (2,))
    oracle_output = OracleOutput(kind="hard_top1", y=labels)
    attack.observe(query_batch, oracle_output, state)

    assert state.attack_state["substitute"] is not None
