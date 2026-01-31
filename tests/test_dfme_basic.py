"""Basic DFME functionality test."""

import pytest
import torch
from mebench.core.state import BenchmarkState
from mebench.core.types import QueryBatch, OracleOutput
from mebench.attackers.dfme import DFME


def test_dfme_basic_initialization():
    """Test DFME basic initialization."""
    config = {
        "batch_size": 16,  # Small batch for quick test
        "student_lr": 0.001,  # Reduced LR
        "generator_lr": 0.0005,
        "n_g_steps": 1,
        "n_s_steps": 1,
    }
    state = BenchmarkState()
    state.metadata = {"device": "cpu", "input_shape": (3, 32, 32)}

    attack = DFME(config, state)

    # Check initialization - generator should be None initially
    assert attack.generator is None
    assert attack.student is None
    assert attack.generator_optimizer is None
    assert attack.student_optimizer is None

    # Propose to initialize both models (generator and student)
    _ = attack._select_query_batch(1, state)

    # Now both models should be initialized
    assert attack.generator is not None
    assert attack.student is not None

    # Check devices
    gen_device = next(attack.generator.parameters()).device
    student_device = next(attack.student.parameters()).device

    print(f"DFME initialized. Generator on {gen_device}, Student on {student_device}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
