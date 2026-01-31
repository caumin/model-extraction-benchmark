import torch
import numpy as np
from mebench.core.state import BenchmarkState
from mebench.attackers.maze import MAZE
from mebench.core.types import QueryBatch, OracleOutput

def test_maze_algorithm_cycle():
    config = {
        "batch_size": 2,
        "n_g_steps": 1,
        "n_c_steps": 2,
        "grad_approx_m": 2,
        "generator_lr": 1e-4,
        "clone_lr": 0.1,
        "grad_approx_epsilon": 1e-3,
        "noise_dim": 8,
        "replay_max": 10
    }
    state = BenchmarkState()
    state.metadata = {
        "device": "cpu",
        "input_shape": (3, 32, 32),
        "num_classes": 2,
        "max_budget": 100,
        "victim_config": {"normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}}
    }
    
    attack = MAZE(config, state)
    
    # 1 Cycle = (1 * (2+1)) + 2 = 5 batches
    # 5 batches * 2 samples = 10 queries
    
    # Propose 10 queries (exactly one cycle)
    qb = attack._select_query_batch(10, state)
    assert qb.x.shape[0] == 10
    assert "blocks" in qb.meta
    blocks = qb.meta["blocks"]
    assert len(blocks) == 5
    
    # Block types should be: G_BASE, G_PERT, G_PERT, C_BASE, C_BASE
    assert blocks[0]["type"] == "G_BASE"
    assert blocks[1]["type"] == "G_PERT"
    assert blocks[2]["type"] == "G_PERT"
    assert blocks[3]["type"] == "C_BASE"
    assert blocks[4]["type"] == "C_BASE"
    
    # Simulate Oracle response
    y = torch.softmax(torch.randn(10, 2), dim=1)
    oo = OracleOutput(kind="soft_prob", y=y)
    
    # Observe should update both models
    attack._handle_oracle_output(qb, oo, state)
    
    # C_BASE count = 2 blocks * 2 samples = 4.
    assert state.attack_state["replay_count"] == 4 
    
    print("MAZE refactor sanity check passed!")

if __name__ == "__main__":
    test_maze_algorithm_cycle()
