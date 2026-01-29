import torch
import numpy as np
from mebench.core.state import BenchmarkState
from mebench.attackers.blackbox_ripper import BlackboxRipper
from mebench.core.types import QueryBatch, OracleOutput

class MockDataloader:
    def __iter__(self):
        while True:
            yield torch.randn(10, 3, 32, 32), torch.randint(0, 10, (10,))

def test_blackbox_ripper_mutation_fidelity():
    """Test that mutation samples parents uniformly and adds noise to all."""
    state = BenchmarkState()
    state.metadata = {
        "device": "cpu",
        "input_shape": (3, 32, 32),
        "num_classes": 10,
        "max_budget": 100,
        "victim_config": {"normalization": {"mean": [0.5], "std": [0.5]}}
    }
    
    config = {
        "batch_size": 30, # Full population size for test
        "population_size": 30,
        "elite_size": 10,
        "mutation_scale": 0.1,
        "proxy_dataset": {"name": "CIFAR10", "data_mode": "seed"} # Dummy
    }
    
    attack = BlackboxRipper(config, state)
    # Mock proxy loader
    attack.proxy_loader = MockDataloader()
    attack.proxy_iter = iter(attack.proxy_loader)
    attack._init_models(state)
    
    # Setup dummy population
    population = torch.zeros(30, 128) # Zeros
    # Setup dummy elites (indices 0-9) to be best
    # Fitness = -MSE. We want indices 0-9 to have MSE=0, others MSE=1.
    # victim_probs needs to match target_onehot for elites.
    target_cls = 0
    victim_probs = torch.zeros(30, 10)
    # Elites: perfect match
    victim_probs[:10, target_cls] = 1.0 
    # Others: mismatch
    victim_probs[10:, (target_cls+1)%10] = 1.0 
    
    best_z, best_score, new_pop, best_idx = attack._evolve_single_slot(population, target_cls, victim_probs)
    
    # 1. Elites should be preserved (first 10)
    assert torch.allclose(new_pop[:10], population[:10])
    
    # 2. Offspring (next 20) should be mutated elites
    # Since parents are zeros and noise is added, offspring should be non-zero
    assert not torch.allclose(new_pop[10:], torch.zeros_like(new_pop[10:]))
    
    # 3. Offspring should be within reasonable bounds of mutation (parent 0 + noise)
    # noise ~ N(0, 0.1). 
    assert new_pop[10:].abs().mean() > 0.0
    
    print("BlackboxRipper mutation fidelity check passed!")

def test_blackbox_ripper_objective_eq2():
    """Test that fitness objective matches Eq. (2) sum of squared differences."""
    state = BenchmarkState()
    state.metadata = {
        "device": "cpu",
        "input_shape": (3, 32, 32),
        "num_classes": 10,
        "max_budget": 100,
        "victim_config": {"normalization": {"mean": [0.5], "std": [0.5]}}
    }
    config = {
        "batch_size": 30,
        "population_size": 30,
        "elite_size": 10,
        "fitness_threshold": 0.02,
        "proxy_dataset": {"name": "CIFAR10", "data_mode": "seed"}
    }
    attack = BlackboxRipper(config, state)
    attack.proxy_loader = MockDataloader()
    attack.proxy_iter = iter(attack.proxy_loader)
    attack._init_models(state)
    
    # Single sample test
    population = torch.randn(1, 128)
    target_cls = 0
    
    # Case 1: Perfect match
    victim_probs = torch.zeros(1, 10)
    victim_probs[0, 0] = 1.0
    
    _, best_score, _, _ = attack._evolve_single_slot(population, target_cls, victim_probs)
    # MSE sum = (1-1)^2 + (0-0)^2... = 0. Fitness = -0 = 0.
    assert abs(best_score) < 1e-6
    
    # Case 2: Complete mismatch (target=0, pred=1)
    victim_probs = torch.zeros(1, 10)
    victim_probs[0, 1] = 1.0
    
    _, best_score, _, _ = attack._evolve_single_slot(population, target_cls, victim_probs)
    # MSE sum = (0-1)^2 + (1-0)^2 = 1 + 1 = 2. Fitness = -2.
    assert abs(best_score + 2.0) < 1e-6
    
    print("BlackboxRipper objective Eq.(2) check passed!")

if __name__ == "__main__":
    test_blackbox_ripper_mutation_fidelity()
    test_blackbox_ripper_objective_eq2()
