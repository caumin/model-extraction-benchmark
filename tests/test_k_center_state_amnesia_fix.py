"""Integration test for K-Center state amnesia fix in ActiveThief."""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from mebench.attackers.activethief import ActiveThief
from mebench.core.state import BenchmarkState


def test_k_center_state_amnesia_integration():
    """Test that K-Center correctly uses labeled data as centers."""
    torch.manual_seed(42)
    
    # Clear test: single labeled center, multiple unlabeled points
    # Labeled: [0.9, 0.1] (clear center)
    # Unlabeled: mix of near and far points
    
    labeled_probs = torch.tensor([[0.9, 0.1]])  # Single center
    unlabeled_probs = torch.tensor([
        [0.85, 0.15],  # Near (distance: 0.07)
        [0.1, 0.9],    # Far (distance: 1.13) 
        [0.95, 0.05],  # Very near (distance: 0.07)
        [0.2, 0.8],    # Far (distance: 0.99)
    ])
    
    # Setup attack
    from mebench.attackers.activethief import ActiveThief
    attack_config = {"strategy": "k_center", "batch_size": 16}
    state = BenchmarkState()
    state.metadata = {"num_classes": 2, "device": "cpu"}
    
    attack = ActiveThief(attack_config, state)
    
    # Select 2 farthest samples
    selected_indices = attack._select_k_center(unlabeled_probs, labeled_probs, 2)
    
    print(f"Selected indices: {selected_indices}")
    
    # The farthest points should be selected
    # Distances: [0.07, 1.13, 0.07, 0.99]
    # Top 2 farthest: indices 1 and 3
    expected = {1, 3}  # The two farthest points
    actual = set(selected_indices)
    
    print(f"Expected (farthest): {expected}")
    print(f"Actual: {actual}")
    
    assert actual == expected, f"K-Center should select farthest points {expected}, got {actual}"
    
    print("K-Center state amnesia integration test passed!")


if __name__ == "__main__":
    test_k_center_state_amnesia_integration()