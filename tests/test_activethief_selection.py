"""Deterministic selection tests for ActiveThief strategies."""

import types
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from mebench.attackers.activethief import ActiveThief
from mebench.core.state import BenchmarkState


class IndexedDataset(Dataset):
    """Dataset backed by a list of tensors."""

    def __init__(self, items):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx], 0


class EntropyModel(nn.Module):
    """Model that returns high/low entropy logits based on input mean."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        means = x.view(x.size(0), -1).mean(dim=1)
        logits = torch.zeros(x.size(0), 2)
        for i, m in enumerate(means):
            if m.item() < 1.0:
                logits[i] = torch.tensor([0.0, 0.0])
            else:
                logits[i] = torch.tensor([10.0, 0.0])
        return logits


class FeatureModel(nn.Module):
    """Model exposing a simple features method."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Linear(1, 1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.view(x.size(0), -1)
        return torch.zeros(flat.size(0), 2)


def _make_attack() -> ActiveThief:
    attack = ActiveThief.__new__(ActiveThief)
    attack.strategy = "uncertainty"
    attack.pool_dataset = None
    return attack


def test_select_uncertainty_prefers_high_entropy() -> None:
    attack = _make_attack()
    state = BenchmarkState()

    high_entropy = torch.zeros(1, 1, 2)
    low_entropy = torch.ones(1, 1, 2) * 10.0
    attack.pool_dataset = IndexedDataset([high_entropy, low_entropy])

    state.attack_state["unlabeled_indices"] = [0, 1]
    state.attack_state["substitute"] = EntropyModel()

    selected = attack._select_uncertainty(1, state)

    assert selected == [0]


def test_select_k_center_prefers_farthest_point() -> None:
    attack = _make_attack()
    state = BenchmarkState()

    # Test with labeled data as centers
    labeled_probs = torch.tensor([[0.9, 0.1]])  # High confidence in class 0
    unlabeled_probs = torch.tensor([
        [0.9, 0.1],  # Similar to labeled (near)
        [0.1, 0.9],  # Different from labeled (far)
    ])

    # Should select the farthest point from existing centers
    selected = attack._select_k_center(unlabeled_probs, labeled_probs, 1)
    assert selected == [1]  # Index of far point


def test_select_k_center_state_amnesia_fix() -> None:
    """Test that K-Center uses labeled data as initial centers (state amnesia fix)."""
    attack = _make_attack()
    
    # Create probability vectors with clear structure
    # Labeled points: cluster around [0.9, 0.1]
    labeled_probs = torch.tensor([
        [0.9, 0.1],  # Center 1
        [0.85, 0.15],  # Center 2 (close to center 1)
    ])
    
    # Unlabeled points: one near labeled cluster, one far
    unlabeled_probs = torch.tensor([
        [0.88, 0.12],  # Near labeled cluster
        [0.1, 0.9],   # Far from labeled cluster
        [0.2, 0.8],   # Also far, but closer to [0.1, 0.9]
    ])
    
    # Select 2 points - should pick the farthest ones from existing centers
    selected = attack._select_k_center(unlabeled_probs, labeled_probs, 2)
    
    # Should select the points farthest from existing centers
    # The farthest points are indices 1 and 2 (both far from labeled cluster)
    assert set(selected) == {1, 2}


def test_select_k_center_no_labeled_fallback() -> None:
    """Test K-Center behavior when no labeled data exists yet."""
    attack = _make_attack()
    
    # No labeled data
    labeled_probs = torch.empty(0, 2)
    unlabeled_probs = torch.tensor([
        [0.9, 0.1],
        [0.1, 0.9],
        [0.5, 0.5],
    ])
    
    # Should fall back to random initialization within unlabeled
    selected = attack._select_k_center(unlabeled_probs, labeled_probs, 2)
    
    # Should select 2 distinct indices from unlabeled
    assert len(selected) == 2
    assert len(set(selected)) == 2
    assert all(0 <= idx < len(unlabeled_probs) for idx in selected)


def test_select_dfal_prefers_smallest_perturbation() -> None:
    attack = _make_attack()
    state = BenchmarkState()

    close = torch.tensor([[[0.1, 0.1]]])
    far = torch.tensor([[[1.0, 1.0]]])
    attack.pool_dataset = IndexedDataset([close, far])

    state.attack_state["unlabeled_indices"] = [0, 1]
    state.attack_state["substitute"] = FeatureModel()

    def fake_compute_input_gradient(self, model, x, target_class):
        scale = float(x.mean().item())
        return torch.ones_like(x) * scale

    attack._compute_input_gradient = types.MethodType(fake_compute_input_gradient, attack)

    selected = attack._select_dfal(1, state)

    assert selected == [0]
