import torch

from mebench.attackers.blackbox_ripper import (
    BlackboxRipper,
    _clamp_to_unit_range,
    _match_input_shape,
)
from mebench.core.state import BenchmarkState


def _make_min_state() -> BenchmarkState:
    state = BenchmarkState()
    state.metadata = {
        "device": "cpu",
        "input_shape": (3, 32, 32),
        "num_classes": 10,
        "max_budget": 100,
        "substitute_config": {"arch": "resnet18", "optimizer": {"lr": 0.01}},
    }
    return state


def test_blackbox_ripper_mutation_structure_matches_paper() -> None:
    """Mutation matches paper Algorithm 1 (Steps 6-7).

    Pc <- {U(Pe)}_{K-k} (uniformly sample K-k copies from elites, with replacement)
    Pc <- Pc + N(0, 1)
    P  <- Pe U Pc
    """

    state = _make_min_state()
    config = {
        "population_size": 30,
        "elite_size": 10,
        "mutation_scale": 1.0,
        # Not used by this unit test (only required when running the attack).
        "generator_checkpoint": "C:/path/to/weights.pth",
    }
    attack = BlackboxRipper(config, state)

    torch.manual_seed(0)

    # Construct elites far apart so offspring nearest-parent mapping is unambiguous.
    dim = 32
    elites = torch.zeros(10, dim)
    elites[:, 0] = torch.arange(10).float() * 100.0

    new_pop = attack._make_next_population_from_elites(elites)

    assert new_pop.shape == (30, dim)
    assert torch.allclose(new_pop[:10], elites)

    children = new_pop[10:30]
    # Each child should be closest to some elite (since elites are far apart).
    d = torch.cdist(children, elites)
    parent_idx = d.argmin(dim=1)
    min_dist = d.min(dim=1).values
    assert torch.all(min_dist < 50.0)

    # Children should not be exactly equal to their nearest elite (noise added).
    parents = elites[parent_idx]
    assert not torch.allclose(children, parents)


def test_blackbox_ripper_objective_eq2() -> None:
    """Objective matches paper Eq.(2): sum_j (p_j - y_j)^2."""

    state = _make_min_state()
    config = {
        "generator_checkpoint": "C:/path/to/weights.pth",
    }
    attack = BlackboxRipper(config, state)

    target_cls = 0
    target_onehot = torch.nn.functional.one_hot(torch.tensor(target_cls), num_classes=10).float()

    # Perfect match => objective = 0.
    probs = torch.zeros(1, 10)
    probs[0, target_cls] = 1.0
    obj = attack._objective_mse_sum(probs, target_onehot)
    assert obj.shape == (1,)
    assert abs(float(obj.item())) < 1e-6

    # Complete mismatch between two classes => objective = 2.
    probs = torch.zeros(1, 10)
    probs[0, 1] = 1.0
    obj = attack._objective_mse_sum(probs, target_onehot)
    assert abs(float(obj.item()) - 2.0) < 1e-6


def test_clamp_and_shape_helpers() -> None:
    x = torch.tensor([[[[-1.0, -0.2], [0.0, 0.5]]]])
    y = _clamp_to_unit_range(x)
    assert y.min() >= 0.0 and y.max() <= 1.0
    assert torch.allclose(y[0], torch.tensor([[[0.0, 0.4], [0.5, 0.75]]]))

    img = torch.randn(2, 3, 32, 16)
    out = _match_input_shape(img, (3, 64, 64))
    assert tuple(out.shape) == (2, 3, 64, 64)
    img3 = torch.randn(2, 1, 32, 32)
    out3 = _match_input_shape(img3, (3, 32, 32))
    assert tuple(out3.shape) == (2, 3, 32, 32)
