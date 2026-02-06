import torch

from mebench.attackers.blackbox_ripper import BlackboxRipper
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


def test_blackbox_ripper_mutation_structure_matches_upstream() -> None:
    """Mutation should keep elites and add two mutated copies.

    Upstream (temp_ripper/torch_optimizer.py) pattern:
      new_pop = concat([elites,
                        elites + N(0, sigma),
                        elites + N(0, sigma)])
    """

    state = _make_min_state()
    config = {
        "population_size": 30,
        "elite_size": 10,
        "mutation_scale": 0.5,
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

    off1 = new_pop[10:20]
    off2 = new_pop[20:30]

    # Offspring should differ from elites (noise added).
    assert not torch.allclose(off1, elites)
    assert not torch.allclose(off2, elites)

    # Each offspring row should be closest to its corresponding elite row.
    d1 = torch.cdist(off1, elites)
    d2 = torch.cdist(off2, elites)
    assert torch.equal(d1.argmin(dim=1), torch.arange(10))
    assert torch.equal(d2.argmin(dim=1), torch.arange(10))


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
