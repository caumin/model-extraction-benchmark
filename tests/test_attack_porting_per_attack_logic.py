import pytest
import torch
import torch.nn.functional as F

from mebench.attackers.copycatcnn import CopycatCNN
from mebench.attackers.disguide import DisGUIDE
from mebench.attackers.inversenet import InverseNet
from mebench.core.state import BenchmarkState
from mebench.core.types import OracleOutput


def _make_state(*, num_classes: int = 10, output_mode: str = "hard_top1") -> BenchmarkState:
    return BenchmarkState(
        budget_remaining=64,
        metadata={
            "device": "cpu",
            "num_classes": int(num_classes),
            "input_shape": (3, 32, 32),
            "dataset_config": {"data_mode": "surrogate", "surrogate_name": "SVHN"},
            "substitute_config": {
                "arch": "resnet18",
                "optimizer": {"name": "sgd", "lr": 0.01},
            },
            "victim_config": {"output_mode": output_mode, "input_size": [32, 32], "channels": 3},
            "max_budget": 64,
        },
    )


def test_copycatcnn_handles_soft_and_hard_outputs() -> None:
    state = _make_state(output_mode="hard_top1")
    attack = CopycatCNN({"batch_size": 4}, state)

    x = torch.rand(4, 3, 32, 32)
    soft = torch.tensor(
        [
            [0.1, 0.7, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.6, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.6, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.2, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    hard = torch.tensor([3, 1, 4, 2], dtype=torch.long)

    attack._handle_oracle_output(x, OracleOutput(kind="soft_prob", y=soft), state)
    attack._handle_oracle_output(x, OracleOutput(kind="hard_top1", y=hard), state)

    assert len(state.attack_state["query_data_x"]) == 2
    assert len(state.attack_state["query_data_y"]) == 2
    assert torch.equal(state.attack_state["query_data_y"][0], soft.argmax(dim=1))
    assert torch.equal(state.attack_state["query_data_y"][1], hard)


def test_copycatcnn_rejects_problem_domain_pool_config() -> None:
    state = _make_state(output_mode="hard_top1")
    state.metadata["dataset_config"] = {"data_mode": "cifar10"}
    with pytest.raises(ValueError, match="requires NPDD dataset"):
        CopycatCNN({}, state)


def test_inversenet_phase_boundaries_match_default_ratios() -> None:
    state = _make_state(output_mode="hard_top1")
    attack = InverseNet({"phase_ratios": [0.45, 0.45, 0.1]}, state)

    p1_end, p2_end, total = attack._phase_boundaries(1000)
    assert (p1_end, p2_end, total) == (450, 900, 1000)

    assert attack._phase_for_query_count(0, 1000) == 1
    assert attack._phase_for_query_count(449, 1000) == 1
    assert attack._phase_for_query_count(450, 1000) == 2
    assert attack._phase_for_query_count(899, 1000) == 2
    assert attack._phase_for_query_count(900, 1000) == 3

    assert attack._remaining_in_phase(1, 120, 1000) == 330
    assert attack._remaining_in_phase(2, 500, 1000) == 400
    assert attack._remaining_in_phase(3, 920, 1000) == 80


def test_inversenet_routes_observations_by_phase() -> None:
    state = _make_state(output_mode="hard_top1")
    attack = InverseNet({"truncation_k": 1}, state)

    x = torch.rand(2, 3, 32, 32)
    labels = torch.tensor([1, 2], dtype=torch.long)
    output = OracleOutput(kind="hard_top1", y=labels)

    attack._handle_oracle_output(x, {"phase": 1}, output, state)
    assert len(state.attack_state["query_data_x"]) == 1
    assert len(state.attack_state["query_data_y"]) == 1

    attack._handle_oracle_output(x, {"phase": 2}, output, state)
    assert len(state.attack_state["inversion_x"]) == 1
    assert len(state.attack_state["inversion_y"]) == 1
    expected_trunc = F.one_hot(labels, num_classes=attack.num_classes).float()
    assert torch.allclose(state.attack_state["inversion_y"][0], expected_trunc)

    attack._handle_oracle_output(x, {"phase": 3}, output, state)
    assert len(state.attack_state["retrain_x"]) == 1
    assert len(state.attack_state["retrain_y"]) == 1


def test_inversenet_rejects_soft_prob_observation() -> None:
    state = _make_state(output_mode="hard_top1")
    attack = InverseNet({}, state)
    x = torch.rand(2, 3, 32, 32)
    soft = torch.rand(2, 10)

    with pytest.raises(ValueError, match="requires hard_top1"):
        attack._handle_oracle_output(x, {"phase": 1}, OracleOutput(kind="soft_prob", y=soft), state)


def test_disguide_target_format_modes() -> None:
    state = _make_state(num_classes=4, output_mode="soft_prob")

    hl = DisGUIDE.__new__(DisGUIDE)
    hl.state = state
    hl.config = {}
    hl.loss_mode = "hl"
    out_probs = torch.tensor([[0.1, 0.7, 0.2, 0.0]], dtype=torch.float32)
    hl_target = hl._format_teacher_target(out_probs)
    assert torch.equal(hl_target, torch.tensor([1], dtype=torch.long))

    kl = DisGUIDE.__new__(DisGUIDE)
    kl.state = state
    kl.config = {}
    kl.loss_mode = "kl"
    labels = torch.tensor([2, 0], dtype=torch.long)
    kl_target = kl._format_teacher_target(labels)
    assert kl_target.shape == (2, 4)
    assert torch.equal(kl_target.argmax(dim=1), labels)

    l1 = DisGUIDE.__new__(DisGUIDE)
    l1.state = state
    l1.config = {}
    l1.loss_mode = "l1"
    l1_target = l1._format_teacher_target(labels)
    assert l1_target.shape == (2, 4)
    assert torch.allclose(l1_target.mean(dim=1), torch.zeros(2), atol=1e-6)


def test_disguide_hard_mode_requires_hl_loss() -> None:
    state = _make_state(output_mode="hard_top1")
    with pytest.raises(ValueError, match="hard_top1 mode requires loss='hl'"):
        DisGUIDE(
            {
                "output_mode": "hard_top1",
                "loss": "l1",
                "ensemble_size": 2,
                "batch_size": 2,
                "noise_dim": 16,
            },
            state,
        )
