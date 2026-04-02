import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from mebench.attackers.copycatcnn import CopycatCNN
from mebench.attackers.ds import DualStudents, _DualStudentEvalWrapper, _MovingAverageModel
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


class _TinyPoolDataset(Dataset):
    def __init__(self, xs: torch.Tensor) -> None:
        self.xs = xs

    def __len__(self) -> int:
        return int(self.xs.size(0))

    def __getitem__(self, idx: int):
        return self.xs[idx], 0


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


def test_inversenet_phase3_normalization_uses_surrogate_standard_stats() -> None:
    state = _make_state(output_mode="hard_top1")
    state.metadata["dataset_config"] = {
        "data_mode": "surrogate",
        "surrogate_name": "ImageNet",
        "surrogate_normalization": "standard",
        "channels": 3,
    }
    attack = InverseNet({}, state)

    attack._configure_phase3_normalization(state.metadata["dataset_config"], state)
    assert attack._phase3_norm_stats is not None

    x = torch.full((2, 3, 4, 4), 0.5, dtype=torch.float32)
    x_norm = attack._normalize_phase3_queries(x)

    expected = torch.tensor(
        [
            (0.5 - 0.485) / 0.229,
            (0.5 - 0.456) / 0.224,
            (0.5 - 0.406) / 0.225,
        ],
        dtype=torch.float32,
    ).view(1, 3, 1, 1)
    assert torch.allclose(x_norm, expected.expand_as(x_norm), atol=1e-6)


def test_inversenet_phase3_normalization_is_noop_when_disabled() -> None:
    state = _make_state(output_mode="hard_top1")
    state.metadata["dataset_config"] = {
        "data_mode": "surrogate",
        "surrogate_name": "ImageNet",
        "surrogate_normalization": "none",
        "channels": 3,
    }
    attack = InverseNet({}, state)

    attack._configure_phase3_normalization(state.metadata["dataset_config"], state)
    assert attack._phase3_norm_stats is None

    x = torch.rand(2, 3, 4, 4)
    x_norm = attack._normalize_phase3_queries(x)
    assert torch.equal(x_norm, x)


def test_inversenet_pool_preload_disables_pin_memory_on_cuda_runtime(monkeypatch) -> None:
    state = _make_state(output_mode="hard_top1")
    state.metadata["device"] = "cuda:0"
    attack = InverseNet({"cache_pool_to_memory": True, "pool_cache_batch_size": 8}, state)

    dataset = _TinyPoolDataset(torch.rand(4, 3, 32, 32))
    seen_kwargs: dict[str, object] = {}

    class _FakeLoader:
        def __init__(self, dataset_obj, batch_size, shuffle, **kwargs):
            del shuffle
            seen_kwargs.update(kwargs)
            self.dataset_obj = dataset_obj
            self.batch_size = int(batch_size)

        def __iter__(self):
            xs = []
            ys = []
            upper = min(len(self.dataset_obj), self.batch_size)
            for idx in range(upper):
                x_item, y_item = self.dataset_obj[idx]
                xs.append(x_item)
                ys.append(y_item)
            yield torch.stack(xs), torch.tensor(ys)

    monkeypatch.setattr(
        "mebench.attackers.inversenet.create_dataloader",
        lambda *args, **kwargs: type("_LoaderWrapper", (), {"dataset": dataset})(),
    )
    monkeypatch.setattr("mebench.attackers.inversenet.DataLoader", _FakeLoader)

    attack._load_pool(state)

    assert seen_kwargs["pin_memory"] is False
    assert isinstance(attack.pool_data, torch.Tensor)
    assert tuple(attack.pool_data.shape) == (4, 3, 32, 32)


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


class _ConstantLogits(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("_logits", logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._logits.expand(x.size(0), -1)


def test_ds_defaults_follow_official_eval_and_budget_semantics() -> None:
    state = _make_state(output_mode="soft_prob")
    attack = DualStudents({"batch_size": 2, "noise_dim": 16}, state)

    assert attack.combine_student_outputs == "first"
    assert attack.strict_iteration_budget is True
    assert attack.student_momentum == pytest.approx(0.9)
    assert attack.generator_momentum == pytest.approx(0.9)


def test_ds_eval_wrapper_matches_official_combine_modes() -> None:
    x = torch.randn(2, 3, 4, 4)
    first_student = _MovingAverageModel(
        _ConstantLogits(torch.tensor([[3.0, 1.0]], dtype=torch.float32)),
        momentum=0.0,
    )
    second_student = _MovingAverageModel(
        _ConstantLogits(torch.tensor([[1.0, 5.0]], dtype=torch.float32)),
        momentum=0.0,
    )

    first_wrapper = _DualStudentEvalWrapper([first_student, second_student], combine_mode="first")
    first_logits = first_wrapper(x)
    assert torch.allclose(first_logits, torch.tensor([[3.0, 1.0], [3.0, 1.0]]))

    mean_wrapper = _DualStudentEvalWrapper([first_student, second_student], combine_mode="mean")
    mean_logits = mean_wrapper(x)
    assert torch.allclose(mean_logits, torch.tensor([[2.0, 3.0], [2.0, 3.0]]))


def test_ds_moving_average_step_matches_official_update_rule() -> None:
    base = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        base.weight.fill_(1.0)

    wrapped = _MovingAverageModel(base, momentum=0.9)
    assert wrapped.test_model is not None

    with torch.no_grad():
        wrapped.train_model.weight.fill_(2.0)
    wrapped.step()
    assert torch.allclose(wrapped.test_model.weight, torch.full((2, 2), 2.0))

    with torch.no_grad():
        wrapped.train_model.weight.fill_(4.0)
    wrapped.step()
    assert torch.allclose(wrapped.test_model.weight, torch.full((2, 2), 2.2))


def test_ds_moving_average_uses_train_model_until_first_sync() -> None:
    base = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        base.weight.fill_(1.0)

    wrapped = _MovingAverageModel(base, momentum=0.9)
    x = torch.ones(1, 2)

    with torch.no_grad():
        wrapped.train_model.weight.fill_(3.0)
    before_sync = wrapped(x, test=True)
    assert torch.allclose(before_sync, torch.full((1, 2), 6.0))

    wrapped.step()
    after_sync = wrapped(x, test=True)
    assert torch.allclose(after_sync, torch.full((1, 2), 6.0))
