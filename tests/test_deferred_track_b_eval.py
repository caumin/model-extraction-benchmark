import torch

import mebench.attackers.runner as runner_module
from mebench.attackers.runner import AttackRunner
from mebench.core.context import BenchmarkContext
from mebench.core.state import BenchmarkState
from mebench.oracles.oracle import Oracle


class _DummyAttack(AttackRunner):
    def run(self, ctx: BenchmarkContext) -> None:
        self.victim = ctx.oracle.model


def _make_state() -> BenchmarkState:
    state = BenchmarkState(budget_remaining=8)
    state.metadata = {
        "device": "cpu",
        "dataset_config": {"name": "CIFAR10"},
        "victim_config": {"channels": 3, "input_size": [32, 32]},
    }
    return state


def _make_oracle(state: BenchmarkState) -> Oracle:
    victim = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 32 * 32, 10),
        torch.nn.Softmax(dim=1),
    )
    return Oracle(
        victim,
        {
            "output_mode": "soft_prob",
            "temperature": 1.0,
            "channels": 3,
            "input_size": [32, 32],
            "num_classes": 10,
        },
        state,
    )


def test_evaluate_current_substitute_dedupes_before_metrics(monkeypatch) -> None:
    state = _make_state()
    attack = _DummyAttack({"output_mode": "soft_prob"}, state)
    attack.victim = _make_oracle(state).model
    attack.test_loader = [object()]
    substitute = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 32 * 32, 10),
        torch.nn.Softmax(dim=1),
    )

    calls = []

    def fake_evaluate_substitute(**kwargs):
        calls.append(kwargs["device"])
        return {
            "acc_gt": 0.1,
            "agreement": 0.2,
            "kl_mean": 0.3,
            "l1_mean": 0.4,
            "binary_precision": 0.0,
            "binary_recall": 0.0,
            "binary_f1": 0.0,
            "binary_roc_auc": 0.0,
        }

    monkeypatch.setattr(runner_module, "evaluate_substitute", fake_evaluate_substitute)

    attack._evaluate_current_substitute(substitute, "cpu", query_count=4)
    attack._evaluate_current_substitute(substitute, "cpu", query_count=4)

    assert calls == ["cpu"]


def test_checkpoint_eval_is_deferred_until_safe_point(monkeypatch) -> None:
    state = _make_state()
    oracle = _make_oracle(state)
    attack = _DummyAttack({"output_mode": "soft_prob"}, state)
    attack.victim = oracle.model
    attack.test_loader = [object()]
    substitute = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 32 * 32, 10),
        torch.nn.Softmax(dim=1),
    )
    attack.state.attack_state["substitute"] = substitute

    seen = []

    def fake_evaluate_substitute(**kwargs):
        seen.append(int(kwargs["substitute"] is substitute))
        return {
            "acc_gt": 0.1,
            "agreement": 0.2,
            "kl_mean": 0.3,
            "l1_mean": 0.4,
            "binary_precision": 0.0,
            "binary_recall": 0.0,
            "binary_f1": 0.0,
            "binary_roc_auc": 0.0,
        }

    monkeypatch.setattr(runner_module, "evaluate_substitute", fake_evaluate_substitute)

    ctx = BenchmarkContext(
        state=state,
        oracle=oracle,
        logger=None,
        config={"budget": {"checkpoints": [3, 5]}},
        checkpoint_callback=None,
    )

    ctx.query(torch.rand(6, 3, 32, 32))
    assert state.attack_state.get("checkpoint_reached") == [3, 5]
    assert seen == []

    attack._drain_deferred_track_b_checkpoints("cpu")
    assert seen == [1, 1]
