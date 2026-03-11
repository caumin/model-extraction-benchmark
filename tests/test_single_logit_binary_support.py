import tempfile
from pathlib import Path

import torch
import torch.optim as optim

from mebench.core.query_storage import QueryStorage
from mebench.core.state import BenchmarkState
from mebench.eval.evaluator import Evaluator
from mebench.oracles.oracle import Oracle
from mebench.models.substitute_factory import create_substitute


def test_oracle_single_logit_binary_outputs_are_not_degenerate() -> None:
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 1))
    with torch.no_grad():
        model[1].weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        model[1].bias.zero_()

    state = BenchmarkState(budget_remaining=10, metadata={"num_classes": 1})
    x = torch.tensor([[-2.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    oracle_soft = Oracle(model, {"output_mode": "soft_prob", "temperature": 1.0, "input_size": (4,), "num_classes": 1}, state)
    soft = oracle_soft.query(x).y
    assert soft.shape == (2, 1)
    assert float(soft[0, 0]) < 0.5 < float(soft[1, 0])

    oracle_hard = Oracle(model, {"output_mode": "hard_top1", "temperature": 1.0, "input_size": (4,), "num_classes": 1}, state)
    hard = oracle_hard.query(x).y
    assert torch.equal(hard, torch.tensor([0, 1], dtype=torch.long))


def test_track_a_training_supports_single_logit_binary_soft_prob() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "run": {"name": "binary", "seeds": [0], "device": "cpu"},
            "victim": {"output_mode": "soft_prob", "temperature": 1.0, "num_classes": 1, "channels": 1},
            "substitute": {
                "arch": "lenet_mnist",
                "init_seed": 123,
                "trackA": {"batch_size": 4, "steps_coeff_c": 0.2},
                "optimizer": {"lr": 0.01, "momentum": 0.9, "weight_decay": 0.0},
                "patience": 5,
            },
        }

        x = torch.randn(20, 1, 28, 28)
        y = torch.randint(0, 2, (20, 1), dtype=torch.long).float()

        storage = QueryStorage(Path(tmpdir) / "cache", output_mode="soft_prob")
        storage.add_batch(x, y)

        state = BenchmarkState(metadata={"device": "cpu"})
        evaluator = Evaluator(config, state, storage)
        model = create_substitute("lenet_mnist", num_classes=1, input_channels=1).to("cpu")
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        evaluator._train_track_a(model, optimizer, 5, 4)
        after = model.state_dict()

        assert any(not torch.allclose(before[name], after[name]) for name in before)
