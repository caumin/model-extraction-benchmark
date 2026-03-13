import torch

from mebench.core.state import BenchmarkState
from mebench.oracles.oracle import Oracle


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
