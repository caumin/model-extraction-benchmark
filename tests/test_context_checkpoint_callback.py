import torch

from mebench.core.context import BenchmarkContext
from mebench.core.state import BenchmarkState
from mebench.oracles.oracle import Oracle


def test_checkpoint_callback_fires_once_per_checkpoint() -> None:
    state = BenchmarkState(budget_remaining=6)
    state.metadata = {"max_budget": 6}

    victim = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 32 * 32, 10),
    )
    victim_cfg = {
        "output_mode": "soft_prob",
        "temperature": 1.0,
        "channels": 3,
        "input_size": [32, 32],
    }
    oracle = Oracle(victim, victim_cfg, state)

    reached = []

    ctx = BenchmarkContext(
        state=state,
        oracle=oracle,
        logger=None,
        config={"budget": {"checkpoints": [3, 5]}},
        checkpoint_callback=lambda q: reached.append(int(q)),
    )

    # 0 -> 2 (no checkpoint)
    ctx.query(torch.rand(2, 3, 32, 32))
    # 2 -> 4 (crosses 3)
    ctx.query(torch.rand(2, 3, 32, 32))
    # 4 -> 6 (crosses 5)
    ctx.query(torch.rand(2, 3, 32, 32))

    assert reached == [3, 5]
    assert state.attack_state.get("checkpoint_reached") == [3, 5]
