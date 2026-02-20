import torch
import torch.nn as nn
from torch.utils.data import Dataset

from mebench.attackers.dfme import DFME
from mebench.attackers.knockoff_nets import KnockoffNets
from mebench.attackers.random_baseline import RandomBaseline
from mebench.core.context import BenchmarkContext
from mebench.core.state import BenchmarkState
from mebench.oracles.oracle import Oracle


class _TinyVictim(nn.Module):
    def __init__(self, in_dim: int = 3 * 32 * 32, num_classes: int = 10) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.view(x.size(0), -1))


class _TinyPool(Dataset):
    def __init__(self, n: int = 16) -> None:
        self.x = torch.rand(n, 3, 32, 32)
        self.y = torch.arange(n) % 10

    def __len__(self) -> int:
        return int(self.x.size(0))

    def __getitem__(self, idx: int):
        return self.x[idx], int(self.y[idx].item())


def _ctx_for_budget(budget: int = 8) -> BenchmarkContext:
    state = BenchmarkState(
        budget_remaining=budget,
        metadata={
            "device": "cpu",
            "num_classes": 10,
            "input_shape": (3, 32, 32),
            "dataset_config": {},
            "substitute_config": {"arch": "resnet18", "optimizer": {"name": "sgd", "lr": 0.01}},
            "victim_config": {"input_size": [32, 32], "channels": 3, "output_mode": "soft_prob"},
            "max_budget": budget,
        },
    )
    victim = _TinyVictim()
    oracle = Oracle(victim, {"output_mode": "soft_prob", "temperature": 1.0, "input_size": [32, 32], "channels": 3}, state)
    return BenchmarkContext(state=state, oracle=oracle, logger=None, config={"budget": {"checkpoints": [budget]}})


def test_dfme_tiny_end_to_end_smoke() -> None:
    ctx = _ctx_for_budget(6)
    attack = DFME(
        {
            "output_mode": "soft_prob",
            "batch_size": 2,
            "n_g_steps": 1,
            "n_s_steps": 1,
            "noise_dim": 32,
            "generator_lr": 1e-4,
        },
        ctx.state,
    )
    attack.run(ctx)
    assert ctx.state.query_count == 6
    assert ctx.state.attack_state.get("substitute") is not None


def test_random_baseline_tiny_end_to_end_smoke() -> None:
    ctx = _ctx_for_budget(5)
    attack = RandomBaseline({"batch_size": 2, "output_mode": "soft_prob"}, ctx.state)
    attack.pool_dataset = _TinyPool(12)
    ctx.state.attack_state["unqueried_indices"] = list(range(12))
    attack.run(ctx)
    assert ctx.state.query_count == 5
    assert len(ctx.state.attack_state.get("query_data_x", [])) > 0


def test_knockoffnets_random_policy_refills_pool() -> None:
    ctx = _ctx_for_budget(9)
    attack = KnockoffNets(
        {
            "batch_size": 4,
            "train_every": 1000,
            "policy": "random",
            "output_mode": "soft_prob",
        },
        ctx.state,
    )
    attack.pool_dataset = _TinyPool(3)
    ctx.state.attack_state["unqueried_indices"] = [0, 1, 2]
    attack.run(ctx)
    assert ctx.state.query_count == 9
    assert len(ctx.state.attack_state.get("query_data_x", [])) > 0
    assert int(ctx.state.attack_state.get("random_refill_count", 0)) >= 1
