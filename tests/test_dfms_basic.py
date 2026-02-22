"""Basic DFMS-HL attack tests."""

import torch
from mebench.core.state import BenchmarkState
from mebench.core.types import OracleOutput
from mebench.attackers.dfms import DFMSHL


def test_dfms_basic_flow(tmp_path) -> None:
    config = {
        "batch_size": 2,
        "noise_dim": 16,
        "num_classes": 10,
        "clone_arch": "lenet",  # Use small model for fast CPU testing
        "proxy_dataset": {
            "name": "CIFAR10",
            "data_mode": "seed",
            "seed_size": 10,
            "num_workers": 0,  # Avoid multiprocessing hang on Windows
        },
    }
    state = BenchmarkState()
    state.metadata = {"device": "cpu", "input_shape": (3, 32, 32)}

    attack = DFMSHL(config, state)
    query_batch = attack._select_query_batch(2, state)
    x_query = query_batch.x
    labels = torch.randint(0, 10, (2,))
    oracle_output = OracleOutput(kind="hard_top1", y=labels)
    attack._handle_oracle_output(x_query, oracle_output, state)

    assert state.attack_state["substitute"] is not None



def test_dfms_required_alternate_epochs_scales_with_budget() -> None:
    attack = DFMSHL(
        {"alternate_epochs": 2},
        BenchmarkState(),
    )

    assert attack._required_alternate_epochs(budget_remaining=11, subset_size=3) == 4
    assert attack._required_alternate_epochs(budget_remaining=11, subset_size=20) == 1


def test_dfms_official_stage_alternate_consumes_full_budget() -> None:
    import torch
    import torch.nn as nn
    from torch import optim
    from tqdm import tqdm
    from mebench.oracles.oracle import Oracle
    from mebench.core.context import BenchmarkContext

    class DummyGenerator(nn.Module):
        def __init__(self, noise_dim: int) -> None:
            super().__init__()
            self.proj = nn.Linear(noise_dim, 3 * 32 * 32)

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            x = self.proj(z)
            return x.view(int(z.shape[0]), 3, 32, 32)

    class DummyDiscriminator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = nn.Parameter(torch.tensor(0.0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.bias * torch.ones(int(x.shape[0]), 1, device=x.device)

    class DummyVictim(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(3 * 32 * 32, 10),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    state = BenchmarkState(
        budget_remaining=11,
        metadata={"device": "cpu", "input_shape": (3, 32, 32), "num_classes": 10},
    )
    attack = DFMSHL(
        {
            "noise_dim": 8,
            "num_classes": 10,
            "batch_size": 4,
            "alternate_epochs": 2,
            "alternate_auto_augment": False,
            "clone_arch": "lenet",
            "query_input_scale_mode": "unit",
            "internal_input_scale_mode": "unit",
            "budget_planner": {"mode": "paper_fair", "enforce_exact_budget": False},
        },
        state,
    )
    attack.proxy_data = torch.rand(3, 3, 32, 32)
    attack.clone = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10)).to("cpu")
    attack.generator = DummyGenerator(noise_dim=8)
    attack.generator_optimizer = optim.Adam(attack.generator.parameters(), lr=1e-4)
    attack.discriminator = DummyDiscriminator()
    attack.discriminator_optimizer = optim.Adam(attack.discriminator.parameters(), lr=1e-3)

    victim = DummyVictim()
    oracle_cfg = {
        "output_mode": "hard_top1",
        "temperature": 1.0,
        "input_size": [32, 32],
        "channels": 3,
    }
    oracle = Oracle(victim, oracle_cfg, state)
    ctx = BenchmarkContext(state=state, oracle=oracle, config={"budget": {"checkpoints": []}})

    pbar = tqdm(total=state.budget_remaining, disable=True)
    attack._official_stage_alternate(ctx=ctx, device="cpu", pbar=pbar)
    pbar.close()

    assert state.query_count == 11
    assert state.budget_remaining == 0
