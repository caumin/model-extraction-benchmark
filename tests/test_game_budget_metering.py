import torch
import torch.nn as nn
import torch.optim as optim

from mebench.attackers.game import GAME
from mebench.core.context import BenchmarkContext
from mebench.core.state import BenchmarkState
from mebench.oracles.oracle import Oracle


class _CountingVictim(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(8 * 8, num_classes)
        self.images_seen = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.images_seen += int(x.size(0))
        return self.fc(x.view(x.size(0), -1))


class _TinyGenerator(nn.Module):
    def __init__(self, noise_dim: int, num_classes: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_classes, noise_dim)
        self.fc = nn.Linear(noise_dim, 8 * 8)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = z + self.embed(y)
        x = torch.tanh(self.fc(h))
        return x.view(-1, 1, 8, 8)


class _TinyDiscriminator(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = nn.Linear(8 * 8, 32)
        self.validity = nn.Linear(32, 1)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.relu(self.backbone(x.view(x.size(0), -1)))
        return self.validity(h), self.classifier(h)


class _TinyStudent(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(8 * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.view(x.size(0), -1))


def _make_state(budget: int) -> BenchmarkState:
    state = BenchmarkState(budget_remaining=budget)
    state.metadata = {
        "device": "cpu",
        "num_classes": 3,
        "input_shape": (1, 8, 8),
        "dataset_config": {},
        "substitute_config": {
            "arch": "resnet18",
            "optimizer": {"name": "sgd", "lr": 0.01},
        },
        "max_budget": budget,
    }
    return state


def test_game_run_uses_only_metered_oracle_queries() -> None:
    budget = 8
    state = _make_state(budget)
    attack = GAME(
        {
            "batch_size": 4,
            "querybudget": budget,
            "noise_dim": 8,
            "num_classes": 3,
            "proxy_num_classes": 3,
            "tdl_steps": 0,
            "attack_train_epoch": 1,
            "round_train_epochs": 1,
            "train_on_full_buffer": False,
            "eval_interval_queries": 0,
            "acs_strategy": "random",
            "agu_loss_terms": ["res", "bou", "dif"],
        },
        state,
    )

    def _stub_init_models(_state: BenchmarkState) -> None:
        if attack.generator is not None:
            return
        attack.generator = _TinyGenerator(noise_dim=attack.noise_dim, num_classes=attack.proxy_num_classes)
        attack.discriminator = _TinyDiscriminator(num_classes=attack.proxy_num_classes)
        attack.student = _TinyStudent(num_classes=attack.num_classes)
        attack.generator_optimizer = optim.Adam(attack.generator.parameters(), lr=1e-3)
        attack.discriminator_optimizer = optim.Adam(attack.discriminator.parameters(), lr=1e-3)
        attack._student_opt_config = {
            "name": "sgd",
            "lr": 0.01,
            "momentum": 0.0,
            "weight_decay": 0.0,
        }
        attack.student_optimizer = optim.SGD(attack.student.parameters(), lr=0.01)

    attack._init_models = _stub_init_models  # type: ignore[method-assign]
    attack._evaluate_current_substitute = lambda *args, **kwargs: None  # type: ignore[method-assign]

    victim = _CountingVictim(num_classes=3)
    oracle = Oracle(
        victim,
        {
            "output_mode": "soft_prob",
            "temperature": 1.0,
            "channels": 1,
            "input_size": [8, 8],
            "return_outputs_on_cpu": True,
        },
        state,
    )
    ctx = BenchmarkContext(
        state=state,
        oracle=oracle,
        config={"attack": {}, "budget": {"checkpoints": []}},
    )

    attack.run(ctx)

    assert int(state.query_count) == budget
    assert int(state.budget_remaining) == 0
    assert int(victim.images_seen) == budget
