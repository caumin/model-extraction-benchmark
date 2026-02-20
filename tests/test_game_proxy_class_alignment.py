import torch

from mebench.attackers.game import GAME
from mebench.core.state import BenchmarkState


class _DummyGenerator(torch.nn.Module):
    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        b = z.size(0)
        return torch.tanh(torch.randn(b, 3, 32, 32, device=z.device))


class _DummyStudent(torch.nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(3 * 32 * 32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.view(x.size(0), -1))


def test_game_uses_proxy_class_space_for_generator_labels() -> None:
    state = BenchmarkState()
    state.metadata = {
        "device": "cpu",
        "input_shape": (3, 32, 32),
        "num_classes": 10,
        "dataset_config": {},
        "substitute_config": {"arch": "resnet18", "optimizer": {"name": "sgd", "lr": 0.01}},
    }

    cfg = {
        "batch_size": 8,
        "noise_dim": 16,
        "num_classes": 10,
        "proxy_dataset": {"surrogate_name": "CIFAR100"},
    }
    atk = GAME(cfg, state)

    # Avoid dataset/model init side effects and force deterministic lightweight components.
    atk._init_models = lambda _state: None
    atk.proxy_num_classes = 100
    atk.generator = _DummyGenerator()
    atk.student = _DummyStudent(num_classes=10)

    qb = atk._select_query_batch(8, state)
    y_g = qb.meta["y_g"]

    assert int(y_g.min().item()) >= 0
    assert int(y_g.max().item()) < 100
