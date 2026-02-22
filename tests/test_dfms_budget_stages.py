import torch
import torch.nn as nn

from mebench.attackers.dfms import DFMSHL
from mebench.core.context import BenchmarkContext
from mebench.core.state import BenchmarkState
from mebench.oracles.oracle import Oracle


class _EventLogger:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def log_history(self, step: int, metrics: dict) -> None:
        self.rows.append({"step": int(step), **dict(metrics)})


class _DummyPBar:
    def __init__(self) -> None:
        self.updates: list[int] = []

    def update(self, count: int) -> None:
        self.updates.append(int(count))

    def close(self) -> None:
        pass


class _Victim(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def test_dfms_official_stages_emit_events_and_respect_stage_plan() -> None:
    num_classes = 10
    state = BenchmarkState(budget_remaining=9)
    state.metadata = {
        "device": "cpu",
        "input_shape": (3, 32, 32),
        "num_classes": num_classes,
        "max_budget": 9,
        "dataset_config": {"name": "CIFAR10"},
        "substitute_config": {"arch": "lenet"},
        "victim_config": {"input_size": [32, 32], "channels": 3},
    }

    logger = _EventLogger()
    victim_config = {
        "output_mode": "hard_top1",
        "temperature": 1.0,
        "channels": 3,
        "input_size": [32, 32],
        "num_classes": num_classes,
    }
    victim = _Victim(num_classes=num_classes)
    oracle = Oracle(victim, victim_config, state)
    ctx = BenchmarkContext(
        state=state,
        oracle=oracle,
        logger=logger,
        config={"budget": {"checkpoints": [4, 8]}},
    )

    attack = DFMSHL(
        {
            "batch_size": 2,
            "noise_dim": 4,
            "num_classes": num_classes,
            "use_official_stages": True,
            "dcgan_epochs": 0,
            "student_init_epochs": 1,
            "degan_epochs": 0,
            "student_degan_epochs": 1,
            "alternate_epochs": 1,
            "dcgan_data_ratio": 0.0,
            "div_gan_data_ratio": 0.0,
            "proxy_data_ratio": 1.0,
            "student_batch_size": 2,
            "oracle_batch_size": 2,
            "proxy_pad_crop": False,
            "alternate_auto_augment": False,
            "alternate_pad_crop": False,
            "budget_planner": {"mode": "legacy_fixed_epochs", "enforce_exact_budget": False},
        },
        state,
    )

    # Preload tiny proxy cache: enough for fixed stage sizing without dataset IO.
    attack.proxy_data = torch.rand(4, 3, 32, 32)

    attack.run(ctx)

    stage_rows = [
        row
        for row in logger.rows
        if row.get("event") == "dfmshl_stage"
    ]
    expected_stage_rows = [
        {"step": 0, "stage": "dcgan_pretrain", "queries": 0},
        {"step": 0, "stage": "student_init_dcgan", "queries": "labels"},
        {"step": 4, "stage": "train_degan", "queries": 0},
        {"step": 4, "stage": "student_init_degan", "queries": "labels"},
        {"step": 8, "stage": "alternate", "queries": "labels"},
    ]
    observed_stage_rows = [
        {k: row[k] for k in ("step", "stage", "queries") if k in row}
        for row in stage_rows
    ]

    assert observed_stage_rows == expected_stage_rows

    checkpoints = [
        row["checkpoint"]
        for row in logger.rows
        if row.get("event") == "checkpoint_reached"
    ]
    assert checkpoints == [4, 8]

    assert state.query_count == 9
    assert state.budget_remaining == 0
    assert state.attack_state.get("checkpoint_reached") == [4, 8]


def test_dfms_alternate_stage_uses_tail_budget_without_overquerying() -> None:
    num_classes = 10
    state = BenchmarkState(budget_remaining=1)
    state.metadata = {
        "device": "cpu",
        "input_shape": (3, 32, 32),
        "num_classes": num_classes,
        "max_budget": 1,
        "dataset_config": {"name": "CIFAR10"},
        "substitute_config": {"arch": "lenet"},
        "victim_config": {"input_size": [32, 32], "channels": 3},
    }

    victim = _Victim(num_classes=num_classes)
    victim_config = {
        "output_mode": "hard_top1",
        "temperature": 1.0,
        "channels": 3,
        "input_size": [32, 32],
        "num_classes": num_classes,
    }
    oracle = Oracle(victim, victim_config, state)
    ctx = BenchmarkContext(
        state=state,
        oracle=oracle,
        logger=None,
        config={"budget": {"checkpoints": []}},
    )

    attack = DFMSHL(
        {
            "batch_size": 2,
            "noise_dim": 4,
            "num_classes": num_classes,
            "use_official_stages": True,
            "dcgan_epochs": 0,
            "student_init_epochs": 1,
            "degan_epochs": 0,
            "student_degan_epochs": 1,
            "alternate_epochs": 1,
            "dcgan_data_ratio": 0.0,
            "div_gan_data_ratio": 0.0,
            "proxy_data_ratio": 1.0,
            "student_batch_size": 2,
            "oracle_batch_size": 2,
            "proxy_pad_crop": False,
            "alternate_auto_augment": False,
            "alternate_pad_crop": False,
            "budget_planner": {"mode": "legacy_fixed_epochs", "enforce_exact_budget": False},
        },
        state,
    )

    # Directly initialize internal models and tiny proxy cache to avoid dataset/IO
    original_budget = int(state.budget_remaining)
    state.budget_remaining = 0
    try:
        attack._init_models(state)
    finally:
        state.budget_remaining = original_budget
    
    attack.proxy_data = torch.rand(4, 3, 32, 32)

    pbar = _DummyPBar()
    attack._official_stage_alternate(ctx, "cpu", pbar=pbar)

    assert state.query_count == 1
    assert state.budget_remaining == 0
    assert pbar.updates == [1]
