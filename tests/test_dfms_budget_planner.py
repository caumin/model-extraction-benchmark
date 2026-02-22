import torch

from mebench.attackers.dfms import DFMSHL
from mebench.core.state import BenchmarkState


def _state() -> BenchmarkState:
    state = BenchmarkState()
    state.metadata = {
        "device": "cpu",
        "input_shape": (3, 32, 32),
        "num_classes": 10,
        "dataset_config": {"name": "CIFAR10"},
        "substitute_config": {"arch": "lenet"},
    }
    return state


def test_dfms_budget_plan_paper_fair_targets_8m_with_10class_subset() -> None:
    attack = DFMSHL(
        {
            "proxy_data_ratio": 1.0,
            "dcgan_data_ratio": 0.8,
            "div_gan_data_ratio": 0.8,
            "max_synth_samples": 50_000,
            "budget_planner": {"mode": "paper_fair", "nC_target": 50_000},
        },
        _state(),
    )
    attack.proxy_data = torch.rand(5_000, 3, 32, 32)

    plan = attack._build_budget_plan(8_000_000)

    assert plan.stage2_target_queries == 50_000
    assert plan.stage4_target_queries == 50_000
    assert plan.stage5_target_queries == 7_900_000
    assert plan.proxy_subset_size == 5_000
    assert plan.stage5_planned_epochs == 1_580


def test_dfms_budget_plan_legacy_keeps_ratio_based_stage_targets() -> None:
    attack = DFMSHL(
        {
            "proxy_data_ratio": 1.0,
            "dcgan_data_ratio": 0.8,
            "div_gan_data_ratio": 0.8,
            "max_synth_samples": 50_000,
            "alternate_epochs": 800,
            "budget_planner": {"mode": "legacy_fixed_epochs"},
        },
        _state(),
    )
    attack.proxy_data = torch.rand(5_000, 3, 32, 32)

    plan = attack._build_budget_plan(8_000_000)

    assert plan.stage2_target_queries == 45_000
    assert plan.stage4_target_queries == 45_000
    assert plan.stage5_target_queries == 7_910_000
    assert plan.stage5_planned_epochs == 1_582
