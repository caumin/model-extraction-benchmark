from __future__ import annotations

import pytest

from mebench.attackers.blackbox_ripper import BlackboxRipper
from mebench.attackers.dfme import DFME
from mebench.attackers.dfms import DFMSHL
from mebench.attackers.game import GAME
from mebench.attackers.maze import MAZE
from mebench.core.state import BenchmarkState


def _state() -> BenchmarkState:
    state = BenchmarkState()
    state.metadata = {
        "device": "cpu",
        "input_shape": (3, 32, 32),
        "num_classes": 10,
        "dataset_config": {"name": "CIFAR10", "num_classes": 10},
        "substitute_config": {"arch": "lenet", "optimizer": {"name": "sgd", "lr": 0.01}},
    }
    return state


def test_dfme_legacy_scale_keys_are_ignored() -> None:
    attack = DFME(
        {
            "internal_input_scale_mode": "unit",
            "query_input_scale_mode": "unit",
            "batch_size": 2,
            "noise_dim": 16,
        },
        _state(),
    )
    assert not hasattr(attack, "internal_input_scale_mode")
    assert not hasattr(attack, "query_input_scale_mode")


def test_maze_legacy_scale_keys_are_ignored() -> None:
    attack = MAZE(
        {
            "clone_input_scale_mode": "unit",
            "query_input_scale_mode": "unit",
            "batch_size": 2,
            "noise_dim": 8,
        },
        _state(),
    )
    assert not hasattr(attack, "clone_input_scale_mode")
    assert not hasattr(attack, "query_input_scale_mode")


def test_dfms_legacy_scale_keys_are_ignored() -> None:
    attack = DFMSHL(
        {
            "internal_input_scale_mode": "unit",
            "query_input_scale_mode": "unit",
            "batch_size": 2,
            "noise_dim": 8,
        },
        _state(),
    )
    assert not hasattr(attack, "internal_input_scale_mode")
    assert not hasattr(attack, "query_input_scale_mode")


def test_game_legacy_scale_keys_are_ignored() -> None:
    attack = GAME({"repro_input_scale_mode": "unit", "batch_size": 2}, _state())
    assert not hasattr(attack, "repro_input_scale_mode")


def test_blackbox_ripper_rejects_non_tanh_generator_output_range() -> None:
    with pytest.raises(ValueError, match="generator_output_range must be 'tanh'"):
        BlackboxRipper({"generator_output_range": "unit", "output_mode": "soft_prob"}, _state())
