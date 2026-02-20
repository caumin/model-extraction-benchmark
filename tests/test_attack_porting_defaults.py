import torch

from mebench.attackers.dfme import DFME
from mebench.attackers.game import GAME
from mebench.attackers.knockoff_nets import KnockoffNets
from mebench.attackers.maze import MAZE
from mebench.attackers.swiftthief import SwiftThief
from mebench.core.state import BenchmarkState


def _state() -> BenchmarkState:
    s = BenchmarkState()
    s.metadata = {
        "device": "cpu",
        "num_classes": 10,
        "input_shape": (3, 32, 32),
        "dataset_config": {"name": "CIFAR10"},
        "substitute_config": {"arch": "resnet18"},
    }
    return s


def test_knockoffnets_defaults_match_official_transfer_defaults() -> None:
    atk = KnockoffNets({}, _state())
    assert atk.batch_size == 8
    assert atk.policy == "adaptive"


def test_maze_default_schedule_is_cosine() -> None:
    atk = MAZE({}, _state())
    assert atk.lr_schedule == "cosine"


def test_swiftthief_kd_lr_default_matches_official_sl_lr() -> None:
    atk = SwiftThief({}, _state())
    assert abs(atk.kd_lr - 1e-2) < 1e-12
    assert atk.query_fraction_per_round == 0.1
    assert atk.imbalance_kde_splits == 5


def test_game_default_batch_size_matches_official() -> None:
    atk = GAME({}, _state())
    assert atk.batch_size == 1024
    assert atk.querybudget == 2000
    assert atk.attack_train_epoch == 40
    assert atk.acs_strategy == "random"


def test_dfme_student_optimizer_defaults_match_official() -> None:
    atk = DFME({}, _state())
    assert isinstance(atk.s_opt, torch.optim.SGD)
    assert abs(float(atk.s_opt.param_groups[0]["lr"]) - 0.1) < 1e-12
