"""Basic GAME attack tests."""

import pytest
import torch
from mebench.core.state import BenchmarkState
from mebench.core.types import OracleOutput
from mebench.attackers.game import GAME
from mebench.models.gan import ACGANDiscriminator


def test_game_basic_flow() -> None:
    config = {
        "batch_size": 2,
        "noise_dim": 16,
        "num_classes": 10,
        "acs_strategy": "uncertainty",
        "proxy_dataset": {
            "name": "CIFAR10",
            "data_mode": "seed",
            "seed_size": 10,
        },
    }
    state = BenchmarkState()
    state.metadata = {"device": "cpu", "input_shape": (3, 32, 32)}

    attack = GAME(config, state)
    query_batch = attack._select_query_batch(2, state)
    assert "y_g" in query_batch.meta
    assert query_batch.meta["y_g"].shape[0] == 2
    probs = torch.softmax(torch.randn(2, 10), dim=1)
    oracle_output = OracleOutput(kind="soft_prob", y=probs)
    attack._handle_oracle_output(query_batch.x, query_batch.meta, oracle_output, state)

    assert state.attack_state["substitute"] is not None
    assert len(state.attack_state.get("query_data_x", [])) == 1
    assert len(state.attack_state.get("query_data_y", [])) == 1


def test_game_agu_loss_terms_parses_csv_and_dedupes() -> None:
    state = BenchmarkState()
    state.metadata = {"device": "cpu", "input_shape": (3, 32, 32)}

    attack = GAME(
        {
            "batch_size": 2,
            "noise_dim": 16,
            "num_classes": 10,
            "agu_loss_terms": "res,bou,dif,adv,res",
        },
        state,
    )

    assert attack.agu_loss_terms == ("res", "bou", "dif", "adv")


def test_game_agu_loss_terms_rejects_unknown_term() -> None:
    state = BenchmarkState()
    state.metadata = {"device": "cpu", "input_shape": (3, 32, 32)}

    with pytest.raises(ValueError, match="unsupported term"):
        GAME(
            {
                "batch_size": 2,
                "noise_dim": 16,
                "num_classes": 10,
                "agu_loss_terms": ["res", "invalid"],
            },
            state,
        )


def test_acgan_discriminator_has_fake_class_logit() -> None:
    disc = ACGANDiscriminator(input_channels=1, base_channels=16, num_classes=10, input_size=32)
    x = torch.randn(4, 1, 32, 32)
    src, cls = disc(x)
    assert src.shape[0] == 4
    assert cls.shape == (4, 11)


def test_game_default_noise_dim_matches_official_formula() -> None:
    state = BenchmarkState()
    state.metadata = {"device": "cpu", "input_shape": (3, 32, 32)}
    attack = GAME({"proxy_num_classes": 7, "use_acgan": True}, state)
    assert attack.noise_dim == 70
