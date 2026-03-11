"""Test contract validation (fail-fast on invalid configs)."""

import pytest
from mebench.core.validate import validate_config


def test_dfme_requires_data_free():
    """DFME requires data_free mode."""
    config = {
        "attack": {"name": "dfme", "output_mode": "soft_prob"},
        "dataset": {"data_mode": "surrogate", "seed_name": "MNIST"},  # Invalid for DFME
        "victim": {"output_mode": "soft_prob", "temperature": 1.0},
        "budget": {"max_budget": 1000000, "checkpoints": [1000, 10000]},
    }

    with pytest.raises(ValueError, match="DFME requires data_free mode"):
        validate_config(config)


def test_temperature_must_be_1_for_default_oracle():
    """Default oracle requires T=1.0 in v1.0."""
    config = {
        "attack": {"name": "random", "output_mode": "soft_prob"},
        "dataset": {"data_mode": "seed", "seed_name": "MNIST"},
        "victim": {"output_mode": "soft_prob", "temperature": 2.0},  # Invalid
        "substitute": {"max_epochs": 200, "patience": 20},
        "budget": {"max_budget": 1000000, "checkpoints": [1000, 10000]},
    }

    with pytest.raises(ValueError, match="Default oracle requires T=1.0 in v1.0"):
        validate_config(config)


def test_output_mode_mismatch():
    """Victim and attack output modes must match."""
    config = {
        "attack": {"name": "random", "output_mode": "hard_top1"},
        "dataset": {"data_mode": "seed", "seed_name": "MNIST"},
        "victim": {"output_mode": "soft_prob"},  # Mismatch
        "temperature": 1.0,
        "budget": {"max_budget": 1000000, "checkpoints": [1000, 10000]},
    }

    with pytest.raises(ValueError, match="Mode mismatch"):
        validate_config(config)


def test_checkpoint_exceeds_max_budget():
    """Checkpoints cannot exceed max_budget."""
    config = {
        "attack": {"name": "random", "output_mode": "soft_prob"},
        "dataset": {"data_mode": "seed", "seed_name": "MNIST"},
        "victim": {"output_mode": "soft_prob", "temperature": 1.0},
        "substitute": {"max_epochs": 200, "patience": 20},
        "budget": {"max_budget": 1000, "checkpoints": [1000, 10000]},  # 10000 > 1000
    }

    with pytest.raises(ValueError, match="Checkpoint exceeds max_budget"):
        validate_config(config)


def test_checkpoints_must_be_increasing():
    """Checkpoints must be in increasing order."""
    config = {
        "attack": {"name": "random", "output_mode": "soft_prob"},
        "dataset": {"data_mode": "seed", "seed_name": "MNIST"},
        "victim": {"output_mode": "soft_prob", "temperature": 1.0},
        "substitute": {"max_epochs": 200, "patience": 20},
        "budget": {"max_budget": 1000000, "checkpoints": [10000, 1000]},  # Not increasing
    }

    with pytest.raises(ValueError, match="Checkpoints must be increasing"):
        validate_config(config)


def test_valid_config_passes():
    """Valid configuration should not raise errors."""
    config = {
        "attack": {"name": "random", "output_mode": "soft_prob"},
        "dataset": {"data_mode": "seed", "seed_name": "MNIST"},
        "victim": {"output_mode": "soft_prob", "temperature": 1.0},
        "substitute": {"max_epochs": 200, "patience": 20},
        "budget": {"max_budget": 1000000, "checkpoints": [1000, 10000, 100000]},
    }

    # Should not raise
    validate_config(config)


def test_valid_sewerml_config_passes() -> None:
    config = {
        "attack": {"name": "random", "output_mode": "soft_prob"},
        "dataset": {"data_mode": "surrogate", "name": "SewerML", "surrogate_name": "ImageNet"},
        "victim": {"output_mode": "soft_prob", "temperature": 1.0},
        "substitute": {"max_epochs": 200, "patience": 20},
        "budget": {"max_budget": 1000000, "checkpoints": [1000, 10000, 100000]},
    }

    validate_config(config)


def test_valid_sewerml_config_with_binary_label_mode() -> None:
    config = {
        "attack": {"name": "random", "output_mode": "soft_prob"},
        "dataset": {
            "data_mode": "surrogate",
            "name": "SewerML",
            "surrogate_name": "ImageNet",
            "sewerml_label_mode": "binary",
        },
        "victim": {"output_mode": "soft_prob", "temperature": 1.0, "num_classes": 1},
        "substitute": {"max_epochs": 200, "patience": 20},
        "budget": {"max_budget": 1000000, "checkpoints": [1000, 10000, 100000]},
    }

    validate_config(config)


def test_sewerml_binary_mode_requires_single_logit_victim() -> None:
    config = {
        "attack": {"name": "random", "output_mode": "soft_prob"},
        "dataset": {
            "data_mode": "surrogate",
            "name": "SewerML",
            "surrogate_name": "ImageNet",
            "sewerml_label_mode": "binary",
        },
        "victim": {"output_mode": "soft_prob", "temperature": 1.0, "num_classes": 2},
        "substitute": {"max_epochs": 200, "patience": 20},
        "budget": {"max_budget": 1000000, "checkpoints": [1000, 10000, 100000]},
    }

    with pytest.raises(ValueError, match="victim.num_classes=1"):
        validate_config(config)


def test_cloudleak_allowed_for_single_logit_binary() -> None:
    config = {
        "attack": {"name": "cloudleak", "output_mode": "soft_prob"},
        "dataset": {"data_mode": "surrogate", "name": "SewerML", "surrogate_name": "ImageNet", "sewerml_label_mode": "binary"},
        "victim": {"output_mode": "soft_prob", "temperature": 1.0, "num_classes": 1},
        "substitute": {"max_epochs": 200, "patience": 20},
        "budget": {"max_budget": 1000000, "checkpoints": [1000, 10000]},
    }

    validate_config(config)


@pytest.mark.parametrize(
    ("attack_name", "output_mode"),
    [
        ("copycatcnn", "hard_top1"),
        ("blackbox_dissector", "hard_top1"),
        ("marich", "hard_top1"),
        ("dfme", "soft_prob"),
        ("ds", "soft_prob"),
        ("maze", "soft_prob"),
        ("es", "soft_prob"),
        ("disguide", "soft_prob"),
        ("dfms", "hard_top1"),
        ("game", "soft_prob"),
        ("inversenet", "hard_top1"),
        ("blackbox_ripper", "soft_prob"),
        ("swiftthief", "soft_prob"),
    ],
)
def test_single_logit_binary_supported_attacks_validate(attack_name: str, output_mode: str) -> None:
    data_mode = "data_free" if attack_name in {"dfme", "ds", "maze", "es", "disguide", "dfms", "game", "blackbox_ripper"} else "surrogate"
    config = {
        "attack": {"name": attack_name, "output_mode": output_mode},
        "dataset": {
            "data_mode": data_mode,
            "name": "SewerML",
            "surrogate_name": "ImageNet",
            "sewerml_label_mode": "binary",
        },
        "victim": {"output_mode": output_mode, "temperature": 1.0, "num_classes": 1},
        "substitute": {"max_epochs": 10, "patience": 2},
        "budget": {"max_budget": 1000, "checkpoints": [1000]},
    }

    if attack_name == "blackbox_ripper":
        config["attack"]["generator_checkpoint"] = "dummy.ckpt"

    validate_config(config)


def test_invalid_sewerml_label_mode_rejected() -> None:
    config = {
        "attack": {"name": "random", "output_mode": "soft_prob"},
        "dataset": {
            "data_mode": "surrogate",
            "name": "SewerML",
            "surrogate_name": "ImageNet",
            "sewerml_label_mode": "not_a_mode",
        },
        "victim": {"output_mode": "soft_prob", "temperature": 1.0},
        "substitute": {"max_epochs": 200, "patience": 20},
        "budget": {"max_budget": 1000000, "checkpoints": [1000, 10000, 100000]},
    }

    with pytest.raises(ValueError, match="Unsupported Sewerml label mode"):
        validate_config(config)


def test_sewerml_label_mode_is_ignored_for_non_sewerml_dataset() -> None:
    """Non-SewerML datasets should ignore SewerML-specific keys."""
    config = {
        "attack": {"name": "random", "output_mode": "soft_prob"},
        "dataset": {
            "data_mode": "seed",
            "seed_name": "MNIST",
            "sewerml_label_mode": "not_a_mode",  # should not be validated here
        },
        "victim": {"output_mode": "soft_prob", "temperature": 1.0},
        "substitute": {"max_epochs": 200, "patience": 20},
        "budget": {"max_budget": 1000000, "checkpoints": [1000, 10000, 100000]},
    }

    validate_config(config)


def test_benchmark_policy_allows_missing_preprocess_declaration():
    """Benchmark inference policy no longer requires preprocess declaration."""
    config = {
        "attack": {"name": "random", "output_mode": "soft_prob"},
        "dataset": {"data_mode": "seed", "seed_name": "MNIST"},
        "victim": {
            "output_mode": "soft_prob",
            "temperature": 1.0,
            "inference_policy": "benchmark",
        },
        "substitute": {"max_epochs": 200, "patience": 20},
        "budget": {"max_budget": 1000000, "checkpoints": [1000, 10000]},
    }

    validate_config(config)


def test_unknown_official_profile_does_not_fail_validation():
    """Unknown official profile is ignored when wrapper preprocessing is disabled."""
    config = {
        "attack": {"name": "random", "output_mode": "soft_prob"},
        "dataset": {"data_mode": "seed", "seed_name": "MNIST"},
        "victim": {
            "output_mode": "soft_prob",
            "temperature": 1.0,
            "official_preprocess_profile": "unknown_profile",
        },
        "substitute": {"max_epochs": 200, "patience": 20},
        "budget": {"max_budget": 1000000, "checkpoints": [1000, 10000]},
    }

    validate_config(config)


def test_benchmark_policy_allows_explicit_no_normalization():
    """Benchmark policy accepts explicit 'no normalization' declaration."""
    config = {
        "attack": {"name": "random", "output_mode": "soft_prob"},
        "dataset": {"data_mode": "seed", "seed_name": "MNIST"},
        "victim": {
            "output_mode": "soft_prob",
            "temperature": 1.0,
            "inference_policy": "benchmark",
            "normalization": None,
        },
        "substitute": {"max_epochs": 200, "patience": 20},
        "budget": {"max_budget": 1000000, "checkpoints": [1000, 10000]},
    }

    validate_config(config)


def test_data_free_does_not_require_input_scale_mode_key():
    config = {
        "attack": {"name": "dfme", "output_mode": "soft_prob"},
        "dataset": {"data_mode": "data_free"},
        "victim": {
            "output_mode": "soft_prob",
            "temperature": 1.0,
            "inference_policy": "benchmark",
            "normalization": None,
        },
        "budget": {"max_budget": 1000000, "checkpoints": [1000, 10000]},
    }

    validate_config(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
