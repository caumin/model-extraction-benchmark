"""Tests for new attack configurations."""

import pytest

from mebench.core.validate import validate_config


def test_swiftthief_allows_soft_prob():
    """SwiftThief supports soft_prob output mode."""
    config = {
        "run": {"name": "test", "seeds": [0]},
        "victim": {"output_mode": "soft_prob", "temperature": 1.0},
        "dataset": {"data_mode": "surrogate", "seed_size": 10000, "seed_name": "CIFAR10"},
        "attack": {"name": "swiftthief", "output_mode": "soft_prob"},
        "budget": {"max_budget": 10000, "checkpoints": [1000, 10000]},
    }
    validate_config(config)  # Should not raise


def test_swiftthief_allows_hard_top1():
    """SwiftThief supports hard_top1 output mode."""
    config = {
        "run": {"name": "test", "seeds": [0]},
        "victim": {"output_mode": "hard_top1", "temperature": 1.0},
        "dataset": {"data_mode": "surrogate", "seed_size": 10000, "seed_name": "CIFAR10"},
        "attack": {"name": "swiftthief", "output_mode": "hard_top1"},
        "budget": {"max_budget": 10000, "checkpoints": [1000, 10000]},
    }
    validate_config(config)  # Should not raise


def test_copycatcnn_requires_hard_top1():
    """CopycatCNN is evaluated under hard_top1 only."""
    ok = {
        "run": {"name": "test", "seeds": [0]},
        "victim": {"output_mode": "hard_top1", "temperature": 1.0},
        "dataset": {"data_mode": "surrogate", "seed_size": 10000, "seed_name": "CIFAR10"},
        "attack": {"name": "copycatcnn", "output_mode": "hard_top1"},
        "budget": {"max_budget": 10000, "checkpoints": [1000, 10000]},
    }
    validate_config(ok)

    bad = dict(ok)
    bad["victim"] = {"output_mode": "soft_prob", "temperature": 1.0}
    bad["attack"] = {"name": "copycatcnn", "output_mode": "soft_prob"}
    with pytest.raises(ValueError, match="copycatcnn requires hard_top1"):
        validate_config(bad)


def test_inversenet_requires_hard_top1():
    """InverseNet is evaluated under hard_top1 only."""
    ok = {
        "run": {"name": "test", "seeds": [0]},
        "victim": {"output_mode": "hard_top1", "temperature": 1.0},
        "dataset": {"data_mode": "surrogate", "seed_size": 10000, "seed_name": "CIFAR10"},
        "attack": {"name": "inversenet", "output_mode": "hard_top1"},
        "budget": {"max_budget": 10000, "checkpoints": [1000, 10000]},
    }
    validate_config(ok)

    bad = dict(ok)
    bad["victim"] = {"output_mode": "soft_prob", "temperature": 1.0}
    bad["attack"] = {"name": "inversenet", "output_mode": "soft_prob"}
    with pytest.raises(ValueError, match="inversenet requires hard_top1"):
        validate_config(bad)


def test_dfms_requires_hard_top1():
    """DFMS-HL is evaluated under hard_top1 only."""
    ok = {
        "run": {"name": "test", "seeds": [0]},
        "victim": {"output_mode": "hard_top1", "temperature": 1.0},
        "dataset": {"data_mode": "data_free"},
        "attack": {"name": "dfms", "output_mode": "hard_top1"},
        "budget": {"max_budget": 10000, "checkpoints": [1000, 10000]},
    }
    validate_config(ok)

    bad = dict(ok)
    bad["victim"] = {"output_mode": "soft_prob", "temperature": 1.0}
    bad["attack"] = {"name": "dfms", "output_mode": "soft_prob"}
    with pytest.raises(ValueError, match="dfms requires hard_top1"):
        validate_config(bad)


def test_knockoff_nets_requires_soft_prob():
    """KnockoffNets is evaluated under soft_prob only."""
    ok = {
        "run": {"name": "test", "seeds": [0]},
        "victim": {"output_mode": "soft_prob", "temperature": 1.0},
        "dataset": {"data_mode": "surrogate", "seed_size": 10000, "seed_name": "CIFAR10"},
        "attack": {"name": "knockoff_nets", "output_mode": "soft_prob"},
        "budget": {"max_budget": 10000, "checkpoints": [1000, 10000]},
    }
    validate_config(ok)

    bad = dict(ok)
    bad["victim"] = {"output_mode": "hard_top1", "temperature": 1.0}
    bad["attack"] = {"name": "knockoff_nets", "output_mode": "hard_top1"}
    with pytest.raises(ValueError, match="knockoff_nets requires soft_prob"):
        validate_config(bad)


def test_dissector_requires_hard_top1():
    """Dissector requires hard_top1 output mode."""
    config = {
        "run": {"name": "test", "seeds": [0]},
        "victim": {"output_mode": "hard_top1", "temperature": 1.0},
        "dataset": {"data_mode": "surrogate", "seed_size": 10000, "seed_name": "CIFAR10"},
        "attack": {"name": "blackbox_dissector", "output_mode": "hard_top1"},
        "budget": {"max_budget": 10000, "checkpoints": [1000, 10000]},
    }
    validate_config(config)  # Should not raise


def test_dissector_rejects_soft_prob():
    """Dissector rejects soft_prob output mode."""
    config = {
        "run": {"name": "test", "seeds": [0]},
        "victim": {"output_mode": "soft_prob", "temperature": 1.0},
        "dataset": {"data_mode": "surrogate", "seed_size": 10000, "seed_name": "CIFAR10"},
        "attack": {"name": "blackbox_dissector", "output_mode": "soft_prob"},
        "budget": {"max_budget": 10000, "checkpoints": [1000, 10000]},
    }
    with pytest.raises(ValueError, match="blackbox_dissector requires hard_top1"):
        validate_config(config)


def test_cloudleak_requires_soft_prob():
    """CloudLeak requires soft_prob output mode."""
    config = {
        "run": {"name": "test", "seeds": [0]},
        "victim": {"output_mode": "soft_prob", "temperature": 1.0},
        "dataset": {"data_mode": "seed", "seed_size": 1000, "seed_name": "MNIST"},
        "attack": {"name": "cloudleak", "output_mode": "soft_prob"},
        "budget": {"max_budget": 10000, "checkpoints": [1000, 10000]},
    }
    validate_config(config)  # Should not raise


def test_cloudleak_rejects_hard_top1():
    """CloudLeak rejects hard_top1 output mode."""
    config = {
        "run": {"name": "test", "seeds": [0]},
        "victim": {"output_mode": "hard_top1", "temperature": 1.0},
        "dataset": {"data_mode": "seed", "seed_size": 1000, "seed_name": "MNIST"},
        "attack": {"name": "cloudleak", "output_mode": "hard_top1"},
        "budget": {"max_budget": 10000, "checkpoints": [1000, 10000]},
    }
    with pytest.raises(ValueError, match="cloudleak requires soft_prob"):
        validate_config(config)


def test_new_attacks_mode_mismatch():
    """New attacks reject mode mismatch."""
    # SwiftThief with hard mode
    config = {
        "run": {"name": "test", "seeds": [0]},
        "victim": {"output_mode": "soft_prob", "temperature": 1.0},
        "dataset": {"data_mode": "surrogate", "seed_size": 10000, "seed_name": "CIFAR10"},
        "attack": {"name": "swiftthief", "output_mode": "hard_top1"},  # Mismatch
        "budget": {"max_budget": 10000, "checkpoints": [1000, 10000]},
    }
    with pytest.raises(ValueError, match="Mode mismatch"):
        validate_config(config)
