"""Test victim checkpoint loading."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile

from mebench.oracles.victim_loader import load_victim_checkpoint, load_victim_from_config
from mebench.models.substitute_factory import create_substitute


def test_load_victim_from_checkpoint_basic():
    """Test loading victim model from checkpoint file."""
    # Create a temporary checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple model and save it
        model = create_substitute(
            arch="resnet18",
            num_classes=10,
            input_channels=3,
        )

        checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"
        torch.save(model.state_dict(), checkpoint_path)

        # Load the checkpoint
        loaded_model = load_victim_checkpoint(
            checkpoint_path=str(checkpoint_path),
            arch="resnet18",
            num_classes=10,
            input_channels=3,
            device="cpu",
        )

        # Check model is loaded correctly
        assert loaded_model is not None
        assert isinstance(loaded_model, nn.Module)
        assert loaded_model.training is False  # Should be in eval mode


def test_load_victim_from_config_with_checkpoint():
    """Test loading victim using config dict with checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create checkpoint file
        model = create_substitute(arch="resnet18", num_classes=10, input_channels=3)
        checkpoint_path = Path(tmpdir) / "victim.pt"
        torch.save(model.state_dict(), checkpoint_path)

        config = {
            "checkpoint_ref": str(checkpoint_path),
            "arch": "resnet18",
            "channels": 3,
            "num_classes": 10,
        }

        # Load using config
        loaded_model = load_victim_from_config(config, device="cpu")

        assert loaded_model is not None
        assert loaded_model.training is False


def test_load_victim_from_config_placeholder():
    """Test loading victim with placeholder checkpoint path."""
    config = {
        "checkpoint_ref": "/path/to/ckpt.pt",  # Placeholder path
        "arch": "resnet18",
        "channels": 3,
        "num_classes": 10,
    }

    # Should create placeholder model with warning
    loaded_model = load_victim_from_config(config, device="cpu")

    assert loaded_model is not None
    assert loaded_model.training is False


def test_load_victim_from_config_with_dropout_checkpoint():
    """Test loading victim checkpoint that was trained with dropout-enabled arch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = create_substitute(
            arch="classifier",
            num_classes=10,
            input_channels=1,
            dropout_prob=0.2,
        )
        checkpoint_path = Path(tmpdir) / "victim_classifier_dropout.pt"
        torch.save(model.state_dict(), checkpoint_path)

        config = {
            "checkpoint_ref": str(checkpoint_path),
            "arch": "classifier",
            "dropout_prob": 0.2,
            "channels": 1,
            "num_classes": 10,
        }

        loaded_model = load_victim_from_config(config, device="cpu")
        assert loaded_model is not None
        assert loaded_model.training is False


def test_load_victim_nonexistent_checkpoint():
    """Test that loading nonexistent checkpoint raises error."""
    config = {
        "checkpoint_ref": "/nonexistent/path/checkpoint.pt",
        "arch": "resnet18",
        "channels": 3,
        "num_classes": 10,
    }

    with pytest.raises(FileNotFoundError, match="Victim checkpoint not found"):
        load_victim_from_config(config, device="cpu")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
