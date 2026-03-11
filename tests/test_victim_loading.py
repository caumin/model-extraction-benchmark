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


def test_load_victim_checkpoint_accepts_extensionless_path_with_pt_and_pth_suffixes():
    """Support checkpoint paths that omit extension (.pt / .pth)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = create_substitute(arch="resnet18", num_classes=10, input_channels=3)
        stem = Path(tmpdir) / "victim_no_ext"
        torch.save(model.state_dict(), stem.with_suffix(".pt"))

        loaded_model = load_victim_checkpoint(
            checkpoint_path=str(stem),
            arch="resnet18",
            num_classes=10,
            input_channels=3,
            device="cpu",
        )

        assert loaded_model is not None
        assert loaded_model.training is False

        model_pth = create_substitute(arch="resnet18", num_classes=10, input_channels=3)
        # make a different model so extension preference is observable if both exist
        with torch.no_grad():
            for param in model_pth.parameters():
                param.add_(1.0)
        torch.save(model_pth.state_dict(), stem.with_suffix(".pth"))

        loaded_model_from_stem = load_victim_checkpoint(
            checkpoint_path=str(stem),
            arch="resnet18",
            num_classes=10,
            input_channels=3,
            device="cpu",
        )
        # Resolve should prefer .pt before .pth when both are present
        assert all(
            torch.equal(a, b)
            for a, b in zip(loaded_model.parameters(), loaded_model_from_stem.parameters())
        )


def test_create_and_load_xie2019_checkpoint() -> None:
    """Xie2019 architecture can be created and loaded from checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = create_substitute(
            arch="xie2019",
            num_classes=17,
            input_channels=3,
            dropout_prob=0.6,
        )
        model.eval()

        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2, 17)

        checkpoint_path = Path(tmpdir) / "victim_xie2019.pt"
        torch.save(model.state_dict(), checkpoint_path)

        loaded_model = load_victim_checkpoint(
            checkpoint_path=str(checkpoint_path),
            arch="xie2019",
            num_classes=17,
            input_channels=3,
            dropout_prob=0.6,
            device="cpu",
        )
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


def test_load_victim_checkpoint_ignores_official_preprocess_profile():
    """Victim inference uses direct model input without runtime wrapper transforms."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = create_substitute(arch="resnet18", num_classes=10, input_channels=3)
        model.eval()
        checkpoint_path = Path(tmpdir) / "victim_preprocess.pt"
        torch.save(model.state_dict(), checkpoint_path)

        wrapped = load_victim_checkpoint(
            checkpoint_path=str(checkpoint_path),
            arch="resnet18",
            num_classes=10,
            input_channels=3,
            official_preprocess_profile="dfme_cifar10_test",
            device="cpu",
        )

        x = torch.rand(2, 3, 32, 32)
        with torch.no_grad():
            y_wrapped = wrapped(x)
            y_ref = model(x)

        assert torch.allclose(y_wrapped, y_ref, atol=1e-6, rtol=1e-5)


def test_load_victim_checkpoint_preserves_tanh_queries() -> None:
    """Victim path no longer converts tanh-scale queries to unit scale."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = create_substitute(arch="resnet18", num_classes=10, input_channels=3)
        model.eval()
        checkpoint_path = Path(tmpdir) / "victim_unit_wrapper.pt"
        torch.save(model.state_dict(), checkpoint_path)

        wrapped = load_victim_checkpoint(
            checkpoint_path=str(checkpoint_path),
            arch="resnet18",
            num_classes=10,
            input_channels=3,
            official_preprocess_profile=None,
            device="cpu",
        )

        x_tanh = torch.empty(2, 3, 32, 32).uniform_(-1.0, 1.0)
        with torch.no_grad():
            y_wrapped = wrapped(x_tanh)
            y_ref = model(x_tanh)

        assert torch.allclose(y_wrapped, y_ref, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
