from pathlib import Path

import pytest
import torch
import torch.nn as nn

from mebench.models.blackbox_ripper.factory import load_blackbox_ripper_generator_weights


class _TinyGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)


def test_load_weights_accepts_checkpoint_without_extension(tmp_path: Path) -> None:
    model = _TinyGenerator()
    checkpoint = tmp_path / "cifar_100_6_classes_gan.pth"
    torch.save(model.state_dict(), checkpoint)

    reloaded = _TinyGenerator()
    load_blackbox_ripper_generator_weights(
        reloaded,
        str(checkpoint.with_suffix("")),
        device="cpu",
        strict=True,
    )


def test_load_weights_missing_checkpoint_raises(tmp_path: Path) -> None:
    model = _TinyGenerator()
    with pytest.raises(FileNotFoundError):
        load_blackbox_ripper_generator_weights(
            model,
            str(tmp_path / "missing_checkpoint"),
            device="cpu",
            strict=True,
        )
