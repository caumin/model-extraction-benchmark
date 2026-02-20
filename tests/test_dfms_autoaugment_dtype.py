import torch
import torchvision.transforms.functional as TF

from mebench.attackers.dfms import DFMSHL
from mebench.core.state import BenchmarkState


def test_dfms_autoaugment_handles_float_tensor_equalize_path() -> None:
    state = BenchmarkState()
    state.metadata = {
        "device": "cpu",
        "input_shape": (3, 32, 32),
        "num_classes": 10,
        "dataset_config": {"name": "CIFAR10"},
        "substitute_config": {"arch": "resnet18"},
    }

    atk = DFMSHL({"alternate_auto_augment": True, "auto_augment_policy": "cifar10"}, state)
    atk._auto_augment = lambda img: TF.equalize(img)

    x = torch.rand(4, 3, 32, 32, dtype=torch.float32)
    y = atk._augment_auto_augment(x)

    assert y.dtype == torch.float32
    assert tuple(y.shape) == tuple(x.shape)
    assert float(y.min().item()) >= 0.0
    assert float(y.max().item()) <= 1.0
