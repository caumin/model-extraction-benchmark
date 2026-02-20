import torch
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop

from mebench.data.preprocessing import (
    apply_official_preprocess_batch,
    get_official_preprocess,
    list_official_preprocess_profiles,
)


def test_official_preprocess_profiles_registered() -> None:
    names = set(list_official_preprocess_profiles())
    expected = {
        "dfme_cifar10_test",
        "maze_rgb_test",
        "swiftthief_cifar_test",
        "knockoffnets_default_test",
        "dfms_hl_train_student",
    }
    assert expected.issubset(names)


def test_dfme_profile_matches_reference_normalize() -> None:
    x = torch.rand(3, 32, 32)
    profile = get_official_preprocess("dfme_cifar10_test")
    ref = Compose(
        [
            lambda t: t.clamp(0.0, 1.0),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    y_a = profile(x)
    y_b = ref(x)
    assert torch.allclose(y_a, y_b, atol=1e-7, rtol=1e-6)


def test_knockoffnets_profile_matches_reference_transform_chain() -> None:
    x = torch.rand(1, 3, 320, 320)
    y_a = apply_official_preprocess_batch(x, "knockoffnets_default_test")

    ref = Compose(
        [
            Resize(256),
            CenterCrop(224),
            lambda t: t.clamp(0.0, 1.0),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    y_b = torch.stack([ref(img) for img in x], dim=0)
    assert y_a.shape == y_b.shape == (1, 3, 224, 224)
    assert torch.allclose(y_a, y_b, atol=1e-7, rtol=1e-6)
