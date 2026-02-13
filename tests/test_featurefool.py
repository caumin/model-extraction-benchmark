"""Unit tests for CloudLeak FeatureFool implementation.

These tests are intentionally lightweight and only validate core invariants:
- Default feature layer selection should use a representation layer (not logits)
  for LeNet-style models.
- Generated adversarial samples must satisfy L_inf bounds around the source
  image and remain within [0,1].
"""

import torch

from mebench.attackers.cloudleak import FeatureFool
from mebench.models.substitute_factory import create_substitute


def test_featurefool_lenet_default_feature_layer_is_penultimate_linear() -> None:
    model = create_substitute("lenet_mnist", num_classes=10, input_channels=1)
    ff = FeatureFool(model, device="cpu")
    # LeNet5MNIST defines fc1/fc2/fc3; we should default to fc2 (penultimate).
    assert ff._feature_layer is getattr(model, "fc2")


def test_featurefool_epsilon_bound_and_range_constraints() -> None:
    torch.manual_seed(0)

    model = create_substitute("lenet_mnist", num_classes=10, input_channels=1)
    ff = FeatureFool(
        model,
        device="cpu",
        objective="euclidean",
        max_iters=1,
        epsilon=8.0 / 255.0,
    )

    x_source = torch.rand(2, 1, 28, 28)
    x_target = torch.rand(2, 1, 28, 28)
    x_adv = ff.generate_batch(x_source, x_target, to_cpu=True)

    assert x_adv.shape == x_source.shape
    assert float(x_adv.min()) >= 0.0
    assert float(x_adv.max()) <= 1.0

    eps = float(ff.epsilon)
    linf = (x_adv - x_source).abs().amax(dim=(1, 2, 3))
    assert bool(torch.all(linf <= eps + 1e-6))
