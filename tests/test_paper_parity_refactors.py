import itertools

import numpy as np
import torch
import torch.nn as nn

from mebench.core.state import BenchmarkState
from mebench.core.types import OracleOutput


class _TinyPool:
    def __init__(self, images_by_idx: dict[int, torch.Tensor]):
        self._images = images_by_idx

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int):
        return self._images[int(idx)], 0


def test_blackbox_dissector_variant_selection_uses_original_label(monkeypatch) -> None:
    """Eq.4 directionality: choose erased variant minimizing prob of original class."""

    from mebench.attackers.blackbox_dissector import BlackboxDissector
    import mebench.attackers.blackbox_dissector as dis_mod

    state = BenchmarkState()
    state.metadata = {"device": "cpu", "input_shape": (3, 32, 32), "num_classes": 3, "max_budget": 100}

    attack = BlackboxDissector({"n_variants": 2}, state)
    attack.pool_dataset = _TinyPool({0: torch.zeros(3, 32, 32)})

    # Transfer set contains idx=0 with victim original label y0=1.
    state.attack_state["labeled_indices"] = [0]
    state.attack_state["unlabeled_indices"] = []
    state.attack_state["victim_labels"][0] = 1

    # Two deterministic variants; variant_b should be chosen because it makes p(y0) smaller.
    variant_a = torch.zeros(3, 32, 32)
    variant_b = torch.ones(3, 32, 32)
    variants = itertools.cycle([variant_a, variant_b])

    def _fake_cam_erase(img, model, **_kwargs):
        return next(variants)

    monkeypatch.setattr(dis_mod, "cam_erase", _fake_cam_erase)

    class DummySub(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # If input is all ones -> low prob on class 1; if zeros -> high prob on class 1.
            s = float(x.mean().item())
            if s > 0.5:
                return torch.tensor([[5.0, -5.0, 0.0]], dtype=torch.float32)
            return torch.tensor([[-5.0, 5.0, 0.0]], dtype=torch.float32)

    state.attack_state["substitute"] = DummySub()

    np.random.seed(0)
    qb = attack.propose(3, state)

    chosen = state.attack_state["best_variant_img"][0]
    assert torch.allclose(chosen, variant_b)
    assert "erased" in qb.meta["variant_types"]


def test_inversenet_phase3_augment_before_query_and_train_on_oracle(monkeypatch) -> None:
    from mebench.attackers.inversenet import InverseNet

    state = BenchmarkState()
    state.metadata = {"device": "cpu", "input_shape": (3, 32, 32), "num_classes": 4, "max_budget": 1000}

    attack = InverseNet({"batch_size": 2, "num_classes": 4}, state)
    # Force phase 3.
    state.query_count = 999
    state.attack_state["phase"] = 3

    class DummyInv(nn.Module):
        def forward(self, y: torch.Tensor) -> torch.Tensor:
            # Encode y into pixels so augmentation can be detected.
            return y[:, :1].view(-1, 1, 1, 1).expand(-1, 3, 32, 32)

    attack.inversion_model = DummyInv()
    state.attack_state["inversion_y"] = [torch.eye(4)[:2]]

    monkeypatch.setattr(attack, "_augment_inversion", lambda x: x * 0.0)

    qb = attack.propose(2, state)
    assert qb.meta.get("phase") == 3
    assert qb.meta.get("augmented") is True
    assert float(qb.x.abs().sum().item()) == 0.0

    probs = torch.softmax(torch.randn(2, 4), dim=1)
    attack.observe(qb, OracleOutput(kind="soft_prob", y=probs), state)
    assert state.attack_state.get("substitute") is not None


def test_swiftthief_kde_selects_highest_density_candidate() -> None:
    from mebench.attackers.swiftthief import SwiftThief

    state = BenchmarkState()
    state.metadata = {"device": "cpu", "input_shape": (3, 32, 32), "num_classes": 2, "max_budget": 100}

    attack = SwiftThief({"num_classes": 2, "kde_sigma": 0.1, "max_pool_eval": 10}, state)

    # Pool values: idx 1 is closest to rare class feature (0.0), so should be selected.
    pool = {
        0: torch.full((3, 32, 32), 0.0),
        1: torch.full((3, 32, 32), 0.0),
        2: torch.full((3, 32, 32), 1.0),
        3: torch.full((3, 32, 32), 2.0),
    }
    attack.pool_dataset = _TinyPool(pool)

    class DummySwift(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))

        def features(self, x: torch.Tensor) -> torch.Tensor:
            return x.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Not used in KDE test.
            return torch.zeros(x.size(0), 2)

    state.attack_state["substitute"] = DummySwift()
    state.attack_state["labeled_indices"] = [0]
    state.attack_state["unlabeled_indices"] = [1, 2, 3]

    # Rare class is 0.
    state.attack_state["class_counts"] = {0: 1, 1: 10}
    state.attack_state["victim_outputs"][0] = torch.tensor([1.0, 0.0])

    np.random.seed(0)
    selected = attack._select_rare_class(1, state)
    assert selected == [1]
