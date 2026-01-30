import torch

from mebench.core.state import BenchmarkState
from mebench.attackers.blackbox_dissector import BlackboxDissector


class _TinyPool:
    def __init__(self, images_by_idx: dict[int, torch.Tensor]):
        self._images = images_by_idx

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int):
        return self._images[int(idx)], 0


def test_dissector_stage_a_and_b_are_separate_batches(monkeypatch) -> None:
    state = BenchmarkState()
    state.metadata = {
        "device": "cpu",
        "input_shape": (3, 8, 8),
        "num_classes": 3,
        "max_budget": 4,
        "dataset_config": {"pool_size": 4},
        "victim_config": {"normalization": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}},
        "substitute_config": {"arch": "lenet", "optimizer": {"lr": 0.1}},
    }

    attack = BlackboxDissector({"n_variants": 2, "max_epochs": 1, "patience": 1, "batch_size": 2}, state)
    attack.pool_dataset = _TinyPool(
        {
            0: torch.zeros(3, 8, 8),
            1: torch.zeros(3, 8, 8),
            2: torch.zeros(3, 8, 8),
            3: torch.zeros(3, 8, 8),
        }
    )

    # Force small pool indices.
    state.attack_state["unlabeled_indices"] = [0, 1, 2, 3]
    state.attack_state["labeled_indices"] = []

    class DummySub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Deterministic logits; MSP is constant.
            b = x.size(0)
            return torch.zeros(b, 3)

    state.attack_state["substitute"] = DummySub()

    # Monkeypatch CAM erasing to avoid Grad-CAM dependency in unit test.
    monkeypatch.setattr(
        "mebench.attackers.blackbox_dissector.cam_erase",
        lambda img, model, **kwargs: img + 0.1,
    )

    qb_a = attack.propose(100, state)
    assert qb_a.meta["stage"] == "A"
    assert all(t == "original" for t in qb_a.meta["variant_types"])
    assert qb_a.x.shape[0] == 2

    # Provide victim labels so Stage B can run.
    for idx in qb_a.meta["indices"]:
        state.attack_state["victim_labels"][int(idx)] = 0

    qb_b = attack.propose(100, state)
    assert qb_b.meta["stage"] == "B"
    assert all(t == "erased" for t in qb_b.meta["variant_types"])
    assert qb_b.x.shape[0] == 2


def test_random_erase_fills_with_uniform_0_1() -> None:
    from mebench.attackers.blackbox_dissector import random_erase

    img = torch.zeros(3, 16, 16)
    variants = random_erase(img, n=5, sl=0.02, sh=0.4, r1=0.3, r2=3.3, fill_min=0.0, fill_max=1.0)
    assert len(variants) == 5
    for v in variants:
        assert float(v.min().item()) >= 0.0
        assert float(v.max().item()) <= 1.0
