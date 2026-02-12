import pytest
import torch

from mebench.attackers.swiftthief import SoftSupSimSiamLossV17


def test_softsup_eta_entropy_factor_decreases_with_entropy() -> None:
    """Official SwiftThief repo uses reversed normalized entropy (1 - H(y)/logK).

    With identical targets across the batch, the entropy weight controls whether
    any pair contributes to the loss.
    """

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    k = 2
    loss_fn = SoftSupSimSiamLossV17(device=device, num_classes=k).to(device)

    # 2q = 4 views, d = 8
    torch.manual_seed(0)
    z = torch.randn(4, 8, device=device)
    p = torch.randn(4, 8, device=device)

    # Low entropy (one-hot): weight factor ~= 1
    y_onehot = torch.tensor([[1.0, 0.0]] * 4, device=device)
    loss_onehot = loss_fn(p=p, z=z, targets=y_onehot)

    # High entropy (uniform): reversed entropy weight ~= 0 -> loss becomes 0
    y_uniform = torch.tensor([[0.5, 0.5]] * 4, device=device)
    loss_uniform = loss_fn(p=p, z=z, targets=y_uniform)

    assert loss_onehot.abs().item() > 0.0
    assert float(loss_uniform.item()) == pytest.approx(0.0, abs=1e-6)
