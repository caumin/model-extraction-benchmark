import pytest
import torch

from mebench.attackers.swiftthief import SoftSupSimSiamLossV17


def test_softsup_eta_entropy_factor_increases_with_entropy() -> None:
    """SwiftThief Eq.(3): (1 + H(y)/logK) increases with entropy.

    If we keep (p, z) fixed and use identical targets across the batch,
    the loss magnitude should scale with the square of this factor.
    """

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    k = 2
    loss_fn = SoftSupSimSiamLossV17(device=device, num_classes=k).to(device)

    # 2q = 4 views, d = 8
    torch.manual_seed(0)
    z = torch.randn(4, 8, device=device)
    p = torch.randn(4, 8, device=device)

    # Low entropy (one-hot): H=0 -> factor a=1
    y_onehot = torch.tensor([[1.0, 0.0]] * 4, device=device)
    loss_onehot = loss_fn(p=p, z=z, targets=y_onehot)

    # High entropy (uniform): H=logK -> factor a=2
    y_uniform = torch.tensor([[0.5, 0.5]] * 4, device=device)
    loss_uniform = loss_fn(p=p, z=z, targets=y_uniform)

    # For identical targets, cos(y_i, y_j)=1.0 and eta scales by a_i*a_j.
    # Thus eta should scale by 4x and the (negative) loss should scale by ~4x.
    ratio = float(loss_uniform.item() / loss_onehot.item())
    assert ratio == pytest.approx(4.0, rel=1e-3, abs=1e-3)
