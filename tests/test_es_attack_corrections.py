import torch

from mebench.attackers.es_attack import ESAttack
from mebench.core.state import BenchmarkState


def _make_state(input_shape=(3, 32, 32)) -> BenchmarkState:
    state = BenchmarkState()
    state.metadata = {
        "device": "cpu",
        "input_shape": input_shape,
        "num_classes": 10,
        "substitute_config": {"arch": "lenet", "optimizer": {"lr": 0.01}},
    }
    state.attack_state = {"step": 0, "syn_index": 0}
    return state


def test_es_attack_step0_returns_gaussian_noise_dnn_syn() -> None:
    torch.manual_seed(0)
    state = _make_state()
    attack = ESAttack(
        {
            "synthesis_mode": "dnn_syn",
            "batch_size": 8,
            "student_epochs": 1,
            "synthesis_steps": 1,
        },
        state,
    )
    x, meta = attack._select_query_batch(8, state)
    assert x.shape == (8, 3, 32, 32)
    assert float(x.min()) >= 0.0
    assert float(x.max()) <= 1.0
    assert meta.get("mode") == "init_gaussian"
    assert meta.get("step") == 0


def test_es_attack_step0_returns_gaussian_noise_opt_syn() -> None:
    torch.manual_seed(0)
    state = _make_state()
    attack = ESAttack(
        {
            "synthesis_mode": "opt_syn",
            "batch_size": 8,
            "student_epochs": 1,
            "opt_steps": 30,
        },
        state,
    )
    x, meta = attack._select_query_batch(8, state)
    assert x.shape == (8, 3, 32, 32)
    assert float(x.min()) >= 0.0
    assert float(x.max()) <= 1.0
    assert meta.get("mode") == "init_gaussian"


def test_es_attack_dirichlet_targets_valid_simplex() -> None:
    torch.manual_seed(0)
    state = _make_state()
    attack = ESAttack({"synthesis_mode": "opt_syn"}, state)
    y = attack._sample_dirichlet_targets(batch_size=16, device="cpu")
    assert y.shape == (16, 10)
    assert torch.isfinite(y).all()
    assert float(y.min()) >= 0.0
    # Dirichlet samples are on the simplex.
    row_sums = y.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4, rtol=1e-4)


def test_es_attack_generator_update_has_classification_grad() -> None:
    """Regression: if student forward is under no_grad, l_img cannot update generator.

    Set mode_seeking_weight=0 to ensure only l_img drives the update.
    """

    torch.manual_seed(0)
    state = _make_state()
    attack = ESAttack(
        {
            "synthesis_mode": "dnn_syn",
            "batch_size": 4,
            "noise_dim": 16,
            "student_epochs": 1,
            "synthesis_steps": 1,
            "mode_seeking_weight": 0.0,
            "acgan_weight": 1.0,
            "generator_lr": 1e-3,
        },
        state,
    )
    attack._init_models(state)
    assert attack.generator is not None
    assert attack.student is not None

    z_cpu = torch.randn(4, 16)
    y_g_cpu = torch.randint(0, 10, (4,))

    # Snapshot one parameter tensor.
    p0 = next(iter(attack.generator.parameters())).detach().clone()
    attack._train_generator(z_cpu=z_cpu, y_g_cpu=y_g_cpu, device="cpu")
    p1 = next(iter(attack.generator.parameters())).detach().clone()

    assert not torch.allclose(p0, p1)
