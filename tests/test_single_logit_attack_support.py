import torch

from mebench.attackers.blackbox_ripper import BlackboxRipper
from mebench.attackers.disguide import DisGUIDE
from mebench.attackers.dfme import DFME
from mebench.attackers.dfms import DFMSHL
from mebench.attackers.ds import DualStudents
from mebench.attackers.es_attack import ESAttack
from mebench.attackers.game import GAME
from mebench.attackers.inversenet import InverseNet
from mebench.attackers.maze import MAZE
from mebench.attackers.copycatcnn import CopycatCNN
from mebench.attackers.swiftthief import SwiftThief
from mebench.core.state import BenchmarkState
from mebench.core.types import OracleOutput


def _state(*, output_mode: str = "soft_prob") -> BenchmarkState:
    return BenchmarkState(
        budget_remaining=64,
        metadata={
            "device": "cpu",
            "num_classes": 1,
            "input_shape": (1, 28, 28),
            "dataset_config": {"data_mode": "surrogate", "surrogate_name": "ImageNet"},
            "substitute_config": {"arch": "lenet_mnist", "optimizer": {"name": "sgd", "lr": 0.01}},
            "victim_config": {"output_mode": output_mode, "input_size": [28, 28], "channels": 1, "num_classes": 1},
            "max_budget": 64,
        },
    )


def test_copycatcnn_single_logit_soft_prob_maps_to_binary_labels() -> None:
    state = _state(output_mode="hard_top1")
    attack = CopycatCNN({"batch_size": 2}, state)
    x = torch.rand(2, 1, 28, 28)
    y = torch.tensor([[0.2], [0.8]], dtype=torch.float32)

    attack._handle_oracle_output(x, OracleOutput(kind="soft_prob", y=y), state)

    assert torch.equal(state.attack_state["query_data_y"][-1], torch.tensor([0, 1], dtype=torch.long))


def test_dfme_single_logit_recovers_log_odds() -> None:
    state = _state(output_mode="soft_prob")
    attack = DFME({"output_mode": "soft_prob", "batch_size": 2, "noise_dim": 8}, state)
    logits = attack._recover_logits(torch.tensor([[0.2], [0.8]], dtype=torch.float32))

    assert logits.shape == (2, 1)
    assert float(logits[0, 0]) < 0.0 < float(logits[1, 0])


def test_ds_single_logit_formats_teacher_targets() -> None:
    state = _state(output_mode="soft_prob")
    attack = DualStudents({"output_mode": "soft_prob", "batch_size": 2, "noise_dim": 8}, state)

    attack.loss_mode = "kl"
    kl_target = attack._format_teacher_target(torch.tensor([[0.3], [0.7]], dtype=torch.float32))
    assert kl_target.shape == (2, 1)

    attack.loss_mode = "l1"
    l1_target = attack._format_teacher_target(torch.tensor([[0.3], [0.7]], dtype=torch.float32))
    assert l1_target.shape == (2, 1)
    assert float(l1_target[0, 0]) < 0.0 < float(l1_target[1, 0])


def test_maze_single_logit_clone_probs_eval_uses_sigmoid() -> None:
    state = _state(output_mode="soft_prob")
    attack = MAZE({"output_mode": "soft_prob", "batch_size": 2, "noise_dim": 8}, state)
    attack.clone = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(28 * 28, 1))
    with torch.no_grad():
        attack.clone[1].weight.zero_()
        attack.clone[1].bias.fill_(2.0)

    probs = attack._clone_probs_eval(torch.zeros(2, 1, 28, 28))
    assert probs.shape == (2, 1)
    assert torch.all(probs > 0.5)


def test_es_attack_single_logit_soft_prob_targets_preserved() -> None:
    state = _state(output_mode="soft_prob")
    attack = ESAttack({"output_mode": "soft_prob", "batch_size": 2, "noise_dim": 8, "student_epochs": 1}, state)
    attack._init_models(state)
    attack._train_student = lambda x, y: None
    attack._train_generator = lambda z, y, d: None
    x = torch.rand(2, 1, 28, 28)
    meta = {"mode": "dnn_syn", "z": torch.randn(2, 8)}
    out = OracleOutput(kind="soft_prob", y=torch.tensor([[0.2], [0.8]], dtype=torch.float32))

    attack._handle_oracle_output(x, meta, out, state)

    assert attack.replay_buffer_y[-1].shape == (2, 1)


def test_disguide_single_logit_target_format_modes() -> None:
    state = _state(output_mode="soft_prob")
    attack = DisGUIDE({"output_mode": "soft_prob", "batch_size": 2, "noise_dim": 8}, state)

    attack.loss_mode = "hl"
    assert torch.equal(attack._format_teacher_target(torch.tensor([[0.2], [0.8]])).cpu(), torch.tensor([0, 1]))

    attack.loss_mode = "kl"
    kl_target = attack._format_teacher_target(torch.tensor([[0.2], [0.8]]))
    assert kl_target.shape == (2, 1)

    attack.loss_mode = "l1"
    l1_target = attack._format_teacher_target(torch.tensor([[0.2], [0.8]]))
    assert l1_target.shape == (2, 1)


def test_dfms_single_logit_soft_prob_thresholds_to_hard_labels() -> None:
    state = _state(output_mode="hard_top1")
    attack = DFMSHL({"output_mode": "hard_top1", "batch_size": 2, "noise_dim": 8}, state)
    attack.generator = torch.nn.Identity()
    attack.discriminator = torch.nn.Identity()
    attack.clone = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(28 * 28, 1))
    attack.clone_optimizer = torch.optim.SGD(attack.clone.parameters(), lr=0.01)
    state.attack_state["phase"] = "alternate"
    x = torch.rand(2, 1, 28, 28)
    out = OracleOutput(kind="soft_prob", y=torch.tensor([[0.2], [0.8]], dtype=torch.float32))
    attack._train_generator = lambda x: None
    attack._train_discriminator = lambda real_x, fake_x: None
    attack._next_proxy_batch = lambda device: torch.rand_like(x)
    attack._handle_oracle_output(x, out, state)
    assert state.attack_state["substitute"] is attack.clone


def test_inversenet_single_logit_hard_observation_uses_binary_distribution() -> None:
    state = _state(output_mode="hard_top1")
    attack = InverseNet({"output_mode": "hard_top1"}, state)
    qb_x = torch.rand(2, 1, 28, 28)
    attack._handle_oracle_output(qb_x, {"phase": 2}, OracleOutput(kind="hard_top1", y=torch.tensor([0, 1])), state)
    stored = state.attack_state["inversion_y"][-1]
    assert stored.shape == (2, 2)


def test_game_single_logit_hard_labels_become_binary_distribution() -> None:
    state = _state(output_mode="soft_prob")
    attack = GAME({"output_mode": "soft_prob", "batch_size": 2, "noise_dim": 8}, state)
    attack.generator = torch.nn.Identity()
    attack.discriminator = torch.nn.Identity()
    attack.student = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(28 * 28, 1))
    attack._train_student_from_buffer = lambda epochs: None
    attack._agu_phase = lambda victim_probs, device, z: None
    x = torch.rand(2, 1, 28, 28)
    attack._handle_oracle_output(x, {"z": torch.randn(2, 8)}, OracleOutput(kind="hard_top1", y=torch.tensor([0, 1])), state)
    stored = state.attack_state["query_data_y"][-1]
    assert stored.shape == (2, 2)


def test_blackbox_ripper_single_logit_uses_binary_semantic_classes() -> None:
    state = _state(output_mode="soft_prob")
    attack = BlackboxRipper({"output_mode": "soft_prob", "generator_checkpoint": "dummy"}, state)
    assert attack.semantic_num_classes == 2


def test_swiftthief_single_logit_f1_uses_binary_thresholds() -> None:
    state = _state(output_mode="soft_prob")
    attack = SwiftThief({"output_mode": "soft_prob", "batch_size": 2}, state)
    attack.normalize = lambda x: x
    attack._ensure_normalizers = lambda state, device: None

    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(28 * 28, 1))
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].bias.fill_(1.0)

    x = torch.zeros(2, 1, 28, 28)
    y = torch.tensor([[1.0], [1.0]])
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=2)

    f1 = attack._compute_f1(model, loader, torch.device("cpu"), state)
    assert float(f1) == 1.0
