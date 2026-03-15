import torch
from torch.utils.data import Dataset

from mebench.attackers.blackbox_ripper import BlackboxRipper
from mebench.attackers.blackbox_dissector import BlackboxDissector
from mebench.attackers.cloudleak import CloudLeak
from mebench.attackers.knockoff_nets import KnockoffNets
from mebench.attackers.disguide import DisGUIDE
from mebench.attackers.dfme import DFME
from mebench.attackers.dfms import DFMSHL
from mebench.attackers.ds import DualStudents
from mebench.attackers.es_attack import ESAttack
from mebench.attackers.game import GAME
from mebench.attackers.inversenet import InverseNet
from mebench.attackers.marich import MARICH
from mebench.attackers.maze import MAZE
from mebench.attackers.copycatcnn import CopycatCNN
from mebench.attackers.random_baseline import RandomBaseline
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


class _TinyIndexedPool(Dataset):
    def __init__(self, xs: torch.Tensor) -> None:
        self.xs = xs

    def __len__(self) -> int:
        return int(self.xs.size(0))

    def __getitem__(self, idx: int):
        return self.xs[idx], int(idx % 2)


class _BinaryFlattenSum(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1).sum(dim=1, keepdim=True)


def test_copycatcnn_single_logit_soft_prob_maps_to_binary_labels() -> None:
    state = _state(output_mode="hard_top1")
    attack = CopycatCNN({"batch_size": 2}, state)
    x = torch.rand(2, 1, 28, 28)
    y = torch.tensor([[0.2], [0.8]], dtype=torch.float32)

    attack._handle_oracle_output(x, OracleOutput(kind="soft_prob", y=y), state)

    assert torch.equal(state.attack_state["query_data_y"][-1], torch.tensor([0, 1], dtype=torch.long))


def test_random_baseline_single_logit_soft_prob_checkpoint_training_uses_bce() -> None:
    state = _state(output_mode="soft_prob")
    state.metadata["substitute_config"] = {
        "arch": "lenet_mnist",
        "batch_size": 2,
        "max_epochs": 1,
        "patience": 1,
        "optimizer": {"name": "sgd", "lr": 0.01},
    }
    attack = RandomBaseline({"output_mode": "soft_prob", "batch_size": 2}, state)

    substitute = attack._train_substitute_checkpoint(
        x_all=torch.rand(8, 1, 28, 28),
        y_all=torch.tensor([[0.1], [0.9], [0.2], [0.8], [0.3], [0.7], [0.4], [0.6]], dtype=torch.float32),
        device="cpu",
    )

    assert substitute is not None


def test_random_baseline_single_logit_hard_top1_checkpoint_training_uses_bce() -> None:
    state = _state(output_mode="hard_top1")
    state.metadata["substitute_config"] = {
        "arch": "lenet_mnist",
        "batch_size": 2,
        "max_epochs": 1,
        "patience": 1,
        "optimizer": {"name": "sgd", "lr": 0.01},
    }
    attack = RandomBaseline({"output_mode": "hard_top1", "batch_size": 2}, state)

    substitute = attack._train_substitute_checkpoint(
        x_all=torch.rand(8, 1, 28, 28),
        y_all=torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long),
        device="cpu",
    )

    assert substitute is not None


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


def test_blackbox_ripper_single_logit_target_labels_preserve_positive_class() -> None:
    state = _state(output_mode="soft_prob")
    attack = BlackboxRipper({"output_mode": "soft_prob", "generator_checkpoint": "dummy"}, state)

    labels_from_prob = attack._semantic_hard_labels_from_targets(
        torch.tensor([[0.2], [0.8]], dtype=torch.float32)
    )
    labels_from_dist = attack._semantic_hard_labels_from_targets(
        torch.tensor([[0.8, 0.2], [0.2, 0.8]], dtype=torch.float32)
    )

    assert torch.equal(labels_from_prob.cpu(), torch.tensor([0, 1]))
    assert torch.equal(labels_from_dist.cpu(), torch.tensor([0, 1]))


def test_knockoffnets_single_logit_soft_prob_uses_semantic_binary_rewards() -> None:
    state = _state(output_mode="soft_prob")
    state.metadata["substitute_config"] = {"arch": "lenet_mnist", "optimizer": {"name": "sgd", "lr": 0.01}}
    attack = KnockoffNets(
        {"output_mode": "soft_prob", "batch_size": 2, "train_every": 1000, "policy": "adaptive"},
        state,
    )
    attack.class_to_indices = {0: [0, 1]}
    state.attack_state["online_substitute"] = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(28 * 28, 1))

    x = torch.rand(2, 1, 28, 28)
    y = torch.tensor([[0.2], [0.8]], dtype=torch.float32)

    attack._handle_oracle_output(x, OracleOutput(kind="soft_prob", y=y), [0, 0], state)

    assert state.attack_state["query_count"] == 2
    assert len(state.attack_state["recent_victim_probs"]) == 2
    assert state.attack_state["recent_victim_probs"][0].shape == (2,)
    assert len(state.attack_state["recent_loss_rewards"]) == 2


def test_cloudleak_single_logit_training_loss_uses_bce() -> None:
    state = _state(output_mode="soft_prob")
    attack = CloudLeak({"output_mode": "soft_prob", "batch_size": 2}, state)

    outputs = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    targets = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

    loss = attack._compute_training_loss(outputs, targets)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_blackbox_dissector_single_logit_pseudo_labels_become_binary_distributions() -> None:
    state = _state(output_mode="hard_top1")
    attack = BlackboxDissector.__new__(BlackboxDissector)
    attack.config = {"output_mode": "hard_top1", "batch_size": 2, "n_variants": 2}
    attack.state = state
    attack.logger = None
    attack.n_variants = 2
    attack.selection_batch_size = 2
    attack.sl = 0.02
    attack.sh = 0.4
    attack.r1 = 0.3
    attack.r2 = 3.3
    attack.fill_min = 0.0
    attack.fill_max = 1.0
    attack.is_single_logit_binary = True
    xs = torch.stack(
        [
            torch.zeros(1, 28, 28),
            torch.ones(1, 28, 28),
        ],
        dim=0,
    )
    attack.pool_dataset = _TinyIndexedPool(xs)
    state.attack_state["unlabeled_indices"] = [0, 1]

    attack._generate_pseudo_labels(state, _BinaryFlattenSum())

    pseudo = state.attack_state["pseudo_labels"]
    assert set(pseudo.keys()) == {0, 1}
    assert pseudo[0].shape == (2,)
    assert pseudo[1].shape == (2,)


def test_marich_single_logit_entropy_select_prefers_boundary_samples() -> None:
    state = _state(output_mode="hard_top1")
    attack = MARICH({"output_mode": "hard_top1", "batch_size": 2, "selection_batch_size": 2}, state)
    attack.show_scoring_progress = False
    attack.pool_dataset = _TinyIndexedPool(
        torch.tensor(
            [
                [[[0.0, 0.0]]],
                [[[3.0, 3.0]]],
            ],
            dtype=torch.float32,
        )
    )
    attack.substitute = _BinaryFlattenSum()

    selected = attack._entropy_select([0, 1], budget=1, device="cpu")

    assert selected == [0]


def test_marich_single_logit_loss_dep_select_uses_bce_losses() -> None:
    state = _state(output_mode="hard_top1")
    attack = MARICH({"output_mode": "hard_top1", "batch_size": 2, "selection_batch_size": 2, "num_clusters": 1}, state)
    attack.show_scoring_progress = False
    attack.pool_dataset = _TinyIndexedPool(
        torch.tensor(
            [
                [[[0.0, 0.0]]],
                [[[1.0, 1.0]]],
            ],
            dtype=torch.float32,
        )
    )
    attack.substitute = _BinaryFlattenSum()
    state.attack_state["labeled_indices"] = [0, 1]
    state.attack_state["query_data_x"] = [torch.tensor([[[[0.0, 0.0]]], [[[1.0, 1.0]]]], dtype=torch.float32)]
    state.attack_state["query_data_y"] = [torch.tensor([[0.0], [1.0]], dtype=torch.float32)]

    selected = attack._loss_dep_select([0, 1], budget=1, device="cpu")

    assert len(selected) == 1


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
