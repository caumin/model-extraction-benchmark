from pathlib import Path

import pytest
import yaml

from generate_configs import generate_configs, generate_paperlike_configs


def _generate_matrix(out_dir: Path) -> int:
    return generate_configs(
        out_dir=out_dir,
        device="cpu",
        seeds=[0],
        pool_budget=30_000,
        synthetic_budget=30_000_000,
        set_a_pool_budget=30_000,
        set_a_synthetic_budget=30_000_000,
        set_b_pool_budget=30_000,
        set_b_synthetic_budget=30_000_000,
        include_both_hard=False,
        clean=True,
        pool_num_workers=0,
        substitute_num_workers=0,
        substitute_train_num_workers=0,
        substitute_val_num_workers=0,
        imagenet_root="D:/imagenet",
    )


def _generate_paperlike(out_dir: Path) -> int:
    return generate_paperlike_configs(
        out_dir=out_dir,
        device="cpu",
        seeds=[0],
        clean=True,
        pool_num_workers=0,
        substitute_num_workers=0,
        substitute_train_num_workers=0,
        substitute_val_num_workers=0,
        imagenet_root="D:/imagenet",
    )


def test_generate_configs_aligns_lr_per_sample_for_explicit_set_b_resnet18_image(tmp_path: Path) -> None:
    out_dir = tmp_path / "cfg"
    _generate_matrix(out_dir)

    def _load(name: str) -> dict:
        return yaml.safe_load((out_dir / name).read_text(encoding="utf-8"))

    dfme = _load("SET-B1_dfme_soft_30m_seed0.yaml")
    assert int(dfme["attack"]["batch_size"]) == 512
    assert int(dfme["substitute"]["batch_size"]) == 512
    assert dfme["substitute"]["optimizer"]["name"] == "sgd"
    assert float(dfme["substitute"]["optimizer"]["lr"]) == pytest.approx(0.2)
    assert float(dfme["attack"]["student_lr"]) == pytest.approx(0.2)

    disguide = _load("SET-B1_disguide_soft_30m_seed0.yaml")
    assert int(disguide["attack"]["batch_size"]) == 512
    assert int(disguide["substitute"]["batch_size"]) == 512
    assert float(disguide["substitute"]["optimizer"]["lr"]) == pytest.approx(0.06)
    assert float(disguide["attack"]["student_lr"]) == pytest.approx(0.06)

    ds = _load("SET-B1_ds_soft_30m_seed0.yaml")
    assert int(ds["attack"]["batch_size"]) == 512
    assert int(ds["substitute"]["batch_size"]) == 512
    assert float(ds["attack"]["student_lr"]) == pytest.approx(0.6)

    maze = _load("SET-B1_maze_soft_30m_seed0.yaml")
    assert int(maze["attack"]["batch_size"]) == 512
    assert int(maze["substitute"]["batch_size"]) == 512
    assert float(maze["substitute"]["optimizer"]["lr"]) == pytest.approx(0.4)

    knockoff = _load("SET-B1_knockoff_nets_soft_30k_seed0.yaml")
    assert int(knockoff["attack"]["batch_size"]) == 8
    assert int(knockoff["attack"]["train_every"]) == 8
    assert int(knockoff["substitute"]["batch_size"]) == 512
    assert knockoff["substitute"]["optimizer"]["name"] == "sgd"
    assert float(knockoff["substitute"]["optimizer"]["momentum"]) == pytest.approx(0.5)
    assert float(knockoff["substitute"]["optimizer"]["lr"]) == pytest.approx(0.08)
    assert float(knockoff["attack"]["paper_train_lr"]) == pytest.approx(0.08)

    dissector = _load("SET-B1_blackbox_dissector_hard_30k_seed0.yaml")
    assert int(dissector["substitute"]["batch_size"]) == 512
    assert float(dissector["substitute"]["optimizer"]["lr"]) == pytest.approx(0.08)
    assert float(dissector["attack"]["lr"]) == pytest.approx(0.08)

    ripper = _load("SET-B1_blackbox_ripper_soft_30m_seed0.yaml")
    assert int(ripper["attack"]["train_batch_size"]) == 512
    assert int(ripper["substitute"]["batch_size"]) == 512
    assert float(ripper["attack"]["substitute_lr"]) == pytest.approx(0.08)

    swift = _load("SET-B1_swiftthief_soft_30k_seed0.yaml")
    assert int(swift["attack"]["batch_size"]) == 512
    assert int(swift["substitute"]["batch_size"]) == 512
    assert swift["substitute"]["optimizer"]["name"] == "sgd"
    assert float(swift["substitute"]["optimizer"]["lr"]) == pytest.approx(0.0512)
    assert float(swift["attack"]["lr"]) == pytest.approx(0.0512)
    assert float(swift["attack"]["kd_lr"]) == pytest.approx(0.0512)

    # Not explicitly aligned: keep previous defaults.
    activethief = _load("SET-B1_activethief_soft_30k_seed0.yaml")
    assert int(activethief["attack"]["batch_size"]) == 150
    assert int(activethief["substitute"]["batch_size"]) == 512
    assert float(activethief["substitute"]["optimizer"]["lr"]) == pytest.approx(0.04)

    # Guard condition: non-resnet18 setup should not be aligned.
    set_a_dfme = _load("SET-A1_dfme_soft_30m_seed0.yaml")
    assert "batch_size" not in set_a_dfme["attack"]
    assert int(set_a_dfme["substitute"]["batch_size"]) == 512
    assert float(set_a_dfme["substitute"]["optimizer"]["lr"]) == pytest.approx(0.04)


def test_generate_configs_emits_only_matrix_variants(tmp_path: Path) -> None:
    out_dir = tmp_path / "cfg"
    count = _generate_matrix(out_dir)

    assert count == 32
    assert not (out_dir / "SET-A1_maze_paper_soft_30m_seed0.yaml").exists()
    assert not (out_dir / "SET-B1_maze_paper_soft_30m_seed0.yaml").exists()
    assert not (out_dir / "SET-A1_dfms_paper_hard_30m_seed0.yaml").exists()
    assert not (out_dir / "SET-B1_dfms_paper_c100_40c_resnet18_hard_8m_seed0.yaml").exists()
    assert (out_dir / "SET-B1_maze_soft_30m_seed0.yaml").exists()
    assert (out_dir / "SET-A1_disguide_soft_30m_seed0.yaml").exists()
    assert (out_dir / "SET-B1_disguide_soft_30m_seed0.yaml").exists()
    assert (out_dir / "SET-A1_ds_soft_30m_seed0.yaml").exists()
    assert (out_dir / "SET-B1_ds_soft_30m_seed0.yaml").exists()
    assert (out_dir / "SET-A1_marich_hard_30k_seed0.yaml").exists()
    assert (out_dir / "SET-B1_marich_hard_30k_seed0.yaml").exists()

    set_b_cfg_path = out_dir / "SET-B1_random_soft_30k_seed0.yaml"
    set_b_cfg = yaml.safe_load(set_b_cfg_path.read_text(encoding="utf-8"))
    assert set_b_cfg["victim"]["inference_policy"] == "benchmark"


def test_generate_paperlike_configs_emits_only_paper_variants(tmp_path: Path) -> None:
    out_dir = tmp_path / "cfg"
    count = _generate_paperlike(out_dir)

    assert count == 10

    inversenet_cfg_path = out_dir / "SET-A1_inversenet_paper_hard_30k_seed0.yaml"
    maze_cfg_path = out_dir / "SET-B1_maze_paper_soft_30m_seed0.yaml"
    disguide_cfg_path = out_dir / "SET-B1_disguide_paper_soft_20m_seed0.yaml"
    ds_cfg_path = out_dir / "SET-B1_ds_paper_soft_20m_seed0.yaml"
    dfms_cfg_path = out_dir / "SET-B1_dfms_paper_c100_40c_resnet18_hard_8m_seed0.yaml"
    blackbox_dissector_cfg_path = out_dir / "SET-B1_blackbox_dissector_paper_hard_30k_seed0.yaml"
    marich_cfg_path = out_dir / "SET-B1_marich_paper_hard_30k_seed0.yaml"
    blackbox_ripper_cfg_path = out_dir / "SET-B1_blackbox_ripper_paper_soft_30k_seed0.yaml"
    assert inversenet_cfg_path.exists()
    assert maze_cfg_path.exists()
    assert disguide_cfg_path.exists()
    assert ds_cfg_path.exists()
    assert dfms_cfg_path.exists()
    assert (out_dir / "SET-B1_dfms_paper_c100_10c_resnet18_hard_8m_seed0.yaml").exists()
    assert (out_dir / "SET-B1_dfms_paper_c100_40c_resnet34_hard_8m_seed0.yaml").exists()
    assert blackbox_dissector_cfg_path.exists()
    assert marich_cfg_path.exists()
    assert blackbox_ripper_cfg_path.exists()
    assert not (out_dir / "SET-B1_maze_soft_30m_seed0.yaml").exists()

    inversenet_cfg = yaml.safe_load(inversenet_cfg_path.read_text(encoding="utf-8"))
    maze_cfg = yaml.safe_load(maze_cfg_path.read_text(encoding="utf-8"))
    disguide_cfg = yaml.safe_load(disguide_cfg_path.read_text(encoding="utf-8"))
    ds_cfg = yaml.safe_load(ds_cfg_path.read_text(encoding="utf-8"))
    dfms_cfg = yaml.safe_load(dfms_cfg_path.read_text(encoding="utf-8"))
    blackbox_dissector_cfg = yaml.safe_load(blackbox_dissector_cfg_path.read_text(encoding="utf-8"))
    marich_cfg = yaml.safe_load(marich_cfg_path.read_text(encoding="utf-8"))
    blackbox_ripper_cfg = yaml.safe_load(blackbox_ripper_cfg_path.read_text(encoding="utf-8"))

    assert inversenet_cfg["victim"]["arch"] == "classifier"
    assert float(inversenet_cfg["victim"]["dropout_prob"]) == 0.2
    assert inversenet_cfg["substitute"]["arch"] == "cnn32"
    assert inversenet_cfg["dataset"]["surrogate_name"] == "EMNIST"
    assert inversenet_cfg["dataset"]["surrogate_split"] == "letters"
    assert inversenet_cfg["attack"]["output_mode"] == "hard_top1"
    assert int(inversenet_cfg["attack"]["max_budget"]) == 30_000
    assert inversenet_cfg["attack"]["phase_ratios"] == [0.45, 0.45, 0.1]
    assert int(inversenet_cfg["attack"]["truncation_k"]) == 1

    assert maze_cfg["victim"]["arch"] == "resnet20"
    assert maze_cfg["substitute"]["arch"] == "wideresnet22"
    assert int(maze_cfg["attack"]["max_budget"]) == 30_000_000
    assert maze_cfg["attack"]["n_g_steps"] == 1
    assert maze_cfg["attack"]["n_c_steps"] == 5
    assert maze_cfg["attack"]["n_r_steps"] == 10
    assert maze_cfg["attack"]["lr_schedule"] == "cosine"
    assert float(maze_cfg["substitute"]["optimizer"]["lr"]) == 0.1

    assert disguide_cfg["victim"]["arch"] == "resnet34"
    assert disguide_cfg["victim"]["victim_id"] == "cifar10_resnet34_8x_official"
    assert int(disguide_cfg["victim"]["width_mult"]) == 1
    assert disguide_cfg["victim"]["checkpoint_ref"] == "runs/victims/cifar10-resnet34_8x.pt"
    assert int(disguide_cfg["attack"]["max_budget"]) == 20_000_000
    assert int(disguide_cfg["attack"]["batch_size"]) == 256
    assert int(disguide_cfg["attack"]["ensemble_size"]) == 2
    assert disguide_cfg["attack"]["replay"] == "Classic"
    assert float(disguide_cfg["attack"]["lambda_div"]) == 0.2
    assert float(disguide_cfg["attack"]["student_lr"]) == 0.03
    assert disguide_cfg["attack"]["strict_iteration_budget"] is True
    assert int(disguide_cfg["substitute"].get("width_mult", 1)) == 1

    assert ds_cfg["victim"]["arch"] == "resnet34"
    assert ds_cfg["victim"]["victim_id"] == "cifar10_resnet34_8x_official"
    assert ds_cfg["victim"]["checkpoint_ref"] == "runs/victims/cifar10-resnet34_8x.pt"
    assert int(ds_cfg["attack"]["max_budget"]) == 20_000_000
    assert int(ds_cfg["attack"]["batch_size"]) == 256
    assert int(ds_cfg["attack"]["num_students"]) == 2
    assert int(ds_cfg["attack"]["g_iter"]) == 1
    assert int(ds_cfg["attack"]["d_iter"]) == 5
    assert ds_cfg["attack"]["loss"] == "l1"
    assert ds_cfg["attack"]["generator_loss"] == "l1"
    assert float(ds_cfg["attack"]["student_lr"]) == 0.3
    assert float(ds_cfg["attack"]["generator_lr"]) == 1e-4

    assert int(dfms_cfg["attack"]["max_budget"]) == 8_000_000
    assert dfms_cfg["attack"]["use_official_stages"] is True
    assert int(dfms_cfg["attack"]["batch_size"]) == 64
    assert int(dfms_cfg["attack"]["alternate_epochs"]) == 800
    assert dfms_cfg["attack"]["proxy_dataset"]["surrogate_name"] == "CIFAR100"
    assert int(dfms_cfg["attack"]["proxy_dataset"]["surrogate_class_subset_size"]) == 40

    assert blackbox_dissector_cfg["attack"]["name"] == "blackbox_dissector"
    assert blackbox_dissector_cfg["attack"]["output_mode"] == "hard_top1"
    assert int(blackbox_dissector_cfg["attack"]["max_budget"]) == 30_000
    assert int(blackbox_dissector_cfg["attack"]["n_variants"]) == 10
    assert float(blackbox_dissector_cfg["attack"]["erase_rate"]) == 0.25
    assert blackbox_dissector_cfg["attack"]["iterative_budgets"] == [1000, 5000, 10000, 20000, 30000]
    assert blackbox_dissector_cfg["dataset"]["surrogate_name"] == "ImageNet"

    assert marich_cfg["attack"]["name"] == "marich"
    assert marich_cfg["attack"]["output_mode"] == "hard_top1"
    assert int(marich_cfg["attack"]["max_budget"]) == 30_000
    assert marich_cfg["attack"]["sampling"] == "all_elg"
    assert int(marich_cfg["attack"]["budget"]) == 300
    assert float(marich_cfg["attack"]["gamma1"]) == 0.8
    assert float(marich_cfg["attack"]["gamma2"]) == 0.8

    assert blackbox_ripper_cfg["attack"]["name"] == "blackbox_ripper"
    assert blackbox_ripper_cfg["attack"]["output_mode"] == "soft_prob"
    assert int(blackbox_ripper_cfg["attack"]["max_budget"]) == 30_000
    assert blackbox_ripper_cfg["attack"]["generator_name"] == "cifar_progan"
    assert blackbox_ripper_cfg["attack"]["generator_checkpoint"] == "checkpoints/blackbox_ripper/official/cifar_100_6_classes_gan"
    assert int(blackbox_ripper_cfg["attack"]["population_size"]) == 30
    assert int(blackbox_ripper_cfg["attack"]["elite_size"]) == 10
    assert int(blackbox_ripper_cfg["attack"]["max_evolve_iters"]) == 10
