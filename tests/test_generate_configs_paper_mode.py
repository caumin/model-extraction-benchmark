from pathlib import Path

import yaml

from generate_configs import generate_configs, generate_paperlike_configs


def _generate_matrix(out_dir: Path) -> int:
    return generate_configs(
        out_dir=out_dir,
        device="cpu",
        seeds=[0],
        pool_budget=30_000,
        synthetic_budget=30_000_000,
        include_both_hard=False,
        clean=True,
        pool_num_workers=0,
        substitute_num_workers=0,
        substitute_train_num_workers=0,
        substitute_val_num_workers=0,
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
    )


def test_generate_configs_emits_only_matrix_variants(tmp_path: Path) -> None:
    out_dir = tmp_path / "cfg"
    count = _generate_matrix(out_dir)

    assert count == 26
    assert not (out_dir / "SET-A1_maze_paper_soft_30m_seed0.yaml").exists()
    assert not (out_dir / "SET-B1_maze_paper_soft_30m_seed0.yaml").exists()
    assert not (out_dir / "SET-A1_dfms_paper_hard_30m_seed0.yaml").exists()
    assert not (out_dir / "SET-B1_dfms_paper_c100_40c_resnet18_hard_8m_seed0.yaml").exists()
    assert (out_dir / "SET-B1_maze_soft_30m_seed0.yaml").exists()


def test_generate_paperlike_configs_emits_only_paper_variants(tmp_path: Path) -> None:
    out_dir = tmp_path / "cfg"
    count = _generate_paperlike(out_dir)

    assert count == 5

    inversenet_cfg_path = out_dir / "SET-A1_inversenet_paper_hard_30k_seed0.yaml"
    maze_cfg_path = out_dir / "SET-B1_maze_paper_soft_30m_seed0.yaml"
    dfms_cfg_path = out_dir / "SET-B1_dfms_paper_c100_40c_resnet18_hard_8m_seed0.yaml"
    assert inversenet_cfg_path.exists()
    assert maze_cfg_path.exists()
    assert dfms_cfg_path.exists()
    assert (out_dir / "SET-B1_dfms_paper_c100_10c_resnet18_hard_8m_seed0.yaml").exists()
    assert (out_dir / "SET-B1_dfms_paper_c100_40c_resnet34_hard_8m_seed0.yaml").exists()
    assert not (out_dir / "SET-B1_maze_soft_30m_seed0.yaml").exists()

    inversenet_cfg = yaml.safe_load(inversenet_cfg_path.read_text(encoding="utf-8"))
    maze_cfg = yaml.safe_load(maze_cfg_path.read_text(encoding="utf-8"))
    dfms_cfg = yaml.safe_load(dfms_cfg_path.read_text(encoding="utf-8"))

    assert inversenet_cfg["victim"]["arch"] == "classifier"
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

    assert int(dfms_cfg["attack"]["max_budget"]) == 8_000_000
    assert dfms_cfg["attack"]["use_official_stages"] is True
    assert int(dfms_cfg["attack"]["batch_size"]) == 64
    assert int(dfms_cfg["attack"]["alternate_epochs"]) == 800
    assert dfms_cfg["attack"]["proxy_dataset"]["surrogate_name"] == "CIFAR100"
    assert int(dfms_cfg["attack"]["proxy_dataset"]["surrogate_class_subset_size"]) == 40
