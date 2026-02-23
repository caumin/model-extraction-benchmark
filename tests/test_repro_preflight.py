from __future__ import annotations

from pathlib import Path

import yaml

import repro.run_experiment as run_experiment


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _set_root(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(run_experiment, "ROOT", root)
    monkeypatch.setattr(run_experiment, "PAPERS_ROOT", root / "repro" / "papers")


def test_preflight_ripper_passes_with_extension_fallback(monkeypatch, tmp_path: Path) -> None:
    _set_root(monkeypatch, tmp_path)

    paper_id = "2020_barbalau_blackbox_ripper"
    paper_dir = tmp_path / "repro" / "papers" / paper_id

    _write_yaml(
        paper_dir / "configs" / "experiment.yaml",
        {
            "run": {"name": "repro_test", "seeds": [0]},
            "victim": {
                "output_mode": "soft_prob",
                "checkpoint_ref": "runs/victims/cifar10-resnet34_8x.pt",
            },
            "dataset": {"data_mode": "data_free"},
            "attack": {
                "name": "blackbox_ripper",
                "output_mode": "soft_prob",
                "max_budget": 30000,
                "generator_checkpoint": "checkpoints/blackbox_ripper/official/cifar_100_6_classes_gan",
            },
            "budget": {"max_budget": 30000},
        },
    )
    _write_yaml(
        paper_dir / "configs" / "victim_eval.yaml",
        {
            "victim": {
                "checkpoint_ref": "runs/victims/cifar10-resnet34_8x.pt",
            }
        },
    )
    _write_yaml(
        paper_dir / "extracted_spec.yaml",
        {
            "reported_results": {
                "targets": [
                    {
                        "id": "target",
                        "metric_name": "acc_gt",
                        "query_budget": 30000,
                        "paper_value": 0.5,
                    }
                ]
            }
        },
    )

    victim_ckpt = tmp_path / "runs" / "victims" / "cifar10-resnet34_8x.pt"
    victim_ckpt.parent.mkdir(parents=True, exist_ok=True)
    victim_ckpt.write_bytes(b"ok")

    generator_ckpt = (
        tmp_path
        / "checkpoints"
        / "blackbox_ripper"
        / "official"
        / "cifar_100_6_classes_gan.pth"
    )
    generator_ckpt.parent.mkdir(parents=True, exist_ok=True)
    generator_ckpt.write_bytes(b"ok")

    ok, messages = run_experiment._preflight_single(paper_id=paper_id, profile="full")
    assert ok is True
    assert any("[OK] generator checkpoint:" in m for m in messages)
    assert any("official generator stem verified" in m for m in messages)


def test_preflight_ripper_fails_for_non_official_generator_stem(monkeypatch, tmp_path: Path) -> None:
    _set_root(monkeypatch, tmp_path)

    paper_id = "2020_barbalau_blackbox_ripper"
    paper_dir = tmp_path / "repro" / "papers" / paper_id

    _write_yaml(
        paper_dir / "configs" / "experiment.yaml",
        {
            "run": {"name": "repro_test", "seeds": [0]},
            "victim": {
                "output_mode": "soft_prob",
                "checkpoint_ref": "runs/victims/cifar10-resnet34_8x.pt",
            },
            "dataset": {"data_mode": "data_free"},
            "attack": {
                "name": "blackbox_ripper",
                "output_mode": "soft_prob",
                "max_budget": 30000,
                "generator_checkpoint": "checkpoints/blackbox_ripper/trained/cifar10_cifar_progan_seed0",
            },
            "budget": {"max_budget": 30000},
        },
    )
    _write_yaml(
        paper_dir / "configs" / "victim_eval.yaml",
        {
            "victim": {
                "checkpoint_ref": "runs/victims/cifar10-resnet34_8x.pt",
            }
        },
    )
    _write_yaml(
        paper_dir / "extracted_spec.yaml",
        {
            "reported_results": {
                "targets": [
                    {
                        "id": "target",
                        "metric_name": "acc_gt",
                        "query_budget": 30000,
                        "paper_value": 0.5,
                    }
                ]
            }
        },
    )

    victim_ckpt = tmp_path / "runs" / "victims" / "cifar10-resnet34_8x.pt"
    victim_ckpt.parent.mkdir(parents=True, exist_ok=True)
    victim_ckpt.write_bytes(b"ok")

    generator_ckpt = (
        tmp_path
        / "checkpoints"
        / "blackbox_ripper"
        / "trained"
        / "cifar10_cifar_progan_seed0.pth"
    )
    generator_ckpt.parent.mkdir(parents=True, exist_ok=True)
    generator_ckpt.write_bytes(b"ok")

    ok, messages = run_experiment._preflight_single(paper_id=paper_id, profile="full")
    assert ok is False
    assert any("requires official generator checkpoint stem" in m for m in messages)


def test_preflight_marich_fails_when_targets_missing(monkeypatch, tmp_path: Path) -> None:
    _set_root(monkeypatch, tmp_path)

    paper_id = "2023_karmakar_marich"
    paper_dir = tmp_path / "repro" / "papers" / paper_id

    surrogate_root = tmp_path / "imagenet"
    (surrogate_root / "train").mkdir(parents=True, exist_ok=True)

    _write_yaml(
        paper_dir / "configs" / "experiment.yaml",
        {
            "run": {"name": "repro_test", "seeds": [0]},
            "victim": {
                "output_mode": "hard_top1",
                "checkpoint_ref": "runs/victims/cifar10-resnet34_8x.pt",
            },
            "dataset": {
                "surrogate_name": "IMAGENET",
                "surrogate_root": str(surrogate_root),
                "train_split": True,
            },
            "attack": {
                "name": "marich",
                "output_mode": "hard_top1",
                "max_budget": 30000,
            },
            "budget": {"max_budget": 30000},
        },
    )
    _write_yaml(
        paper_dir / "configs" / "victim_eval.yaml",
        {
            "victim": {
                "checkpoint_ref": "runs/victims/cifar10-resnet34_8x.pt",
            }
        },
    )
    _write_yaml(paper_dir / "extracted_spec.yaml", {"reported_results": {"targets": []}})

    victim_ckpt = tmp_path / "runs" / "victims" / "cifar10-resnet34_8x.pt"
    victim_ckpt.parent.mkdir(parents=True, exist_ok=True)
    victim_ckpt.write_bytes(b"ok")

    ok, messages = run_experiment._preflight_single(paper_id=paper_id, profile="full")
    assert ok is False
    assert any("extracted_spec targets missing" in m for m in messages)


def test_preflight_dissector_fails_on_output_mode_mismatch(monkeypatch, tmp_path: Path) -> None:
    _set_root(monkeypatch, tmp_path)

    paper_id = "2021_wang_blackbox_dissector"
    paper_dir = tmp_path / "repro" / "papers" / paper_id

    surrogate_root = tmp_path / "imagenet"
    (surrogate_root / "train").mkdir(parents=True, exist_ok=True)

    _write_yaml(
        paper_dir / "configs" / "experiment.yaml",
        {
            "run": {"name": "repro_test", "seeds": [0]},
            "victim": {
                "output_mode": "soft_prob",
                "checkpoint_ref": "runs/victims/cifar10-resnet34_8x.pt",
            },
            "dataset": {
                "surrogate_name": "ImageNet",
                "surrogate_root": str(surrogate_root),
                "train_split": True,
            },
            "attack": {
                "name": "blackbox_dissector",
                "output_mode": "hard_top1",
                "max_budget": 30000,
            },
            "budget": {"max_budget": 30000},
        },
    )
    _write_yaml(
        paper_dir / "configs" / "victim_eval.yaml",
        {
            "victim": {
                "checkpoint_ref": "runs/victims/cifar10-resnet34_8x.pt",
            }
        },
    )
    _write_yaml(
        paper_dir / "extracted_spec.yaml",
        {"reported_results": {"targets": [{"metric_name": "acc_gt", "query_budget": 30000, "paper_value": 0.5}]}},
    )

    victim_ckpt = tmp_path / "runs" / "victims" / "cifar10-resnet34_8x.pt"
    victim_ckpt.parent.mkdir(parents=True, exist_ok=True)
    victim_ckpt.write_bytes(b"ok")

    ok, messages = run_experiment._preflight_single(paper_id=paper_id, profile="full")
    assert ok is False
    assert any("output mode mismatch" in m for m in messages)


def test_preflight_disguide_passes_with_shared_victim_profile(monkeypatch, tmp_path: Path) -> None:
    _set_root(monkeypatch, tmp_path)

    paper_id = "2023_tan_disguide"
    paper_dir = tmp_path / "repro" / "papers" / paper_id

    _write_yaml(
        paper_dir / "configs" / "experiment.yaml",
        {
            "run": {"name": "repro_test", "seeds": [0]},
            "victim": {
                "output_mode": "soft_prob",
                "checkpoint_ref": "runs/victims/cifar10-resnet34_8x.pt",
            },
            "dataset": {"data_mode": "data_free"},
            "attack": {
                "name": "disguide",
                "output_mode": "soft_prob",
                "max_budget": 20000000,
            },
            "budget": {"max_budget": 20000000},
        },
    )
    _write_yaml(
        paper_dir / "configs" / "victim_eval.yaml",
        {
            "victim": {
                "checkpoint_ref": "runs/victims/cifar10-resnet34_8x.pt",
            }
        },
    )
    _write_yaml(
        paper_dir / "extracted_spec.yaml",
        {
            "reported_results": {
                "targets": [
                    {
                        "metric_name": "acc_gt",
                        "query_budget": 20000000,
                        "paper_value": 0.9402,
                    }
                ]
            }
        },
    )

    victim_ckpt = tmp_path / "runs" / "victims" / "cifar10-resnet34_8x.pt"
    victim_ckpt.parent.mkdir(parents=True, exist_ok=True)
    victim_ckpt.write_bytes(b"ok")

    ok, messages = run_experiment._preflight_single(paper_id=paper_id, profile="full")
    assert ok is True


def test_preflight_ds_passes_with_shared_victim_profile(monkeypatch, tmp_path: Path) -> None:
    _set_root(monkeypatch, tmp_path)

    paper_id = "2023_beetham_dual_students"
    paper_dir = tmp_path / "repro" / "papers" / paper_id

    _write_yaml(
        paper_dir / "configs" / "experiment.yaml",
        {
            "run": {"name": "repro_test", "seeds": [0]},
            "victim": {
                "output_mode": "soft_prob",
                "checkpoint_ref": "runs/victims/cifar10-resnet34_8x.pt",
            },
            "dataset": {"data_mode": "data_free"},
            "attack": {
                "name": "ds",
                "output_mode": "soft_prob",
                "max_budget": 20000000,
            },
            "budget": {"max_budget": 20000000},
        },
    )
    _write_yaml(
        paper_dir / "configs" / "victim_eval.yaml",
        {
            "victim": {
                "checkpoint_ref": "runs/victims/cifar10-resnet34_8x.pt",
            }
        },
    )
    _write_yaml(
        paper_dir / "extracted_spec.yaml",
        {
            "reported_results": {
                "targets": [
                    {
                        "metric_name": "acc_gt",
                        "query_budget": 20000000,
                        "paper_value": 0.9134,
                    }
                ]
            }
        },
    )

    victim_ckpt = tmp_path / "runs" / "victims" / "cifar10-resnet34_8x.pt"
    victim_ckpt.parent.mkdir(parents=True, exist_ok=True)
    victim_ckpt.write_bytes(b"ok")

    ok, messages = run_experiment._preflight_single(paper_id=paper_id, profile="full")
    assert ok is True


def test_targeted_attacks_share_cifar10_resnet34_checkpoint() -> None:
    papers_root = Path(__file__).resolve().parent.parent / "repro" / "papers"

    expected_checkpoint = "runs/victims/cifar10-resnet34_8x.pt"
    expected_victim_id = "cifar10_resnet34_8x_official"

    paper_ids = [
        "2023_karmakar_marich",
        "2021_wang_blackbox_dissector",
        "2020_barbalau_blackbox_ripper",
        "2023_tan_disguide",
        "2023_beetham_dual_students",
    ]

    for paper_id in paper_ids:
        paper_dir = papers_root / paper_id
        victim_train = run_experiment._load_yaml(paper_dir / "configs" / "victim_train.yaml")
        victim_eval = run_experiment._load_yaml(paper_dir / "configs" / "victim_eval.yaml")
        experiment = run_experiment._load_yaml(paper_dir / "configs" / "experiment.yaml")

        assert victim_train["victim_id"] == expected_victim_id
        assert victim_train["out"] == expected_checkpoint

        assert victim_eval["victim"]["checkpoint_ref"] == expected_checkpoint

        assert experiment["victim"]["victim_id"] == expected_victim_id
        assert experiment["victim"]["checkpoint_ref"] == expected_checkpoint

