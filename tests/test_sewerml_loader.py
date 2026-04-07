from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
import pytest

from mebench.data.loaders import SEWERML_LABELS, create_dataloader, get_test_dataloader


def _write_dummy_sewerml(tmp_path: Path) -> tuple[Path, Path]:
    ann_root = tmp_path / "annotations"
    data_root = tmp_path / "Data"
    ann_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    image_names = ["img_a.jpg", "img_b.jpg"]
    for idx, name in enumerate(image_names):
        img = Image.new("RGB", (320, 240), color=(20 + idx * 30, 40, 60))
        img.save(data_root / name)

    rows = []
    first = {k: 0 for k in SEWERML_LABELS}
    first["DE"] = 1
    first["Filename"] = image_names[0]
    first["Defect"] = 1
    rows.append(first)

    second = {k: 0 for k in SEWERML_LABELS}
    second["OB"] = 1
    second["PH"] = 1
    second["Filename"] = image_names[1]
    second["Defect"] = 1
    rows.append(second)

    df = pd.DataFrame(rows)
    df.to_csv(ann_root / "Train13.csv", index=False)
    df.to_csv(ann_root / "Test13.csv", index=False)
    df.to_csv(ann_root / "Valid13.csv", index=False)
    return ann_root, data_root


def _move_images_into_split_dir(data_root: Path, split_dir: str) -> None:
    split_root = data_root / split_dir
    split_root.mkdir(parents=True, exist_ok=True)
    for image_path in list(data_root.glob("*.jpg")):
        image_path.rename(split_root / image_path.name)


def _write_dummy_sewerml_many(tmp_path: Path, image_count: int = 6) -> tuple[Path, Path]:
    ann_root = tmp_path / "annotations_many"
    data_root = tmp_path / "Data_many"
    ann_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx in range(int(image_count)):
        name = f"img_{idx}.jpg"
        img = Image.new("RGB", (320, 240), color=(20 + idx * 10, 40, 60))
        img.save(data_root / name)

        row = {k: 0 for k in SEWERML_LABELS}
        row[SEWERML_LABELS[idx % len(SEWERML_LABELS)]] = 1
        row["Filename"] = name
        row["Defect"] = 1
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(ann_root / "Train13.csv", index=False)
    df.to_csv(ann_root / "Test13.csv", index=False)
    df.to_csv(ann_root / "Valid13.csv", index=False)
    return ann_root, data_root


def test_get_test_dataloader_sewerml(monkeypatch, tmp_path: Path) -> None:
    ann_root, data_root = _write_dummy_sewerml(tmp_path)
    monkeypatch.setenv("SEWERML_ANN_ROOT", str(ann_root))
    monkeypatch.setenv("SEWERML_DATA_ROOT", str(data_root))

    loader = get_test_dataloader(name="SewerML", batch_size=2, num_workers=0, input_size=(224, 224))
    x, y = next(iter(loader))

    assert tuple(x.shape) == (2, 3, 224, 224)
    assert y.dtype == torch.int64
    assert set(y.tolist()).issubset(set(range(len(SEWERML_LABELS))))
    assert y.tolist() == [3, 1]


def test_get_test_dataloader_sewerml_uses_valid_split_by_default(monkeypatch, tmp_path: Path) -> None:
    ann_root, data_root = _write_dummy_sewerml(tmp_path)
    valid_df = pd.read_csv(ann_root / "Valid13.csv")
    valid_df["Defect"] = [0, 1]
    valid_df.loc[0, SEWERML_LABELS] = 0
    valid_df.loc[0, "RB"] = 1
    valid_df.loc[1, SEWERML_LABELS] = 0
    valid_df.loc[1, "DE"] = 1
    valid_df.to_csv(ann_root / "Valid13.csv", index=False)

    monkeypatch.setenv("SEWERML_ANN_ROOT", str(ann_root))
    monkeypatch.setenv("SEWERML_DATA_ROOT", str(data_root))

    loader = get_test_dataloader(name="SewerML", batch_size=2, num_workers=0, input_size=(224, 224))
    _, y = next(iter(loader))

    assert y.tolist() == [0, 3]


def test_get_test_dataloader_sewerml_accepts_official_val_csv_name(monkeypatch, tmp_path: Path) -> None:
    ann_root, data_root = _write_dummy_sewerml(tmp_path)
    valid_df = pd.read_csv(ann_root / "Valid13.csv")
    valid_df["Defect"] = [0, 1]
    valid_df.loc[0, SEWERML_LABELS] = 0
    valid_df.loc[0, "RB"] = 1
    valid_df.loc[1, SEWERML_LABELS] = 0
    valid_df.loc[1, "DE"] = 1
    (ann_root / "Valid13.csv").unlink()
    valid_df.to_csv(ann_root / "SewerML_Val.csv", index=False)

    monkeypatch.setenv("SEWERML_ANN_ROOT", str(ann_root))
    monkeypatch.setenv("SEWERML_DATA_ROOT", str(data_root))

    loader = get_test_dataloader(name="SewerML", batch_size=2, num_workers=0, input_size=(224, 224))
    _, y = next(iter(loader))

    assert y.tolist() == [0, 3]


def test_get_test_dataloader_sewerml_reads_images_from_valid_subdirectory(monkeypatch, tmp_path: Path) -> None:
    ann_root, data_root = _write_dummy_sewerml(tmp_path)
    _move_images_into_split_dir(data_root, "valid")

    monkeypatch.setenv("SEWERML_ANN_ROOT", str(ann_root))
    monkeypatch.setenv("SEWERML_DATA_ROOT", str(data_root))

    loader = get_test_dataloader(name="SewerML", batch_size=2, num_workers=0, input_size=(224, 224))
    x, y = next(iter(loader))

    assert tuple(x.shape) == (2, 3, 224, 224)
    assert y.tolist() == [3, 1]


def test_get_test_dataloader_sewerml_skips_missing_images(monkeypatch, tmp_path: Path) -> None:
    ann_root, data_root = _write_dummy_sewerml(tmp_path)
    missing_df = pd.read_csv(ann_root / "Valid13.csv")
    missing_df.loc[len(missing_df)] = {**{k: 0 for k in SEWERML_LABELS}, "Filename": "missing.png", "Defect": 0}
    missing_df.loc[len(missing_df) - 1, "RB"] = 1
    missing_df.to_csv(ann_root / "Valid13.csv", index=False)

    monkeypatch.setenv("SEWERML_ANN_ROOT", str(ann_root))
    monkeypatch.setenv("SEWERML_DATA_ROOT", str(data_root))

    loader = get_test_dataloader(name="SewerML", batch_size=4, num_workers=0, input_size=(224, 224))
    x, y = next(iter(loader))

    assert len(loader.dataset) == 2
    assert tuple(x.shape) == (2, 3, 224, 224)
    assert y.tolist() == [3, 1]


def test_get_test_dataloader_sewerml_binary_mode(monkeypatch, tmp_path: Path) -> None:
    ann_root, data_root = _write_dummy_sewerml(tmp_path)
    monkeypatch.setenv("SEWERML_ANN_ROOT", str(ann_root))
    monkeypatch.setenv("SEWERML_DATA_ROOT", str(data_root))

    loader = get_test_dataloader(
        name="SewerML",
        batch_size=2,
        num_workers=0,
        input_size=(224, 224),
        sewerml_label_mode="binary",
    )
    x, y = next(iter(loader))

    assert tuple(x.shape) == (2, 3, 224, 224)
    assert y.dtype == torch.int64
    assert y.tolist() == [1, 1]
    assert set(y.tolist()).issubset({0, 1})


def test_get_test_dataloader_sewerml_explicit_roots(tmp_path: Path, monkeypatch) -> None:
    ann_root, data_root = _write_dummy_sewerml(tmp_path)
    monkeypatch.delenv("SEWERML_ANN_ROOT", raising=False)
    monkeypatch.delenv("SEWERML_DATA_ROOT", raising=False)

    loader = get_test_dataloader(
        name="SewerML",
        batch_size=2,
        num_workers=0,
        input_size=(224, 224),
        sewerml_label_mode="binary",
        sewerml_ann_root=str(ann_root),
        sewerml_data_root=str(data_root),
    )
    x, y = next(iter(loader))

    assert tuple(x.shape) == (2, 3, 224, 224)
    assert y.tolist() == [1, 1]


def test_get_test_dataloader_sewerml_random_subsample_is_deterministic(tmp_path: Path, monkeypatch) -> None:
    ann_root, data_root = _write_dummy_sewerml_many(tmp_path, image_count=6)
    monkeypatch.delenv("SEWERML_ANN_ROOT", raising=False)
    monkeypatch.delenv("SEWERML_DATA_ROOT", raising=False)

    loader_a = get_test_dataloader(
        name="SewerML",
        batch_size=3,
        num_workers=0,
        input_size=(224, 224),
        sewerml_ann_root=str(ann_root),
        sewerml_data_root=str(data_root),
        sewerml_max_samples=3,
        sewerml_subset_seed=7,
    )
    loader_b = get_test_dataloader(
        name="SewerML",
        batch_size=3,
        num_workers=0,
        input_size=(224, 224),
        sewerml_ann_root=str(ann_root),
        sewerml_data_root=str(data_root),
        sewerml_max_samples=3,
        sewerml_subset_seed=7,
    )
    loader_c = get_test_dataloader(
        name="SewerML",
        batch_size=3,
        num_workers=0,
        input_size=(224, 224),
        sewerml_ann_root=str(ann_root),
        sewerml_data_root=str(data_root),
        sewerml_max_samples=3,
        sewerml_subset_seed=11,
    )

    assert len(loader_a.dataset) == 3
    assert loader_a.dataset.img_paths == loader_b.dataset.img_paths
    assert loader_a.dataset.targets == loader_b.dataset.targets
    assert loader_a.dataset.img_paths != loader_c.dataset.img_paths


def test_get_test_dataloader_sewerml_random_subsample_stays_deterministic_after_missing_filter(
    tmp_path: Path, monkeypatch
) -> None:
    ann_root, data_root = _write_dummy_sewerml_many(tmp_path, image_count=6)
    (data_root / "img_2.jpg").unlink()
    monkeypatch.delenv("SEWERML_ANN_ROOT", raising=False)
    monkeypatch.delenv("SEWERML_DATA_ROOT", raising=False)

    loader_a = get_test_dataloader(
        name="SewerML",
        batch_size=3,
        num_workers=0,
        input_size=(224, 224),
        sewerml_ann_root=str(ann_root),
        sewerml_data_root=str(data_root),
        sewerml_max_samples=3,
        sewerml_subset_seed=7,
    )
    loader_b = get_test_dataloader(
        name="SewerML",
        batch_size=3,
        num_workers=0,
        input_size=(224, 224),
        sewerml_ann_root=str(ann_root),
        sewerml_data_root=str(data_root),
        sewerml_max_samples=3,
        sewerml_subset_seed=7,
    )

    assert len(loader_a.dataset) == 3
    assert loader_a.dataset.img_paths == loader_b.dataset.img_paths
    assert loader_a.dataset.targets == loader_b.dataset.targets
    assert "img_2.jpg" not in loader_a.dataset.img_paths


def test_get_test_dataloader_sewerml_binary_mode_derives_target_without_defect_column(
    monkeypatch, tmp_path: Path
) -> None:
    ann_root, data_root = _write_dummy_sewerml(tmp_path)
    ann_df = pd.read_csv(ann_root / "Test13.csv")
    ann_df = ann_df.drop(columns=["Defect"])
    ann_df.to_csv(ann_root / "Test13.csv", index=False)
    ann_df = pd.read_csv(ann_root / "Train13.csv")
    ann_df = ann_df.drop(columns=["Defect"])
    ann_df.to_csv(ann_root / "Train13.csv", index=False)

    monkeypatch.setenv("SEWERML_ANN_ROOT", str(ann_root))
    monkeypatch.setenv("SEWERML_DATA_ROOT", str(data_root))

    loader = get_test_dataloader(
        name="SewerML",
        batch_size=2,
        num_workers=0,
        input_size=(224, 224),
        sewerml_label_mode="binary",
    )
    _, y = next(iter(loader))

    assert y.tolist() == [1, 1]


def test_get_test_dataloader_sewerml_invalid_binary_annotation(monkeypatch, tmp_path: Path) -> None:
    ann_root, data_root = _write_dummy_sewerml(tmp_path)
    ann_df = pd.read_csv(ann_root / "Valid13.csv")
    ann_df = ann_df.drop(columns=["Defect", *SEWERML_LABELS])
    ann_df.to_csv(ann_root / "Valid13.csv", index=False)

    monkeypatch.setenv("SEWERML_ANN_ROOT", str(ann_root))
    monkeypatch.setenv("SEWERML_DATA_ROOT", str(data_root))

    with pytest.raises(ValueError, match="requires Defect column or all defect label columns"):
        get_test_dataloader(
            name="SewerML",
            batch_size=2,
            num_workers=0,
            input_size=(224, 224),
            sewerml_label_mode="binary",
        )


def test_create_dataloader_seed_mode_sewerml(monkeypatch, tmp_path: Path) -> None:
    ann_root, data_root = _write_dummy_sewerml(tmp_path)
    monkeypatch.setenv("SEWERML_ANN_ROOT", str(ann_root))
    monkeypatch.setenv("SEWERML_DATA_ROOT", str(data_root))

    cfg = {
        "name": "SewerML",
        "data_mode": "seed",
        "seed_size": 2,
        "seed_split": "balanced",
        "train_split": False,
        "num_workers": 0,
        "channels": 3,
        "input_size": [224, 224],
        "sewerml_ann_root": str(ann_root),
        "sewerml_data_root": str(data_root),
        "sewerml_split": "Test",
    }
    loader = create_dataloader(cfg, batch_size=2, shuffle=False)
    x, y = next(iter(loader))

    assert tuple(x.shape) == (2, 3, 224, 224)
    assert y.dtype == torch.int64


def test_create_dataloader_seed_mode_sewerml_binary(monkeypatch, tmp_path: Path) -> None:
    ann_root, data_root = _write_dummy_sewerml(tmp_path)
    ann_df = pd.read_csv(ann_root / "Train13.csv")
    ann_df.loc[1, "Defect"] = 0
    ann_df.to_csv(ann_root / "Train13.csv", index=False)
    ann_df.to_csv(ann_root / "Test13.csv", index=False)

    monkeypatch.setenv("SEWERML_ANN_ROOT", str(ann_root))
    monkeypatch.setenv("SEWERML_DATA_ROOT", str(data_root))

    cfg = {
        "name": "SewerML",
        "data_mode": "seed",
        "seed_size": 2,
        "seed_split": "balanced",
        "train_split": False,
        "num_workers": 0,
        "channels": 3,
        "input_size": [224, 224],
        "sewerml_ann_root": str(ann_root),
        "sewerml_data_root": str(data_root),
        "sewerml_split": "Test",
        "sewerml_label_mode": "binary",
    }
    loader = create_dataloader(cfg, batch_size=2, shuffle=False)
    x, y = next(iter(loader))

    assert tuple(x.shape) == (2, 3, 224, 224)
    assert y.dtype == torch.int64
    assert set(y.tolist()).issubset({0, 1})


def test_get_test_dataloader_sewerml_allows_explicit_test_split(monkeypatch, tmp_path: Path) -> None:
    ann_root, data_root = _write_dummy_sewerml(tmp_path)
    test_df = pd.read_csv(ann_root / "Test13.csv")
    test_df["Defect"] = [0, 1]
    test_df.loc[0, SEWERML_LABELS] = 0
    test_df.loc[0, "AF"] = 1
    test_df.to_csv(ann_root / "Test13.csv", index=False)

    monkeypatch.setenv("SEWERML_ANN_ROOT", str(ann_root))
    monkeypatch.setenv("SEWERML_DATA_ROOT", str(data_root))

    loader = get_test_dataloader(
        name="SewerML",
        batch_size=2,
        num_workers=0,
        input_size=(224, 224),
        sewerml_eval_split="Test",
    )
    _, y = next(iter(loader))

    assert y.tolist() == [8, 1]
