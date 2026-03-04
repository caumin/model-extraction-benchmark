from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image

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
