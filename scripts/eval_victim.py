"""Evaluate a victim checkpoint and export metrics."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from mebench.core.seed import set_seed
from mebench.data.loaders import get_test_dataloader
from mebench.eval.metrics import compute_accuracy
from mebench.oracles.victim_loader import load_victim_checkpoint


def _cfg_get(config: dict[str, Any], *path: str, default: Any = None) -> Any:
    cur: Any = config
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _resolve_device(user_device: str | None, cfg: dict[str, Any]) -> str:
    if user_device is not None:
        return str(user_device)
    cfg_device = _cfg_get(cfg, "device", default=None)
    if cfg_device is not None:
        return str(cfg_device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a victim checkpoint")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("YAML root must be a mapping")

    seed = int(_cfg_get(cfg, "seed", default=0))
    set_seed(seed)

    victim_cfg = _cfg_get(cfg, "victim", default={})
    dataset_cfg = _cfg_get(cfg, "dataset", default={})

    arch = str(_cfg_get(victim_cfg, "arch", default=_cfg_get(cfg, "arch", default="resnet18")))
    checkpoint = str(
        _cfg_get(victim_cfg, "checkpoint_ref", default=_cfg_get(cfg, "checkpoint", default=""))
    )
    if checkpoint == "":
        raise ValueError("checkpoint path is required (victim.checkpoint_ref or checkpoint)")

    num_classes = int(_cfg_get(victim_cfg, "num_classes", default=_cfg_get(cfg, "num_classes", default=10)))
    channels = int(_cfg_get(victim_cfg, "channels", default=_cfg_get(cfg, "channels", default=3)))
    width_mult = int(_cfg_get(victim_cfg, "width_mult", default=_cfg_get(cfg, "width_mult", default=1)))
    dropout_prob = float(
        _cfg_get(victim_cfg, "dropout_prob", default=_cfg_get(cfg, "dropout_prob", default=0.0))
    )
    input_scale_mode = str(
        _cfg_get(victim_cfg, "input_scale_mode", default=_cfg_get(cfg, "input_scale_mode", default="unit"))
    ).lower()
    dataset_name = str(_cfg_get(dataset_cfg, "name", default=_cfg_get(cfg, "dataset", default="CIFAR10")))
    batch_size = int(_cfg_get(cfg, "batch_size", default=128))
    num_workers = int(_cfg_get(cfg, "num_workers", default=0))

    input_size_raw = _cfg_get(victim_cfg, "input_size", default=_cfg_get(cfg, "input_size", default=None))
    input_size = None
    if isinstance(input_size_raw, (list, tuple)) and len(input_size_raw) == 2:
        input_size = (int(input_size_raw[0]), int(input_size_raw[1]))

    device = _resolve_device(args.device, cfg)
    model = load_victim_checkpoint(
        checkpoint_path=checkpoint,
        arch=arch,
        num_classes=num_classes,
        input_channels=channels,
        width_mult=width_mult,
        dropout_prob=dropout_prob,
        input_scale_mode=input_scale_mode,
        device=device,
    )
    test_loader = get_test_dataloader(
        name=dataset_name,
        batch_size=batch_size,
        num_workers=num_workers,
        input_size=input_size,
        channels=channels,
    )
    acc = float(compute_accuracy(model, test_loader, device))

    result = {
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "dataset": dataset_name,
        "arch": arch,
        "checkpoint": checkpoint,
        "num_classes": num_classes,
        "channels": channels,
        "input_scale_mode": input_scale_mode,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "device": device,
        "metric_name": "acc_gt",
        "metric_value": acc,
    }

    output_json = Path(str(_cfg_get(cfg, "output", "json", default="victim_eval.json")))
    output_csv = Path(str(_cfg_get(cfg, "output", "csv", default="victim_eval.csv")))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)

    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
