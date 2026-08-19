#!/usr/bin/env python3
"""Generate SET-C1 DFME failure-mode exploration configs (seed 0, 5M budget).

8 variants sweeping student LR/optimizer, generator LR, generator/student
step ratio, and substitute architecture. All start from the seed-0 SGD
baseline and only patch the fields that differ. The reduced budget (5M vs
20M) keeps per-run wall clock around 2.5h so we can cycle through the whole
matrix in ~10h with 2-3 parallel jobs.
"""
from __future__ import annotations
import copy
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / "configs/matrix"
BASE = CFG / "SET-C1_dfme_soft_20m_seed0.yaml"

BUDGET = 5_000_000
CHECKPOINTS = [500_000, 1_000_000, 2_000_000, 3_000_000, 5_000_000]

# (tag, patch_fn)
VARIANTS = []


def _patch(base, run_name, *, sub=None, attack=None):
    cfg = copy.deepcopy(base)
    cfg["run"]["name"] = run_name
    cfg["budget"]["max_budget"] = BUDGET
    cfg["budget"]["checkpoints"] = list(CHECKPOINTS)
    if sub:
        for k, v in sub.items():
            cfg["substitute"][k] = v
    if attack:
        for k, v in attack.items():
            cfg["attack"][k] = v
    return cfg


def build():
    with open(BASE) as f:
        base = yaml.safe_load(f)

    outs = {}

    # 1) baseline (SGD 0.05, lr_G 1e-4, n_g=1, n_s=5) at 5M
    outs["SET-C1_dfme_explore_baseline_5m_seed0"] = _patch(
        base, "SET-C1_dfme_explore_baseline_5m_seed0",
    )

    # 2) SGD lr=0.01 (smaller student step)
    outs["SET-C1_dfme_explore_sgd_lr1e-2_5m_seed0"] = _patch(
        base, "SET-C1_dfme_explore_sgd_lr1e-2_5m_seed0",
        sub={"optimizer": {"name": "sgd", "lr": 0.01,
                           "momentum": 0.9, "weight_decay": 5e-4}},
    )

    # 3) SGD lr=0.001 (much smaller)
    outs["SET-C1_dfme_explore_sgd_lr1e-3_5m_seed0"] = _patch(
        base, "SET-C1_dfme_explore_sgd_lr1e-3_5m_seed0",
        sub={"optimizer": {"name": "sgd", "lr": 0.001,
                           "momentum": 0.9, "weight_decay": 5e-4}},
    )

    # 4) AdamW lr=1e-3 (larger than any tried before)
    outs["SET-C1_dfme_explore_adamw_lr1e-3_5m_seed0"] = _patch(
        base, "SET-C1_dfme_explore_adamw_lr1e-3_5m_seed0",
        sub={"optimizer": {"name": "adamw", "lr": 1e-3,
                           "weight_decay": 0.01, "betas": [0.9, 0.999]},
             "scheduler": {"name": "cosine", "warmup_epochs": 1}},
    )

    # 5) generator lr up (1e-3, 10x default)
    outs["SET-C1_dfme_explore_lrG_1e-3_5m_seed0"] = _patch(
        base, "SET-C1_dfme_explore_lrG_1e-3_5m_seed0",
        attack={"generator_lr": 1e-3},
    )

    # 6) generator lr down (1e-5)
    outs["SET-C1_dfme_explore_lrG_1e-5_5m_seed0"] = _patch(
        base, "SET-C1_dfme_explore_lrG_1e-5_5m_seed0",
        attack={"generator_lr": 1e-5},
    )

    # 7) n_g=5 (equal weight to generator vs student updates)
    outs["SET-C1_dfme_explore_ng5_5m_seed0"] = _patch(
        base, "SET-C1_dfme_explore_ng5_5m_seed0",
        attack={"n_g_steps": 5, "n_s_steps": 5},
    )

    # 8) substitute swap: resnet18 (isolate xie2019 arch)
    outs["SET-C1_dfme_explore_resnet18_5m_seed0"] = _patch(
        base, "SET-C1_dfme_explore_resnet18_5m_seed0",
        sub={"arch": "resnet18"},
    )

    return outs


def main():
    outs = build()
    for name, cfg in outs.items():
        dst = CFG / f"{name}.yaml"
        with open(dst, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"  + {dst.name}")
    print(f"\nwrote {len(outs)} configs")


if __name__ == "__main__":
    main()
