#!/usr/bin/env python3
"""Generate phase 3-5 configs for SET-A/B/C 4(5)-column unification rerun.

Phase 3 (SET-B, resnet34 substitute): SGD / SGD+Aug / AdamW / AdamW+Aug per pool attack
Phase 4 (SET-A): existing SGD/SGD+Aug/SGD+Aug-soft + AdamW / AdamW+Aug
Phase 5 (SET-C): existing SGD/SGD+Aug + AdamW / AdamW+Aug

SwiftThief: SGD baseline + AdamW only (no Aug columns; baseline already augmented).

LR placeholders: SET-B SGD lr=0.1 (legacy), AdamW lr=1e-3 (R1 best). After R2 sweep
completes, run `scripts/patch_lr_from_r2.py` to overwrite placeholders.

Usage:
  python3 scripts/gen_phase_rerun.py --phase 3   # SET-B
  python3 scripts/gen_phase_rerun.py --phase 4   # SET-A
  python3 scripts/gen_phase_rerun.py --phase 5   # SET-C
  python3 scripts/gen_phase_rerun.py --phase all
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import argparse, copy, sys
import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG_DIR = ROOT / "configs/matrix"

POOL_ATTACKS = [
    "random_hard", "random_soft", "knockoff_nets_soft",
    "activethief_hard", "activethief_soft",
    "activethief_dfal_hard", "activethief_dfal_soft",
    "activethief_uncertainty_hard", "activethief_uncertainty_soft",
    "cloudleak_soft", "marich_hard", "blackbox_dissector_hard",
]
# NOTE: DS (DualStudents) is data-free per ICLR 2023 paper + ds.py implementation
# (uses DFMEGenerator). Excluded from phase 3-5 rerun; data-free attacks stay
# SGD-only per Q3 decision.
SWIFTTHIEF = ["swiftthief_hard", "swiftthief_soft"]
SEEDS = [0, 1, 2]

# Per-set base spec
SETS = {
    "A": {  # SET-A1, MNIST, LeNet, 10k
        "set_id": "SET-A1",
        "budget": "10k",
        "sub_arch_target": "lenet_mnist",   # same as victim, no change
        "name_suffix": "",                  # no _sub_* suffix
        "aug_strong": {
            "pipeline": ["random_resized_crop", "hflip", "color_jitter", "grayscale"],
            "random_resized_crop": {"scale": [0.2, 1.0]},
            "color_jitter": {"brightness": 0.4, "contrast": 0.4,
                             "saturation": 0.4, "hue": 0.1, "p": 0.8},
            "grayscale": {"p": 0.2},
        },
    },
    "B": {  # SET-B1, CIFAR-10, ResNet-34 sub (new), 20k
        "set_id": "SET-B1",
        "budget": "20k",
        "sub_arch_target": "resnet34",
        "name_suffix": "_sub_resnet34",
        "aug_strong": {
            "pipeline": ["random_resized_crop", "hflip", "color_jitter", "grayscale"],
            "random_resized_crop": {"scale": [0.2, 1.0]},
            "color_jitter": {"brightness": 0.4, "contrast": 0.4,
                             "saturation": 0.4, "hue": 0.1, "p": 0.8},
            "grayscale": {"p": 0.2},
        },
    },
    "C": {  # SET-C1, SewerML binary, xie2019, 20k
        "set_id": "SET-C1",
        "budget": "20k",
        "sub_arch_target": "xie2019",
        "name_suffix": "",
        "aug_strong": {
            "pipeline": ["random_resized_crop", "hflip", "color_jitter", "grayscale"],
            "random_resized_crop": {"scale": [0.2, 1.0]},
            "color_jitter": {"brightness": 0.4, "contrast": 0.4,
                             "saturation": 0.0, "hue": 0.0, "p": 0.8},  # binary branch
            "grayscale": {"p": 0.2},
        },
    },
}

# Placeholder LR values; patched after R2 sweep completes.
ADAMW_SPEC = {
    "optimizer": {"name": "adamw", "lr": 0.001, "weight_decay": 0.01, "betas": [0.9, 0.999]},
    "scheduler": {"name": "cosine", "warmup_epochs": 1},
}


DS_BUDGET = {"A": "10m", "B": "20m", "C": "20m"}  # DualStudents uses larger budget


def attack_budget(set_cfg, attack: str) -> str:
    """Return the budget string used in this attack's filename."""
    if attack.startswith("ds_"):
        return DS_BUDGET[set_cfg['set_id'][4]]
    return set_cfg["budget"]


def base_cfg_path(set_cfg, attack: str, seed: int) -> Path:
    """Path to existing baseline config (SGD/legacy substitute)."""
    budget = attack_budget(set_cfg, attack)
    return CFG_DIR / f"{set_cfg['set_id']}_{attack}_{budget}_seed{seed}.yaml"


def write_cfg(cfg: dict, name: str) -> Path:
    cfg["run"]["name"] = name
    out = CFG_DIR / f"{name}.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def make_sgd_baseline(base: dict, set_cfg, attack: str, seed: int) -> Path | None:
    """For SET-B only: copy baseline SGD into _sub_resnet34 variant.

    For SET-A/C: SGD baseline already exists at base path; skip.
    """
    if not set_cfg["name_suffix"]:
        return None  # SET-A/C: SGD baseline already exists
    cfg = copy.deepcopy(base)
    cfg["substitute"]["arch"] = set_cfg["sub_arch_target"]
    cfg["run"]["seeds"] = [seed]
    name = f"{set_cfg['set_id']}_{attack}_{attack_budget(set_cfg, attack)}_seed{seed}{set_cfg['name_suffix']}"
    return write_cfg(cfg, name)


def make_sgd_aug(base: dict, set_cfg, attack: str, seed: int, soft: bool = False) -> Path | None:
    """SGD + augmentation block. soft=True for SET-A optional variant."""
    if attack.startswith("swiftthief"):
        return None  # SwiftThief: no Aug column
    cfg = copy.deepcopy(base)
    if set_cfg["name_suffix"]:
        cfg["substitute"]["arch"] = set_cfg["sub_arch_target"]
    cfg["run"]["seeds"] = [seed]
    if soft:
        cfg["substitute"]["augmentation"] = {
            "pipeline": ["random_crop"],
            "random_crop": {"size": [28, 28], "padding": 2, "padding_mode": "reflect"},
        }
        suffix = "_aug_soft"
    else:
        cfg["substitute"]["augmentation"] = copy.deepcopy(set_cfg["aug_strong"])
        suffix = "_aug"
    name = f"{set_cfg['set_id']}_{attack}_{attack_budget(set_cfg, attack)}_seed{seed}{set_cfg['name_suffix']}{suffix}"
    return write_cfg(cfg, name)


def make_adamw(base: dict, set_cfg, attack: str, seed: int) -> Path:
    cfg = copy.deepcopy(base)
    if set_cfg["name_suffix"]:
        cfg["substitute"]["arch"] = set_cfg["sub_arch_target"]
    cfg["run"]["seeds"] = [seed]
    cfg["substitute"]["optimizer"] = copy.deepcopy(ADAMW_SPEC["optimizer"])
    cfg["substitute"]["scheduler"] = copy.deepcopy(ADAMW_SPEC["scheduler"])
    name = f"{set_cfg['set_id']}_{attack}_{attack_budget(set_cfg, attack)}_seed{seed}{set_cfg['name_suffix']}_adamw"
    return write_cfg(cfg, name)


def make_adamw_aug(base: dict, set_cfg, attack: str, seed: int) -> Path | None:
    if attack.startswith("swiftthief"):
        return None
    cfg = copy.deepcopy(base)
    if set_cfg["name_suffix"]:
        cfg["substitute"]["arch"] = set_cfg["sub_arch_target"]
    cfg["run"]["seeds"] = [seed]
    cfg["substitute"]["optimizer"] = copy.deepcopy(ADAMW_SPEC["optimizer"])
    cfg["substitute"]["scheduler"] = copy.deepcopy(ADAMW_SPEC["scheduler"])
    cfg["substitute"]["augmentation"] = copy.deepcopy(set_cfg["aug_strong"])
    name = f"{set_cfg['set_id']}_{attack}_{attack_budget(set_cfg, attack)}_seed{seed}{set_cfg['name_suffix']}_adamw_aug"
    return write_cfg(cfg, name)


def gen_set(set_key: str) -> int:
    set_cfg = SETS[set_key]
    print(f"=== Generating phase configs for SET-{set_key} ({set_cfg['set_id']}) ===")
    n = 0
    skipped = 0
    attacks = POOL_ATTACKS + SWIFTTHIEF
    for attack in attacks:
        for seed in SEEDS:
            base_path = base_cfg_path(set_cfg, attack, seed)
            if not base_path.exists():
                print(f"  [skip] no base: {base_path.name}")
                skipped += 1
                continue
            with open(base_path) as f:
                base = yaml.safe_load(f)

            # SET-B: also make SGD baseline (_sub_resnet34, no other suffix)
            p = make_sgd_baseline(base, set_cfg, attack, seed)
            if p: n += 1

            # SGD+Aug (Phase 3 SET-B only; SET-A/C already have these)
            if set_key == "B":
                p = make_sgd_aug(base, set_cfg, attack, seed)
                if p: n += 1
            # SET-A: SGD+Aug-soft was already generated separately; skip
            # SET-C: SGD+Aug already generated

            # AdamW (always new for phase 3/4/5)
            p = make_adamw(base, set_cfg, attack, seed)
            if p: n += 1
            # AdamW+Aug (skip SwiftThief)
            p = make_adamw_aug(base, set_cfg, attack, seed)
            if p: n += 1
    print(f"  generated: {n}, skipped (no base): {skipped}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["3", "4", "5", "all"],
                    help="3=SET-B, 4=SET-A, 5=SET-C, all=all three")
    args = ap.parse_args()
    mapping = {"3": ["B"], "4": ["A"], "5": ["C"], "all": ["B", "A", "C"]}
    total = 0
    for key in mapping[args.phase]:
        total += gen_set(key)
    print(f"\nTotal generated: {total}")


if __name__ == "__main__":
    main()
