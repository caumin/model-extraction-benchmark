#!/usr/bin/env python3
"""Generate SET-A1 pool-based AdamW + soft-aug configs.

The paper table has 5 pool-based variants: SGD / SGD+Aug / SGD+Aug(soft) /
AdamW / AdamW+Aug. The AdamW + Aug(soft) column is currently missing.

Base: each existing SET-A1_*_aug_soft.yaml (36 files: 12 attacks x 3 seeds).
Patch: swap substitute.optimizer to AdamW and substitute.scheduler to cosine
(matching the pattern established by *_adamw_aug.yaml). Keep the augmentation
block unchanged.

Naming: SET-A1_{attack}_{mode}_10k_seed{s}_adamw_aug_soft.yaml

Usage:
  python3 scripts/gen_seta_adamw_aug_soft.py --dry-run
  python3 scripts/gen_seta_adamw_aug_soft.py --apply
"""
from __future__ import annotations
import argparse, copy, re
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / "configs/matrix"

ADAMW_SPEC = {"name": "adamw", "lr": 0.001, "weight_decay": 0.01,
              "betas": [0.9, 0.999]}
COSINE_SPEC = {"name": "cosine", "warmup_epochs": 1}


def patch(base_cfg: dict, new_name: str) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["substitute"]["optimizer"] = dict(ADAMW_SPEC)
    cfg["substitute"]["scheduler"] = dict(COSINE_SPEC)
    cfg.setdefault("run", {})
    cfg["run"]["name"] = new_name
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    pat = re.compile(r"^SET-A1_(?P<body>.+)_aug_soft\.yaml$")
    bases = sorted(p for p in CFG.iterdir()
                   if pat.match(p.name) and "_adamw" not in p.name)

    n_new = n_skip = 0
    for base in bases:
        m = pat.match(base.name)
        body = m.group("body")
        new_name = f"SET-A1_{body}_adamw_aug_soft"
        dst = CFG / f"{new_name}.yaml"
        if dst.exists():
            n_skip += 1
            print(f"  [exists] {new_name}.yaml")
            continue
        with open(base) as f:
            base_cfg = yaml.safe_load(f)
        new_cfg = patch(base_cfg, new_name)
        if args.apply:
            with open(dst, "w") as f:
                yaml.safe_dump(new_cfg, f, sort_keys=False)
            print(f"  +  {new_name}.yaml")
        else:
            print(f"  [dry] {new_name}.yaml")
        n_new += 1

    print(f"\nnew={n_new}  exists={n_skip}  base_count={len(bases)}")
    if not args.apply:
        print("Rerun with --apply to write.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
