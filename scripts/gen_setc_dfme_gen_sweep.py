#!/usr/bin/env python3
"""SET-C1 DFME generator-LR sweep.

Motivated by the previous 8-config exploration where lrG=1e-5 (10x lower
than default 1e-4) was the only variant escaping the trivial baseline
(acc 0.540, precision 0.572, recall 0.207) — everything else stayed at
acc=0.515/agr=0.5011. Two goals:

  (a) Verify lrG=1e-5 signal is not a seed accident (seeds 1, 2).
  (b) Extend the LR sweep — is 1e-5 already the optimum, or does going
      even lower / slightly higher improve further?  Points added:
      lrG ∈ {1e-6, 3e-6, 3e-5}.

All jobs use budget=5M, single seed each (seed 0 for the sweep points,
seeds 1 & 2 for the reproducibility check).
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


def patch(base, run_name, *, seed, lrG):
    cfg = copy.deepcopy(base)
    cfg["run"]["name"] = run_name
    cfg["run"]["seeds"] = [seed]
    cfg["budget"]["max_budget"] = BUDGET
    cfg["budget"]["checkpoints"] = list(CHECKPOINTS)
    cfg["attack"]["generator_lr"] = lrG
    return cfg


def build():
    with open(BASE) as f:
        base = yaml.safe_load(f)
    outs = {}

    # (a) reproducibility of lrG=1e-5 on seeds 1, 2
    for s in (1, 2):
        name = f"SET-C1_dfme_explore_lrG_1e-5_5m_seed{s}"
        outs[name] = patch(base, name, seed=s, lrG=1e-5)

    # (b) extended LR sweep at seed 0
    for lrG, tag in [(1e-6, "1e-6"), (3e-6, "3e-6"), (3e-5, "3e-5")]:
        name = f"SET-C1_dfme_explore_lrG_{tag}_5m_seed0"
        outs[name] = patch(base, name, seed=0, lrG=lrG)

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
