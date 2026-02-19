"""Run prioritized reproduction queue for four attacks.

Priority order:
1) DFME
2) MAZE
3) DFMS
4) InverseNet
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

PRIORITY = [
    "2021_truong_dfme",
    "2021_kariyappa_maze",
    "2022_sanyal_dfms",
    "2021_gong_inversenet",
]

STAGES_BY_PAPER = {
    # trains shared CIFAR10 victim once
    "2021_truong_dfme": "victim_train,victim_eval,attack,collect,compare",
    # reuse CIFAR10 victim checkpoint from DFME stage
    "2021_kariyappa_maze": "attack,collect,compare",
    "2022_sanyal_dfms": "attack,collect,compare",
    # separate MNIST victim
    "2021_gong_inversenet": "victim_train,victim_eval,attack,collect,compare",
}


def _run(cmd: list[str], dry_run: bool) -> None:
    shown = shlex.join(cmd)
    print(f"$ {shown}")
    if dry_run:
        return
    proc = subprocess.run(cmd, cwd=ROOT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {shown}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DFME/MAZE/DFMS/InverseNet queue")
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=2)
    args = parser.parse_args()

    for idx, paper_id in enumerate(PRIORITY, start=1):
        print(f"\n=== [{idx}/{len(PRIORITY)}] {paper_id} ===")
        cmd = [
            sys.executable,
            "repro/run_experiment.py",
            "run",
            "--paper-id",
            paper_id,
            "--profile",
            args.profile,
            "--device",
            args.device,
            "--smoke-epochs",
            str(int(args.smoke_epochs)),
            "--stages",
            STAGES_BY_PAPER.get(paper_id, "victim_train,victim_eval,attack,collect,compare"),
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        _run(cmd, args.dry_run)


if __name__ == "__main__":
    main()
