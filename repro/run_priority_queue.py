"""Run prioritized reproduction queue for four attacks.

Priority order:
1) DFME
2) MAZE
3) DFMS
4) InverseNet
"""

from __future__ import annotations

import argparse
import queue
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

PRIORITY = [
    "2021_truong_dfme",
    "2021_kariyappa_maze",
    "2022_sanyal_dfms",
    "2021_gong_inversenet",
]

STAGES_BY_PAPER = {
    # strict per-paper victim training/eval
    "2021_truong_dfme": "victim_train,victim_eval,attack,collect,compare",
    "2021_kariyappa_maze": "victim_train,victim_eval,attack,collect,compare",
    "2022_sanyal_dfms": "victim_train,victim_eval,attack,collect,compare",
    "2021_gong_inversenet": "victim_train,victim_eval,attack,collect,compare",
}

STAGE_HINTS = {
    "train_victim.py": "victim_train",
    "eval_victim.py": "victim_eval",
    "-m mebench run": "attack",
    "skip collect": "collect",
    "skip compare": "compare",
}


def _infer_stage(line: str, current: str) -> str:
    stripped = line.strip()
    for marker, stage in STAGE_HINTS.items():
        if marker in stripped:
            return stage
    return current


def _run(cmd: list[str], dry_run: bool, label: str, heartbeat_sec: int) -> None:
    shown = shlex.join(cmd)
    print(f"$ {shown}")
    if dry_run:
        return

    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None

    out_q: queue.Queue[str | None] = queue.Queue()

    def _reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            out_q.put(line)
        out_q.put(None)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    started = time.monotonic()
    last_output = started
    current_stage = "pipeline"

    while True:
        try:
            line = out_q.get(timeout=max(1, int(heartbeat_sec)))
        except queue.Empty:
            elapsed = int(time.monotonic() - started)
            quiet_for = int(time.monotonic() - last_output)
            print(
                f"[{label}] heartbeat: running stage={current_stage} elapsed={elapsed}s quiet={quiet_for}s"
            )
            continue

        if line is None:
            break

        current_stage = _infer_stage(line, current_stage)
        last_output = time.monotonic()
        sys.stdout.write(line)

    rc = proc.wait()
    t.join(timeout=1.0)
    if rc != 0:
        raise RuntimeError(f"Command failed ({rc}): {shown}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DFME/MAZE/DFMS/InverseNet queue")
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--smoke-batch-size", type=int, default=64)
    parser.add_argument(
        "--heartbeat-sec",
        type=int,
        default=60,
        help="Heartbeat interval while child emits no output",
    )
    args = parser.parse_args()

    queue_start = time.monotonic()
    for idx, paper_id in enumerate(PRIORITY, start=1):
        stages = STAGES_BY_PAPER.get(paper_id, "victim_train,victim_eval,attack,collect,compare")
        stage_list = [s.strip() for s in stages.split(",") if s.strip()]
        label = f"{idx}/{len(PRIORITY)} {paper_id}"
        print(f"\n=== [{label}] start ===")
        print(f"[{label}] planned_stages={','.join(stage_list)}")
        paper_start = time.monotonic()
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
            "--smoke-batch-size",
            str(int(args.smoke_batch_size)),
            "--stages",
            stages,
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        _run(cmd, args.dry_run, label=label, heartbeat_sec=args.heartbeat_sec)
        elapsed = int(time.monotonic() - paper_start)
        print(f"=== [{label}] done ({elapsed}s) ===")

    total_elapsed = int(time.monotonic() - queue_start)
    print(f"\nQueue finished: {len(PRIORITY)} papers, elapsed={total_elapsed}s")


if __name__ == "__main__":
    main()
