#!/usr/bin/env python3
"""Runner for SET-A1 AdamW + soft-aug pool-based experiments (36 jobs).

12 attacks x 3 seeds. All MNIST/lenet_mnist so each job is light on VRAM
(~2-5 GB) and short (~few min). Share the box with the existing user
(their gpu_autoyield job takes ~50 GB) - stay under 90 GB committed to
be safe.
"""
from __future__ import annotations
import re, subprocess, sys, time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
CFG_DIR = ROOT / "configs/matrix"
LOG_DIR = ROOT / "logs"; LOG_DIR.mkdir(exist_ok=True)
RUNS_DIR = ROOT / "runs"
PYTHON = "/opt/conda/envs/mebench/bin/python3"

MAX_CONCURRENT = 8
TOTAL_BUDGET_GB = 90
PER_JOB_VRAM_GB = 6
MARGIN_GB = 4
POLL_INTERVAL_S = 30
LAUNCH_INTERVAL_S = 15

ATTACKS = [
    "activethief_dfal_hard", "activethief_dfal_soft",
    "activethief_hard", "activethief_soft",
    "activethief_uncertainty_hard", "activethief_uncertainty_soft",
    "blackbox_dissector_hard",
    "cloudleak_soft",
    "knockoff_nets_soft",
    "marich_hard",
    "random_hard", "random_soft",
]
SEEDS = [0, 1, 2]


@dataclass
class Job:
    config: str
    vram_gb: int = PER_JOB_VRAM_GB


def build_queue() -> List[Job]:
    jobs = []
    for atk in ATTACKS:
        for s in SEEDS:
            name = f"SET-A1_{atk}_10k_seed{s}_adamw_aug_soft"
            if (CFG_DIR / f"{name}.yaml").exists():
                jobs.append(Job(name))
    return jobs


def already_done(j: Job) -> bool:
    rd = RUNS_DIR / j.config
    if not rd.exists(): return False
    for ts in sorted(rd.iterdir(), reverse=True):
        if not ts.is_dir(): continue
        for sd in ts.iterdir():
            if (sd / "summary.json").exists(): return True
    return False


def running_configs() -> set:
    try:
        out = subprocess.check_output(
            ["pgrep", "-af", "python.*-m mebench run --config"],
            text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return set()
    names = set()
    for line in out.splitlines():
        m = re.search(r"--config\s+\S*?([^/\s]+)\.yaml", line)
        if m: names.add(m.group(1))
    return names


def gpu_free_gb() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True, timeout=10)
        return int(out.strip().splitlines()[0]) / 1024
    except Exception:
        return 0.0


def now() -> str: return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def log(m: str) -> None: print(f"[{now()}] {m}", flush=True)


def launch(j: Job) -> subprocess.Popen:
    cfg = CFG_DIR / f"{j.config}.yaml"
    lp = LOG_DIR / f"{j.config}.log"
    f = open(lp, "w")
    proc = subprocess.Popen(
        [PYTHON, "-m", "mebench", "run", "--config", str(cfg)],
        stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    proc._log_file = f
    return proc


def main() -> int:
    queue = build_queue()
    log(f"==== SET-A1 AdamW+aug_soft: {len(queue)} jobs ====")
    log(f"MAX={MAX_CONCURRENT} VRAM_per_job={PER_JOB_VRAM_GB}G budget={TOTAL_BUDGET_GB}G")
    running: List[tuple[Job, subprocess.Popen]] = []
    done = failed = skipped = 0
    while queue or running:
        for j, proc in list(running):
            rc = proc.poll()
            if rc is None: continue
            proc._log_file.close(); running.remove((j, proc))
            if rc == 0:
                done += 1; log(f"DONE  {j.config}  (done={done})")
            else:
                failed += 1; log(f"FAIL  {j.config}  rc={rc}")
        ext = running_configs() - set(j.config for j, _ in running)
        active = set(j.config for j, _ in running) | ext
        committed = sum(j.vram_gb for j, _ in running) + 6 * len(ext)
        for j in list(queue):
            if already_done(j):
                queue.remove(j); skipped += 1; log(f"SKIP  {j.config}"); continue
            if j.config in active:
                queue.remove(j); skipped += 1; log(f"SKIP  {j.config} (running)"); continue
            if len(running) + len(ext) >= MAX_CONCURRENT: break
            if committed + j.vram_gb + MARGIN_GB > TOTAL_BUDGET_GB: continue
            free = gpu_free_gb()
            if free < j.vram_gb + MARGIN_GB: continue
            queue.remove(j); proc = launch(j); running.append((j, proc))
            committed += j.vram_gb; active.add(j.config)
            log(f"START {j.config}  committed={committed}G  free={free:.1f}G  in_flight={len(running)}")
            time.sleep(LAUNCH_INTERVAL_S)
        if not queue and not running: break
        time.sleep(POLL_INTERVAL_S)
    log(f"==== finished: done={done} failed={failed} skipped={skipped} ====")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
