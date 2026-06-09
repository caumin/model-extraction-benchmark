#!/usr/bin/env python3
"""Generic phase runner for SET-A/B/C 4(5)-column rerun.

Usage:
  python3 scripts/run_phase.py --phase 3   # SET-B (180 runs)
  python3 scripts/run_phase.py --phase 4   # SET-A (90 runs)
  python3 scripts/run_phase.py --phase 5   # SET-C (90 runs)

Per-phase concurrency / VRAM budget tuned for set's input size and arch.
"""
from __future__ import annotations
import argparse, re, subprocess, sys, time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
CFG_DIR = ROOT / "configs/matrix"
LOG_DIR = ROOT / "logs"; LOG_DIR.mkdir(exist_ok=True)
RUNS_DIR = ROOT / "runs"
PYTHON = sys.executable

# Per-phase tunables
PHASE_CONFIG = {
    3: {  # SET-B (32x32x3, resnet34 ~21M)
        "set_id": "SET-B1",
        "name_suffix": "_sub_resnet34",  # main marker for phase 3
        "max_concurrent": 5,
        "per_job_vram_gb": 5,
        "budget_gb": 135,
        "launch_interval_s": 30,
    },
    4: {  # SET-A (28x28x1, lenet ~60k)
        "set_id": "SET-A1",
        "name_suffix": "",   # SET-A doesn't change arch
        "main_pattern_suffixes": ["_adamw", "_adamw_aug"],  # phase 4 only adds these
        "max_concurrent": 8,  # very light
        "per_job_vram_gb": 2,
        "budget_gb": 135,
        "launch_interval_s": 20,
    },
    5: {  # SET-C (224x224x3, xie2019 ~9M, heavy CPU/IO)
        "set_id": "SET-C1",
        "name_suffix": "",
        "main_pattern_suffixes": ["_adamw", "_adamw_aug"],
        "max_concurrent": 3,
        "per_job_vram_gb": 15,
        "budget_gb": 135,
        "launch_interval_s": 90,
    },
}

POLL_S = 30
MARGIN_GB = 5


@dataclass
class Job:
    config: str
    vram_gb: int


def collect_jobs(phase: int) -> List[Job]:
    """Collect all phase configs in execution order (light → heavy)."""
    cfg = PHASE_CONFIG[phase]
    set_id = cfg["set_id"]
    vram = cfg["per_job_vram_gb"]
    jobs = []

    if phase == 3:
        # SET-B: all 4 cells (SGD baseline + SGD+Aug + AdamW + AdamW+Aug) on resnet34
        # cell suffixes: "" (SGD baseline), "_aug", "_adamw", "_adamw_aug"
        # name pattern: SET-B1_<attack>_<budget>_seed<s>_sub_resnet34<cell_suffix>.yaml
        for cell in ("", "_aug", "_adamw", "_adamw_aug"):
            pattern = f"{set_id}_*_sub_resnet34{cell}.yaml"
            for p in sorted(CFG_DIR.glob(pattern)):
                # exclude sweep variants (_sub_resnet34_sgd_lr* / _adamw_lr*)
                if re.search(r"_sub_resnet34_(sgd|adamw)_lr", p.name):
                    continue
                # exclude misclassification — strict cell match
                stem = p.stem
                if cell == "":
                    # must end exactly in _sub_resnet34 (no further suffix)
                    if not stem.endswith("_sub_resnet34"):
                        continue
                else:
                    if not stem.endswith(f"_sub_resnet34{cell}"):
                        continue
                jobs.append(Job(stem, vram))
    else:
        # SET-A/C: only add AdamW + AdamW+Aug new cells
        for suffix in cfg["main_pattern_suffixes"]:
            for p in sorted(CFG_DIR.glob(f"{set_id}_*{suffix}.yaml")):
                stem = p.stem
                # strict-end match
                if not stem.endswith(suffix):
                    continue
                # exclude ablation / sub_resnet34 / aug_soft etc
                if "_ablation" in stem or "_sub_" in stem:
                    continue
                # avoid double-counting (_adamw_aug also matches _adamw glob)
                if suffix == "_adamw" and stem.endswith("_adamw_aug"):
                    continue
                jobs.append(Job(stem, vram))
    return jobs


def already_done(j: Job) -> bool:
    rd = RUNS_DIR / j.config
    if not rd.exists(): return False
    for ts in sorted(rd.iterdir(), reverse=True):
        if not ts.is_dir(): continue
        for sd in ts.iterdir():
            if (sd / "summary.json").exists(): return True
    return False


def running_configs() -> set[str]:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, required=True, choices=[3, 4, 5])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = PHASE_CONFIG[args.phase]
    queue = collect_jobs(args.phase)
    log(f"==== Phase {args.phase} runner: {len(queue)} jobs queued ====")
    log(f"set={cfg['set_id']} MAX={cfg['max_concurrent']} VRAM_per_job={cfg['per_job_vram_gb']}G")

    if args.dry_run:
        for j in queue: print(f"  {j.config}")
        return 0

    running: List[tuple[Job, subprocess.Popen]] = []
    done = failed = skipped = 0

    while queue or running:
        for j, proc in list(running):
            rc = proc.poll()
            if rc is None: continue
            proc._log_file.close()
            running.remove((j, proc))
            if rc == 0:
                done += 1
                log(f"DONE  {j.config}  (done={done}/{len(queue)+done+failed+len(running)})")
            else:
                failed += 1
                log(f"FAIL  {j.config}  rc={rc}")

        ext = running_configs() - set(j.config for j, _ in running)
        active = set(j.config for j, _ in running) | ext
        committed = sum(j.vram_gb for j, _ in running) + 10 * len(ext)

        for j in list(queue):
            if already_done(j):
                queue.remove(j); skipped += 1
                log(f"SKIP  {j.config}  (already done)"); continue
            if j.config in active:
                queue.remove(j); skipped += 1
                log(f"SKIP  {j.config}  (already running)"); continue
            if len(running) + len(ext) >= cfg["max_concurrent"]: break
            if committed + j.vram_gb + MARGIN_GB > cfg["budget_gb"]: continue
            free = gpu_free_gb()
            if free < j.vram_gb + MARGIN_GB: continue
            queue.remove(j)
            proc = launch(j)
            running.append((j, proc))
            committed += j.vram_gb
            active.add(j.config)
            log(f"START {j.config}  committed={committed}G  free={free:.1f}G  in_flight={len(running)}")
            time.sleep(cfg["launch_interval_s"])

        if not queue and not running: break
        time.sleep(POLL_S)

    log(f"==== finished: done={done} failed={failed} skipped={skipped} ====")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
