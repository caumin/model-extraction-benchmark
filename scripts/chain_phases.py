#!/usr/bin/env python3
"""Chain phase 3 → 4 → 5 sequentially with auto-retry of failed runs.

Per-phase behavior:
  1. Wait for the initial run (run_phase.py --phase N) to finish.
  2. After it terminates, count how many of that phase's jobs lack a summary.json.
  3. If any failures, rename the current runner log and re-launch run_phase.py;
     run_phase's own `already_done(j)` skip ensures only failed/missing jobs re-launch.
  4. Repeat up to MAX_RETRIES_PER_PHASE times. If failures remain, log a warning
     and proceed to the next phase (do not abort the whole chain).

Detection: pgrep + tail-grep "finished:" marker, same as before.

Usage:
  nohup python3 scripts/chain_phases.py > logs/_chain_phases.log 2>&1 &
"""
from __future__ import annotations
import re, shutil, subprocess, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
PYTHON = sys.executable

POLL_S = 60
CHAIN = [3, 4, 5]
MAX_RETRIES_PER_PHASE = 2  # extra attempts beyond initial run (so total = 3)

sys.path.insert(0, str(ROOT / "scripts"))
from run_phase import collect_jobs, already_done  # noqa: E402


def now() -> str: return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def log(m: str) -> None: print(f"[{now()}] {m}", flush=True)


def runner_alive(phase: int) -> bool:
    try:
        out = subprocess.check_output(
            ["pgrep", "-af", f"run_phase\\.py --phase {phase}"],
            text=True, stderr=subprocess.DEVNULL)
        return bool(out.strip())
    except subprocess.CalledProcessError:
        return False


def runner_log_finished(phase: int) -> bool:
    lp = LOG_DIR / f"_phase{phase}_runner.log"
    if not lp.exists(): return False
    try:
        out = subprocess.check_output(["tail", "-5", str(lp)], text=True)
    except subprocess.CalledProcessError:
        return False
    return bool(re.search(r"^\[.*?\] ==== finished: done=\d+ failed=\d+ skipped=\d+ ====", out, re.M))


def count_failures(phase: int) -> tuple[int, list]:
    """Return (count, list-of-Job-objects) for jobs lacking summary.json."""
    jobs = collect_jobs(phase)
    failed = [j for j in jobs if not already_done(j)]
    return len(failed), failed


def wait_for_phase(phase: int) -> bool:
    """Block until phase N runner terminates with finished marker.
    Returns True if completed normally, False if log/process state is ambiguous.
    """
    log(f"Waiting for phase {phase} to finish (poll every {POLL_S}s)…")
    while True:
        alive = runner_alive(phase)
        done_marker = runner_log_finished(phase)
        if not alive and done_marker:
            log(f"Phase {phase} run terminated (process dead + finished marker present)")
            return True
        if not alive and not done_marker:
            log(f"WARN  Phase {phase} runner not running but log lacks 'finished:' marker — abnormal exit")
            return False
        time.sleep(POLL_S)


def rotate_log(phase: int, attempt: int) -> None:
    """Rename current runner log so a fresh run can write a new 'finished:' marker."""
    lp = LOG_DIR / f"_phase{phase}_runner.log"
    if not lp.exists(): return
    backup = LOG_DIR / f"_phase{phase}_runner.log.attempt{attempt}"
    shutil.move(str(lp), str(backup))
    log(f"  rotated {lp.name} → {backup.name}")


def launch_phase(phase: int) -> bool:
    """Launch a phase runner. Returns True on launch success."""
    lp = LOG_DIR / f"_phase{phase}_runner.log"
    if runner_alive(phase):
        log(f"Phase {phase} runner already alive — skipping launch")
        return True
    log(f"Launching phase {phase} runner…")
    proc = subprocess.Popen(
        ["nohup", PYTHON, "scripts/run_phase.py", "--phase", str(phase)],
        stdout=open(lp, "w"), stderr=subprocess.STDOUT, cwd=str(ROOT),
        start_new_session=True)
    time.sleep(5)
    if proc.poll() is not None and proc.returncode != 0:
        log(f"FAIL  Phase {phase} runner exited immediately rc={proc.returncode}")
        return False
    log(f"Phase {phase} runner PID={proc.pid}, log={lp.name}")
    return True


def run_phase_with_retry(phase: int, initial_already_running: bool = False) -> bool:
    """Wait for phase + auto-retry failed jobs up to MAX_RETRIES_PER_PHASE times.

    initial_already_running=True means caller assumes the phase is already
    launched externally (e.g., phase 3). Skip the first launch_phase call.
    """
    if not initial_already_running:
        if not launch_phase(phase):
            return False

    # Attempt 0 (initial) + retries
    for attempt in range(MAX_RETRIES_PER_PHASE + 1):
        if not wait_for_phase(phase):
            log(f"Phase {phase} attempt {attempt} terminated abnormally; aborting phase")
            return False

        n_failed, failed_jobs = count_failures(phase)
        if n_failed == 0:
            log(f"Phase {phase} complete — no failed/missing jobs after attempt {attempt}")
            return True

        if attempt >= MAX_RETRIES_PER_PHASE:
            log(f"Phase {phase} reached max retries ({MAX_RETRIES_PER_PHASE}); "
                f"{n_failed} jobs still failing. Continuing chain anyway.")
            for j in failed_jobs:
                log(f"  STILL FAILING: {j.config}")
            return True  # don't block chain

        # Retry round
        log(f"Phase {phase} attempt {attempt} ended with {n_failed} failed/missing jobs. "
            f"Launching retry (attempt {attempt + 1}/{MAX_RETRIES_PER_PHASE})")
        for j in failed_jobs[:10]:
            log(f"  retry: {j.config}")
        if n_failed > 10:
            log(f"  ... and {n_failed - 10} more")
        rotate_log(phase, attempt)
        if not launch_phase(phase):
            log(f"Phase {phase} retry launch failed")
            return False
        # loop continues; wait_for_phase will fire on the new run


def main() -> int:
    log("==== chain_phases (with auto-retry): waiting for phase 3 → 4 → 5 ====")
    log(f"MAX_RETRIES_PER_PHASE={MAX_RETRIES_PER_PHASE} (initial + 2 retries)")

    # phase 3 is launched externally; just monitor + auto-retry
    if not run_phase_with_retry(3, initial_already_running=True):
        log("Phase 3 chain abort. Stopping.")
        return 1

    if not run_phase_with_retry(4):
        log("Phase 4 chain abort. Stopping.")
        return 2

    if not run_phase_with_retry(5):
        log("Phase 5 chain abort. Stopping.")
        return 3

    log("==== chain complete: phase 3 + 4 + 5 done ====")
    log("Next: run `python3 analyze_results.py` to refresh paper tables")
    return 0


if __name__ == "__main__":
    sys.exit(main())
