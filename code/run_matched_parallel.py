#!/usr/bin/env python3
"""Parallel completion of the six matched-information suites.

Replaces the serial tail of run_matched_information_queue.py (whose parent
process is stopped before this runs). Measured basis for the width choice:
one suite uses ~1 CPU core of 12 and ~32% GPU, so three concurrent suites
saturate the GPU without CPU contention. No matched arm is wall-clock
budgeted (no rolling-OR), and per-process numerics do not depend on machine
load, so parallel suites remain statistically and bit-comparable;
validate_matched_information.py stays the per-suite integrity gate.

Preserved queue discipline: BOTH seed-12345 control suites must complete AND
validate before any extra-seed suite starts (fail-fast on a systemic break).
A suite already running under the old queue is ADOPTED (waited on, then
validated), never restarted - matched_information_eval.py cannot resume.

  python run_matched_parallel.py                       # adopt h8:12345, width 3
  python run_matched_parallel.py --adopt "" --width 3  # nothing external

NOTE: never run this while run_matched_information_queue.py is alive - they
write the same output files. Resumable: complete suites are skipped.
"""
from __future__ import annotations

import argparse
import ctypes
import queue as queue_mod
import subprocess
import sys
import threading
import time

from run_matched_information_queue import (
    HERE,
    PYTHON,
    RESULTS,
    matched_complete,
    output_path,
    validate,
)

CONTROLS = ((8, 12345), (4, 12345))
EXTRAS = ((8, 13345), (8, 14345), (4, 13345), (4, 14345))


def log_path(horizon: int, seed: int):
    return RESULTS / f"matched_information_h{horizon}_seed_{seed}.log"


def run_suite(horizon: int, seed: int) -> bool:
    destination = output_path(horizon, seed)
    if matched_complete(destination):
        print(f"[MATCHED-PAR] H={horizon} seed={seed}: already complete", flush=True)
        return True
    command = [
        PYTHON, "-u", str(HERE / "matched_information_eval.py"),
        "--base-seed", str(seed),
        "--horizon-hours", str(horizon),
        "--output", str(destination),
    ]
    lp = log_path(horizon, seed)
    print(f"[MATCHED-PAR] H={horizon} seed={seed}: starting (log: {lp.name})", flush=True)
    started = time.time()
    with lp.open("a", encoding="utf-8") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(command)}\n")
        log.flush()
        return_code = subprocess.run(
            command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT
        ).returncode
    minutes = (time.time() - started) / 60
    ok = return_code == 0 and matched_complete(destination)
    status = "done" if ok else f"FAILED rc={return_code}"
    print(f"[MATCHED-PAR] H={horizon} seed={seed}: {status} after {minutes:.0f} min", flush=True)
    return ok


def adopt_external(horizon: int, seed: int, stale_minutes: float = 25.0) -> bool:
    """Wait for a suite another process is running; rerun only if it dies."""
    lp = log_path(horizon, seed)
    while True:
        if matched_complete(output_path(horizon, seed)):
            print(f"[MATCHED-PAR] H={horizon} seed={seed}: external run complete; adopted", flush=True)
            return True
        age = (time.time() - lp.stat().st_mtime) / 60 if lp.exists() else float("inf")
        if age > stale_minutes:
            print(f"[MATCHED-PAR] H={horizon} seed={seed}: external run stale "
                  f"({age:.0f} min); rerunning here", flush=True)
            return run_suite(horizon, seed)
        time.sleep(60)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=3,
                        help="concurrent extra-seed suites (3 saturates the GPU)")
    parser.add_argument("--adopt", default="8:12345",
                        help="'h:seed' suite already running externally ('' = none)")
    args = parser.parse_args()

    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        print("[MATCHED-PAR] keep-awake armed", flush=True)

    external = None
    if args.adopt:
        h, s = args.adopt.split(":")
        external = (int(h), int(s))

    # Phase 1: both seed-12345 controls, concurrently, each validated on completion.
    control_ok: dict = {}

    def control_job(horizon: int, seed: int) -> None:
        if external == (horizon, seed):
            finished = adopt_external(horizon, seed)
        else:
            finished = run_suite(horizon, seed)
        control_ok[(horizon, seed)] = finished and validate(horizon, seed)
        verdict = "validated" if control_ok[(horizon, seed)] else "GATE FAIL"
        print(f"[MATCHED-PAR] control H={horizon} seed={seed}: {verdict}", flush=True)

    control_threads = [
        threading.Thread(target=control_job, args=(h, s)) for h, s in CONTROLS
    ]
    for t in control_threads:
        t.start()
        time.sleep(5)  # stagger CUDA init
    for t in control_threads:
        t.join()
    if not all(control_ok.get(c) for c in CONTROLS):
        print("[MATCHED-PAR] CONTROL GATE FAILED - extra-seed suites NOT started. "
              "Investigate with gate_matched_frozen_identity.py before rerunning.", flush=True)
        raise SystemExit(1)
    print("[MATCHED-PAR] both controls complete + validated; starting extra seeds", flush=True)

    # Phase 2: four extra-seed suites, width-wide, each validated on completion.
    failures: list = []
    jobs: "queue_mod.Queue" = queue_mod.Queue()
    for job in EXTRAS:
        jobs.put(job)

    def worker() -> None:
        while True:
            try:
                horizon, seed = jobs.get_nowait()
            except queue_mod.Empty:
                return
            if not (run_suite(horizon, seed) and validate(horizon, seed)):
                failures.append(f"h{horizon}/{seed}")
            jobs.task_done()

    workers = [threading.Thread(target=worker) for _ in range(max(1, args.width))]
    for t in workers:
        t.start()
        time.sleep(5)
    for t in workers:
        t.join()

    if failures:
        print(f"[MATCHED-PAR] FINISHED WITH FAILURES: {failures} - "
              "rerun this script to resume.", flush=True)
        raise SystemExit(1)

    if subprocess.run([PYTHON, str(HERE / "aggregate_matched_information.py")],
                      cwd=HERE).returncode != 0:
        print("[MATCHED-PAR] aggregation FAILED", flush=True)
        raise SystemExit(1)
    print("[MATCHED-PAR] all six suites complete; validation and aggregation passed. "
          "Next: S6.3a write-in per MATCHED_SELECTOR_PLAN.md S5.", flush=True)


if __name__ == "__main__":
    main()
