#!/usr/bin/env python3
"""Parallel-lane replacement for run_p3_extra_seeds.py — same jobs, ~30-40% less wall time.

Why this exact structure (do not "optimize" it further):
- N=100 seeds and hstress_h4 feed rolling-OR comparisons in the paper (Table 3, §6.8).
  rolling-OR is wall-clock-budgeted (30 ms/step), so those runs must be EXCLUSIVE —
  CPU contention would silently weaken the baseline and taint the headline table.
- The six obs_* (job × seed) units feed §6.4a only (oracle−repair, Δ vs live control;
  no rolling-OR numbers used), and schedules are presampled from the seed, so mutual
  contention is statistically safe. They run WIDTH-wide (default 2).

Order: wait for N=100 gate (exclusive) → obs pool 2-wide → hstress_h4 seeds
sequentially, exclusive. Resumable: completed JSONs are skipped by run_job.

  python run_p3_parallel.py               # wait for N=100 runs, then go
  python run_p3_parallel.py --start-now   # GPU already free
  python run_p3_parallel.py --skip-obs-extras   # only obs_base (+hstress): fastest
                                                # defensible tier; obs_mask/traffic
                                                # keep their 1-seed caveat
NOTE: only ONE of run_p3_parallel.py / run_p3_extra_seeds.py may run at a time —
they write the same output files.
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading

from run_p3_extra_seeds import JOBS, SEEDS, run_job, wait_for_gpu

OBS_JOBS = ["obs_base", "obs_mask", "obs_traffic"]


def worker(q: "queue.Queue", fails: list) -> None:
    while True:
        try:
            name, seed = q.get_nowait()
        except queue.Empty:
            return
        if not run_job(name, JOBS[name], seed):
            fails.append(f"{name}/{seed}")
        q.task_done()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-now", action="store_true",
                    help="skip the N=100 gate (GPU already free)")
    ap.add_argument("--width", type=int, default=2,
                    help="parallel lanes for the obs pool (2 recommended)")
    ap.add_argument("--skip-obs-extras", action="store_true",
                    help="extend only obs_base (skip obs_mask/obs_traffic)")
    args = ap.parse_args()

    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        print("[p3p] keep-awake armed")

    if not args.start_now:
        wait_for_gpu()  # N=100 lane stays exclusive until both seeds finish

    obs_jobs = ["obs_base"] if args.skip_obs_extras else OBS_JOBS
    fails: list = []
    q: "queue.Queue" = queue.Queue()
    for name in obs_jobs:
        for seed in SEEDS:
            q.put((name, seed))
    print(f"[p3p] obs pool: {q.qsize()} units, {args.width}-wide "
          f"(safe: §6.4a uses no rolling-OR numbers)")
    threads = [threading.Thread(target=worker, args=(q, fails), daemon=True)
               for _ in range(args.width)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("[p3p] obs pool drained -> hstress_h4 EXCLUSIVE (S6.8 uses rolling-OR)")
    for seed in SEEDS:
        if not run_job("hstress_h4", JOBS["hstress_h4"], seed):
            fails.append(f"hstress_h4/{seed}")

    if fails:
        print(f"[p3p] FINISHED WITH FAILURES: {fails} — rerun this script to resume.")
    else:
        print("[p3p] all jobs complete. Next: /p3-upgrade (pool 3-seed stats, "
              "upgrade §6.8/§6.4a wording).")


if __name__ == "__main__":
    main()
