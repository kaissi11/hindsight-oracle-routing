#!/usr/bin/env python3
"""Review-experiment queue (fix-plan Problems 5, 6, 8, 19) - one GPU, run
everything sequentially, resumable, safe to start while the rush-hour run is
still going (it WAITS for cyc_msa4 to finish before touching the GPU).

  python run_review_experiments.py                 # wait for GPU, run all
  python run_review_experiments.py --start-now     # skip the GPU wait
  python run_review_experiments.py --only obs_base dec_msa4
  python run_review_experiments.py --list

Every job uses the SAME settings as the Damascus control suite
(osrm_s5, seed 12345: OSRM pool, N=20, 4 instances/episode, 40 episodes,
lookahead K=8) so its episodes pair 1:1 with the control's - identical
instances and pre-sampled disruption schedules. Monitor with:

  python watch_paper_progress.py            (once)
  python watch_paper_progress.py --watch 120  (refresh loop)

Jobs (priority order - each ~ one evening on the RTX 3060; MSA jobs longer):
  obs_base / obs_mask / obs_traffic  observability ablation (sec.6.4a): what the
                                     policy SEES (env stays truthful); live
                                     control = osrm_s5 itself, no rerun needed
  dec_2opt / dec_msa4                online decoder strength under OU dynamics
                                     (answers the "weak strawman" objection)
  hstress_h6 / hstress_h4            horizon stress - releases the completion
                                     ceiling (delivered ≥97.5% of max at H=8h)
  budget_10ms / budget_100ms / budget_300ms
                                     rolling-OR budget sensitivity at v2
                                     (30 ms control = osrm_s5)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PY = sys.executable
SEED = 12345
BUCKETS = ("low", "medium", "high")

CONTROL = RESULTS / f"scenario_bucket_v2_osrm_s5_seed_{SEED}.json"
RUSHHOUR_GATE = RESULTS / f"scenario_bucket_v2_cyc_msa4_seed_{SEED}.json"

# name -> extra CLI args (base command is identical to the osrm_s5 control)
JOBS: dict[str, list[str]] = {
    "obs_base":     ["--policy-matrix-mode", "base"],
    "obs_mask":     ["--policy-matrix-mode", "mask_only"],
    "obs_traffic":  ["--policy-matrix-mode", "traffic_only"],
    "dec_2opt":     ["--lookahead-2opt"],
    "dec_msa4":     ["--lookahead-scenarios", "4"],
    "hstress_h6":   ["--horizon-hours", "6"],
    "hstress_h4":   ["--horizon-hours", "4"],
    "budget_10ms":  ["--ortools-time-limit-ms", "10"],
    "budget_100ms": ["--ortools-time-limit-ms", "100"],
    "budget_300ms": ["--ortools-time-limit-ms", "300"],
}


def out_json(name: str) -> Path:
    return RESULTS / f"scenario_bucket_v2_{name}_seed_{SEED}.json"


def buckets_done(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return sum(1 for b in BUCKETS if b in d.get("buckets", {}))
    except (json.JSONDecodeError, OSError):
        return 0  # mid-write


def wait_for_gpu() -> None:
    """Block until the rush-hour MSA run has written all three buckets."""
    if buckets_done(RUSHHOUR_GATE) >= 3:
        return
    print(f"[queue] waiting for rush-hour run to finish "
          f"({RUSHHOUR_GATE.name}: {buckets_done(RUSHHOUR_GATE)}/3 buckets)...")
    while buckets_done(RUSHHOUR_GATE) < 3:
        time.sleep(120)
    print("[queue] rush-hour run complete - GPU free. "
          "Remember: fill sec.6.7 per RUSHHOUR_TRACKING.md (python check_rushhour.py).")
    time.sleep(30)  # let the process exit and release VRAM


def run_job(name: str, extra: list[str]) -> bool:
    dst = out_json(name)
    if buckets_done(dst) >= 3:
        print(f"[queue] {name}: already complete - skip")
        return True
    cmd = [PY, "-u", str(HERE / "scenario_bucket_eval_v2.py"),
           "--base-seed", str(SEED),
           "--save-json", str(dst), *extra]
    log = RESULTS / f"review_run_{name}.log"
    print(f"[queue] {name}: starting -> {dst.name}  (log: {log.name})")
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as lf:
        lf.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        lf.flush()
        rc = subprocess.run(cmd, cwd=HERE, stdout=lf, stderr=subprocess.STDOUT).returncode
    mins = (time.time() - t0) / 60
    status = "OK" if rc == 0 and buckets_done(dst) >= 3 else f"FAILED (rc={rc})"
    print(f"[queue] {name}: {status} after {mins:.0f} min")
    return status == "OK"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(JOBS), help="run only these jobs")
    ap.add_argument("--start-now", action="store_true",
                    help="skip waiting for the rush-hour run (GPU already free)")
    ap.add_argument("--list", action="store_true", help="show job status and exit")
    args = ap.parse_args()

    names = args.only or list(JOBS)

    if args.list:
        for n in JOBS:
            print(f"  {n:14s} {buckets_done(out_json(n))}/3 buckets  "
                  f"{'<- selected' if n in names else ''}")
        return

    if not CONTROL.exists():
        print(f"[queue] WARNING: control suite {CONTROL.name} missing - "
              "paired analysis against osrm_s5 will not be possible.")

    if sys.platform == "win32":
        # prevent IDLE sleep while the queue runs (cannot block manual
        # sleep / lid close - leave the machine on overnight!)
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        print("[queue] keep-awake armed (idle sleep blocked while queue runs)")

    if not args.start_now:
        wait_for_gpu()

    failures = [n for n in names if not run_job(n, JOBS[n])]
    print("\n[queue] finished.",
          f"failures: {failures}" if failures else "all jobs OK.")
    print("[queue] next: python watch_paper_progress.py  (paired deltas vs control)")


if __name__ == "__main__":
    main()
