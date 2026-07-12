#!/usr/bin/env python3
"""Extend the single-seed review ablations that carry abstract-level claims
(REVIEW_FABLE_2026-07-09.md item P3) from 1 seed to 3: hstress_h4 first (the
sec.6.8 inversion), then obs_base / obs_mask / obs_traffic (the sec.6.4a
feasibility-observability decomposition). Waits for the N=100 extra-seed runs
(run_n100_extra_seeds.py) to finish before touching the GPU; sequential on
purpose (rolling-OR is wall-clock-budgeted). Resumable: skips complete JSONs.

  python run_p3_extra_seeds.py              # wait for N=100 runs, then go
  python run_p3_extra_seeds.py --start-now  # GPU already free

Pairing notes:
- obs_* at seed S pairs cross-run with the live control
  scenario_bucket_v2_osrm_s5_seed_S.json (controls exist for 13345/14345).
- hstress_h4 has a different horizon: within-run pairs only.
After all runs land: pool per-seed paired deltas (episode bootstrap per seed,
then across seeds); to fold into formal equivalence stats add the suites to
SUITES in equivalence_analysis.py (>=2 seeds) and rerun. Then update the
sec.6.8 / sec.6.4a wording per HANDOFF_AFTER_EXPERIMENTS.md sec.3 (single-seed
labels drop; 'consistent with' may upgrade once CIs pool).
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
SEEDS = [13345, 14345]
BUCKETS = ("low", "medium", "high")

# priority order per REVIEW_FABLE_2026-07-09.md P3: hstress_h4 first, then obs_*
JOBS: dict[str, list[str]] = {
    "hstress_h4":  ["--horizon-hours", "4"],
    "obs_base":    ["--policy-matrix-mode", "base"],
    "obs_mask":    ["--policy-matrix-mode", "mask_only"],
    "obs_traffic": ["--policy-matrix-mode", "traffic_only"],
}

N100_GATES = [RESULTS / f"scenario_bucket_v2_n100_s5_seed_{s}.json" for s in SEEDS]


def buckets_done(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return sum(1 for b in BUCKETS if b in d.get("buckets", {}))
    except (json.JSONDecodeError, OSError):
        return 0  # mid-write


def wait_for_gpu() -> None:
    """Block until both N=100 extra-seed runs have written all three buckets."""
    while True:
        states = {g.name: buckets_done(g) for g in N100_GATES}
        if all(v >= 3 for v in states.values()):
            break
        print(f"[p3] waiting for N=100 runs: " +
              ", ".join(f"{k}: {v}/3" for k, v in states.items()))
        time.sleep(300)
    print("[p3] N=100 runs complete - GPU free. Remember the N=100 ripple: "
          "aggregate_stage2_seeds.py n100_s5 && equivalence_analysis.py && "
          "make_paper_assets.py, then Table 3 / A.4 / A.5 / limitation (6).")
    time.sleep(30)  # let the process exit and release VRAM


def run_job(name: str, extra: list[str], seed: int) -> bool:
    dst = RESULTS / f"scenario_bucket_v2_{name}_seed_{seed}.json"
    if buckets_done(dst) >= 3:
        print(f"[p3] {name} seed {seed}: already complete - skip")
        return True
    cmd = [PY, "-u", str(HERE / "scenario_bucket_eval_v2.py"),
           "--base-seed", str(seed),
           "--save-json", str(dst), *extra]
    log = RESULTS / f"p3_run_{name}_seed_{seed}.log"
    print(f"[p3] {name} seed {seed}: starting -> {dst.name}  (log: {log.name})")
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as lf:
        lf.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        lf.flush()
        rc = subprocess.run(cmd, cwd=HERE, stdout=lf, stderr=subprocess.STDOUT).returncode
    mins = (time.time() - t0) / 60
    status = "OK" if rc == 0 and buckets_done(dst) >= 3 else f"FAILED (rc={rc})"
    print(f"[p3] {name} seed {seed}: {status} after {mins:.0f} min")
    return status == "OK"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-now", action="store_true",
                    help="skip the N=100 gate (GPU already free)")
    ap.add_argument("--only", nargs="*", choices=sorted(JOBS),
                    help="run only these jobs")
    args = ap.parse_args()

    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        print("[p3] keep-awake armed")

    if not args.start_now:
        wait_for_gpu()

    jobs = {k: v for k, v in JOBS.items() if not args.only or k in args.only}
    failures = []
    for name, extra in jobs.items():          # job-major: finish hstress_h4 fully first
        for seed in SEEDS:
            if not run_job(name, extra, seed):
                failures.append(f"{name}/{seed}")

    if failures:
        print(f"[p3] FINISHED WITH FAILURES: {failures} - rerun this script to resume.")
    else:
        print("[p3] all jobs complete. Next: pool per-seed deltas (3 seeds now for "
              "hstress_h4 + obs_*), add suites to equivalence_analysis.py SUITES, "
              "then upgrade sec.6.8/sec.6.4a wording + limitations (8)/(9) per "
              "REVIEW_FABLE_2026-07-09.md P3.")


if __name__ == "__main__":
    main()
