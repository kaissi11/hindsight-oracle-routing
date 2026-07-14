#!/usr/bin/env python3
"""Waves 2-3 exclusive-machine queue (final pre-submission plan, 2026-07-12).

Chains, in order, everything that still needs the machine to itself
(every step is resumable -- outputs that already exist are skipped):

  0. dual-count pilot gate     analyze_dualcount_pilot.py (prints verdict)
  1. Wave 2: n100_s5           online seeds 15345, 16345 (rolling-OR arm)
  2.         aggregate n100_s5 (5 seeds) + equivalence re-check
  3. Wave 3: synth_s5          3rd seed 14345 (rolling-OR arm)
  4.         aggregate synth_s5 (3 seeds)
  5. Wave 3: KPI-aligned OR at H=4, seeds 12345/13345/14345 (wall-clock OR)
  6. oracle-K sweep            osrm_h8 + hstress_h4, 3 seeds (GPU-only arm,
                               contention-safe, but machine is free by then)
  7. equivalence_analysis.py + all figure/table regeneration

Launch DETACHED on an otherwise idle machine (golden rule 3):

  Start-Process -WindowStyle Hidden python -ArgumentList
    "-u","run_exclusive_queue.py" -WorkingDirectory $PWD
    -RedirectStandardOutput results\\exclusive_queue.log
    -RedirectStandardError results\\exclusive_queue.err.log

Progress: python watch_paper_progress.py   (or tail the logs)
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PY = sys.executable

if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    print("[queue] keep-awake armed", flush=True)


def sh(cmd: list[str], log_name: str) -> int:
    log = RESULTS / log_name
    print(f"[queue] $ {' '.join(str(c) for c in cmd)}  (log: {log.name})", flush=True)
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as lf:
        rc = subprocess.run(cmd, cwd=str(HERE), stdout=lf,
                            stderr=subprocess.STDOUT).returncode
    print(f"[queue]   -> rc={rc} after {(time.time() - t0) / 60:.0f} min", flush=True)
    return rc


def json_complete(path: Path, min_buckets: int = 3) -> bool:
    """Complete = all buckets present AND the script's own complete flag
    (absent in scenario_bucket_eval_v2 outputs -> bucket count decides)."""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return len(d.get("buckets", {})) >= min_buckets and bool(d.get("complete", True))


def wait_for_pilot() -> None:
    """Block until both pilot JSONs are complete (run_dualcount_pilot.py)."""
    targets = [RESULTS / "pilot_dualcount_osrm_s5_seed_12345.json",
               RESULTS / "pilot_dualcount_hstress_h4_seed_12345.json"]
    while True:
        states = {t.name: json_complete(t) for t in targets}
        if all(states.values()):
            print("[queue] pilot JSONs complete - proceeding", flush=True)
            return
        print(f"[queue] waiting on pilot: {states}", flush=True)
        time.sleep(600)


def main() -> None:
    wait_for_pilot()
    time.sleep(120)  # let the pilot process fully exit before claiming the GPU

    # -- 0. pilot decision gate (prints KEEP-WORDING or FLIPS-FOUND) ---------
    sh([PY, "-u", "analyze_dualcount_pilot.py"], "exclusive_queue_gate.log")
    print("[queue] pilot gate output in exclusive_queue_gate.log "
          "(FLIPS-FOUND would need author sign-off on wording; queue continues "
          "-- the remaining suites are needed under either branch)", flush=True)

    # -- 1. Wave 2: n100_s5 seeds 15345, 16345 (serial; rolling-OR arm) ------
    for seed in (15345, 16345):
        out = RESULTS / f"scenario_bucket_v2_n100_s5_seed_{seed}.json"
        if json_complete(out):
            print(f"[queue] n100_s5 seed {seed}: already complete - skip", flush=True)
            continue
        rc = sh([PY, "-u", "scenario_bucket_eval_v2.py",
                 "--instance-pool", "",
                 "--n-nodes", "100", "--num-instances", "2",
                 "--n-episodes", "40",
                 "--base-seed", str(seed),
                 "--save-json", f"results/scenario_bucket_v2_n100_s5_seed_{seed}.json"],
                f"stage5_run_n100_s5_seed_{seed}.log")
        if rc != 0:
            print(f"[queue] n100_s5 seed {seed} FAILED - stopping queue", flush=True)
            return

    # -- 2. aggregate n100_s5 at 5 seeds ------------------------------------
    sh([PY, "aggregate_stage2_seeds.py", "n100_s5"], "exclusive_queue_agg.log")

    # -- 3. Wave 3: synth_s5 3rd seed ----------------------------------------
    out = RESULTS / "scenario_bucket_v2_synth_s5_seed_14345.json"
    if json_complete(out):
        print("[queue] synth_s5 seed 14345: already complete - skip", flush=True)
    else:
        rc = sh([PY, "-u", "scenario_bucket_eval_v2.py",
                 "--instance-pool", "",
                 "--n-nodes", "20", "--num-instances", "4",
                 "--n-episodes", "40",
                 "--base-seed", "14345",
                 "--save-json", "results/scenario_bucket_v2_synth_s5_seed_14345.json"],
                "stage5_run_synth_s5_seed_14345.log")
        if rc != 0:
            print("[queue] synth_s5 seed 14345 FAILED - stopping queue", flush=True)
            return

    # -- 4. aggregate synth_s5 ------------------------------------------------
    sh([PY, "aggregate_stage2_seeds.py", "synth_s5"], "exclusive_queue_agg.log")

    # -- 5. Wave 3: completion-first rolling-OR at H=4, 3 seeds --------------
    for seed in (12345, 13345, 14345):
        out = RESULTS / f"kpi_aligned_rolling_or_h4_seed_{seed}.json"
        if json_complete(out):
            print(f"[queue] kpi-OR H4 seed {seed}: already complete - skip", flush=True)
            continue
        rc = sh([PY, "-u", "kpi_aligned_or_eval.py",
                 "--horizon-hours", "4.0",
                 "--base-seed", str(seed),
                 "--reference", f"results/scenario_bucket_v2_hstress_h4_seed_{seed}.json",
                 "--output", f"results/kpi_aligned_rolling_or_h4_seed_{seed}.json"],
                f"kpi_or_h4_seed_{seed}.log")
        if rc != 0:
            print(f"[queue] kpi-OR H4 seed {seed} FAILED - continuing "
                  "(exploratory control)", flush=True)

    # -- 6. oracle-K sweep (GPU sampling arm only) ----------------------------
    sh([PY, "-u", "oracle_k_sweep.py",
        "--suites", "osrm_h8", "hstress_h4",
        "--seeds", "12345", "13345", "14345",
        "--policies", "v1"],
       "oracle_k_sweep_run.log")

    # -- 7. statistics + figures + tables -------------------------------------
    sh([PY, "equivalence_analysis.py"], "exclusive_queue_agg.log")
    sh([PY, "make_stage_figures.py"], "exclusive_queue_figs.log")
    sh([PY, "make_paper_assets.py"], "exclusive_queue_figs.log")
    sh([PY, "make_fig_horizon.py"], "exclusive_queue_figs.log")
    sh([PY, "make_fig_pareto.py"], "exclusive_queue_figs.log")
    sh([PY, "make_fig_ksweep.py"], "exclusive_queue_figs.log")

    print("[queue] ALL DONE - next: review gate log, update master numbers "
          "(python watch_paper_progress.py), rebuild the bundle.", flush=True)


if __name__ == "__main__":
    main()
