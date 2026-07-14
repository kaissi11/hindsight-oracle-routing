#!/usr/bin/env python3
"""Extend the generic-hindsight control (Table 3c) from 1 seed to 3.

Runs generic_hindsight_eval.py for seeds 13345 and 14345 against the matching
recorded osrm_s5 reference suites, sequentially, resumable. CPU-only, no
wall-clock-budgeted arm (repair 2-opt runs to convergence) -- safe to run in
parallel with other non-rolling-OR jobs. Aggregation happens at write-in time.

  python run_generic_extra_seeds.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SEEDS = [13345, 14345]


def main() -> None:
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        print("[GH-EXTRA] keep-awake armed", flush=True)

    for seed in SEEDS:
        out = RESULTS / f"generic_hindsight_repair_seed_{seed}.json"
        try:
            if json.loads(out.read_text(encoding="utf-8")).get("complete"):
                print(f"[GH-EXTRA] seed {seed}: already complete - skip",
                      flush=True)
                continue
        except (OSError, json.JSONDecodeError):
            pass
        cmd = [sys.executable, "-u", str(HERE / "generic_hindsight_eval.py"),
               "--base-seed", str(seed),
               "--reference",
               f"results/scenario_bucket_v2_osrm_s5_seed_{seed}.json",
               "--output", str(out)]
        log = RESULTS / f"generic_hindsight_seed_{seed}.log"
        print(f"[GH-EXTRA] seed {seed}: starting (log: {log.name})", flush=True)
        t0 = time.time()
        with open(log, "a", encoding="utf-8") as lf:
            rc = subprocess.run(
                cmd, cwd=HERE, stdout=lf, stderr=subprocess.STDOUT).returncode
        print(f"[GH-EXTRA] seed {seed}: {'OK' if rc == 0 else f'FAILED rc={rc}'} "
              f"after {(time.time() - t0) / 60:.0f} min", flush=True)

    print("[GH-EXTRA] done. Next: pool 3 seeds at write-in time "
          "(Table 3c, abstract genericity clause, design ledger).", flush=True)


if __name__ == "__main__":
    main()
