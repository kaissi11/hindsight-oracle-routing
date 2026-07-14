#!/usr/bin/env python3
"""Wave-1 dual-count pilot (review P0.1 deadline-semantics decision gate).

Reruns the two headline Damascus suites at seed 12345 with the dual-count
environment (delivered = departure-cutoff, delivered_strict = arrival <= H,
plus late-arrival and wait-step logging):

  * osrm_s5     -- Damascus OSRM, H=8, look-8 arm enabled (paper Table 3);
  * hstress_h4  -- Damascus OSRM, H=4 (paper Table 4a inversion).

Flags replicate the recorded suites exactly (same seeds, same checkpoints,
same pool), so every non-wall-clock arm must reproduce the recorded
delivered/time values bit-identically (the counters are write-only
bookkeeping; test_deadline_semantics.py locks this). Only rolling_or can
drift, being wall-clock budgeted -- run this on an EXCLUSIVE machine.

Outputs go to results/pilot_dualcount_<suite>_seed_12345.json (the recorded
JSONs are never overwritten). Analyze with analyze_dualcount_pilot.py.

  python run_dualcount_pilot.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

SUITES = [
    ("osrm_s5", []),
    ("hstress_h4", ["--horizon-hours", "4.0"]),
]

if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    print("[pilot] keep-awake armed")

for tag, extra in SUITES:
    out = RESULTS / f"pilot_dualcount_{tag}_seed_12345.json"
    try:
        if len(json.loads(out.read_text(encoding="utf-8")).get("buckets", {})) >= 3:
            print(f"[pilot] {tag}: already complete - skip")
            continue
    except (OSError, json.JSONDecodeError):
        pass
    cmd = [sys.executable, "-u", str(HERE / "scenario_bucket_eval_v2.py"),
           "--base-seed", "12345",
           "--save-json", str(out)] + extra
    log = RESULTS / f"pilot_dualcount_{tag}.log"
    print(f"[pilot] {tag}: starting (log: {log.name})")
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as lf:
        rc = subprocess.run(cmd, cwd=HERE, stdout=lf, stderr=subprocess.STDOUT).returncode
    print(f"[pilot] {tag}: {'OK' if rc == 0 else f'FAILED rc={rc}'} "
          f"after {(time.time() - t0) / 60:.0f} min")

print("[pilot] done. Next: python analyze_dualcount_pilot.py")
