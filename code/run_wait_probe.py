#!/usr/bin/env python3
"""Wait-costs-time probe (deadline-semantics follow-up; exploratory).

The dual-count pilot found zero-cost wait steps are common (~16-70/route).
This probe reruns the two headline Damascus suites' HIGH bucket at one seed
with waiting charged 60 s of horizon per wait step (--wait-cost-sec 60,
a dispatcher re-poll interval), to check whether any headline ordering is
an artifact of free waiting:

  * osrm_s5     high bucket, H=8  (loose-horizon regime)
  * hstress_h4  high bucket, H=4  (binding-horizon regime)

Contains a rolling-OR arm -> EXCLUSIVE machine (golden rule 3); run only
after the main queue drains. Outputs:
results/probe_waitcost60_{osrm_s5,hstress_h4}_seed_12345.json

  python run_wait_probe.py
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
    print("[waitprobe] keep-awake armed")

for tag, extra in SUITES:
    out = RESULTS / f"probe_waitcost60_{tag}_seed_12345.json"
    try:
        if len(json.loads(out.read_text(encoding="utf-8")).get("buckets", {})) >= 1:
            print(f"[waitprobe] {tag}: already complete - skip")
            continue
    except (OSError, json.JSONDecodeError):
        pass
    cmd = [sys.executable, "-u", str(HERE / "scenario_bucket_eval_v2.py"),
           "--base-seed", "12345",
           "--buckets", "high",
           "--wait-cost-sec", "60",
           "--save-json", str(out)] + extra
    log = RESULTS / f"probe_waitcost60_{tag}.log"
    print(f"[waitprobe] {tag}: starting (log: {log.name})")
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as lf:
        rc = subprocess.run(cmd, cwd=HERE, stdout=lf, stderr=subprocess.STDOUT).returncode
    print(f"[waitprobe] {tag}: {'OK' if rc == 0 else f'FAILED rc={rc}'} "
          f"after {(time.time() - t0) / 60:.0f} min")

print("[waitprobe] done. Compare per-arm delivered/time vs the recorded high "
      "bucket (paired by episode index) and vs pilot_dualcount_* wait counts.")
