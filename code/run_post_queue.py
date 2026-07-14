#!/usr/bin/env python3
"""Post-queue follow-ups (launched after run_exclusive_queue.py drains).

Waits for the exclusive queue's ALL-DONE sentinel, then runs, serially on the
now-idle machine (both jobs contain the wall-clock rolling-OR arm ->
exclusive, golden rule 3; every step resumable):

  1. Dual-count H=4 seeds 13345 + 14345 (pilot_dualcount_hstress_h4_seed_*.json)
     -> upgrades the strict-rule sensitivity paragraph (Sec 5) from 1 seed to
     the full 3 seeds behind Table 12. (H=8 needs no replication: straddle is
     exactly zero at every arm/bucket, the rules coincide there.)
  2. run_wait_probe.py -- high-bucket wait-costs-time probe at 60 s/wait.

Launch DETACHED:
  Start-Process -WindowStyle Hidden python -ArgumentList "-u","run_post_queue.py"
    -WorkingDirectory $PWD -RedirectStandardOutput results\post_queue.log
    -RedirectStandardError results\post_queue.err.log
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
QUEUE_LOG = RESULTS / "exclusive_queue.log"

if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    print("[postq] keep-awake armed", flush=True)


def queue_state() -> str:
    try:
        text = QUEUE_LOG.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "waiting"
    if "ALL DONE" in text:
        return "done"
    if "stopping queue" in text:
        return "stopped"
    return "waiting"


while True:
    state = queue_state()
    if state == "done":
        print("[postq] queue ALL DONE - starting follow-ups", flush=True)
        break
    if state == "stopped":
        print("[postq] queue STOPPED EARLY (see exclusive_queue.log) - "
              "not starting follow-ups", flush=True)
        sys.exit(1)
    print("[postq] waiting on exclusive queue...", flush=True)
    time.sleep(900)

time.sleep(120)  # let the queue process fully exit

# -- 1. dual-count H=4, remaining seeds ------------------------------------
for seed in (13345, 14345):
    out = RESULTS / f"pilot_dualcount_hstress_h4_seed_{seed}.json"
    try:
        if len(json.loads(out.read_text(encoding="utf-8")).get("buckets", {})) >= 3:
            print(f"[postq] dualcount H4 seed {seed}: already complete - skip",
                  flush=True)
            continue
    except (OSError, json.JSONDecodeError):
        pass
    cmd = [sys.executable, "-u", str(HERE / "scenario_bucket_eval_v2.py"),
           "--base-seed", str(seed),
           "--horizon-hours", "4.0",
           "--save-json", str(out)]
    log = RESULTS / f"pilot_dualcount_hstress_h4_seed_{seed}.log"
    print(f"[postq] dualcount H4 seed {seed}: starting (log: {log.name})",
          flush=True)
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as lf:
        rc = subprocess.run(cmd, cwd=HERE, stdout=lf,
                            stderr=subprocess.STDOUT).returncode
    print(f"[postq] dualcount H4 seed {seed}: "
          f"{'OK' if rc == 0 else f'FAILED rc={rc}'} "
          f"after {(time.time() - t0) / 60:.0f} min", flush=True)

# -- 2. wait-costs-time probe ----------------------------------------------
rc = subprocess.run([sys.executable, "-u", str(HERE / "run_wait_probe.py")],
                    cwd=HERE).returncode
print(f"[postq] wait probe rc={rc}", flush=True)

print("[postq] ALL FOLLOW-UPS DONE - next: extend the Sec 5 sensitivity "
      "paragraph to 3 seeds (analyze_dualcount_pilot.py) and read the wait "
      "probe.", flush=True)
