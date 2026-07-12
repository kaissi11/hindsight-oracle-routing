#!/usr/bin/env python3
"""Extend the N=100 online suite from 1 seed to 3 (the 'at least 3 seeds'
floor). Runs seeds 13345 and 14345 with the exact n100_s5 flags,
sequentially, resumable; then re-aggregates so Table 3 / A.4 / Table A.5 can
be refreshed (aggregate_stage2_seeds.py n100_s5 && equivalence_analysis.py).

  python run_n100_extra_seeds.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SEEDS = [13345, 14345]

if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    print("[n100] keep-awake armed")

for seed in SEEDS:
    out = RESULTS / f"scenario_bucket_v2_n100_s5_seed_{seed}.json"
    try:
        if len(json.loads(out.read_text(encoding="utf-8")).get("buckets", {})) >= 3:
            print(f"[n100] seed {seed}: already complete - skip")
            continue
    except (OSError, json.JSONDecodeError):
        pass
    cmd = [sys.executable, "-u", str(HERE / "scenario_bucket_eval_v2.py"),
           "--instance-pool", "", "--n-nodes", "100", "--num-instances", "2",
           "--n-episodes", "40", "--base-seed", str(seed),
           "--save-json", str(out)]
    log = RESULTS / f"n100_extra_seed_{seed}.log"
    print(f"[n100] seed {seed}: starting (log: {log.name})")
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as lf:
        rc = subprocess.run(cmd, cwd=HERE, stdout=lf, stderr=subprocess.STDOUT).returncode
    print(f"[n100] seed {seed}: {'OK' if rc == 0 else f'FAILED rc={rc}'} "
          f"after {(time.time() - t0) / 60:.0f} min")

print("[n100] done. Next: python aggregate_stage2_seeds.py n100_s5 && "
      "python equivalence_analysis.py && python make_paper_assets.py, then "
      "refresh Table 3 N=100 rows + A.4 + Table A.5 (seed counts 3, n=120).")
