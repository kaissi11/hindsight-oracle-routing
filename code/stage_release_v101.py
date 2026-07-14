#!/usr/bin/env python3
"""Stage files for the v1.0.1 Zenodo/GitHub package release.

Copies completed Phase-1 result JSONs from 01_paper1/code/results into
03_github_zenodo/package/code/results/, then prints a checklist of what is
ready vs still missing. Does NOT commit/tag/push (author must confirm).

  python stage_release_v101.py          # dry-run status
  python stage_release_v101.py --copy   # copy present files into the package
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "results"
PKG = HERE.parents[1] / "03_github_zenodo" / "package" / "code" / "results"

# Files that must land in the package for v1.0.1 (runbook §5).
REQUIRED = [
    "scenario_bucket_v2_n100_s5_seed_15345.json",
    "scenario_bucket_v2_n100_s5_seed_16345.json",
    "scenario_bucket_v2_synth_s5_seed_14345.json",
    "stage2_aggregate_5seeds_n100_s5.json",
    "stage2_aggregate_5seeds_synth_s5.json",
    "kpi_aligned_rolling_or_h4_seed_12345.json",
    "kpi_aligned_rolling_or_h4_seed_13345.json",
    "kpi_aligned_rolling_or_h4_seed_14345.json",
    "generic_hindsight_repair_seed_13345.json",
    "generic_hindsight_repair_seed_14345.json",
    "generic_hindsight_aggregate.json",
    "oracle_k_sweep_osrm_h8_seed_12345.json",
    "oracle_k_sweep_osrm_h8_seed_13345.json",
    "oracle_k_sweep_osrm_h8_seed_14345.json",
    "oracle_k_sweep_hstress_h4_seed_12345.json",
    "oracle_k_sweep_hstress_h4_seed_13345.json",
    "oracle_k_sweep_hstress_h4_seed_14345.json",
    "pilot_dualcount_osrm_s5_seed_12345.json",
    "pilot_dualcount_hstress_h4_seed_12345.json",
    "pilot_dualcount_hstress_h4_seed_13345.json",
    "pilot_dualcount_hstress_h4_seed_14345.json",
    "probe_waitcost60_osrm_s5_seed_12345.json",
    "probe_waitcost60_hstress_h4_seed_12345.json",
    "equivalence_analysis.json",
]

# Code already synced during the plan; listed so the operator can verify.
CODE_SCRIPTS = [
    "run_dualcount_pilot.py",
    "analyze_dualcount_pilot.py",
    "run_exclusive_queue.py",
    "run_post_queue.py",
    "run_wait_probe.py",
    "aggregate_generic_hindsight.py",
    "oracle_k_sweep.py",
    "test_deadline_semantics.py",
    "scenario_bucket_eval_v2.py",
    "kpi_aligned_or_eval.py",
    "make_design_ledger.py",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", action="store_true",
                    help="copy present REQUIRED files into the package")
    args = ap.parse_args()

    if not PKG.exists():
        print(f"[release] package results dir missing: {PKG}")
        return 1

    present, missing = [], []
    for name in REQUIRED:
        src = SRC / name
        if src.exists() and src.stat().st_size > 100:
            present.append(name)
        else:
            missing.append(name)

    print(f"[release] present {len(present)}/{len(REQUIRED)}")
    for n in present:
        print(f"  OK  {n}  sha={sha256(SRC / n)}")
    print(f"[release] missing {len(missing)}/{len(REQUIRED)}")
    for n in missing:
        print(f"  --  {n}")

    pkg_code = PKG.parent
    print("[release] code scripts in package:")
    for n in CODE_SCRIPTS:
        p = pkg_code / n
        print(f"  {'OK' if p.exists() else 'MISSING'}  {n}")

    if args.copy:
        n_copied = 0
        for name in present:
            dst = PKG / name
            shutil.copy2(SRC / name, dst)
            n_copied += 1
            print(f"[release] copied {name}")
        print(f"[release] copied {n_copied} files -> {PKG}")
        if missing:
            print("[release] NOT ready to tag: still missing files above")
            return 2
        print("[release] all REQUIRED files present — next: README/CITATION "
              "bump, commit, tag v1.0.1 (see POST_COMPUTE_RUNBOOK §5)")
        return 0

    print("[release] dry-run only (pass --copy when ready)")
    return 0 if not missing else 2


if __name__ == "__main__":
    sys.exit(main())
