#!/usr/bin/env python3
"""Stage 5 suite runner — chains the 5-seed suites in priority order.

Priority (see STAGE5_PLAN.md / EXTERNAL_REVIEW_2026-07-02.md):
  1. osrm_s5    (b) Damascus OSRM, N=20  — decides the decision gate
  2. synth_s5   (a) synthetic v2 dynamics, N=20 — in-distribution corroboration
  3. n100_s5    (d) synthetic N=100 — where the effect is largest (also
                upgrades the old n=50 caveat to full power n=200)
  4. london_s5  (c) London OSRM — cross-city confirmation

RESUMABLE: a seed whose output JSON already exists is skipped, so the script
can be stopped/restarted at any time (power cut, interruption) and every
completed variant is a complete, quotable table. After each variant it runs
the aggregator and regenerates the figures, then prints the gate pairs.

Usage:
  python run_stage5_suites.py                 # all four, priority order
  python run_stage5_suites.py --only osrm_s5  # a single suite
  python run_stage5_suites.py --dry-run       # show what would run
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
SEEDS = [12345, 13345, 14345, 15345, 16345]
PY = sys.executable

VARIANTS = {
    "osrm_s5": dict(
        pool="results/osrm_instance_pool/pool.npz", n_nodes=20, num_instances=4, n_episodes=40),
    "synth_s5": dict(
        pool="", n_nodes=20, num_instances=4, n_episodes=40),
    "n100_s5": dict(
        pool="", n_nodes=100, num_instances=2, n_episodes=40),
    "london_s5": dict(
        pool="results/osrm_instance_pool_london/pool.npz", n_nodes=20, num_instances=4, n_episodes=40),
    # OPTIONAL journal-tier extra ("fuse b with d"): Damascus real roads at
    # N=100. Not in the default ORDER — it fills no [S5-*] placeholder and
    # needs a NEW pool first (Damascus OSRM backend up, then:
    #   python build_osrm_instance_pool.py --osrm-url http://localhost:5000 \
    #     --core-size 200 --n-nodes 100 --n-instances 160 --seed 556 \
    #     --out-dir results/osrm_instance_pool_n100
    # ). Run with: python run_stage5_suites.py --only osrm_n100_s5
    "osrm_n100_s5": dict(
        pool="results/osrm_instance_pool_n100/pool.npz", n_nodes=100, num_instances=2, n_episodes=40),
}
ORDER = ["osrm_s5", "synth_s5", "n100_s5", "london_s5"]


def run(cmd: list[str], log_path: Path, dry: bool) -> int:
    print(f"  $ {' '.join(cmd)}")
    if dry:
        return 0
    with open(log_path, "a", encoding="utf-8") as log:
        return subprocess.call(cmd, cwd=str(HERE), stdout=log, stderr=subprocess.STDOUT)


def print_gate(variant: str) -> None:
    agg = RESULTS / f"stage2_aggregate_5seeds_{variant}.json"
    if not agg.exists():
        return
    data = json.loads(agg.read_text(encoding="utf-8"))
    print(f"\n===== GATE SUMMARY [{variant}] (delivered deltas, 95% CI, Wilcoxon p) =====")
    for pair in ["v2look_minus_repair", "v2look_minus_rolling_or",
                 "v2look_minus_v2x8", "v2look_minus_v1look", "v2x8_minus_repair"]:
        row = []
        for b in ["low", "medium", "high"]:
            pr = data["buckets"][b]["pairs"].get(pair)
            if pr is None:
                break
            lo, hi = pr["delivered_delta_ci95"]
            row.append(f"{b}: {pr['delivered_delta_mean']:+.3f} [{lo:+.3f},{hi:+.3f}] p={pr['delivered_wilcoxon_p']:.1e}")
        if row:
            print(f"  {pair:26s} " + " | ".join(row))
    print("  (v2look_minus_v2x8 = oracle gap; v2look_minus_repair = the new decision gate)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(VARIANTS), default=None,
                    help="run a single variant instead of all four")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    todo = [args.only] if args.only else ORDER
    t_start = time.time()

    for variant in todo:
        cfg = VARIANTS[variant]
        print(f"\n########## Suite {variant} ##########")
        if cfg["pool"] and not (HERE / cfg["pool"]).exists():
            print(f"  [SKIP] pool missing: {cfg['pool']} — build it first (see comments in this file)")
            continue
        log_path = RESULTS / f"stage5_run_{variant}.log"
        for seed in SEEDS:
            out = RESULTS / f"scenario_bucket_v2_{variant}_seed_{seed}.json"
            if out.exists():
                print(f"  [skip] seed {seed} — {out.name} already exists")
                continue
            cmd = [PY, "-u", "scenario_bucket_eval_v2.py",
                   "--instance-pool", cfg["pool"],
                   "--n-nodes", str(cfg["n_nodes"]),
                   "--num-instances", str(cfg["num_instances"]),
                   "--n-episodes", str(cfg["n_episodes"]),
                   "--base-seed", str(seed),
                   "--save-json", f"results/scenario_bucket_v2_{variant}_seed_{seed}.json"]
            t0 = time.time()
            rc = run(cmd, log_path, args.dry_run)
            if rc != 0:
                print(f"  [FAIL] seed {seed} exited {rc} — see {log_path.name}; stopping this variant")
                break
            print(f"  [done] seed {seed} in {(time.time() - t0) / 3600:.2f} h")
        else:
            if not args.dry_run:
                run([PY, "aggregate_stage2_seeds.py", variant], log_path, False)
                run([PY, "make_stage_figures.py"], log_path, False)
                print_gate(variant)

    print(f"\nAll requested suites processed in {(time.time() - t_start) / 3600:.2f} h total.")


if __name__ == "__main__":
    main()
