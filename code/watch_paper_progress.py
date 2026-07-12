#!/usr/bin/env python3
"""One dashboard for everything still moving in the paper.

  python watch_paper_progress.py              # print once
  python watch_paper_progress.py --watch 120  # refresh every 120 s

Sections:
  1. Rush-hour falsification (sec.6.7): progress; paired MSA-frozen verdict when done.
  2. Review-experiment queue (run_review_experiments.py): per-job progress and,
     once finished, the headline paired deltas vs the osrm_s5 control.
  3. Build artifacts: modification times of draft / bundle / LaTeX / final PDFs
     (the paper sources are not part of this package, so those rows read
     "missing" here — the result sections above are the useful ones).

Read-only - safe to run any time, needs only numpy.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PAPER = HERE / "paper"
SEED = 12345
BUCKETS = ("low", "medium", "high")

CONTROL = RESULTS / f"scenario_bucket_v2_osrm_s5_seed_{SEED}.json"
CYC_PLAIN = RESULTS / f"scenario_bucket_v2_cyc_plain_seed_{SEED}.json"
CYC_MSA = RESULTS / f"scenario_bucket_v2_cyc_msa4_seed_{SEED}.json"

REVIEW_JOBS = ["obs_base", "obs_mask", "obs_traffic", "dec_2opt", "dec_msa4",
               "hstress_h6", "hstress_h4", "budget_10ms", "budget_100ms",
               "budget_300ms"]

ARTIFACTS = [
    PAPER / "paper_draft_v3_submission.md",
    PAPER / "bundle" / "paper_v3_bundle.pdf",
    PAPER / "latex" / "main.pdf",
    PAPER / "final_submission" / "paper_final.pdf",
    RESULTS / "equivalence_analysis.json",
]


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None  # mid-write


def episodes(d: dict, bucket: str, method: str) -> np.ndarray | None:
    b = d.get("buckets", {}).get(bucket)
    if not b:
        return None
    vals = [e[method]["delivered_mean"] for e in b["episodes"] if method in e]
    return np.array(vals) if vals else None


def boot_ci(x: np.ndarray, n=5000, seed=0) -> tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    means = x[rng.randint(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fmt(mean, lo, hi) -> str:
    star = " *" if lo > 0 or hi < 0 else ""
    return f"{mean:+.3f} [{lo:+.3f},{hi:+.3f}]{star}"


def within(d: dict, bucket: str, a: str, b: str) -> str:
    xa, xb = episodes(d, bucket, a), episodes(d, bucket, b)
    if xa is None or xb is None or len(xa) != len(xb):
        return "-"
    return fmt(*boot_ci(xa - xb))


def across(d1: dict, d2: dict, bucket: str, method: str) -> str:
    """Paired delta of one method between two runs (same seed → same episodes)."""
    x1, x2 = episodes(d1, bucket, method), episodes(d2, bucket, method)
    if x1 is None or x2 is None or len(x1) != len(x2):
        return "-"
    return fmt(*boot_ci(x1 - x2))


def progress(path: Path) -> str:
    d = load(path)
    if d is None:
        return "not started" if not path.exists() else "writing..."
    done = [b for b in BUCKETS if b in d.get("buckets", {})]
    n_ep = len(d["buckets"][done[-1]]["episodes"]) if done else 0
    return f"{len(done)}/3 buckets (last: {n_ep} ep)" if len(done) < 3 else "DONE"


def section_rushhour() -> None:
    print("== 1. Rush-hour falsification (sec.6.7) " + "=" * 42)
    print(f"  frozen  (cyc_plain): {progress(CYC_PLAIN)}")
    print(f"  MSA-4   (cyc_msa4) : {progress(CYC_MSA)}")
    p, m = load(CYC_PLAIN), load(CYC_MSA)
    if p and m:
        common = [b for b in BUCKETS if b in p.get("buckets", {}) and b in m.get("buckets", {})]
        confirmed = False
        for b in common:
            x1, x2 = episodes(m, b, "policy_v1_lookahead"), episodes(p, b, "policy_v1_lookahead")
            if x1 is None or x2 is None:
                continue
            mean, lo, hi = boot_ci(x1 - x2)
            confirmed |= lo > 0
            print(f"  MSA - frozen v1look, {b:6s}: {fmt(mean, lo, hi)}   "
                  f"(context: frozen look-repair {within(p, b, 'policy_v1_lookahead', 'repair_nn2opt')})")
        if len(common) == 3:
            print("  VERDICT: " + ("PREDICTION CONFIRMED - anticipation pays under cycles"
                                   if confirmed else
                                   "NOT CONFIRMED at amp 0.4 - matrix-sufficiency holds")
                  + "  [paper sec.6.7 updated with this result 2026-07-07 - no action needed]")


def job_summary(name: str, d: dict, ctrl: dict | None) -> None:
    for b in BUCKETS:
        if b not in d.get("buckets", {}):
            continue
        if name.startswith("obs_"):
            line = (f"oracle-repair {within(d, b, 'policy_v1_samplexN', 'repair_nn2opt')}  "
                    f"look-repair {within(d, b, 'policy_v1_lookahead', 'repair_nn2opt')}")
            if ctrl:
                line += f"  oracle vs live-ctrl {across(d, ctrl, b, 'policy_v1_samplexN')}"
        elif name.startswith("dec_"):
            line = (f"look-repair {within(d, b, 'policy_v1_lookahead', 'repair_nn2opt')}  "
                    f"gap look-oracle {within(d, b, 'policy_v1_lookahead', 'policy_v1_samplexN')}")
        elif name.startswith("hstress"):
            rep = episodes(d, b, "repair_nn2opt")
            ceil = f"repair mean {rep.mean():.2f}/19" if rep is not None else ""
            line = (f"{ceil}  look-repair {within(d, b, 'policy_v1_lookahead', 'repair_nn2opt')}  "
                    f"oracle-repair {within(d, b, 'policy_v1_samplexN', 'repair_nn2opt')}")
        else:  # budget_*
            line = (f"look-rollOR {within(d, b, 'policy_v1_lookahead', 'rolling_or')}  "
                    f"repair-rollOR {within(d, b, 'repair_nn2opt', 'rolling_or')}")
        print(f"      {b:6s} {line}")


def section_queue() -> None:
    print("\n== 2. Review-experiment queue (run_review_experiments.py) " + "=" * 20)
    ctrl = load(CONTROL)
    any_done = False
    for name in REVIEW_JOBS:
        path = RESULTS / f"scenario_bucket_v2_{name}_seed_{SEED}.json"
        st = progress(path)
        print(f"  {name:13s} {st}")
        d = load(path)
        if d and st == "DONE":
            any_done = True
            job_summary(name, d, ctrl)
    if not any_done:
        print("  (headline paired deltas appear here as each job finishes; "
              "* = 95% CI excludes 0)")


def section_live() -> None:
    """What is running RIGHT NOW: any results/*.log written in the last 15 min,
    with its last progress line."""
    print("\n== 0. Live now " + "=" * 64)
    now = time.time()
    active = sorted((p for p in RESULTS.glob("*.log")
                     if now - p.stat().st_mtime < 15 * 60),
                    key=lambda p: p.stat().st_mtime, reverse=True)
    if not active:
        print("  no log activity in the last 15 min - nothing is running "
              "(or a run just crashed: check results/review_queue_err.log)")
    for p in active:
        try:
            lines = [ln.strip() for ln in
                     p.read_text(encoding="utf-8", errors="replace").splitlines()
                     if ln.strip()]
            tail = lines[-1][:110] if lines else "(empty)"
            tail = tail.encode("ascii", "replace").decode()  # console-safe
        except OSError:
            tail = "(unreadable)"
        age = int(now - p.stat().st_mtime)
        print(f"  {p.name}  ({age}s ago)")
        print(f"    {tail}")


def section_artifacts() -> None:
    print("\n== 3. Build artifacts " + "=" * 57)
    for p in ARTIFACTS:
        stamp = (time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))
                 if p.exists() else "missing")
        print(f"  {stamp:17s}  {p.relative_to(HERE)}")


def dashboard() -> None:
    print(f"\n######## paper progress - {time.strftime('%Y-%m-%d %H:%M:%S')} ########")
    section_live()
    section_rushhour()
    section_queue()
    section_artifacts()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, metavar="SEC",
                    help="refresh every SEC seconds (Ctrl+C to stop)")
    args = ap.parse_args()
    dashboard()
    while args.watch:
        time.sleep(args.watch)
        dashboard()


if __name__ == "__main__":
    main()
