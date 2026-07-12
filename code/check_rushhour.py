#!/usr/bin/env python3
"""Rush-hour falsification (paper §6.7) — one-command progress + result checker.

Usage:  python check_rushhour.py

While the runs are going: prints per-run progress (buckets/episodes done, log tail).
When BOTH runs are complete: computes the paired comparison the hypothesis is
about — v1 lookahead with MSA scenario scoring (anticipates the cycle) minus
v1 lookahead with frozen-matrix scoring (cannot anticipate) — per bucket, with
bootstrap 95% CI and Wilcoxon p, plus each run's look-vs-repair for context.

Hypothesis (§6.7): under cyclical (non-Markovian) traffic, MSA > frozen.
  CONFIRMED  -> MSA − frozen > 0 with CI excluding 0 (any bucket, esp. high)
  NOT CONFIRMED at this amplitude -> CIs include 0 everywhere
Either way the result fills the single §6.7 sentence (see RUSHHOUR_TRACKING.md §4).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RUNS = {
    "frozen (cyc_plain)": RESULTS / "scenario_bucket_v2_cyc_plain_seed_12345.json",
    "MSA-4  (cyc_msa4)": RESULTS / "scenario_bucket_v2_cyc_msa4_seed_12345.json",
}
LOG = RESULTS / "stage5_run_cyc.log"
BUCKETS = ["low", "medium", "high"]


def bootstrap_ci(arr, n_boot=20000, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def progress():
    done = {}
    for name, path in RUNS.items():
        d = load(path)
        if d is None:
            print(f"  {name:18s} not started (no JSON yet)")
            done[name] = None
            continue
        bk = list(d["buckets"].keys())
        n_ep = sum(len(d["buckets"][b]["episodes"]) for b in bk)
        complete = len(bk) >= 3
        done[name] = d if complete else False
        print(f"  {name:18s} buckets {bk} ({n_ep} episodes){' -- COMPLETE' if complete else ' -- running'}")
    if LOG.exists():
        tail = LOG.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1]
        print(f"  log tail: {tail[:110]}")
    return done


def result(plain, msa):
    print("\n=== RESULT: MSA − frozen, v1 lookahead delivered (paired, n=40/bucket) ===")
    verdict_cells = []
    for b in BUCKETS:
        ep_p = plain["buckets"][b]["episodes"]
        ep_m = msa["buckets"][b]["episodes"]
        d = np.array([m["policy_v1_lookahead"]["delivered_mean"] - p["policy_v1_lookahead"]["delivered_mean"]
                      for m, p in zip(ep_m, ep_p)])
        mean, lo, hi = bootstrap_ci(d, seed=1)
        try:
            from scipy.stats import wilcoxon
            pval = 1.0 if np.allclose(d, 0) else float(wilcoxon(d, zero_method="wilcox").pvalue)
        except Exception:
            pval = float("nan")
        sig = "SIG" if (lo > 0 or hi < 0) else "ns"
        verdict_cells.append(lo > 0)
        print(f"  {b:7s} {mean:+.3f} [{lo:+.3f},{hi:+.3f}] p={pval:.2e}  [{sig}]")
        # context: each variant vs repair within its own run
        for tag, run in (("frozen", plain), ("MSA   ", msa)):
            s = run["buckets"][b]["summary"]["paired_v1look_minus_repair"]
            c = s["delivered_delta_ci95"]
            print(f"          {tag} look-repair {s['delivered_delta_mean']:+.3f} [{c[0]:+.3f},{c[1]:+.3f}]")
    print("\n=== VERDICT ===")
    if any(verdict_cells):
        print("  PREDICTION CONFIRMED: scenario-scored (anticipating) lookahead beats the")
        print("  frozen-matrix lookahead under cyclical dynamics -> fill outcome (a) in RUSHHOUR_TRACKING.md section 4.")
    else:
        print("  NOT CONFIRMED at amplitude 0.4: MSA ties frozen under the cycle")
        print("  -> fill outcome (b) in RUSHHOUR_TRACKING.md section 4 (still a reportable, honest result).")


if __name__ == "__main__":
    print("Rush-hour falsification status (paper section 6.7)\n")
    done = progress()
    runs = list(done.values())
    if all(isinstance(r, dict) for r in runs):
        result(runs[0], runs[1])
    else:
        print("\n  (result computed automatically once BOTH runs show COMPLETE)")
