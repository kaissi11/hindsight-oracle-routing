#!/usr/bin/env python3
"""Aggregate Stage 1 paired results across suite seeds.

Pools episode-level paired deltas (policy_samplexN minus each baseline) across
all seeds per bucket, reports per-seed means and pooled bootstrap 95% CIs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

# Usage: python aggregate_stage1_seeds.py [synthetic|osrm]
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
SEEDS = [12345, 13345, 14345, 15345, 16345]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
METHODS = ["policy_greedy", "policy_samplexN", "rolling_or", "repair_nn2opt", "reactive_nn"]
PAIRS = {
    "policy_samplexN_minus_repair": ("policy_samplexN", "repair_nn2opt"),
    "policy_samplexN_minus_rolling_or": ("policy_samplexN", "rolling_or"),
    "policy_greedy_minus_repair": ("policy_greedy", "repair_nn2opt"),
    "repair_minus_rolling_or": ("repair_nn2opt", "rolling_or"),
    "repair_minus_reactive_nn": ("repair_nn2opt", "reactive_nn"),
}


def bootstrap_ci(arr: np.ndarray, n_boot: int = 20000, seed: int = 0):
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon_p(deltas: np.ndarray) -> float:
    """Wilcoxon signed-rank p (two-sided); 1.0 when all paired deltas are zero."""
    if np.allclose(deltas, 0.0):
        return 1.0
    return float(wilcoxon(deltas, zero_method="wilcox").pvalue)


def main() -> None:
    runs = {s: json.load(open(RESULTS_DIR / f"scenario_bucket_repair_{VARIANT}_seed_{s}.json")) for s in SEEDS}
    buckets = list(next(iter(runs.values()))["buckets"].keys())
    out = {"variant": VARIANT, "seeds": SEEDS, "buckets": {}}

    for bucket in buckets:
        episodes = {s: runs[s]["buckets"][bucket]["episodes"] for s in SEEDS}
        pooled = {m: [] for m in METHODS}
        for s in SEEDS:
            for ep in episodes[s]:
                for m in METHODS:
                    pooled[m].append(ep[m])

        n = len(pooled["policy_samplexN"])
        bucket_out = {"n_episodes_pooled": n, "method_means": {}, "pairs": {}}
        print(f"\n== {bucket} (pooled n={n}) ==")
        for m in METHODS:
            dlv = float(np.mean([e["delivered_mean"] for e in pooled[m]]))
            tm = float(np.mean([e["time_mean"] for e in pooled[m]]))
            bucket_out["method_means"][m] = {"delivered_mean": dlv, "time_mean": tm}
            print(f"  {m:18s} delivered={dlv:.3f} time={tm:.1f}")

        for name, (ma, mb) in PAIRS.items():
            d_del = np.array([a["delivered_mean"] - b["delivered_mean"] for a, b in zip(pooled[ma], pooled[mb])])
            d_tim = np.array([a["time_mean"] - b["time_mean"] for a, b in zip(pooled[ma], pooled[mb])])
            dm, dlo, dhi = bootstrap_ci(d_del, seed=1)
            tm, tlo, thi = bootstrap_ci(d_tim, seed=2)
            wins = int((d_del > 1e-9).sum())
            losses = int((d_del < -1e-9).sum())
            per_seed = []
            for s in SEEDS:
                ds = np.mean([
                    ep[ma]["delivered_mean"] - ep[mb]["delivered_mean"] for ep in episodes[s]
                ])
                per_seed.append(round(float(ds), 4))
            p_del = wilcoxon_p(d_del)
            p_tim = wilcoxon_p(d_tim)
            bucket_out["pairs"][name] = {
                "delivered_delta_mean": dm,
                "delivered_delta_ci95": [dlo, dhi],
                "delivered_wilcoxon_p": p_del,
                "time_delta_mean": tm,
                "time_delta_ci95": [tlo, thi],
                "time_wilcoxon_p": p_tim,
                "delivered_win_tie_loss": [wins, n - wins - losses, losses],
                "per_seed_delivered_delta": per_seed,
            }
            print(f"  {name}")
            print(f"    delivered {dm:+.3f} CI[{dlo:+.3f},{dhi:+.3f}] p={p_del:.2e}  "
                  f"time {tm:+.1f} CI[{tlo:+.1f},{thi:+.1f}] p={p_tim:.2e}"
                  f"  W/T/L [{wins},{n - wins - losses},{losses}]")
            print(f"    per-seed delivered delta: {per_seed}")

        out["buckets"][bucket] = bucket_out

    save = RESULTS_DIR / f"stage1_aggregate_5seeds_{VARIANT}.json"
    save.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved: {save}")


if __name__ == "__main__":
    main()
