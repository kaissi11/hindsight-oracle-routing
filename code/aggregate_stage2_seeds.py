#!/usr/bin/env python3
"""Aggregate Stage 2 paired results across suite seeds (mirrors stage 1 aggregator)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
SEEDS = [12345, 13345, 14345, 15345, 16345]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
METHODS = ["policy_v2_greedy", "policy_v2_samplexN", "policy_v1_samplexN",
           "policy_v2_lookahead", "policy_v1_lookahead",
           "rolling_or", "repair_nn2opt", "reactive_nn"]
PAIRS = {
    "v2x8_minus_repair": ("policy_v2_samplexN", "repair_nn2opt"),
    "v2x8_minus_rolling_or": ("policy_v2_samplexN", "rolling_or"),
    "v2x8_minus_v1x8": ("policy_v2_samplexN", "policy_v1_samplexN"),
    "v1x8_minus_repair": ("policy_v1_samplexN", "repair_nn2opt"),
    "v1x8_minus_rolling_or": ("policy_v1_samplexN", "rolling_or"),
    "v2greedy_minus_repair": ("policy_v2_greedy", "repair_nn2opt"),
    "repair_minus_rolling_or": ("repair_nn2opt", "rolling_or"),
    "repair_minus_reactive_nn": ("repair_nn2opt", "reactive_nn"),
    # Stage 5 (online lookahead) pairs — skipped if the runs predate Stage 5.
    "v2look_minus_repair": ("policy_v2_lookahead", "repair_nn2opt"),
    "v2look_minus_rolling_or": ("policy_v2_lookahead", "rolling_or"),
    "v2look_minus_v2x8": ("policy_v2_lookahead", "policy_v2_samplexN"),
    "v2look_minus_v1look": ("policy_v2_lookahead", "policy_v1_lookahead"),
    "v1look_minus_repair": ("policy_v1_lookahead", "repair_nn2opt"),
    "v1look_minus_rolling_or": ("policy_v1_lookahead", "rolling_or"),
    "v1look_minus_v1x8": ("policy_v1_lookahead", "policy_v1_samplexN"),
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
    global SEEDS
    # Aggregate over whichever suite seeds have completed (partial suites are
    # quotable with their pooled n; rerun after more seeds land).
    SEEDS = [s for s in SEEDS
             if (RESULTS_DIR / f"scenario_bucket_v2_{VARIANT}_seed_{s}.json").exists()]
    if not SEEDS:
        sys.exit(f"no seed files found for variant '{VARIANT}'")
    print(f"[aggregate] {VARIANT}: {len(SEEDS)} seed(s) found: {SEEDS}")
    runs = {s: json.load(open(RESULTS_DIR / f"scenario_bucket_v2_{VARIANT}_seed_{s}.json")) for s in SEEDS}
    buckets = list(next(iter(runs.values()))["buckets"].keys())
    out = {"variant": VARIANT, "seeds": SEEDS, "buckets": {}}

    for bucket in buckets:
        episodes = {s: runs[s]["buckets"][bucket]["episodes"] for s in SEEDS}
        present = [m for m in METHODS
                   if all(m in ep for s in SEEDS for ep in episodes[s])]
        pooled = {m: [] for m in present}
        for s in SEEDS:
            for ep in episodes[s]:
                for m in present:
                    pooled[m].append(ep[m])

        n = len(pooled[present[0]])
        bucket_out = {"n_episodes_pooled": n, "method_means": {}, "pairs": {}}
        print(f"\n== {bucket} (pooled n={n}) ==")
        for m in present:
            dlv = float(np.mean([e["delivered_mean"] for e in pooled[m]]))
            tm = float(np.mean([e["time_mean"] for e in pooled[m]]))
            bucket_out["method_means"][m] = {"delivered_mean": dlv, "time_mean": tm}
            print(f"  {m:20s} delivered={dlv:.3f} time={tm:.1f}")

        for name, (ma, mb) in PAIRS.items():
            if ma not in pooled or mb not in pooled:
                continue
            d_del = np.array([a["delivered_mean"] - b["delivered_mean"]
                              for a, b in zip(pooled[ma], pooled[mb])])
            d_tim = np.array([a["time_mean"] - b["time_mean"]
                              for a, b in zip(pooled[ma], pooled[mb])])
            dm, dlo, dhi = bootstrap_ci(d_del, seed=1)
            tm, tlo, thi = bootstrap_ci(d_tim, seed=2)
            wins = int((d_del > 1e-9).sum())
            losses = int((d_del < -1e-9).sum())
            per_seed = [round(float(np.mean([
                ep[ma]["delivered_mean"] - ep[mb]["delivered_mean"]
                for ep in episodes[s]])), 4) for s in SEEDS]
            p_del = wilcoxon_p(d_del)
            p_tim = wilcoxon_p(d_tim)
            bucket_out["pairs"][name] = {
                "delivered_delta_mean": dm, "delivered_delta_ci95": [dlo, dhi],
                "delivered_wilcoxon_p": p_del,
                "time_delta_mean": tm, "time_delta_ci95": [tlo, thi],
                "time_wilcoxon_p": p_tim,
                "delivered_win_tie_loss": [wins, n - wins - losses, losses],
                "per_seed_delivered_delta": per_seed,
            }
            print(f"  {name}")
            print(f"    delivered {dm:+.3f} CI[{dlo:+.3f},{dhi:+.3f}] p={p_del:.2e}  time {tm:+.1f} "
                  f"CI[{tlo:+.1f},{thi:+.1f}] p={p_tim:.2e}  W/T/L [{wins},{n - wins - losses},{losses}]")
            print(f"    per-seed: {per_seed}")

        out["buckets"][bucket] = bucket_out

    save = RESULTS_DIR / f"stage2_aggregate_5seeds_{VARIANT}.json"
    save.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved: {save}")


if __name__ == "__main__":
    main()
