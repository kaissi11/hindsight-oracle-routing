#!/usr/bin/env python3
"""Figure 7 (paper sec. 6.8): the horizon-stress inversion. Two panels over
H = 8h -> 6h -> 4h (Damascus, all available seeds per horizon, paired deltas,
hierarchical seed-cluster bootstrap CIs matching the sec. 6.8 table):
  left  : look-8 - repair    (deployable edge appears when the horizon binds)
  right : look-8 - oracle-8  (the gap flips sign: best-of-K stops being a bound)
Reads the run JSONs directly; writes paper/assets/fig_horizon_inversion.png.
Style matches make_stage_figures.py; bucket palette CVD-validated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ASSETS = HERE / "paper" / "assets"

RUNS = [("8 h", "osrm_s5"), ("6 h", "hstress_h6"), ("4 h", "hstress_h4")]
BUCKETS = [("low", "#1f77b4"), ("medium", "#d95f02"), ("high", "#d62728")]
PAIRS = [("policy_v1_lookahead", "repair_nn2opt", "look-8 − repair"),
         ("policy_v1_lookahead", "policy_v1_samplexN", "look-8 − oracle-8 (gap)")]


def seed_files(variant):
    files = sorted(RESULTS.glob(f"scenario_bucket_v2_{variant}_seed_*.json"))
    if not files:
        raise FileNotFoundError(f"no results for {variant}")
    return files


def eps(path, bucket, method):
    d = json.loads(path.read_text(encoding="utf-8"))
    return np.array([e[method]["delivered_mean"]
                     for e in d["buckets"][bucket]["episodes"]])


def ci(per_seed, seed=0):
    """Hierarchical bootstrap: resample seeds, then episodes within each."""
    rng = np.random.RandomState(seed)
    arrays = list(per_seed)
    n_seed = len(arrays)
    boot = np.empty(20000)
    for i in range(20000):
        chosen = rng.randint(0, n_seed, n_seed)
        boot[i] = np.concatenate(
            [arrays[j][rng.randint(0, len(arrays[j]), len(arrays[j]))]
             for j in chosen]).mean()
    pooled = np.concatenate(arrays)
    return pooled.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharex=True)
    xs = np.arange(len(RUNS))
    files = {variant: seed_files(variant) for _, variant in RUNS}
    counts = ", ".join(f"{label} ×{len(files[variant])}" for label, variant in RUNS)
    for ax, (ma, mb, title) in zip(axes, PAIRS):
        for k, (bucket, color) in enumerate(BUCKETS):
            means, los, his = [], [], []
            for _, variant in RUNS:
                per_seed = [eps(p, bucket, ma) - eps(p, bucket, mb)
                            for p in files[variant]]
                m, lo, hi = ci(per_seed)
                means.append(m); los.append(m - lo); his.append(hi - m)
            x = xs + (k - 1) * 0.12
            ax.errorbar(x, means, yerr=[los, his], color=color, marker="o",
                        ms=6, lw=1.6, capsize=3, label=bucket)
            ax.annotate(bucket, (x[-1], means[-1]), textcoords="offset points",
                        xytext=(9, -3), fontsize=7.5, color="#333333")
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_xticks(xs)
        ax.set_xticklabels([h for h, _ in RUNS])
        ax.set_xlabel("Mission horizon $H$")
        ax.set_title(title, fontsize=10)
        ax.margins(x=0.18)
    axes[0].set_ylabel("Delivered stops, paired Δ")
    axes[0].legend(fontsize=8, loc="upper left", title=None)
    fig.suptitle(f"The horizon-stress inversion (Damascus; seeds: {counts}): "
                 "as the deadline binds, the deployable lookahead pulls ahead of repair\n"
                 "and overtakes the trajectory-hindsight metric, which bounds only "
                 "its own K rollouts, not other controllers",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / "fig_horizon_inversion.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[fig] {out}")


if __name__ == "__main__":
    main()
