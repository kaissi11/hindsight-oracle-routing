#!/usr/bin/env python3
"""Figure 6 (paper sec. 6.6): absolute completion-makespan positions per method
per bucket, with the per-bucket Pareto frontier — shows no method dominates:
rolling-OR holds the makespan frontier, the oracle bound alone holds the
completion frontier, and deployable look-8 pays oracle-level makespan for
repair-level completion.

Reads the Stage-5 aggregates; writes paper/assets/fig_pareto.png.
Style matches make_stage_figures.py (same figsize family, colors, markers).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ASSETS = HERE / "paper" / "assets"
BUCKETS = ["low", "medium", "high"]
SIZES = {"low": 40, "medium": 80, "high": 130}

# fixed categorical order (palette validated: dataviz six-checks, light mode)
METHODS = [
    ("policy_v1_samplexN", "oracle-8 (upper bound)", "#9467bd", "^"),
    ("policy_v1_lookahead", "look-8 (online)", "#d62728", "o"),
    ("rolling_or", "rolling-OR", "#1f77b4", "s"),
    ("repair_nn2opt", "repair (free)", "#2ca02c", "D"),
]

PANELS = [
    ("stage2_aggregate_5seeds_osrm_s5.json", "Damascus OSRM, N=20 (n=200)"),
    ("stage2_aggregate_5seeds_n100_s5.json", "Synthetic, N=100 (n=40, 1 seed)"),
]


def frontier(points):
    """Non-dominated set for (min time, max delivered), sorted by time."""
    pts = sorted(points)
    best, out = -1.0, []
    for t, d in pts:
        if d > best:
            out.append((t, d))
            best = d
    return out


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for ax, (fname, title) in zip(axes, PANELS):
        agg = json.loads((RESULTS / fname).read_text(encoding="utf-8"))
        for b in BUCKETS:
            mm = agg["buckets"][b]["method_means"]
            pts = []
            for key, label, color, marker in METHODS:
                t = mm[key]["time_mean"] / 3600.0
                d = mm[key]["delivered_mean"]
                pts.append((t, d))
                ax.scatter(t, d, s=SIZES[b], color=color, marker=marker,
                           label=label if b == "low" else None,
                           edgecolors="k", linewidths=0.4, zorder=3)
            fx, fy = zip(*frontier(pts))
            ax.plot(fx, fy, color="gray", lw=0.8, ls=":", zorder=1)
        # direct labels once per method, at the high-bucket point
        offsets = {"policy_v1_samplexN": (-10, -4, "right"),
                   "policy_v1_lookahead": (8, -3, "left"),
                   "rolling_or": (-10, -4, "right"),
                   "repair_nn2opt": (8, -11, "left")}
        mm_hi = agg["buckets"]["high"]["method_means"]
        for key, label, color, _ in METHODS:
            t = mm_hi[key]["time_mean"] / 3600.0
            d = mm_hi[key]["delivered_mean"]
            dx, dy, ha = offsets[key]
            ax.annotate(label.split(" (")[0], (t, d), textcoords="offset points",
                        xytext=(dx, dy), ha=ha, fontsize=7.5, color="#333333")
        ax.margins(0.1, 0.09)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Makespan (hours)")
    axes[0].set_ylabel("Delivered stops (mean)")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("No Pareto dominance: completion is bought with makespan "
                 "(marker size = disruption bucket; dotted line = per-bucket frontier)",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / "fig_pareto.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[fig] {out}")


if __name__ == "__main__":
    main()
