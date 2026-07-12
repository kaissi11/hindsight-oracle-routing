#!/usr/bin/env python3
"""Summary figures: paired deltas with 95% CIs from the aggregate JSONs.

Reads whatever aggregates exist in results/ and writes PNGs to
results/figures/. Rerun any time; missing variants are skipped.

Figures:
  fig_policy_vs_repair.png   - policy x8 minus repair (delivered) per bucket,
                               one series per stage x instance-distribution
  fig_zero_shot_transfer.png - v2 dynamics: v2x8-v1x8 and v1x8-repair (delivered)
  fig_time_tradeoff.png      - policy x8 minus rolling-OR: delivered vs time
  fig_online_vs_oracle.png   - online (s5): ONLINE lookahead vs repair, and the
                               oracle gap (lookahead - samplexN); appears once
                               the *_s5 aggregates exist
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
FIGS = RESULTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
BUCKETS = ["low", "medium", "high"]


def load(name: str):
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pair_series(agg, pair_name):
    means, los, his = [], [], []
    for b in BUCKETS:
        pr = agg["buckets"][b]["pairs"][pair_name]
        means.append(pr["delivered_delta_mean"])
        lo, hi = pr["delivered_delta_ci95"]
        los.append(lo)
        his.append(hi)
    means, los, his = map(np.array, (means, los, his))
    return means, means - los, his - means


def errbar(ax, x, series, label, color):
    means, lo_err, hi_err = series
    ax.errorbar(x, means, yerr=[lo_err, hi_err], label=label, color=color,
                marker="o", capsize=4, lw=1.6, ms=5)


def fig_policy_vs_repair():
    sources = [
        ("stage1_aggregate_5seeds_synthetic.json", "policy_samplexN_minus_repair",
         "Stage 1 (static dyn) synthetic", "#1f77b4"),
        ("stage1_aggregate_5seeds_osrm.json", "policy_samplexN_minus_repair",
         "Stage 1 (static dyn) OSRM", "#aec7e8"),
        ("stage2_aggregate_5seeds_synthetic.json", "v2x8_minus_repair",
         "Stage 2 (decision-relevant dyn) synthetic", "#d62728"),
        ("stage2_aggregate_5seeds_osrm.json", "v2x8_minus_repair",
         "Stage 2 (decision-relevant dyn) OSRM", "#ff9896"),
    ]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x0 = np.arange(len(BUCKETS), dtype=float)
    plotted = 0
    for i, (fname, pair, label, color) in enumerate(sources):
        agg = load(fname)
        if agg is None:
            continue
        errbar(ax, x0 + (i - 1.5) * 0.08, pair_series(agg, pair), label, color)
        plotted += 1
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xticks(x0)
    ax.set_xticklabels([b.capitalize() for b in BUCKETS])
    ax.set_xlabel("Disruption bucket")
    ax.set_ylabel("Delivered stops: policy x8 − repair (95% CI)")
    ax.set_title("Learned policy vs zero-compute repair heuristic\n(paired, 5 seeds, n=200 episodes/bucket)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_policy_vs_repair.png", dpi=200)
    plt.close(fig)
    return plotted


def fig_zero_shot():
    plotted = 0
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x0 = np.arange(len(BUCKETS), dtype=float)
    sources = [
        ("stage2_aggregate_5seeds_synthetic.json", "synthetic", "#2ca02c", "#d62728", "#9467bd"),
        ("stage2_aggregate_5seeds_osrm.json", "OSRM", "#98df8a", "#ff9896", "#c5b0d5"),
    ]
    for fname, tag, c1, c2, c3 in sources:
        agg = load(fname)
        if agg is None:
            continue
        errbar(ax, x0 - 0.10, pair_series(agg, "v1x8_minus_repair"), f"v1 policy zero-shot − repair ({tag})", c1)
        errbar(ax, x0, pair_series(agg, "v2x8_minus_repair"), f"fine-tuned policy − repair ({tag})", c2)
        errbar(ax, x0 + 0.10, pair_series(agg, "v2x8_minus_v1x8"), f"fine-tuned − zero-shot ({tag})", c3)
        plotted += 1
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xticks(x0)
    ax.set_xticklabels([b.capitalize() for b in BUCKETS])
    ax.set_xlabel("Disruption bucket")
    ax.set_ylabel("Delivered stops delta (95% CI)")
    ax.set_title("Zero-shot transfer: retraining under new dynamics adds nothing\n(Stage 2, paired, 5 seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_zero_shot_transfer.png", dpi=200)
    plt.close(fig)
    return plotted


def fig_time_tradeoff():
    sources = [
        ("stage1_aggregate_5seeds_synthetic.json", "policy_samplexN_minus_rolling_or",
         "Stage 1 synthetic", "#1f77b4", "o"),
        ("stage1_aggregate_5seeds_osrm.json", "policy_samplexN_minus_rolling_or",
         "Stage 1 OSRM", "#aec7e8", "s"),
        ("stage2_aggregate_5seeds_synthetic.json", "v2x8_minus_rolling_or",
         "Stage 2 synthetic", "#d62728", "o"),
        ("stage2_aggregate_5seeds_osrm.json", "v2x8_minus_rolling_or",
         "Stage 2 OSRM", "#ff9896", "s"),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    plotted = 0
    for fname, pair, label, color, marker in sources:
        agg = load(fname)
        if agg is None:
            continue
        for b, size in zip(BUCKETS, (40, 80, 130)):
            pr = agg["buckets"][b]["pairs"][pair]
            mean_time = agg["buckets"][b]["method_means"]["rolling_or"]["time_mean"]
            time_pct = 100.0 * pr["time_delta_mean"] / mean_time
            ax.scatter(time_pct, pr["delivered_delta_mean"], s=size, color=color,
                       marker=marker, label=label if b == "low" else None,
                       edgecolors="k", linewidths=0.4, zorder=3)
        plotted += 1
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Extra makespan vs rolling-OR (%)")
    ax.set_ylabel("Extra delivered stops vs rolling-OR")
    ax.set_title("The completion-vs-makespan tradeoff\n(marker size = disruption bucket: low/med/high)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_time_tradeoff.png", dpi=200)
    plt.close(fig)
    return plotted


def fig_online_vs_oracle():
    """Online (s5): does the completion advantage survive online (no-hindsight)
    selection? One panel: lookahead-repair per setting; plus the oracle gap."""
    sources = [
        ("stage2_aggregate_5seeds_synth_s5.json", "synthetic", "#1f77b4"),
        ("stage2_aggregate_5seeds_osrm_s5.json", "Damascus OSRM", "#d62728"),
        ("stage2_aggregate_5seeds_london_s5.json", "London OSRM", "#2ca02c"),
        ("stage2_aggregate_5seeds_n100_s5.json", "N=100 synthetic", "#9467bd"),
    ]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x0 = np.arange(len(BUCKETS), dtype=float)
    plotted = 0
    seed_tags = []
    for i, (fname, tag, color) in enumerate(sources):
        agg = load(fname)
        # headline policy is zero-shot v1 — plot the v1 pairs (fall back to v2
        # for aggregates that predate the v1look_minus_v1x8 pair)
        if agg is None:
            continue
        pairs = agg["buckets"][BUCKETS[0]]["pairs"]
        rep = "v1look_minus_repair" if "v1look_minus_repair" in pairs else "v2look_minus_repair"
        gap = "v1look_minus_v1x8" if "v1look_minus_v1x8" in pairs else "v2look_minus_v2x8"
        if rep not in pairs:
            continue
        n_seeds = len(agg.get("seeds", [])) or 5
        seed_tags.append(f"{tag}: {n_seeds}")
        errbar(ax, x0 + (i - 1.5) * 0.08, pair_series(agg, rep),
               f"online look-8 − repair ({tag})", color)
        errbar(ax, x0 + (i - 1.5) * 0.08 + 0.04, pair_series(agg, gap),
               f"hindsight gap: look-8 − oracle-8 ({tag})", color)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return 0
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xticks(x0)
    ax.set_xticklabels([b.capitalize() for b in BUCKETS])
    ax.set_xlabel("Disruption bucket")
    ax.set_ylabel("Delivered stops delta (95% CI)")
    # keep the title short enough for the 7-in canvas; per-setting seed counts
    # live in Table 3 / the caption (stale long titles overflowed the edge)
    ax.set_title("Online lookahead vs repair, and the hindsight gap (look-8 − oracle-8)\n"
                 "(paired, zero-shot v1; per-setting seed counts in Table 3)",
                 fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_online_vs_oracle.png", dpi=200)
    plt.close(fig)
    return plotted


if __name__ == "__main__":
    n1 = fig_policy_vs_repair()
    n2 = fig_zero_shot()
    n3 = fig_time_tradeoff()
    n4 = fig_online_vs_oracle()
    print(f"fig_policy_vs_repair.png   ({n1} series sets)")
    print(f"fig_zero_shot_transfer.png ({n2} series sets)")
    print(f"fig_time_tradeoff.png      ({n3} series sets)")
    print(f"fig_online_vs_oracle.png   ({n4} series sets)" if n4 else
          "fig_online_vs_oracle.png   (skipped - no *_s5 aggregates yet)")
    print(f"-> {FIGS}")
