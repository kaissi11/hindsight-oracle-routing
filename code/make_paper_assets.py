#!/usr/bin/env python3
"""Generate EVERY paper table and figure from the recorded aggregates.

Outputs to paper/assets/:
  tables  (Markdown .md + LaTeX .tex, booktabs):
    table2_oracle_vs_repair    zero-shot policy oracle-8 - repair (4 settings x 3 buckets)
    table3_online_lookahead    look-8 pairs (auto-fills when *_s5 aggregates exist)
    table4_zeroshot_vs_finetune  v2x8 - v1x8 across all settings
    table5_transfer            N=100 + London vs repair and rolling-OR
    table6_makespan            time deltas (s and % of rolling-OR makespan)
    tableA_means_<variant>     per-bucket method means (appendix)
  figures (200 dpi PNG):
    fig1_oracle_online_schematic   conceptual: hindsight selection vs online lookahead
    fig_forest_consistency         all policy-repair deltas with 95% CIs (one look)
    fig_effect_by_setting          delta vs repair across settings, per bucket
    fig_tradeoff_v2                completion gain vs makespan premium (% of rolling-OR)

Colors: fixed entity->hue assignment (CVD-validated palette), never cycled;
identity is also carried by markers/linestyles so no meaning is color-alone.
Rerun any time; missing aggregates are skipped, S5 tables appear when ready.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ASSETS = HERE / "paper" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
BUCKETS = ["low", "medium", "high"]

# Fixed entity colors (validated categorical palette, light surface).
C = {
    "policy": "#2a78d6",    # zero-shot policy (headline) - blue
    "v2": "#4a3aa7",        # fine-tuned policy - violet
    "repair": "#eb6834",    # repair heuristic - orange
    "rolling": "#008300",   # rolling-OR - green
    "greedy": "#6b6a63",    # greedy floor - neutral gray
    "grid": "#d9d8d0",
    "ink": "#1a1a19",
    "muted": "#6b6a63",
}
BUCKET_BLUES = ["#9ec7f0", "#2a78d6", "#123f78"]  # sequential: low/med/high
MARKER = {"policy": "o", "v2": "D", "repair": "s", "rolling": "^", "greedy": "v"}

plt.rcParams.update({
    "axes.edgecolor": C["muted"], "axes.labelcolor": C["ink"],
    "text.color": C["ink"], "xtick.color": C["muted"], "ytick.color": C["muted"],
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6,
    "font.size": 9.5, "axes.titlesize": 10.5, "figure.dpi": 110,
})


def load(name: str):
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# Settings registry: display name -> (aggregate file, pair-name prefix for the
# zero-shot policy). Stage 1 aggregates name the policy pair differently.
SETTINGS = [
    ("Synthetic v1 dyn (N=20)", "stage1_aggregate_5seeds_synthetic.json", "policy_samplexN"),
    ("Damascus OSRM v1 dyn (N=20)", "stage1_aggregate_5seeds_osrm.json", "policy_samplexN"),
    ("Synthetic v2 dyn (N=20)", "stage2_aggregate_5seeds_synthetic.json", "v1x8"),
    ("Damascus OSRM v2 dyn (N=20)", "stage2_aggregate_5seeds_osrm.json", "v1x8"),
    ("London OSRM v2 dyn (N=20)", "stage2_aggregate_5seeds_london.json", "v1x8"),
    ("Synthetic v2 dyn (N=100)", "stage2_aggregate_5seeds_n100.json", "v1x8"),
]
S5_SETTINGS = [
    ("Damascus OSRM (S5)", "stage2_aggregate_5seeds_osrm_s5.json"),
    ("Synthetic (S5)", "stage2_aggregate_5seeds_synth_s5.json"),
    ("London OSRM (S5)", "stage2_aggregate_5seeds_london_s5.json"),
    ("N=100 synthetic (S5)", "stage2_aggregate_5seeds_n100_s5.json"),
]


def pair(agg, bucket, name):
    return agg["buckets"][bucket]["pairs"].get(name)


def fmt(pr, what="delivered"):
    if pr is None:
        return "—"
    m = pr[f"{what}_delta_mean"]
    lo, hi = pr[f"{what}_delta_ci95"]
    p = pr.get(f"{what}_wilcoxon_p")
    ps = f", p={p:.1e}" if p is not None else ""
    if what == "time":
        return f"{m:+.0f} [{lo:+.0f}, {hi:+.0f}]{ps}"
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]{ps}"


def write_table(stem: str, caption: str, header: list[str], rows: list[list[str]]):
    md = [f"**{caption}**", "", "| " + " | ".join(header) + " |",
          "|" + "|".join(["---"] * len(header)) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    (ASSETS / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    tex = ["% " + caption, r"\begin{table}[t]\centering\small",
           rf"\caption{{{caption}}}",
           r"\begin{tabular}{" + "l" * len(header) + "}", r"\toprule",
           " & ".join(header) + r" \\", r"\midrule"]
    tex += [" & ".join(c.replace("—", "--") for c in r) + r" \\" for r in rows]
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (ASSETS / f"{stem}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"  [table] {stem} ({len(rows)} rows)")


# ---------------------------------------------------------------- tables ----

def table2():
    rows = []
    for label, fname, prefix in SETTINGS[:4]:
        agg = load(fname)
        if agg is None:
            continue
        rows.append([label] + [fmt(pair(agg, b, f"{prefix}_minus_repair")) for b in BUCKETS])
    write_table("table2_oracle_vs_repair",
                "Zero-shot policy (oracle-8, upper bound) minus repair: delivered stops "
                "(paired, 5 seeds, n=200/bucket)",
                ["Setting", "Low", "Medium", "High"], rows)


def table3():
    rows, have = [], False
    for label, fname in S5_SETTINGS:
        agg = load(fname)
        if agg is None:
            continue
        for pname, plabel in [("v1look_minus_repair", "look-8 − repair"),
                              ("v1look_minus_rolling_or", "look-8 − rolling-OR"),
                              ("v1look_minus_v1x8", "look-8 − oracle-8 (hindsight gap)")]:
            pr = pair(agg, "low", pname)
            if pr is None:
                continue
            have = True
            rows.append([f"{label}: {plabel}"] +
                        [fmt(pair(agg, b, pname)) for b in BUCKETS])
    if not have:
        rows = [["[online (s5) aggregates missing — run run_stage5_suites.py, then rerun this script]", "—", "—", "—"]]
    write_table("table3_online_lookahead",
                "ONLINE lookahead (deployable): delivered stops (paired, 5 seeds)",
                ["Pair", "Low", "Medium", "High"], rows)


def table4():
    rows = []
    for label, fname, prefix in SETTINGS[2:]:
        agg = load(fname)
        if agg is None:
            continue
        rows.append([label] + [fmt(pair(agg, b, "v2x8_minus_v1x8")) for b in BUCKETS])
    write_table("table4_zeroshot_vs_finetune",
                "Fine-tuned (v2) minus zero-shot (v1), oracle-8, delivered stops — "
                "re-training never helps",
                ["Setting", "Low", "Medium", "High"], rows)


def table5():
    rows = []
    for label, fname, prefix in [SETTINGS[5], SETTINGS[4]]:
        agg = load(fname)
        if agg is None:
            continue
        rows.append([f"{label} vs repair"] +
                    [fmt(pair(agg, b, f"{prefix}_minus_repair")) for b in BUCKETS])
        rows.append([f"{label} vs rolling-OR"] +
                    [fmt(pair(agg, b, f"{prefix}_minus_rolling_or")) for b in BUCKETS])
    write_table("table5_transfer",
                "Zero-shot transfer (oracle-8): one un-retrained checkpoint at 5x scale "
                "and on an unseen city",
                ["Setting / baseline", "Low", "Medium", "High"], rows)


def table6():
    rows = []
    for label, fname, prefix in SETTINGS[2:]:
        agg = load(fname)
        if agg is None:
            continue
        for b in BUCKETS:
            pr = pair(agg, b, f"{prefix}_minus_rolling_or")
            rt = agg["buckets"][b]["method_means"]["rolling_or"]["time_mean"]
            pct = 100.0 * pr["time_delta_mean"] / rt if pr else float("nan")
            rows.append([label if b == "low" else "", b,
                         fmt(pr, "time"), f"{pct:+.1f}%",
                         fmt(pair(agg, b, f"{prefix}_minus_repair"), "time")])
    write_table("table6_makespan",
                "The price: makespan deltas of the zero-shot policy (oracle-8), seconds "
                "(and % of rolling-OR makespan)",
                ["Setting", "Bucket", "vs rolling-OR (s)", "vs rolling-OR (%)", "vs repair (s)"], rows)


def tables_means():
    for label, fname, _ in SETTINGS[2:]:
        agg = load(fname)
        if agg is None:
            continue
        stem = "tableA_means_" + fname.split("5seeds_")[1].split(".")[0]
        methods = list(agg["buckets"]["low"]["method_means"].keys())
        rows = []
        for b in BUCKETS:
            mm = agg["buckets"][b]["method_means"]
            rows.append([b] + [f"{mm[m]['delivered_mean']:.3f} | {mm[m]['time_mean']:.0f}"
                               for m in methods])
        write_table(stem, f"Per-bucket method means, {label} (delivered | time s)",
                    ["Bucket"] + methods, rows)


# --------------------------------------------------------------- figures ----

def fig_schematic():
    rng = np.random.RandomState(7)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for ax in axes:
        ax.set_xlim(0, 10); ax.set_ylim(-0.5, 8.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(C["muted"])
        ax.set_xlabel("time →", fontsize=9)

    ax = axes[0]
    ax.set_title("(a) Episode-level best-of-8  —  ORACLE upper bound", fontsize=10)
    ys = np.linspace(1, 7.4, 8)
    best = 5
    for i, y0 in enumerate(ys):
        x = np.linspace(0.4, 8.6, 60)
        y = y0 + np.cumsum(rng.randn(60)) * 0.05
        col = C["policy"] if i == best else C["grid"]
        ax.plot(x, y, color=col, lw=2.2 if i == best else 1.2, zorder=3 if i == best else 2)
    for xe in (2.8, 5.4, 7.2):   # shared disruption events
        ax.axvline(xe, color=C["repair"], lw=1.0, ls=":", zorder=1)
        ax.text(xe, 7.95, "✕", color=C["repair"], ha="center", fontsize=9)
    ax.annotate("select winner AFTER the whole\nepisode (sees the future)",
                xy=(8.8, ys[best]), xytext=(6.1, 0.15), fontsize=8.5, color=C["ink"],
                arrowprops=dict(arrowstyle="->", color=C["muted"], lw=1))
    ax.text(0.4, 8.6, "8 full rollouts, same realized disruptions (✕)", fontsize=8.5,
            color=C["muted"])
    ax.set_ylim(-0.5, 9.0)

    ax = axes[1]
    ax.set_title("(b) Per-step lookahead  —  ONLINE, deployable", fontsize=10)
    xt = np.linspace(0.4, 8.6, 5)
    yt = 4.2 + np.cumsum(rng.randn(5)) * 0.3
    ax.plot(xt, yt, color=C["policy"], lw=2.2, marker="o", ms=5,
            markerfacecolor="white", zorder=4)
    for i in range(len(xt) - 1):
        for k in range(6):   # fan of frozen-matrix completions at each decision
            dx = np.linspace(0, 1.35, 12)
            dy = np.cumsum(rng.randn(12)) * 0.12
            ax.plot(xt[i] + dx, yt[i] + dy, color=C["grid"], lw=0.9, zorder=2)
    ax.annotate("sample K completions under the\nFROZEN current matrix, commit\nbest first action, re-plan",
                xy=(xt[1], yt[1]), xytext=(4.6, 0.15), fontsize=8.5, color=C["ink"],
                arrowprops=dict(arrowstyle="->", color=C["muted"], lw=1))
    ax.text(0.5, 8.15, "no future information at any decision", fontsize=8.5, color=C["muted"])

    fig.tight_layout()
    fig.savefig(ASSETS / "fig1_oracle_online_schematic.png", dpi=200)
    plt.close(fig)
    print("  [fig] fig1_oracle_online_schematic")


def fig_forest():
    rows = []
    for label, fname, prefix in SETTINGS:
        agg = load(fname)
        if agg is None:
            continue
        for b in BUCKETS:
            pr = pair(agg, b, f"{prefix}_minus_repair")
            if pr:
                lo, hi = pr["delivered_delta_ci95"]
                rows.append((f"{label} · {b}", pr["delivered_delta_mean"], lo, hi))
    fig, ax = plt.subplots(figsize=(7.2, 0.34 * len(rows) + 1.2))
    ypos = np.arange(len(rows))[::-1]
    for (lab, m, lo, hi), y in zip(rows, ypos):
        ax.plot([lo, hi], [y, y], color=C["policy"], lw=1.6, zorder=2)
        ax.plot(m, y, MARKER["policy"], color=C["policy"], ms=5,
                markerfacecolor="white", zorder=3)
    ax.axvline(0, color=C["muted"], lw=1.0, ls="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("Delivered stops: zero-shot policy (oracle-8) − repair, 95% CI")
    ax.set_title("Positive in all 18 recorded setting × bucket cells", fontsize=10)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_forest_consistency.png", dpi=200)
    plt.close(fig)
    print(f"  [fig] fig_forest_consistency ({len(rows)} cells)")


def fig_effect_by_setting():
    order = [SETTINGS[2], SETTINGS[3], SETTINGS[4], SETTINGS[5]]
    labels = ["Syn N=20", "Damascus N=20", "London N=20", "Syn N=100"]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    x = np.arange(len(order), dtype=float)
    for bi, b in enumerate(BUCKETS):
        means, los, his = [], [], []
        for _, fname, prefix in order:
            agg = load(fname)
            pr = pair(agg, b, f"{prefix}_minus_repair") if agg else None
            means.append(pr["delivered_delta_mean"] if pr else np.nan)
            lo, hi = pr["delivered_delta_ci95"] if pr else (np.nan, np.nan)
            los.append(lo); his.append(hi)
        means, los, his = map(np.array, (means, los, his))
        ax.errorbar(x + (bi - 1) * 0.09, means, yerr=[means - los, his - means],
                    color=BUCKET_BLUES[bi], marker=["o", "s", "^"][bi], ms=5.5,
                    lw=1.8, capsize=3.5, label=f"{b} disruption")
    ax.axhline(0, color=C["muted"], lw=0.9, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Delivered stops: policy (oracle-8) − repair, 95% CI")
    ax.set_title("The advantage transfers out-of-distribution and grows ~10× with scale\n"
                 "(one un-retrained checkpoint, v2 dynamics)", fontsize=10)
    ax.legend(fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_effect_by_setting.png", dpi=200)
    plt.close(fig)
    print("  [fig] fig_effect_by_setting")


def fig_tradeoff_v2():
    order = [(SETTINGS[3], "Damascus N=20"), (SETTINGS[4], "London N=20"),
             (SETTINGS[5], "Syn N=100")]
    cols = [C["policy"], C["v2"], C["repair"]]   # fixed per setting, not per rank
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for ((_, fname, prefix), lab), col in zip(order, cols):
        agg = load(fname)
        if agg is None:
            continue
        for b, size in zip(BUCKETS, (40, 85, 140)):
            pr = pair(agg, b, f"{prefix}_minus_rolling_or")
            rt = agg["buckets"][b]["method_means"]["rolling_or"]["time_mean"]
            ax.scatter(100.0 * pr["time_delta_mean"] / rt, pr["delivered_delta_mean"],
                       s=size, color=col, marker="o", edgecolors="white", linewidths=1.2,
                       label=lab if b == "low" else None, zorder=3)
    ax.axhline(0, color=C["muted"], lw=0.9, ls="--")
    ax.axvline(0, color=C["muted"], lw=0.9, ls="--")
    ax.set_xlabel("Makespan premium vs rolling-OR (%)")
    ax.set_ylabel("Delivered stops gained vs rolling-OR")
    ax.set_title("Completion is bought with route time — always upper-right\n"
                 "(marker size = disruption bucket; zero-shot policy, oracle-8)", fontsize=10)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_tradeoff_v2.png", dpi=200)
    plt.close(fig)
    print("  [fig] fig_tradeoff_v2")


if __name__ == "__main__":
    print("Tables ->", ASSETS)
    table2(); table3(); table4(); table5(); table6(); tables_means()
    print("Figures ->", ASSETS)
    fig_schematic(); fig_forest(); fig_effect_by_setting(); fig_tradeoff_v2()
    print("Done. Rerun after the online (s5) suites to auto-fill table3.")
