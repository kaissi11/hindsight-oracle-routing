"""Equivalence + cluster-robust statistics for the Stage-5 online "tie" claims.

Addresses two review must-fixes:
  1. "Tie" is not statistically valid without equivalence testing (TOST).
  2. Pooled episode Wilcoxon is vulnerable to pseudo-replication; a seed-level
     cluster bootstrap must be primary (or co-primary).

Predeclared equivalence margins (rationale in table caption):
  N=20 suites : +/- 0.05 delivered stops  (smaller than the smallest measured
                oracle-8 - repair effect at N=20, +0.055 Damascus-low)
  N=100 suite : +/- 0.20 delivered stops  (~half the smallest measured N=100
                oracle effect, +0.375)

Verdict rules per cell (delivered stops, paired deltas):
  equivalent    : 90% cluster-bootstrap CI fully inside [-margin, +margin]
                  (equivalent to TOST at alpha = 0.05)
  different     : 95% cluster-bootstrap CI excludes 0
  inconclusive  : neither

Cluster bootstrap: resample seeds with replacement, then episodes within each
sampled seed (hierarchical). For 1-seed suites this degenerates to an episode
bootstrap and is flagged `cluster=False` — those cells can at best be
"inconclusive-consistent", never "equivalent (clustered)".

Outputs:
  results/equivalence_analysis.json
  paper/assets/table_equiv.md / .tex   (main-text compact table)

Usage: python equivalence_analysis.py
"""

import json
import os
import glob
import numpy as np
from scipy import stats

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper", "assets")

SUITES = {
    "osrm_s5": dict(label="Damascus OSRM (N=20)", margin=0.05),
    "london_s5": dict(label="London OSRM (N=20)", margin=0.05),
    "synth_s5": dict(label="Synthetic (N=20)", margin=0.05),
    "n100_s5": dict(label="Synthetic (N=100)", margin=0.20),
}
BUCKETS = ["low", "medium", "high"]

# (label, method_a, method_b, claim)  -- delta = a - b on delivered_mean
PAIRS = [
    ("look-8 - repair", "policy_v1_lookahead", "repair_nn2opt", "tie"),
    ("look-8 - rolling_or", "policy_v1_lookahead", "rolling_or", "tie"),
    ("look-8 - oracle-8 (gap)", "policy_v1_lookahead", "policy_v1_samplexN", "difference"),
    ("oracle-8 - repair", "policy_v1_samplexN", "repair_nn2opt", "difference"),
]

N_BOOT = 20000
RNG = np.random.default_rng(12345)


def load_suite(variant):
    """-> {bucket: {seed: {method: np.array of per-episode delivered}}}"""
    files = sorted(glob.glob(os.path.join(RESULTS, f"scenario_bucket_v2_{variant}_seed_*.json")))
    out = {b: {} for b in BUCKETS}
    for f in files:
        seed = int(f.rsplit("seed_", 1)[1].split(".")[0])
        d = json.load(open(f))
        for b in BUCKETS:
            eps = d["buckets"][b]["episodes"]
            out[b][seed] = {
                m: np.array([e[m]["delivered_mean"] for e in eps])
                for m in eps[0] if m != "episode"
            }
    return out


def cluster_boot_ci(per_seed_deltas, levels=(0.95, 0.90)):
    """Hierarchical bootstrap: seeds with replacement, then episodes within seed."""
    seeds = list(per_seed_deltas)
    arrs = [per_seed_deltas[s] for s in seeds]
    ns = len(arrs)
    means = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = RNG.integers(0, ns, ns)
        chunks = []
        for j in idx:
            a = arrs[j]
            chunks.append(a[RNG.integers(0, len(a), len(a))])
        means[i] = np.concatenate(chunks).mean()
    cis = {}
    for lv in levels:
        lo, hi = np.percentile(means, [(1 - lv) / 2 * 100, (1 + lv) / 2 * 100])
        cis[lv] = (float(lo), float(hi))
    return cis


def tost_t(deltas, margin):
    """Paired-t TOST p-value (max of the two one-sided p's)."""
    n = len(deltas)
    m, se = deltas.mean(), deltas.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return 0.0 if abs(m) < margin else 1.0
    p_lo = stats.t.sf((m + margin) / se, n - 1)   # H0: delta <= -margin
    p_hi = stats.t.cdf((m - margin) / se, n - 1)  # H0: delta >= +margin
    return float(max(p_lo, p_hi))


def analyze():
    results = {}
    for variant, meta in SUITES.items():
        data = load_suite(variant)
        margin = meta["margin"]
        results[variant] = dict(label=meta["label"], margin=margin, cells=[])
        for label, ma, mb, claim in PAIRS:
            for b in BUCKETS:
                per_seed = {s: v[ma] - v[mb] for s, v in data[b].items()}
                if not per_seed:
                    continue
                pooled = np.concatenate(list(per_seed.values()))
                n_seeds = len(per_seed)
                cis = cluster_boot_ci(per_seed)
                try:
                    w = stats.wilcoxon(pooled, zero_method="pratt")
                    wp = float(w.pvalue)
                except ValueError:  # all zeros
                    wp = 1.0
                tp = tost_t(pooled, margin)
                lo90, hi90 = cis[0.90]
                lo95, hi95 = cis[0.95]
                if claim == "tie":
                    if n_seeds >= 2 and -margin < lo90 and hi90 < margin:
                        verdict = "equivalent"
                    elif lo95 > 0 or hi95 < 0:
                        verdict = "different"
                    else:
                        verdict = "inconclusive"
                else:
                    verdict = "different" if (lo95 > 0 or hi95 < 0) else "not significant"
                results[variant]["cells"].append(dict(
                    pair=label, bucket=b, claim=claim,
                    n_seeds=n_seeds, n_episodes=int(pooled.size), cluster=n_seeds >= 2,
                    mean=float(pooled.mean()),
                    per_seed_means={str(s): float(v.mean()) for s, v in per_seed.items()},
                    ci95_cluster=[lo95, hi95], ci90_cluster=[lo90, hi90],
                    wilcoxon_p=wp, tost_p=tp, margin=margin, verdict=verdict,
                ))
    return results


def fmt_p(p):
    return "<1e-4" if p < 1e-4 else f"{p:.3f}" if p >= 1e-3 else f"{p:.0e}"


def write_tables(results):
    os.makedirs(ASSETS, exist_ok=True)
    md = ["**Table E — Equivalence and cluster-robust analysis of the online tie claims** "
          "(delivered stops, paired deltas; cluster bootstrap resamples seeds, then episodes "
          "within seed; margins predeclared: ±0.05 stops at N=20 — smaller than the smallest "
          "measured N=20 oracle effect — and ±0.20 at N=100, ≈half the smallest N=100 oracle "
          "effect; *equivalent* = 90% cluster CI inside the margin, i.e. TOST at α=0.05).",
          "",
          "| Setting | Pair | Bucket | mean Δ | 95% cluster CI | TOST *p* | Verdict |",
          "|---|---|---|---|---|---|---|"]
    tex = [r"\begin{table*}[t]\centering\small",
           r"\caption{Equivalence and cluster-robust analysis of the online tie claims "
           r"(delivered stops, paired deltas). Cluster bootstrap resamples seeds, then episodes "
           r"within seed. Margins predeclared: $\pm0.05$ stops at $N{=}20$ (smaller than the "
           r"smallest measured $N{=}20$ oracle effect) and $\pm0.20$ at $N{=}100$ ($\approx$half "
           r"the smallest $N{=}100$ oracle effect). \emph{Equivalent} = 90\% cluster CI inside "
           r"the margin (TOST, $\alpha{=}0.05$).}",
           r"\label{tab:equiv}",
           r"\begin{tabular}{llllllr}",
           r"\toprule",
           r"Setting & Pair & Bucket & mean $\Delta$ & 95\% cluster CI & TOST $p$ & Verdict \\",
           r"\midrule"]
    for variant, res in results.items():
        for c in res["cells"]:
            if c["claim"] != "tie":
                continue
            ci = f"[{c['ci95_cluster'][0]:+.3f}, {c['ci95_cluster'][1]:+.3f}]"
            note = "" if c["cluster"] else " (1 seed)"
            verdict = c["verdict"] + note
            md.append(f"| {res['label']} | {c['pair']} | {c['bucket']} | {c['mean']:+.3f} | "
                      f"{ci} | {fmt_p(c['tost_p'])} | {verdict} |")
            tex.append(f"{res['label']} & {c['pair']} & {c['bucket']} & {c['mean']:+.3f} & "
                       f"{ci} & {fmt_p(c['tost_p'])} & {verdict} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    with open(os.path.join(ASSETS, "table_equiv.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(ASSETS, "table_equiv.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")


def main():
    results = analyze()
    out = os.path.join(RESULTS, "equivalence_analysis.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    write_tables(results)
    # console summary
    for variant, res in results.items():
        print(f"\n=== {res['label']} (margin ±{res['margin']}) ===")
        for c in res["cells"]:
            tag = f"{c['pair']:26s} {c['bucket']:6s}"
            ci95 = c["ci95_cluster"]
            print(f"  {tag} mean {c['mean']:+.3f}  CI95c [{ci95[0]:+.3f},{ci95[1]:+.3f}]  "
                  f"TOST p={fmt_p(c['tost_p'])}  W p={fmt_p(c['wilcoxon_p'])}  -> {c['verdict']}"
                  + ("" if c["cluster"] else "  [1 seed: episode bootstrap only]"))
    print(f"\nWrote {out}\n  + paper/assets/table_equiv.md/.tex")


if __name__ == "__main__":
    main()
