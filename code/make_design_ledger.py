"""Generate the experimental design ledger (Supplement S10) from result-file metadata.

Spec: one authoritative table, generated from the result manifest, that makes it
impossible to mistake training seeds for evaluation seeds or batched instances for
independent observations.

Every number in the emitted table is read from the recorded result JSONs:
  - evaluation seeds        <- filenames (suite glob)
  - episodes/bucket/seed    <- config.n_episodes, cross-checked against len(bucket.episodes)
  - N (nodes incl. depot)   <- config.n_nodes
  - horizon                 <- config.horizon_hours (default 8.0 for pre-field files)
  - status                  <- 'complete' flag where present, else bucket-count check

Only Data/city, Methods/selectors, and Role are declarative (they describe design intent,
not measurements). Output: paper/assets/table_design_ledger.md (+ .tex via the normal
markdown->latex build of the master).

Usage: python make_design_ledger.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
ASSETS = os.path.join(HERE, "paper", "assets")

# Declarative design intent per suite. Everything numeric is read from the JSONs.
# role vocabulary: primary | confirmation | boundary | mechanism | exploratory
MANIFEST = [
    # suite_id, glob pattern, data/city, dynamics, methods/selectors, role, paper anchor
    ("osrm-v2", "scenario_bucket_v2_osrm_seed_*.json", "Damascus OSRM",
     "v2", "greedy; oracle-8 (v1, v2); rolling-OR; repair; reactive-NN", "primary", "Table 5"),
    ("london-v2", "scenario_bucket_v2_london_seed_*.json", "London OSRM (zero-shot)",
     "v2", "greedy; oracle-8 (v1, v2); rolling-OR; repair", "confirmation", "Table 5, S8.2"),
    ("synth-v2", "scenario_bucket_v2_synthetic_seed_*.json", "synthetic Euclidean",
     "v2", "greedy; oracle-8 (v1, v2); rolling-OR; repair", "confirmation", "Table 5"),
    ("n100-v2", "scenario_bucket_v2_n100_seed_*.json", "synthetic Euclidean",
     "v2", "greedy; oracle-8 (v1, v2); rolling-OR; repair", "primary", "Table 5, S8.3"),
    ("osrm-s5", "scenario_bucket_v2_osrm_s5_seed_*.json", "Damascus OSRM",
     "v2", "look-8 (online) vs oracle-8, rolling-OR, repair", "primary", "Table 6"),
    ("london-s5", "scenario_bucket_v2_london_s5_seed_*.json", "London OSRM (zero-shot)",
     "v2", "look-8 (online) vs oracle-8, rolling-OR, repair", "confirmation", "Table 6"),
    ("synth-s5", "scenario_bucket_v2_synth_s5_seed_*.json", "synthetic Euclidean",
     "v2", "look-8 (online) vs oracle-8, rolling-OR, repair", "confirmation", "Table 6"),
    ("n100-s5", "scenario_bucket_v2_n100_s5_seed_*.json", "synthetic Euclidean",
     "v2", "look-8 (online) vs oracle-8, rolling-OR, repair", "primary", "Table 6"),
    ("repair-v1-osrm", "scenario_bucket_repair_osrm_seed_*.json", "Damascus OSRM",
     "v1", "greedy; oracle-8; repair; reactive-NN", "confirmation", "Table 5, v1 rows"),
    ("repair-v1-synth", "scenario_bucket_repair_synthetic_seed_*.json", "synthetic Euclidean",
     "v1", "greedy; oracle-8; repair; reactive-NN", "confirmation", "Table 5, v1 rows"),
    ("obs-base", "scenario_bucket_v2_obs_base_seed_*.json", "Damascus OSRM",
     "v2", "look-8 observability: base-matrix mode", "mechanism", "Table 10, §6.6"),
    ("obs-mask", "scenario_bucket_v2_obs_mask_seed_*.json", "Damascus OSRM",
     "v2", "look-8 observability: mask-only mode", "mechanism", "Table 10, §6.6"),
    ("obs-traffic", "scenario_bucket_v2_obs_traffic_seed_*.json", "Damascus OSRM",
     "v2", "look-8 observability: traffic-only mode", "mechanism", "Table 10, §6.6"),
    ("hstress-h4", "scenario_bucket_v2_hstress_h4_seed_*.json", "Damascus OSRM",
     "v2", "look-8; oracle-8; rolling-OR; repair (H=4 h)", "boundary", "§6.10"),
    ("hstress-h6", "scenario_bucket_v2_hstress_h6_seed_*.json", "Damascus OSRM",
     "v2", "look-8; oracle-8; rolling-OR; repair (H=6 h)", "boundary", "§6.10"),
    ("matched-h8", "matched_information_h8_seed_*.json", "Damascus OSRM",
     "v2", "matched clairvoyant vs online vs frozen-arm oracle (K=8)", "mechanism", "Table 8, §6.4"),
    ("matched-h4", "matched_information_h4_seed_*.json", "Damascus OSRM",
     "v2", "matched clairvoyant vs online vs frozen-arm oracle (K=8, H=4 h)", "mechanism", "Table 8, §6.4"),
    ("generic-hindsight", "generic_hindsight_repair*.json", "Damascus OSRM",
     "v2", "best-of-8 randomized repair restarts (no learning)", "mechanism", "Table 7"),
    ("kpi-or", "kpi_aligned_rolling_or.json", "Damascus OSRM",
     "v2", "completion-first (lex served-then-time) rolling-OR vs time-first", "mechanism", "§5, limitation 10"),
    ("kpi-or-h4", "kpi_aligned_rolling_or_h4_seed_*.json", "Damascus OSRM",
     "v2", "completion-first (lex served-then-time) rolling-OR vs time-first (H=4 h)", "mechanism", "§5, limitation 10"),
    ("dec-2opt", "scenario_bucket_v2_dec_2opt_seed_*.json", "Damascus OSRM",
     "v2", "look-8 + 2-opt polish (decoder ablation)", "exploratory", "Table S2"),
    ("dec-msa4", "scenario_bucket_v2_dec_msa4_seed_*.json", "Damascus OSRM",
     "v2", "look-8-MSA(4) scenario scoring (decoder ablation)", "exploratory", "Table S2"),
    ("cyc-plain", "scenario_bucket_v2_cyc_plain_seed_*.json", "Damascus OSRM",
     "v2 + cycle 0.4", "look-8 (frozen matrix) under cyclic traffic", "exploratory", "§6.9"),
    ("cyc-msa4", "scenario_bucket_v2_cyc_msa4_seed_*.json", "Damascus OSRM",
     "v2 + cycle 0.4", "look-8-MSA(4) (cycle-aware scoring)", "exploratory", "§6.9"),
    ("or-budget", "scenario_bucket_v2_budget_*ms_seed_*.json", "Damascus OSRM",
     "v2", "rolling-OR budget sweep (10/100/300 ms; 30 ms in osrm-s5)", "exploratory", "§5"),
    ("oracle-ksweep", "oracle_k_sweep_*_seed_*.json", "Damascus OSRM",
     "v2", "retrospective oracle-K sweep, sampling arm, K in {1,2,4,8}", "exploratory", "Figure 4"),
    ("dualcount-h8", "pilot_dualcount_osrm_s5_seed_*.json", "Damascus OSRM",
     "v2", "dual-count audit rerun of osrm-s5 (both deadline rules logged)", "exploratory", "§5 deadline-rule sensitivity"),
    ("dualcount-h4", "pilot_dualcount_hstress_h4_seed_*.json", "Damascus OSRM",
     "v2", "dual-count audit rerun of hstress-h4 (both deadline rules logged)", "exploratory", "§5 deadline-rule sensitivity"),
]

SEED_RE = re.compile(r"seed_(\d+)")


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def seed_of(path):
    m = SEED_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None


def fmt_seeds(seeds):
    seeds = sorted(s for s in seeds if s is not None)
    if not seeds:
        return "1 (12345)"  # single-file suites keyed by config.base_seed
    return f"{len(seeds)} ({'/'.join(str(s) for s in seeds)})"


def suite_row(suite_id, pattern, city, dyn, methods, role, anchor):
    paths = sorted(glob.glob(os.path.join(RESULTS, pattern)))
    if not paths:
        return None, f"MISSING: {suite_id} ({pattern})"
    n_eps, n_nodes, horizons, seeds, statuses = set(), set(), set(), [], []
    for p in paths:
        d = load(p)
        cfg = d.get("config", {})
        n_eps.add(cfg.get("n_episodes"))
        n_nodes.add(cfg.get("n_nodes"))
        horizons.add(cfg.get("horizon_hours", 8.0))
        s = seed_of(p)
        seeds.append(s if s is not None else cfg.get("base_seed"))
        buckets = d.get("buckets", {})
        ok = bool(buckets) and all(
            len(b.get("episodes", [])) == cfg.get("n_episodes") for b in buckets.values()
        )
        if "complete" in d:
            ok = ok and bool(d["complete"])
        statuses.append(ok)
    eps = n_eps.pop() if len(n_eps) == 1 else "MIXED"
    nodes = n_nodes.pop() if len(n_nodes) == 1 else "MIXED"
    hz = horizons.pop() if len(horizons) == 1 else "/".join(str(h) for h in sorted(horizons))
    status = "complete" if all(statuses) else "PARTIAL"
    uniq = sorted(set(s for s in seeds if s is not None))
    seeds = uniq  # files may differ by arm (e.g. budget level), not seed
    total = eps * len(uniq) if isinstance(eps, int) else "?"
    if isinstance(eps, int) and len(paths) > len(uniq):
        total = f"{total} per arm ({len(paths) // len(uniq)} arms)"
    row = (
        f"| {suite_id} | {city} | {nodes} | {hz} h | {dyn} | {methods} "
        f"| {fmt_seeds(seeds)} | {eps} | {total} | {status} | {role} ({anchor}) |"
    )
    return row, None


def perm_probe_row():
    p = os.path.join(RESULTS, "permutation_invariance.json")
    if not os.path.exists(p):
        return None
    d = load(p)
    cfg = d.get("config", {})
    n_inst = cfg.get("n_episodes", cfg.get("n_instances", "?"))
    n_perm = cfg.get("n_permutations", cfg.get("n_perms", 20))
    return (
        f"| perm-probe | Damascus OSRM | {cfg.get('n_nodes', '?')} | 8.0 h | v2 "
        f"| greedy relabeling probe ({n_inst} instances x {n_perm} depot-fixed permutations) "
        f"| 1 ({cfg.get('base_seed', 12345)}) | {n_inst} | {n_inst} instances | complete "
        f"| exploratory (limitation 12) |"
    )


def main():
    rows, problems = [], []
    for entry in MANIFEST:
        row, err = suite_row(*entry)
        if err:
            problems.append(err)
        if row:
            rows.append(row)
    pp = perm_probe_row()
    if pp:
        rows.append(pp)

    header = (
        "| Suite ID | Data/city | N (nodes incl. depot) | Horizon | Dynamics | Methods/selectors "
        "| Evaluation seeds | Episodes/bucket/seed | Total paired episodes/bucket | Status | Role |\n"
        "|---|---|---:|---:|---|---|---|---:|---:|---|---|"
    )
    note = (
        "*Unit of analysis: the paired episode within a disruption bucket (one pre-sampled "
        "schedule evaluated by every method); CIs pool episodes across seeds with "
        "seed-clustered bootstrap where seeds > 1. All seeds are **evaluation** seeds "
        "(instances + schedules); training used a single run per checkpoint (limitation 5). "
        "Episodes are independent paired observations, not batch replicas. Generated by "
        "`make_design_ledger.py` from the recorded result JSONs.*"
    )
    table = "\n".join(rows)
    md = f"{note}\n\n{header}\n{table}\n"

    os.makedirs(ASSETS, exist_ok=True)
    out_md = os.path.join(ASSETS, "table_design_ledger.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[LEDGER] wrote {out_md} ({len(rows)} suites)")
    for pr in problems:
        print(f"[LEDGER][WARN] {pr}")
    print(md)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
