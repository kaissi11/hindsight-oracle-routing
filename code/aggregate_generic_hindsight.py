#!/usr/bin/env python3
"""Pool the generic-hindsight control (randomized repair restarts, Table 7)
across evaluation seeds and print the three paper rows with hierarchical
seed-cluster bootstrap 95% CIs (falls back to episode bootstrap at 1 seed).

  python aggregate_generic_hindsight.py

Writes results/generic_hindsight_aggregate.json and prints the Markdown rows.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

FILES = {
    12345: "generic_hindsight_repair.json",
    13345: "generic_hindsight_repair_seed_13345.json",
    14345: "generic_hindsight_repair_seed_14345.json",
}
# Canonical repair rows come from the recorded reference suite of the SAME
# seed (identical instances + schedules -> cross-run pairing by episode index).
REFS = {s: f"scenario_bucket_v2_osrm_s5_seed_{s}.json" for s in FILES}


def cluster_ci(per_seed: dict[int, np.ndarray], n_boot: int = 20000,
               seed: int = 0) -> tuple[float, float, float]:
    """Hierarchical bootstrap: resample seeds, then episodes within seed."""
    rng = np.random.RandomState(seed)
    seeds = list(per_seed)
    pooled = np.concatenate([per_seed[s] for s in seeds])
    if len(seeds) == 1:
        vals = per_seed[seeds[0]]
        bs = vals[rng.randint(0, len(vals), size=(n_boot, len(vals)))].mean(axis=1)
    else:
        bs = np.empty(n_boot)
        for i in range(n_boot):
            picked = [per_seed[seeds[j]] for j in rng.randint(0, len(seeds), len(seeds))]
            bs[i] = np.mean([v[rng.randint(0, len(v), len(v))].mean() for v in picked])
    return float(pooled.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main() -> None:
    runs, refs = {}, {}
    for seed, name in FILES.items():
        path, ref_path = RESULTS / name, RESULTS / REFS[seed]
        if path.exists() and ref_path.exists():
            runs[seed] = json.loads(path.read_text(encoding="utf-8"))
            refs[seed] = json.loads(ref_path.read_text(encoding="utf-8"))
    if not runs:
        raise SystemExit("no generic_hindsight JSONs found")
    print(f"[generic] pooling {len(runs)} seed(s): {sorted(runs)}")

    out = {"seeds": sorted(runs), "buckets": {}}
    md_rows = {}
    for bucket in ("low", "medium", "high"):
        comp = {}
        for key in ("online_minus_repair", "hindsight_minus_online",
                    "hindsight_minus_repair", "hindsight_time_minus_repair_sec",
                    "hindsight_time_minus_online_sec"):
            per_seed = {}
            for seed, d in runs.items():
                eps = d["buckets"][bucket]["episodes"]
                ref_eps = refs[seed]["buckets"][bucket]["episodes"]
                assert len(eps) == len(ref_eps), (seed, bucket)
                rep_del = [r["repair_nn2opt"]["delivered_mean"] for r in ref_eps]
                rep_tim = [r["repair_nn2opt"]["time_mean"] for r in ref_eps]
                if key == "online_minus_repair":
                    vals = [e["online_mean"]["delivered_mean"] - rd
                            for e, rd in zip(eps, rep_del)]
                elif key == "hindsight_minus_online":
                    vals = [e["hindsight_best"]["delivered_mean"]
                            - e["online_mean"]["delivered_mean"] for e in eps]
                elif key == "hindsight_minus_repair":
                    vals = [e["hindsight_best"]["delivered_mean"] - rd
                            for e, rd in zip(eps, rep_del)]
                elif key == "hindsight_time_minus_repair_sec":
                    vals = [e["hindsight_best"]["time_mean"] - rt
                            for e, rt in zip(eps, rep_tim)]
                else:
                    vals = [e["hindsight_best"]["time_mean"]
                            - e["online_mean"]["time_mean"] for e in eps]
                per_seed[seed] = np.asarray(vals, dtype=float)
            m, lo, hi = cluster_ci(per_seed, seed=abs(hash((bucket, key))) % 2**31)
            comp[key] = {"mean": m, "ci95": [lo, hi],
                         "n": int(sum(len(v) for v in per_seed.values()))}
        out["buckets"][bucket] = comp
        md_rows[bucket] = comp

    dst = RESULTS / "generic_hindsight_aggregate.json"
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[generic] wrote {dst}")

    def cell(c, key):
        m, (lo, hi) = c[key]["mean"], c[key]["ci95"]
        sig = lo > 0 or hi < 0
        s = f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        return f"**{s}**" if sig else f"{s} ns"

    print("\nTable 7 rows (Markdown, low/medium/high):")
    for key, label in [("online_minus_repair", "online restart (mean of K) − canonical repair"),
                       ("hindsight_minus_online", "hindsight best-of-8 − online mean"),
                       ("hindsight_minus_repair", "hindsight best-of-8 − canonical repair")]:
        cells = " | ".join(cell(md_rows[b], key) for b in ("low", "medium", "high"))
        print(f"| {label} | {cells} |")
    t = [md_rows[b]["hindsight_time_minus_repair_sec"]["mean"] for b in ("low", "medium", "high")]
    print(f"hindsight makespan vs repair (s): {t[0]:+.0f} / {t[1]:+.0f} / {t[2]:+.0f}")
    t2 = [md_rows[b]["hindsight_time_minus_online_sec"]["mean"] for b in ("low", "medium", "high")]
    print(f"hindsight makespan vs online mean (s): {t2[0]:+.0f} / {t2[1]:+.0f} / {t2[2]:+.0f}")


if __name__ == "__main__":
    main()
