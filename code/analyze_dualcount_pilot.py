#!/usr/bin/env python3
"""Decision gate for the Wave-1 dual-count pilot (deadline semantics).

Reads every results/pilot_dualcount_{osrm_s5,hstress_h4}_seed_*.json (the
pilot was seed 12345; the post-queue follow-up adds H=4 seeds 13345/14345)
plus the recorded reference suite of the SAME seed, then reports per bucket:

  1. IDENTITY: pilot delivered/time vs recorded, per non-wall-clock arm
     (must be bit-identical; the dual counters are write-only bookkeeping —
     any drift there means the environment changed and the pilot is void);
  2. STRADDLE SIZE: delivered − delivered_strict per arm (≤ 1 stop/route by
     construction; how big is it in practice, and is it method-differential?);
  3. ORDERING GATE: sign + significance of every headline paired delta
     recomputed under STRICT counting vs the recorded departure-cutoff
     counting — hierarchical seed-cluster bootstrap when >1 seed (the paper's
     primary estimator), episode bootstrap at 1 seed;
  4. WAIT STEPS: nonzero anywhere? (if yes, plan says add a wait-costs-time
     probe on the high bucket).

Verdict line at the end: KEEP-WORDING (no ordering flips → rename estimand +
sensitivity paragraph) or FLIPS-FOUND (list them → author decision on strict
reruns).

  python analyze_dualcount_pilot.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

SUITES = ["osrm_s5", "hstress_h4"]
# Wall-clock-budgeted arm: identity not expected even on an exclusive machine.
WALL_CLOCK_ARMS = {"rolling_or"}
# Headline pairs whose ordering the paper leans on (v1 = zero-shot headline
# policy; v2 kept as secondary).
PAIRS = [
    ("policy_v1_lookahead", "repair_nn2opt"),
    ("policy_v1_lookahead", "policy_v1_samplexN"),
    ("policy_v1_lookahead", "rolling_or"),
    ("policy_v1_samplexN", "repair_nn2opt"),
    ("policy_v2_lookahead", "repair_nn2opt"),
    ("policy_v2_lookahead", "policy_v2_samplexN"),
    ("policy_v2_samplexN", "repair_nn2opt"),
]
SEED_RE = re.compile(r"seed_(\d+)\.json$")


def cluster_ci(per_seed: dict[int, np.ndarray], n: int = 10000, seed: int = 0):
    """Hierarchical bootstrap (seeds, then episodes); episode-only at 1 seed."""
    rng = np.random.RandomState(seed)
    seeds = sorted(per_seed)
    pooled = np.concatenate([per_seed[s] for s in seeds])
    if len(seeds) == 1:
        vals = per_seed[seeds[0]]
        bs = vals[rng.randint(0, len(vals), size=(n, len(vals)))].mean(axis=1)
    else:
        bs = np.empty(n)
        for i in range(n):
            picked = [per_seed[seeds[j]]
                      for j in rng.randint(0, len(seeds), len(seeds))]
            bs[i] = np.mean([v[rng.randint(0, len(v), len(v))].mean()
                             for v in picked])
    return (float(pooled.mean()),
            float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def sig(lo: float, hi: float) -> str:
    if lo > 0:
        return "+"
    if hi < 0:
        return "-"
    return "0"


def main() -> None:
    any_flip = False
    any_wait = False
    max_ident_err = 0.0
    for tag in SUITES:
        paths = sorted(RESULTS.glob(f"pilot_dualcount_{tag}_seed_*.json"))
        pilots: dict[int, dict] = {}
        refs: dict[int, dict] = {}
        for p in paths:
            m = SEED_RE.search(p.name)
            if not m:
                continue
            s = int(m.group(1))
            ref_path = RESULTS / f"scenario_bucket_v2_{tag}_seed_{s}.json"
            if not ref_path.exists():
                print(f"[gate] {tag} seed {s}: no recorded reference - skip")
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            if len(d.get("buckets", {})) < 3:
                print(f"[gate] {tag} seed {s}: incomplete pilot JSON - skip")
                continue
            pilots[s] = d
            refs[s] = json.loads(ref_path.read_text(encoding="utf-8"))
        if not pilots:
            print(f"[gate] {tag}: no complete pilot JSONs - run "
                  "run_dualcount_pilot.py")
            continue
        seeds = sorted(pilots)
        print(f"\n===== {tag} ({len(seeds)} seed(s): "
              f"{'/'.join(str(s) for s in seeds)}) =====")
        buckets = list(pilots[seeds[0]]["buckets"])
        for bucket in buckets:
            arms = [k for k in pilots[seeds[0]]["buckets"][bucket]["episodes"][0]
                    if k != "episode"]

            # -- 1. identity check (non-wall-clock arms), per seed --
            ident_bad = {}
            for s in seeds:
                eps = pilots[s]["buckets"][bucket]["episodes"]
                ref_eps = refs[s]["buckets"][bucket]["episodes"]
                for arm in arms:
                    if arm in WALL_CLOCK_ARMS or arm not in ref_eps[0]:
                        continue
                    err = max(abs(e[arm]["delivered_mean"] - r[arm]["delivered_mean"])
                              for e, r in zip(eps, ref_eps))
                    terr = max(abs(e[arm]["time_mean"] - r[arm]["time_mean"])
                               for e, r in zip(eps, ref_eps))
                    max_ident_err = max(max_ident_err, err, terr)
                    if err > 1e-9 or terr > 1e-6:
                        ident_bad[(s, arm)] = (err, terr)
            print(f"[{bucket}] identity vs recorded: "
                  + ("OK (all non-OR arms bit-identical)" if not ident_bad
                     else f"MISMATCH {ident_bad}"))

            # -- 2. straddle + waits per arm (pooled over seeds) --
            for arm in arms:
                d = np.concatenate([[e[arm]["delivered_mean"]
                                     for e in pilots[s]["buckets"][bucket]["episodes"]]
                                    for s in seeds])
                st = np.concatenate([[e[arm].get("delivered_strict_mean", np.nan)
                                      for e in pilots[s]["buckets"][bucket]["episodes"]]
                                     for s in seeds])
                w = np.concatenate([[e[arm].get("wait_steps_mean", 0.0)
                                     for e in pilots[s]["buckets"][bucket]["episodes"]]
                                    for s in seeds])
                if np.isnan(st).any():
                    continue
                if w.sum() > 0:
                    any_wait = True
                print(f"[{bucket}] {arm:22s} delivered={d.mean():7.3f} "
                      f"strict={st.mean():7.3f} straddle={(d - st).mean():5.3f} "
                      f"waits/route={w.mean():5.2f}")

            # -- 3. ordering gate: recorded vs strict paired deltas --
            for a, b in PAIRS:
                if a not in arms or b not in arms:
                    continue
                rec = {s: np.array([e[a]["delivered_mean"] - e[b]["delivered_mean"]
                                    for e in pilots[s]["buckets"][bucket]["episodes"]])
                       for s in seeds}
                stz = {s: np.array([e[a]["delivered_strict_mean"]
                                    - e[b]["delivered_strict_mean"]
                                    for e in pilots[s]["buckets"][bucket]["episodes"]])
                       for s in seeds}
                rm, rlo, rhi = cluster_ci(rec, seed=7)
                sm, slo, shi = cluster_ci(stz, seed=7)
                flip = sig(rlo, rhi) != sig(slo, shi)
                any_flip |= flip
                mark = "  <-- ORDERING CHANGE" if flip else ""
                print(f"[{bucket}] {a} - {b}:")
                print(f"          recorded {rm:+.3f} [{rlo:+.3f},{rhi:+.3f}] ({sig(rlo, rhi)})"
                      f"  strict {sm:+.3f} [{slo:+.3f},{shi:+.3f}] ({sig(slo, shi)}){mark}")

    print("\n===== VERDICT =====")
    print(f"max identity error (non-OR arms): {max_ident_err:.2e}")
    print(f"wait steps observed: {'YES - consider wait-costs-time probe' if any_wait else 'no'}")
    if any_flip:
        print("FLIPS-FOUND: at least one headline pair changes sign/significance "
              "under strict counting. STOP - author decision on strict reruns.")
    else:
        print("KEEP-WORDING: no ordering changes under strict arrival<=H counting. "
              "Proceed with estimand rename (departure-cutoff) + sensitivity "
              "paragraph; recorded numbers stand.")


if __name__ == "__main__":
    main()
