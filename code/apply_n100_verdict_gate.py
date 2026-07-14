#!/usr/bin/env python3
"""Read the fresh N=100 look-8 - repair medium verdict and print the branch
to apply from POST_COMPUTE_RUNBOOK_2026-07-12.md (CONFIRM vs FLIP), plus
fill-in numbers for the prewritten templates.

  python apply_n100_verdict_gate.py

Does not edit the master — print-only decision aid after equivalence_analysis.py.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EQ = HERE / "results" / "equivalence_analysis.json"


def main() -> None:
    if not EQ.exists():
        raise SystemExit(f"missing {EQ} — run equivalence_analysis.py first")
    data = json.loads(EQ.read_text(encoding="utf-8"))

    # schema: {variant: {cells: [{pair, bucket, verdict, mean, ci95_cluster, ...}]}}
    n100 = data.get("n100_s5") or data.get("n100") or {}
    if not n100:
        # try nested / alternate keys
        for k, v in data.items():
            if "n100" in k.lower() and isinstance(v, dict) and "cells" in v:
                n100 = v
                break
    if not n100:
        raise SystemExit(f"no n100 suite in {EQ.name}; keys={list(data)[:20]}")

    cells = n100.get("cells", [])
    target = None
    for c in cells:
        if c.get("bucket") == "medium" and c.get("pair") == "look-8 - repair":
            target = c
            break
    if target is None:
        print("[gate] medium candidates:")
        for c in cells:
            if c.get("bucket") == "medium":
                print(" ", c.get("pair"), c.get("verdict"), c.get("mean"),
                      c.get("ci95_cluster"), "n_seeds=", c.get("n_seeds"))
        raise SystemExit("could not locate N=100 medium look-8 - repair cell")

    verdict = target.get("verdict")
    mean = target.get("mean")
    ci95 = target.get("ci95_cluster") or target.get("ci95")
    ci90 = target.get("ci90_cluster") or target.get("ci90")
    n_seeds = target.get("n_seeds")
    n_eps = target.get("n_episodes")

    print("===== N=100 look-8 - repair MEDIUM =====")
    print(f"verdict:  {verdict}")
    print(f"mean:     {mean:+.3f}")
    print(f"ci95:     {ci95}")
    print(f"ci90:     {ci90}")
    print(f"n_seeds:  {n_seeds}  n_episodes: {n_eps}")

    if verdict == "different" and mean is not None and mean > 0:
        branch = "CONFIRM"
        note = "keep the modest N=100-medium gain claim; update numbers/seed counts"
    else:
        branch = "FLIP"
        note = "apply FLIP verbatim templates in POST_COMPUTE_RUNBOOK §2"

    print(f"\n>>> BRANCH: {branch}")
    print(f">>> {note}")

    # recount baseline (look vs repair / rolling_or) across all suites
    from collections import Counter
    vc = Counter()
    for suite, v in data.items():
        if not isinstance(v, dict):
            continue
        for c in v.get("cells", []):
            pair = c.get("pair", "")
            if pair.startswith("look-8 - ") and "oracle" not in pair:
                vc[c.get("verdict")] += 1
                print(f"  [{suite}] {pair} / {c['bucket']}: {c['verdict']} "
                      f"{c['mean']:+.3f} seeds={c['n_seeds']}")
    print("\nlook-8 vs baseline verdict counts:")
    for v, n in sorted(vc.items()):
        print(f"  {v}: {n}")
    print(f"  total baseline cells: {sum(vc.values())}  "
          f"(expect 24 = 4 settings x 2 baselines x 3 buckets)")


if __name__ == "__main__":
    main()
