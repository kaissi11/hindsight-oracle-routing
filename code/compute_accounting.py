#!/usr/bin/env python3
"""Compute-accounting extraction (the paper's per-decision compute table).

Pools the recorded per-episode ``wall_sec`` of every method from the 5-seed
Damascus control suites (no new runs, no timing games — these are the wall
times measured during the original recorded evaluations on the one fixed
machine, RTX 3060 + CPU). Emits results/compute_accounting.json and a
paper-ready markdown table paper/assets/table_compute.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ASSETS = HERE / "paper" / "assets"
SEEDS = (12345, 13345, 14345, 15345, 16345)
BUCKETS = ("low", "medium", "high")

# method key -> (paper name, objective, replan trigger, per-decision work, budget, device, deployable)
METHODS = {
    "reactive_nn": (
        "reactive-NN", "nearest feasible", "every step",
        "argmin over row", "—", "CPU", "yes",
    ),
    "repair_nn2opt": (
        "lightweight repair", "NN + 2-opt plan; defer/reinsert", "event-triggered",
        "list ops; 2-opt on events", "none (to convergence)", "CPU", "yes",
    ),
    "rolling_or": (
        "rolling-OR (time-first)", "min route time", "every step",
        "one OR-Tools solve", "30 ms/solve", "CPU", "yes",
    ),
    "policy_v2_greedy": (
        "greedy", "policy argmax", "every step",
        "1 forward pass", "—", "GPU", "yes",
    ),
    "policy_v1_lookahead": (
        "look-8", "lex (completion, time), frozen matrix", "every step",
        "K=8 batched forwards per inner step", "—", "GPU", "yes",
    ),
    "policy_v1_samplexN": (
        "oracle-8 (hindsight-selected best-of-K)", "retrospective lex selection", "n/a",
        "8 complete closed-loop episodes", "—", "GPU", "no",
    ),
}


def main() -> None:
    per_method: dict[str, dict[str, list[float]]] = {
        key: {bucket: [] for bucket in BUCKETS} for key in METHODS
    }
    for seed in SEEDS:
        data = json.loads(
            (RESULTS / f"scenario_bucket_v2_osrm_s5_seed_{seed}.json").read_text(
                encoding="utf-8"
            )
        )
        for bucket in BUCKETS:
            for episode in data["buckets"][bucket]["episodes"]:
                for key in METHODS:
                    if key in episode and "wall_sec" in episode[key]:
                        per_method[key][bucket].append(
                            float(episode[key]["wall_sec"])
                        )

    summary = {
        key: {
            bucket: {
                "mean_wall_sec": float(np.mean(values)),
                "n_episodes": len(values),
            }
            for bucket, values in buckets.items()
        }
        for key, buckets in per_method.items()
    }
    (RESULTS / "compute_accounting.json").write_text(
        json.dumps(
            {
                "seeds": list(SEEDS),
                "source": "scenario_bucket_v2_osrm_s5_seed_*.json wall_sec",
                "hardware": "one RTX 3060 (GPU methods) + host CPU; no batching across episodes",
                "methods": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "**Table 4x — compute, objective, and implementation fairness "
        "(Damascus N=20 v2 control suite, 5 seeds, n=200 episodes/bucket; "
        "mean measured wall time per episode; one RTX 3060 + host CPU; "
        "decision latency excluded from makespan, §5).**",
        "",
        "| Method | Objective | Replan trigger | Per-decision work | Solver budget "
        "| Device | Deployable | Wall s/episode (low/med/high) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, meta in METHODS.items():
        name, objective, trigger, work, budget, device, deployable = meta
        walls = "/".join(
            (
                f"{summary[key][bucket]['mean_wall_sec']:.1f}"
                if summary[key][bucket]["mean_wall_sec"] >= 1
                else f"{summary[key][bucket]['mean_wall_sec']:.3f}"
            )
            for bucket in BUCKETS
        )
        lines.append(
            f"| {name} | {objective} | {trigger} | {work} | {budget} "
            f"| {device} | {deployable} | {walls} |"
        )
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "table_compute.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {RESULTS / 'compute_accounting.json'}")
    print(f"wrote {ASSETS / 'table_compute.md'}")


if __name__ == "__main__":
    main()
