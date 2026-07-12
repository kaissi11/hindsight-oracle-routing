#!/usr/bin/env python3
"""Decision-relevance diagnostics for the disruption process (plan P0.15).

Demonstrates that the synthetic process over real-road matrices genuinely
changes routing decisions rather than merely scaling all costs. Everything is
computed from regenerated pre-sampled schedules (the same deterministic
regeneration the paired suites use) — no model, no GPU, no wall-clock
sensitivity.

Per bucket, over the standard seed-12345 Damascus episode grid, it reports:

  * %% of steps where the nearest-feasible next customer (argmin over the
    effective matrix from a fixed reference node) differs from the
    base-matrix nearest customer;
  * %% of sampled pairwise arc-cost rankings that flip vs the base matrix;
  * %% of OD entries whose cost changes materially (>5%%) or whose
    feasibility changes;
  * action-mask change rate between consecutive steps;
  * mean active blocked-arc share and node-block share;
  * traffic multiplier distribution (mean, sd, p5/p95) and lag-1
    autocorrelation of the per-step mean multiplier.

The reference tour position is the depot at t=0 and a random unserved node
per step thereafter (fixed RNG), so the statistics describe the process, not
any particular policy.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import scenario_bucket_eval_v2 as stage2
from matched_information_eval import (
    _load_instance_pool,
    _make_initial_states,
    _resolve_from_root,
    _write_json_atomic,
)

ROOT = Path(__file__).resolve().parent
COST_CHANGE_THRESHOLD = 0.05  # relative change that counts as "changed"


def analyze_schedule(
    state: stage2.SimStateV2,
    initial_eff: np.ndarray,
    schedule: Sequence[tuple[np.ndarray, np.ndarray]],
    rng: np.random.RandomState,
    n_arc_pairs: int = 200,
) -> dict[str, Any]:
    n = state.n_nodes
    base = state.base_dist
    customers = np.arange(1, n)

    nearest_changed = []
    pair_rank_flips = []
    cost_changed_share = []
    feasibility_changed_share = []
    mask_change_rate = []
    blocked_arc_share = []
    node_block_share = []
    mean_multipliers = []

    previous_feasible = None
    events = [(initial_eff, state.node_blocked)] + list(schedule)
    for step, (eff, node_blocked) in enumerate(events):
        reference = 0 if step == 0 else int(rng.choice(customers))
        finite = np.isfinite(eff[reference, customers])
        available = (node_blocked[customers] < 0.5) & finite
        if available.any():
            eff_costs = np.where(available, eff[reference, customers], np.inf)
            base_costs = np.where(available, base[reference, customers], np.inf)
            nearest_changed.append(
                int(np.argmin(eff_costs)) != int(np.argmin(base_costs))
            )

        rows = rng.randint(0, n, n_arc_pairs)
        cols_a = rng.randint(0, n, n_arc_pairs)
        cols_b = rng.randint(0, n, n_arc_pairs)
        valid = (rows != cols_a) & (rows != cols_b) & (cols_a != cols_b)
        base_order = base[rows, cols_a] < base[rows, cols_b]
        with np.errstate(invalid="ignore"):
            eff_order = eff[rows, cols_a] < eff[rows, cols_b]
        pair_rank_flips.append(
            float(np.mean((base_order != eff_order)[valid]))
        )

        off_diagonal = ~np.eye(n, dtype=bool)
        finite_eff = np.isfinite(eff)
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = np.abs(eff - base) / np.where(base > 0, base, 1.0)
        cost_changed_share.append(
            float(
                np.mean(
                    (relative > COST_CHANGE_THRESHOLD)[off_diagonal & finite_eff]
                )
            )
        )
        feasibility_changed_share.append(
            float(np.mean(~finite_eff[off_diagonal]))
        )
        blocked_arc_share.append(float(np.mean(~finite_eff[off_diagonal])))
        node_block_share.append(float(np.mean(node_blocked[1:] > 0.5)))

        feasible_vector = (node_blocked[1:] < 0.5) & np.isfinite(
            eff[0, 1:]
        )
        if previous_feasible is not None:
            mask_change_rate.append(
                float(np.mean(feasible_vector != previous_feasible))
            )
        previous_feasible = feasible_vector

        with np.errstate(invalid="ignore"):
            ratio = eff[off_diagonal & finite_eff] / base[off_diagonal & finite_eff]
        mean_multipliers.append(float(np.mean(ratio)))

    multipliers = np.asarray(mean_multipliers)
    if len(multipliers) > 2 and multipliers.std() > 0:
        lag1 = float(np.corrcoef(multipliers[:-1], multipliers[1:])[0, 1])
    else:
        lag1 = None

    return {
        "nearest_feasible_changed_rate": float(np.mean(nearest_changed)),
        "pairwise_arc_rank_flip_rate": float(np.mean(pair_rank_flips)),
        "cost_changed_share_gt5pct": float(np.mean(cost_changed_share)),
        "blocked_arc_share": float(np.mean(blocked_arc_share)),
        "node_block_share": float(np.mean(node_block_share)),
        "mask_change_rate": float(np.mean(mask_change_rate)),
        "multiplier_mean": float(multipliers.mean()),
        "multiplier_sd": float(multipliers.std()),
        "multiplier_lag1_autocorr": lag1,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Environment decision-relevance diagnostics (P0.15)"
    )
    parser.add_argument(
        "--instance-pool",
        default="results/osrm_instance_pool/pool.npz",
    )
    parser.add_argument("--n-episodes", type=int, default=40)
    parser.add_argument("--n-nodes", type=int, default=20)
    parser.add_argument("--num-instances", type=int, default=4)
    parser.add_argument(
        "--buckets", nargs="+", choices=("low", "medium", "high"),
        default=["low", "medium", "high"],
    )
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument(
        "--output", default="results/decision_relevance_diagnostics.json"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    pool = _load_instance_pool(args.instance_pool)
    max_steps = args.max_steps or (args.n_nodes * 8 + 64)
    output_path = _resolve_from_root(args.output)

    output: dict[str, Any] = {
        "schema_version": "decision_relevance_diagnostics.v1",
        "complete": False,
        "config": {**vars(args), "max_steps_effective": max_steps},
        "provenance": {
            "semantics": (
                "disruptions act on OSRM-derived OD travel-time matrix "
                "arcs (spatially correlated OD-arc/reachability "
                "disruptions), advancing once per decision step"
            ),
            "reference": (
                "statistics computed on regenerated pre-sampled schedules; "
                "reference node = depot at t=0, random unserved node after"
            ),
        },
        "buckets": {},
    }

    for bucket_index, bucket in enumerate(args.buckets):
        cfg = stage2.apply_bucket_v2(
            stage2.ResearchEnvV2Config(
                n_nodes=args.n_nodes,
                num_instances=args.num_instances,
                device="cpu",
                auto_reset=False,
                use_augmentation=True,
            ),
            bucket,
        )
        started = time.perf_counter()
        per_instance: list[dict[str, Any]] = []
        for episode_index in range(args.n_episodes):
            episode_seed = args.base_seed + episode_index + 10000 * bucket_index
            np.random.seed(episode_seed)
            import torch

            torch.manual_seed(episode_seed)
            initial_states = _make_initial_states(cfg, episode_seed, pool)
            for instance_index, (initial_state, _src) in enumerate(initial_states):
                schedule_seed = episode_seed + 999 + instance_index
                initial_eff, schedule = stage2.presample_schedule_v2(
                    initial_state, cfg, max_steps, schedule_seed
                )
                per_instance.append(
                    analyze_schedule(
                        initial_state,
                        initial_eff,
                        schedule,
                        np.random.RandomState(schedule_seed + 5),
                    )
                )

        keys = per_instance[0].keys()
        output["buckets"][bucket] = {
            "n_instances": len(per_instance),
            **{
                key: float(
                    np.mean(
                        [
                            item[key]
                            for item in per_instance
                            if item[key] is not None
                        ]
                    )
                )
                for key in keys
            },
        }
        print(
            f"[DR] {bucket}: {json.dumps(output['buckets'][bucket], indent=2)} "
            f"({time.perf_counter() - started:.0f}s)",
            flush=True,
        )
        _write_json_atomic(output_path, output)

    output["complete"] = True
    _write_json_atomic(output_path, output)
    print(f"[DR] wrote {output_path}")


if __name__ == "__main__":
    main()
