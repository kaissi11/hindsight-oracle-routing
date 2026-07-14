#!/usr/bin/env python3
"""KPI-aligned (completion-lexicographic) rolling-OR (blueprint SCHEDULED #1, §3.6).

The paper's ``rolling_or`` baseline minimizes route time over all open
customers. The primary KPI, however, is completion before H — so the
baseline optimizes a different objective than the one it is scored on
(limitation (10)). This variant re-solves, at every decision step, the
lexicographic objective the evaluation actually uses:

    1. maximize customers served before the remaining horizon;
    2. among equal completion, minimize route time,

implemented as the standard scalarization  U = M*C - T  with a drop penalty
M = 10^7 s, provably larger than any feasible route-time difference (route
time is bounded by the 8 h horizon = 28,800 s). Served-by-H is enforced with
an OR-Tools time dimension capped at the remaining horizon; blocked arcs get
a transit larger than the horizon, which excludes them through the same
dimension.

Pairing: identical instance and schedule regeneration as the recorded
seed-12345 suite (same machinery as matched_information_eval.py), so every
episode pairs cross-run with the recorded ``rolling_or``, ``repair_nn2opt``
and ``policy_v1_lookahead`` rows.

IMPORTANT: the per-step solve is wall-clock budgeted (default 30 ms, same as
the recorded rolling_or). Run this ONLY on an otherwise quiet machine —
CPU contention starves the solver and biases the comparison.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

import scenario_bucket_eval_v2 as stage2
from generic_hindsight_eval import bootstrap_ci
from matched_information_eval import (
    _load_instance_pool,
    _make_initial_states,
    _resolve_from_root,
    _write_json_atomic,
)

ROOT = Path(__file__).resolve().parent
DROP_PENALTY = 10_000_000  # > any feasible route-time difference (horizon 28,800 s)


def solve_path_kpi_aligned(
    current: int,
    feasible_nodes: Sequence[int],
    travel_cost: np.ndarray,
    horizon_remaining_sec: float,
    time_limit_ms: int,
) -> list[int]:
    """Open path from ``current`` maximizing (served within horizon, -time)."""
    if not feasible_nodes:
        return []
    horizon = max(1, int(horizon_remaining_sec))
    over_horizon = horizon + 1
    real_nodes = [current] + list(feasible_nodes)
    dummy_end = len(real_nodes)
    num_nodes = len(real_nodes) + 1
    mat = np.full((num_nodes, num_nodes), over_horizon, dtype=np.int64)
    for i, ni in enumerate(real_nodes):
        for j, nj in enumerate(real_nodes):
            if i == j:
                mat[i, j] = 0
            else:
                c = float(travel_cost[ni, nj])
                mat[i, j] = max(1, int(round(c))) if np.isfinite(c) else over_horizon
    mat[:, dummy_end] = 0
    mat[dummy_end, :] = over_horizon
    mat[dummy_end, dummy_end] = 0

    manager = pywrapcp.RoutingIndexManager(num_nodes, 1, [0], [dummy_end])
    routing = pywrapcp.RoutingModel(manager)

    def transit(from_index: int, to_index: int) -> int:
        return int(mat[manager.IndexToNode(from_index), manager.IndexToNode(to_index)])

    transit_index = routing.RegisterTransitCallback(transit)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_index)
    routing.AddDimension(transit_index, 0, horizon, True, "Time")
    for node in range(1, dummy_end):
        routing.AddDisjunction([manager.NodeToIndex(node)], DROP_PENALTY)

    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    parameters.time_limit.FromMilliseconds(time_limit_ms)
    solution = routing.SolveWithParameters(parameters)
    if solution is None:
        return []
    route: list[int] = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        if node not in (0, dummy_end):
            route.append(real_nodes[node])
        index = solution.Value(routing.NextVar(index))
    return route


def choose_kpi_rolling_or_action(
    state: stage2.SimStateV2, time_limit_ms: int
) -> int:
    mask = stage2.valid_mask_v2(state)
    feasible = [
        j
        for j in range(1, state.n_nodes)
        if (not state.visited[j]) and state.node_blocked[j] < 0.5
    ]
    if not feasible:
        return state.current_node
    cost = np.where(np.isfinite(state.eff_dist), state.eff_dist, stage2.BIG)
    remaining = max(0.0, state.horizon_sec - state.elapsed_time)
    route = solve_path_kpi_aligned(
        state.current_node, feasible, cost, remaining, time_limit_ms
    )
    for nxt in route:
        if mask[nxt]:
            return int(nxt)
    # Same fallback semantics as the recorded rolling_or.
    feasible_arr = np.flatnonzero(mask)
    if feasible_arr.size == 0:
        return state.current_node
    return int(feasible_arr[np.argmin(cost[state.current_node, feasible_arr])])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Completion-lexicographic rolling-OR on the paired suite"
    )
    parser.add_argument(
        "--instance-pool",
        default="results/osrm_instance_pool/pool.npz",
        help="cached pool; pass an empty string for synthetic smoke instances",
    )
    parser.add_argument("--n-episodes", type=int, default=40)
    parser.add_argument("--n-nodes", type=int, default=20)
    parser.add_argument("--num-instances", type=int, default=4)
    parser.add_argument("--time-limit-ms", type=int, default=30)
    parser.add_argument(
        "--buckets", nargs="+", choices=("low", "medium", "high"),
        default=["low", "medium", "high"],
    )
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument(
        "--horizon-hours", type=float, default=8.0,
        help="mission horizon H in hours (pair H=4 runs with the recorded "
             "hstress_h4 suites)",
    )
    parser.add_argument(
        "--reference",
        default="results/scenario_bucket_v2_osrm_s5_seed_12345.json",
        help="recorded control suite for cross-run pairing; empty to skip",
    )
    parser.add_argument("--output", default="results/kpi_aligned_rolling_or.json")
    return parser


def summarize_bucket(
    episodes: list[dict[str, Any]],
    reference_episodes: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    delivered = np.array([e["kpi_rolling_or"]["delivered_mean"] for e in episodes])
    times = np.array([e["kpi_rolling_or"]["time_mean"] for e in episodes])
    summary: dict[str, Any] = {
        "n_episodes": len(episodes),
        "kpi_rolling_or_delivered": float(delivered.mean()),
        "kpi_rolling_or_time_sec": float(times.mean()),
    }
    if reference_episodes is not None and len(reference_episodes) == len(episodes):
        for method in ("rolling_or", "repair_nn2opt", "policy_v1_lookahead"):
            ref_delivered = np.array(
                [e[method]["delivered_mean"] for e in reference_episodes]
            )
            ref_time = np.array([e[method]["time_mean"] for e in reference_episodes])
            summary[f"paired_kpiOR_minus_{method}"] = {
                "delivered_mean": float((delivered - ref_delivered).mean()),
                "delivered_ci95": bootstrap_ci(delivered - ref_delivered),
                "time_mean_sec": float((times - ref_time).mean()),
            }
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    pool = _load_instance_pool(args.instance_pool)
    max_steps = args.max_steps or (args.n_nodes * 8 + 64)
    output_path = _resolve_from_root(args.output)

    reference = None
    if args.reference:
        reference_path = _resolve_from_root(args.reference)
        if reference_path.exists():
            reference = json.loads(reference_path.read_text(encoding="utf-8"))

    output: dict[str, Any] = {
        "schema_version": "kpi_aligned_rolling_or.v1",
        "complete": False,
        "config": {**vars(args), "max_steps_effective": max_steps},
        "provenance": {
            "objective": (
                "lexicographic (max served before remaining horizon, min time) "
                f"via drop penalty {DROP_PENALTY} with an OR-Tools time "
                "dimension capped at the remaining horizon"
            ),
            "budget": f"{args.time_limit_ms} ms per solve (wall-clock)",
            "quiet_machine_required": True,
            "schedule_pairing": (
                "instances and pre-sampled schedules regenerate from the seed; "
                "cross-run paired with the recorded reference suite"
            ),
        },
        "buckets": {},
    }
    _write_json_atomic(output_path, output)

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
        cfg.time_horizon_sec = args.horizon_hours * 3600.0
        bucket_records: list[dict[str, Any]] = []
        output["buckets"][bucket] = {"episodes": bucket_records}
        started = time.perf_counter()

        for episode_index in range(args.n_episodes):
            episode_seed = args.base_seed + episode_index + 10000 * bucket_index
            np.random.seed(episode_seed)
            torch.manual_seed(episode_seed)
            initial_states = _make_initial_states(cfg, episode_seed, pool)

            init_states, init_effs, schedules = [], [], []
            for instance_index, (initial_state, _source) in enumerate(initial_states):
                schedule_seed = episode_seed + 999 + instance_index
                initial_eff, schedule = stage2.presample_schedule_v2(
                    initial_state, cfg, max_steps, schedule_seed
                )
                init_states.append(initial_state)
                init_effs.append(initial_eff)
                schedules.append(schedule)

            def act_fn(mode, state, ctrl=None):
                if mode == "init":
                    return None
                return choose_kpi_rolling_or_action(state, args.time_limit_ms)

            episode_started = time.perf_counter()
            result = stage2.run_rollout_v2(
                init_states, init_effs, schedules, max_steps, act_fn
            )
            bucket_records.append(
                {
                    "episode_index": episode_index,
                    "episode_seed": episode_seed,
                    "kpi_rolling_or": {
                        "delivered_mean": float(result["delivered_mean"]),
                        "time_mean": float(result["time_mean"]),
                        "wall_sec": time.perf_counter() - episode_started,
                    },
                }
            )
            if (episode_index + 1) % 5 == 0:
                print(
                    f"[KPI-OR] {bucket} {episode_index + 1}/{args.n_episodes} "
                    f"({time.perf_counter() - started:.0f}s)",
                    flush=True,
                )
            _write_json_atomic(output_path, output)

        reference_episodes = None
        if reference is not None:
            reference_episodes = (
                reference.get("buckets", {}).get(bucket, {}).get("episodes")
            )
        output["buckets"][bucket]["summary"] = summarize_bucket(
            bucket_records, reference_episodes
        )
        _write_json_atomic(output_path, output)
        print(
            f"[KPI-OR] {bucket} summary: "
            f"{json.dumps(output['buckets'][bucket]['summary'], indent=2)}",
            flush=True,
        )

    output["complete"] = True
    _write_json_atomic(output_path, output)
    print(f"[KPI-OR] wrote {output_path}")


if __name__ == "__main__":
    main()
