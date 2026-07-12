#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    HAS_ORTOOLS = True
except Exception:
    HAS_ORTOOLS = False

from research_env import ResearchEnv, ResearchEnvConfig


LARGE_COST = 10**9


@dataclass
class RowPlanState:
    planned_route: List[int]
    last_traffic: float
    last_blocked: np.ndarray
    since_replan: int = 0


def route_path_cost(route: List[int], current: int, cost_matrix: np.ndarray) -> float:
    total = 0.0
    prev = current
    for node in route:
        total += float(cost_matrix[prev, node])
        prev = node
    return total


def two_opt_path(route: List[int], current: int, cost_matrix: np.ndarray, max_passes: int = 20) -> List[int]:
    if len(route) < 4:
        return route[:]
    best = route[:]
    best_cost = route_path_cost(best, current, cost_matrix)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        n = len(best)
        for i in range(n - 1):
            for j in range(i + 2, n):
                cand = best[:i] + list(reversed(best[i:j])) + best[j:]
                cand_cost = route_path_cost(cand, current, cost_matrix)
                if cand_cost + 1e-9 < best_cost:
                    best = cand
                    best_cost = cand_cost
                    improved = True
    return best


def solve_path_with_ortools(current: int, feasible_nodes: List[int], travel_cost: np.ndarray, time_limit_ms: int) -> List[int]:
    if not feasible_nodes:
        return []

    real_nodes = [current] + feasible_nodes
    dummy_end = len(real_nodes)
    num_nodes = len(real_nodes) + 1

    mat = np.full((num_nodes, num_nodes), LARGE_COST, dtype=np.int64)
    for i, ni in enumerate(real_nodes):
        for j, nj in enumerate(real_nodes):
            if i == j:
                mat[i, j] = 0
            else:
                c = float(travel_cost[ni, nj])
                mat[i, j] = max(1, int(round(c))) if np.isfinite(c) else LARGE_COST
    for i in range(len(real_nodes)):
        mat[i, dummy_end] = 0
    mat[dummy_end, dummy_end] = 0

    manager = pywrapcp.RoutingIndexManager(num_nodes, 1, [0], [dummy_end])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return int(mat[i, j])

    transit_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_index)

    search = pywrapcp.DefaultRoutingSearchParameters()
    search.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search.time_limit.FromMilliseconds(time_limit_ms)

    solution = routing.SolveWithParameters(search)
    if solution is None:
        return feasible_nodes[:]

    route = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        if node != 0 and node != dummy_end:
            route.append(real_nodes[node])
        index = solution.Value(routing.NextVar(index))
    return route


def should_replan(
    row_state: RowPlanState,
    current_traffic: float,
    current_blocked: np.ndarray,
    route_exhausted: bool,
    replan_every: int,
    replan_on_disruption: bool,
    traffic_delta_threshold: float,
) -> bool:
    if route_exhausted:
        return True
    if replan_every > 0 and row_state.since_replan >= replan_every:
        return True
    if replan_on_disruption:
        traffic_changed = abs(current_traffic - row_state.last_traffic) >= traffic_delta_threshold
        blocked_changed = not np.array_equal((current_blocked > 0.5).astype(np.int8), (row_state.last_blocked > 0.5).astype(np.int8))
        if traffic_changed or blocked_changed:
            return True
    return False


def make_env(args, device: str, static_mode: bool) -> ResearchEnv:
    cfg = ResearchEnvConfig(
        n_nodes=args.n_nodes,
        num_instances=args.num_instances,
        device=device,
        use_osrm_instances=args.use_osrm,
        auto_reset=False,
        use_augmentation=not args.use_osrm,
    )
    if static_mode:
        if hasattr(cfg, "block_prob_per_step"):
            cfg.block_prob_per_step = 0.0
        if hasattr(cfg, "unblock_prob_per_step"):
            cfg.unblock_prob_per_step = 0.0
        if hasattr(cfg, "traffic_rw_std"):
            cfg.traffic_rw_std = 0.0
    return ResearchEnv(cfg)


def run_episode(env: ResearchEnv, args) -> Dict[str, float]:
    obs, info = env.reset()
    num_rows = env.num_envs
    done = torch.zeros(num_rows, dtype=torch.bool, device=env.device)
    rewards_total = torch.zeros(num_rows, device=env.device)
    row_plans: List[Optional[RowPlanState]] = [None for _ in range(num_rows)]

    step = 0
    max_steps = env.n_nodes * 8 + 64

    while not done.all():
        if step >= max_steps:
            raise RuntimeError(f"Episode exceeded max_steps={max_steps}")

        actions = env.current_node.clone()

        for row in range(num_rows):
            if done[row]:
                continue

            current = int(env.current_node[row].item())
            visited = env.visited[row].detach().cpu().numpy().astype(bool)
            blocked = env.blocked[row].detach().cpu().numpy().copy()
            traffic = float(env.traffic[row].item())
            base_dist = env.base_dist[row].detach().cpu().numpy()
            feasible_nodes = [j for j in range(1, env.n_nodes) if (not visited[j]) and blocked[j] < 0.5]
            travel_cost = base_dist * traffic

            state = row_plans[row]
            route_exhausted = state is None or len(state.planned_route) == 0

            replan = True if state is None else should_replan(
                state, traffic, blocked, route_exhausted,
                args.replan_every, args.replan_on_disruption, args.traffic_delta_threshold,
            )

            if replan:
                route = solve_path_with_ortools(current, feasible_nodes, travel_cost, args.ortools_time_limit_ms)
                if args.apply_two_opt and route:
                    route = two_opt_path(route, current, travel_cost, max_passes=args.two_opt_max_passes)
                row_plans[row] = RowPlanState(
                    planned_route=route,
                    last_traffic=traffic,
                    last_blocked=blocked.copy(),
                    since_replan=0,
                )
                state = row_plans[row]

            while state is not None and state.planned_route:
                nxt = state.planned_route[0]
                if nxt < env.n_nodes and (not visited[nxt]) and blocked[nxt] < 0.5:
                    break
                state.planned_route.pop(0)

            if state is not None and state.planned_route:
                actions[row] = int(state.planned_route.pop(0))
                state.since_replan += 1
            else:
                actions[row] = current

        obs, rewards, done_step, info = env.step(actions)
        rewards_total += rewards * (~done).float()
        done = done | done_step
        step += 1

    b = env.cfg.num_instances
    p = getattr(env, "pomo_size", 1)

    if b > 0 and env.num_envs == b * p:
        rewards_reshaped = rewards_total.view(b, p)
        best_rewards = rewards_reshaped.max(dim=1).values
        delivered = env.visited.sum(dim=1).float().view(b, p).max(dim=1).values.mean().item()
        elapsed = env.elapsed_time.view(b, p).min(dim=1).values.mean().item()
    else:
        best_rewards = rewards_total
        delivered = env.visited.sum(dim=1).float().mean().item()
        elapsed = env.elapsed_time.mean().item()

    return {
        "reward_mean": float(best_rewards.mean().item()),
        "delivered_mean": float(delivered),
        "time_mean": float(elapsed),
    }


def aggregate(results: List[Dict[str, float]]) -> Dict[str, float]:
    return {
        "reward_mean": float(np.mean([r["reward_mean"] for r in results])),
        "delivered_mean": float(np.mean([r["delivered_mean"] for r in results])),
        "time_mean": float(np.mean([r["time_mean"] for r in results])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["static", "dynamic", "both"], default="both")
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--n-nodes", type=int, default=20)
    parser.add_argument("--num-instances", type=int, default=4)
    parser.add_argument("--use-osrm", action="store_true")
    parser.add_argument("--replan-every", type=int, default=1)
    parser.add_argument("--replan-on-disruption", action="store_true")
    parser.add_argument("--traffic-delta-threshold", type=float, default=0.02)
    parser.add_argument("--ortools-time-limit-ms", type=int, default=30)
    parser.add_argument("--apply-two-opt", action="store_true")
    parser.add_argument("--two-opt-max-passes", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--save-json", default="results/rolling_horizon_or_baseline.json")
    args = parser.parse_args()

    if not HAS_ORTOOLS:
        raise RuntimeError("OR-Tools is not available in this Python environment.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"config": vars(args), "ortools_available": HAS_ORTOOLS}

    print(f"[ROLLING OR] Device: {device}")
    print(f"[ROLLING OR] mode={args.mode}, episodes={args.n_episodes}, replan_every={args.replan_every}, replan_on_disruption={args.replan_on_disruption}, ortools_time_limit_ms={args.ortools_time_limit_ms}, apply_two_opt={args.apply_two_opt}")

    for mode_name, static_flag in [("static", True), ("dynamic", False)]:
        if args.mode not in [mode_name, "both"]:
            continue

        results = []
        for ep in range(args.n_episodes):
            seed = args.base_seed + ep
            np.random.seed(seed)
            torch.manual_seed(seed)
            env = make_env(args, device, static_flag)
            metrics = run_episode(env, args)
            results.append(metrics)
            print(f"[ROLLING OR] {mode_name} episode {ep+1}/{args.n_episodes} complete | reward={metrics['reward_mean']:.4f} | delivered={metrics['delivered_mean']:.2f} | time={metrics['time_mean']:.2f}")

        out[mode_name] = aggregate(results)

    save_path = Path(args.save_json)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[ROLLING OR] Saved JSON: {save_path}")


if __name__ == "__main__":
    main()
