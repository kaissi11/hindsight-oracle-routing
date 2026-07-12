#!/usr/bin/env python3
"""Stage 1: repair baselines vs learned policy on the v6.2 scenario-bucket harness.

Adds two cheap reactive baselines to the existing paired evaluation:

- repair_nn2opt : plan once (nearest-neighbor + 2-opt), then *repair* on
  disruptions: defer blocked stops, 2-opt-patch the remaining route,
  cheapest-reinsert stops when they unblock. No solver, no GPU.
- reactive_nn   : myopic nearest feasible neighbor at every step.

All methods are evaluated on IDENTICAL instances and IDENTICAL presampled
disruption schedules (same machinery as v6.2/scenario_bucket_eval.py), so
episode-level differences are paired. Reports paired deltas with bootstrap
95% CIs for the Stage 1 decision gate.

v6.2 is imported read-only; nothing there is modified.

Run from v6.3:
    python scenario_bucket_eval_with_repair.py \
        --policy-checkpoint ../v6.2/checkpoints_research_pomo/research_best.pt
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

V62_DIR = Path(__file__).resolve().parent  # merged layout: frozen deps live alongside
sys.path.insert(0, str(V62_DIR))

from scenario_bucket_eval import (  # noqa: E402  (v6.2, frozen)
    SimState,
    apply_action_and_advance,
    apply_bucket,
    choose_rolling_or_action,
    init_states_from_env,
    load_policy,
    presample_events,
    run_policy_best_of_k,
    run_single_rollout,
    valid_mask,
)
from rolling_horizon_or_baseline import two_opt_path  # noqa: E402  (v6.2, frozen)
from research_env import ResearchEnv, ResearchEnvConfig  # noqa: E402  (v6.2, frozen)
from connected_instance_builder import normalize_lonlat  # noqa: E402  (v6.2, frozen)
from osrm_client import DAMASCUS_BBOX  # noqa: E402  (v6.2, frozen)


def init_states_from_pool(pool_lonlats, pool_durations, idx, cfg) -> list[SimState]:
    """Build SimStates from cached OSRM instances (mirrors init_states_from_env)."""
    states = []
    for k in idx:
        n = pool_lonlats.shape[1]
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        states.append(SimState(
            coords=normalize_lonlat(pool_lonlats[k], DAMASCUS_BBOX).astype(np.float32),
            base_dist=pool_durations[k].astype(np.float64).copy(),
            visited=visited,
            blocked=np.zeros(n, dtype=np.float32),
            current_node=0,
            traffic=float(cfg.traffic_init),
            elapsed_time=0.0,
            horizon_sec=float(cfg.time_horizon_sec),
            n_nodes=n,
        ))
    return states


# ---------------------------------------------------------------------------
# Cheap route construction helpers
# ---------------------------------------------------------------------------

def nearest_neighbor_order(current: int, nodes: list[int], cost: np.ndarray) -> list[int]:
    remaining = list(nodes)
    route: list[int] = []
    cur = current
    while remaining:
        nxt = min(remaining, key=lambda j: float(cost[cur, j]))
        route.append(nxt)
        remaining.remove(nxt)
        cur = nxt
    return route


def cheapest_insertion(route: list[int], node: int, current: int, cost: np.ndarray) -> list[int]:
    if not route:
        return [node]
    best_pos, best_delta = 0, float("inf")
    for pos in range(len(route) + 1):
        prev = current if pos == 0 else route[pos - 1]
        if pos == len(route):
            delta = float(cost[prev, node])  # open path: appending adds one arc
        else:
            nxt = route[pos]
            delta = float(cost[prev, node]) + float(cost[node, nxt]) - float(cost[prev, nxt])
        if delta < best_delta:
            best_delta, best_pos = delta, pos
    return route[:best_pos] + [node] + route[best_pos:]


# ---------------------------------------------------------------------------
# Controllers (per-instance persistent state)
# ---------------------------------------------------------------------------

class ReactiveNNController:
    """Myopic: always go to the nearest feasible node."""

    def __init__(self, state: SimState):
        pass  # stateless; signature matches the controller factory

    def act(self, state: SimState) -> int:
        mask = valid_mask(state)
        feasible = np.flatnonzero(mask)
        if feasible.size == 0:
            return state.current_node  # wait
        d = state.base_dist[state.current_node, feasible]
        return int(feasible[int(np.argmin(d))])


class RepairController:
    """Plan once (NN + 2-opt), then repair on blocking changes.

    Traffic in this env is a global scalar multiplier, so it never changes the
    optimal visit order; only blocking changes trigger a repair.
    """

    def __init__(self, state: SimState, two_opt_passes: int = 5):
        self.two_opt_passes = two_opt_passes
        self.deferred: list[int] = []
        self.last_blocked = (state.blocked > 0.5).copy()
        nodes = [j for j in range(1, state.n_nodes) if not state.visited[j]]
        cost = state.base_dist
        route = nearest_neighbor_order(state.current_node, nodes, cost)
        self.route = two_opt_path(route, state.current_node, cost, max_passes=20)
        self._repair(state)

    def _repair(self, state: SimState) -> None:
        blocked = state.blocked > 0.5
        cost = state.base_dist
        cur = state.current_node

        self.route = [j for j in self.route if not state.visited[j]]
        self.deferred = [j for j in self.deferred if not state.visited[j]]

        newly_blocked = [j for j in self.route if blocked[j]]
        if newly_blocked:
            self.route = [j for j in self.route if not blocked[j]]
            self.deferred.extend(j for j in newly_blocked if j not in self.deferred)
            if len(self.route) >= 4:
                self.route = two_opt_path(self.route, cur, cost, max_passes=self.two_opt_passes)

        unblocked = [j for j in self.deferred if not blocked[j]]
        for j in unblocked:
            self.deferred.remove(j)
            self.route = cheapest_insertion(self.route, j, cur, cost)

        self.last_blocked = blocked.copy()

    def act(self, state: SimState) -> int:
        blocked = state.blocked > 0.5
        if not np.array_equal(blocked, self.last_blocked):
            self._repair(state)
        else:
            self.route = [j for j in self.route if not state.visited[j]]

        mask = valid_mask(state)
        while self.route and not mask[self.route[0]]:
            j = self.route.pop(0)
            if blocked[j] and not state.visited[j] and j not in self.deferred:
                self.deferred.append(j)
        if self.route:
            return self.route.pop(0)

        feasible = np.flatnonzero(mask)
        if feasible.size == 0:
            return state.current_node  # everything left is blocked: wait
        d = state.base_dist[state.current_node, feasible]
        return int(feasible[int(np.argmin(d))])


# ---------------------------------------------------------------------------
# Rollout for controller-based methods (mirrors v6.2 run_single_rollout)
# ---------------------------------------------------------------------------

def run_controller_rollout(init_states, schedule, make_controller, max_steps: int):
    states = [copy.deepcopy(s) for s in init_states]
    controllers = [make_controller(s) for s in states]
    decision_time = 0.0
    decisions = 0
    for step_idx in range(max_steps):
        all_done = True
        for i, state in enumerate(states):
            if state.elapsed_time >= state.horizon_sec or state.visited[1:].all():
                continue
            all_done = False
            t0 = time.perf_counter()
            action = controllers[i].act(state)
            decision_time += time.perf_counter() - t0
            decisions += 1
            apply_action_and_advance(state, action, schedule[i][step_idx])
        if all_done:
            break
    times = [s.elapsed_time for s in states]
    delivered = [int(s.visited[1:].sum()) for s in states]
    return {
        "time_mean": float(np.mean(times)),
        "delivered_mean": float(np.mean(delivered)),
        "decision_ms_mean": float(1000.0 * decision_time / max(1, decisions)),
    }


# ---------------------------------------------------------------------------
# Paired statistics
# ---------------------------------------------------------------------------

def bootstrap_ci(deltas: list[float], n_boot: int = 10000, seed: int = 0):
    arr = np.asarray(deltas, dtype=np.float64)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_summary(a: list[dict], b: list[dict], seed: int = 0) -> dict:
    """Per-episode paired deltas of method a minus method b."""
    d_del = [x["delivered_mean"] - y["delivered_mean"] for x, y in zip(a, b)]
    d_time = [x["time_mean"] - y["time_mean"] for x, y in zip(a, b)]
    del_mean, del_lo, del_hi = bootstrap_ci(d_del, seed=seed)
    t_mean, t_lo, t_hi = bootstrap_ci(d_time, seed=seed + 1)
    wins = sum(1 for d in d_del if d > 1e-9)
    losses = sum(1 for d in d_del if d < -1e-9)
    return {
        "delivered_delta_mean": del_mean,
        "delivered_delta_ci95": [del_lo, del_hi],
        "time_delta_mean": t_mean,
        "time_delta_ci95": [t_lo, t_hi],
        "delivered_win_tie_loss": [wins, len(d_del) - wins - losses, losses],
    }


def mean_key(xs, key):
    vals = [x[key] for x in xs if key in x]
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--n-episodes", type=int, default=40)
    parser.add_argument("--n-nodes", type=int, default=20)
    parser.add_argument("--num-instances", type=int, default=4)
    parser.add_argument("--use-osrm", action="store_true")
    parser.add_argument("--instance-pool", default="", help="Path to cached OSRM pool.npz (overrides --use-osrm)")
    parser.add_argument("--policy-n-samples", type=int, default=8)
    parser.add_argument("--ortools-time-limit-ms", type=int, default=30)
    parser.add_argument("--buckets", nargs="+", default=["low", "medium", "high"])
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--save-json", default="results/scenario_bucket_repair.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, policy_ckpt = load_policy(args.policy_checkpoint, device)

    pool_lonlats = pool_durations = None
    if args.instance_pool:
        pool = np.load(args.instance_pool)
        pool_lonlats, pool_durations = pool["lonlats"], pool["durations"]
        print(f"[STAGE1] Instance pool: {args.instance_pool} ({len(pool_lonlats)} instances)")

    print(f"[STAGE1] Device: {device}")
    print(f"[STAGE1] Policy: {args.policy_checkpoint}")
    print(f"[STAGE1] Buckets={args.buckets}, episodes={args.n_episodes}, samples={args.policy_n_samples}")

    out = {
        "config": vars(args),
        "policy_meta": {"epoch": policy_ckpt.get("epoch"), "best_of_pomo": policy_ckpt.get("best_of_pomo")},
        "buckets": {},
    }
    max_steps = args.n_nodes * 8 + 64
    method_keys = ["policy_greedy", "policy_samplexN", "rolling_or", "repair_nn2opt", "reactive_nn"]

    for bidx, bucket in enumerate(args.buckets):
        cfg = ResearchEnvConfig(
            n_nodes=args.n_nodes, num_instances=args.num_instances, device=device.type,
            use_osrm_instances=args.use_osrm, auto_reset=False, use_augmentation=not args.use_osrm,
        )
        cfg = apply_bucket(cfg, bucket)

        per_method: dict[str, list[dict]] = {k: [] for k in method_keys}
        episodes = []

        for ep in range(args.n_episodes):
            seed = args.base_seed + ep + 10000 * bidx
            np.random.seed(seed)
            torch.manual_seed(seed)
            if pool_lonlats is not None:
                inst_idx = np.random.RandomState(seed).choice(
                    len(pool_lonlats), args.num_instances, replace=False)
                init_states = init_states_from_pool(pool_lonlats, pool_durations, inst_idx, cfg)
            else:
                env = ResearchEnv(cfg)
                env.reset()
                init_states = init_states_from_env(env)
            schedule = presample_events(init_states, cfg, max_steps, seed + 999)

            t0 = time.perf_counter()
            greedy = run_single_rollout(init_states, schedule, "policy", max_steps, policy, device,
                                        sampling=False, ortools_time_limit_ms=args.ortools_time_limit_ms)
            greedy["wall_sec"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            samplexN = run_policy_best_of_k(init_states, schedule, max_steps, policy, device,
                                            args.policy_n_samples, None, None, None, 0.0, 0.0,
                                            args.ortools_time_limit_ms, seed * 1000 + 100)
            samplexN["wall_sec"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            rolling = run_single_rollout(init_states, schedule, "rolling_or", max_steps, None, device,
                                         sampling=False, ortools_time_limit_ms=args.ortools_time_limit_ms)
            rolling["wall_sec"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            repair = run_controller_rollout(init_states, schedule, RepairController, max_steps)
            repair["wall_sec"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            reactive = run_controller_rollout(init_states, schedule, ReactiveNNController, max_steps)
            reactive["wall_sec"] = time.perf_counter() - t0

            results = {
                "policy_greedy": greedy, "policy_samplexN": samplexN, "rolling_or": rolling,
                "repair_nn2opt": repair, "reactive_nn": reactive,
            }
            for k in method_keys:
                per_method[k].append(results[k])
            episodes.append({"episode": ep + 1, **results})
            print(
                f"[STAGE1] {bucket} ep {ep+1}/{args.n_episodes} | "
                f"samplexN del={samplexN['delivered_mean']:.2f} t={samplexN['time_mean']:.0f} | "
                f"repair del={repair['delivered_mean']:.2f} t={repair['time_mean']:.0f} | "
                f"rollingOR del={rolling['delivered_mean']:.2f} t={rolling['time_mean']:.0f} | "
                f"reactNN del={reactive['delivered_mean']:.2f} t={reactive['time_mean']:.0f}"
            )

        summary = {
            k: {
                "time_mean": mean_key(per_method[k], "time_mean"),
                "delivered_mean": mean_key(per_method[k], "delivered_mean"),
                "episode_wall_sec_mean": mean_key(per_method[k], "wall_sec"),
                "decision_ms_mean": mean_key(per_method[k], "decision_ms_mean"),
            }
            for k in method_keys
        }
        summary["paired_policy_samplexN_minus_repair"] = paired_summary(
            per_method["policy_samplexN"], per_method["repair_nn2opt"], seed=seed)
        summary["paired_policy_samplexN_minus_rolling_or"] = paired_summary(
            per_method["policy_samplexN"], per_method["rolling_or"], seed=seed + 7)
        summary["paired_repair_minus_rolling_or"] = paired_summary(
            per_method["repair_nn2opt"], per_method["rolling_or"], seed=seed + 13)
        summary["paired_repair_minus_reactive_nn"] = paired_summary(
            per_method["repair_nn2opt"], per_method["reactive_nn"], seed=seed + 19)

        out["buckets"][bucket] = {"summary": summary, "episodes": episodes}

        # Incremental save so partial results survive interruption.
        save_path = Path(__file__).resolve().parent / args.save_json
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

        p = summary["paired_policy_samplexN_minus_repair"]
        print(
            f"[STAGE1] === {bucket} === policy_samplexN - repair: "
            f"delivered {p['delivered_delta_mean']:+.3f} CI{p['delivered_delta_ci95']} | "
            f"time {p['time_delta_mean']:+.1f} CI{p['time_delta_ci95']} | "
            f"W/T/L {p['delivered_win_tie_loss']}"
        )

    save_path = Path(__file__).resolve().parent / args.save_json
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[STAGE1] Saved JSON: {save_path}")


if __name__ == "__main__":
    main()
