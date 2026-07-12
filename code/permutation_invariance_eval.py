#!/usr/bin/env python3
"""Permutation-invariance evaluation (blueprint #5 / §3.13).

For each test instance, customer order (node 0 = depot stays fixed) and all
corresponding matrix rows/columns are permuted consistently; the policy runs
on the permuted representation; actions/distributions are mapped back to the
original node identities and compared against the identity run.

Reported per the blueprint:
  - KL divergence between mapped action distributions (per decision state);
  - first-action (argmax) agreement;
  - completion variance across permutations (closed-loop greedy rollouts);
  - route-time variance across permutations.

CPU-only friendly: single forward passes and greedy rollouts, no sampling.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

import scenario_bucket_eval_v2 as stage2
from matched_information_eval import (
    _load_instance_pool,
    _make_initial_states,
    _resolve_from_root,
    _write_json_atomic,
)

ROOT = Path(__file__).resolve().parent
ScheduleEvent = tuple[np.ndarray, np.ndarray]


def sample_customer_permutation(n_nodes: int, rng: np.random.RandomState) -> np.ndarray:
    """A node permutation that fixes the depot (index 0)."""
    perm = np.arange(n_nodes)
    perm[1:] = rng.permutation(np.arange(1, n_nodes))
    return perm


def permute_state(state: stage2.SimStateV2, perm: np.ndarray) -> stage2.SimStateV2:
    """Relabel nodes so that new index i holds old node perm[i]."""
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size)
    permuted = copy.deepcopy(state)
    permuted.coords = state.coords[perm].copy()
    permuted.base_dist = state.base_dist[np.ix_(perm, perm)].copy()
    permuted.eff_dist = state.eff_dist[np.ix_(perm, perm)].copy()
    permuted.visited = state.visited[perm].copy()
    permuted.node_blocked = state.node_blocked[perm].copy()
    permuted.current_node = int(inv[state.current_node])
    return permuted


def permute_event(event: ScheduleEvent, perm: np.ndarray) -> ScheduleEvent:
    matrix, blocked = event
    return matrix[np.ix_(perm, perm)].copy(), blocked[perm].copy()


def policy_distribution(
    state: stage2.SimStateV2, policy, device: torch.device
) -> np.ndarray | None:
    """Masked softmax action distribution, mirroring choose_policy_action_v2."""
    n = state.n_nodes
    x = torch.tensor(state.coords[:, 0], device=device, dtype=torch.float32).unsqueeze(0)
    y = torch.tensor(state.coords[:, 1], device=device, dtype=torch.float32).unsqueeze(0)
    visited = torch.tensor(state.visited.astype(np.float32), device=device).unsqueeze(0)
    blocked = torch.tensor(state.node_blocked.astype(np.float32), device=device).unsqueeze(0)
    is_current = torch.zeros((1, n), device=device)
    is_current[0, state.current_node] = 1.0
    t_rem = max(0.0, state.horizon_sec - state.elapsed_time)
    t_frac = torch.full((1, n), float(t_rem / state.horizon_sec), device=device)
    obs = torch.stack([x, y, visited, is_current, blocked, t_frac], dim=-1).reshape(1, n * 6)

    eff = torch.tensor(
        stage2.policy_view_matrix(state), device=device, dtype=torch.float32
    ).unsqueeze(0)
    with torch.no_grad():
        logits = policy(obs, n, dist_matrix=eff).float().squeeze(0)
    mask = torch.tensor(stage2.valid_mask_v2(state), device=device)
    if mask.sum().item() == 0:
        return None
    logits[~mask] = -1e9
    return torch.softmax(logits, dim=0).cpu().numpy()


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def greedy_rollout(
    state: stage2.SimStateV2,
    initial_eff: np.ndarray,
    schedule: Sequence[ScheduleEvent],
    max_steps: int,
    policy,
    device: torch.device,
) -> dict[str, float]:
    act_fn = stage2.make_act_fn_policy(policy, device, sampling=False)
    return stage2.run_rollout_v2([state], [initial_eff], [schedule], max_steps, act_fn)


def collect_snapshots(
    state: stage2.SimStateV2,
    initial_eff: np.ndarray,
    schedule: Sequence[ScheduleEvent],
    snapshot_steps: Sequence[int],
    max_steps: int,
    policy,
    device: torch.device,
) -> list[tuple[int, stage2.SimStateV2]]:
    """Deep-copied decision states of the identity greedy rollout."""
    wanted = sorted(set(int(s) for s in snapshot_steps))
    snapshots: list[tuple[int, stage2.SimStateV2]] = []
    sim = copy.deepcopy(state)
    sim.eff_dist = initial_eff.copy()
    for step in range(max_steps):
        if sim.elapsed_time >= sim.horizon_sec or sim.visited[1:].all():
            break
        if step in wanted:
            snapshots.append((step, copy.deepcopy(sim)))
        action = stage2.choose_policy_action_v2(sim, policy, device, sampling=False)
        stage2.apply_action_and_advance_v2(sim, action, schedule[step])
    return snapshots


def evaluate_instance(
    state: stage2.SimStateV2,
    initial_eff: np.ndarray,
    schedule: Sequence[ScheduleEvent],
    *,
    permutations: Sequence[np.ndarray],
    snapshot_steps: Sequence[int],
    max_steps: int,
    policy,
    device: torch.device,
) -> dict[str, Any]:
    # Closed-loop outcomes under each relabeling of the same instance+schedule.
    delivered: list[float] = []
    times: list[float] = []
    identity = np.arange(state.n_nodes)
    for perm in [identity, *permutations]:
        result = greedy_rollout(
            permute_state(state, perm),
            initial_eff[np.ix_(perm, perm)].copy(),
            [permute_event(event, perm) for event in schedule],
            max_steps,
            policy,
            device,
        )
        delivered.append(float(result["delivered_mean"]))
        times.append(float(result["time_mean"]))

    # Same-state action-distribution comparison on identity-rollout snapshots.
    kls: list[float] = []
    agreements: list[bool] = []
    snapshot_records: list[dict[str, Any]] = []
    for step, snapshot in collect_snapshots(
        state, initial_eff, schedule, snapshot_steps, max_steps, policy, device
    ):
        base_probs = policy_distribution(snapshot, policy, device)
        if base_probs is None:
            continue
        base_action = int(np.argmax(base_probs))
        for perm in permutations:
            probs = policy_distribution(permute_state(snapshot, perm), policy, device)
            if probs is None:
                continue
            # new index i holds old node perm[i] -> mapped_to_old[perm] = probs
            mapped = np.empty_like(probs)
            mapped[perm] = probs
            kl = kl_divergence(base_probs, mapped)
            agree = int(np.argmax(mapped)) == base_action
            kls.append(kl)
            agreements.append(agree)
            snapshot_records.append(
                {
                    "decision_step": int(step),
                    "kl_mapped_vs_identity": kl,
                    "first_action_agreement": bool(agree),
                }
            )

    return {
        "closed_loop": {
            "delivered_by_permutation": delivered,
            "time_by_permutation": times,
            "delivered_std": float(np.std(delivered)),
            "delivered_range": float(np.max(delivered) - np.min(delivered)),
            "time_std_sec": float(np.std(times)),
            "time_range_sec": float(np.max(times) - np.min(times)),
        },
        "distribution": {
            "n_comparisons": len(kls),
            "kl_mean": float(np.mean(kls)) if kls else None,
            "kl_max": float(np.max(kls)) if kls else None,
            "first_action_agreement_rate": (
                float(np.mean(agreements)) if agreements else None
            ),
            "snapshots": snapshot_records,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Policy permutation-invariance test")
    parser.add_argument(
        "--policy-checkpoint",
        default=str(stage2.V62_DIR / "checkpoints_research_pomo" / "research_best.pt"),
    )
    parser.add_argument(
        "--instance-pool",
        default="results/osrm_instance_pool/pool.npz",
        help="cached pool; pass an empty string for synthetic instances",
    )
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--n-nodes", type=int, default=20)
    parser.add_argument("--num-instances", type=int, default=1)
    parser.add_argument("--n-permutations", type=int, default=8)
    parser.add_argument(
        "--snapshot-steps", nargs="+", type=int, default=[0, 3, 6, 9, 12]
    )
    parser.add_argument("--bucket", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", default="results/permutation_invariance.json")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    stage2.POLICY_MATRIX_MODE = "live"
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    policy, checkpoint = stage2.load_policy(
        str(_resolve_from_root(args.policy_checkpoint)), device
    )
    policy.eval()
    pool = _load_instance_pool(args.instance_pool)
    max_steps = args.max_steps or (args.n_nodes * 8 + 64)
    output_path = _resolve_from_root(args.output)

    cfg = stage2.apply_bucket_v2(
        stage2.ResearchEnvV2Config(
            n_nodes=args.n_nodes,
            num_instances=args.num_instances,
            device=device.type,
            auto_reset=False,
            use_augmentation=True,
        ),
        args.bucket,
    )

    output: dict[str, Any] = {
        "schema_version": "permutation_invariance_eval.v1",
        "complete": False,
        "config": {**vars(args), "max_steps_effective": max_steps},
        "provenance": {
            "policy_checkpoint_epoch": checkpoint.get("epoch"),
            "depot_fixed": True,
            "decoding": "greedy (deterministic); distributions are masked softmax",
        },
        "episodes": [],
    }

    started = time.perf_counter()
    for episode_index in range(args.n_episodes):
        episode_seed = args.base_seed + episode_index
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        initial_states = _make_initial_states(cfg, episode_seed, pool)
        perm_rng = np.random.RandomState(episode_seed + 777)
        for instance_index, (initial_state, source) in enumerate(initial_states):
            schedule_seed = episode_seed + 999 + instance_index
            initial_eff, schedule = stage2.presample_schedule_v2(
                initial_state, cfg, max_steps, schedule_seed
            )
            permutations = [
                sample_customer_permutation(initial_state.n_nodes, perm_rng)
                for _ in range(args.n_permutations)
            ]
            record = evaluate_instance(
                initial_state,
                initial_eff,
                schedule,
                permutations=permutations,
                snapshot_steps=args.snapshot_steps,
                max_steps=max_steps,
                policy=policy,
                device=device,
            )
            output["episodes"].append(
                {
                    "episode_index": episode_index,
                    "episode_seed": episode_seed,
                    "instance_index": instance_index,
                    **source,
                    **record,
                }
            )
        _write_json_atomic(output_path, output)
        print(
            f"[PERM] episode {episode_index + 1}/{args.n_episodes} done "
            f"({time.perf_counter() - started:.0f}s elapsed)"
        )

    episodes = output["episodes"]
    kl_values = [
        e["distribution"]["kl_mean"]
        for e in episodes
        if e["distribution"]["kl_mean"] is not None
    ]
    all_kls = [
        snap["kl_mapped_vs_identity"]
        for e in episodes
        for snap in e["distribution"]["snapshots"]
    ]
    agreement = [
        e["distribution"]["first_action_agreement_rate"]
        for e in episodes
        if e["distribution"]["first_action_agreement_rate"] is not None
    ]
    output["summary"] = {
        "n_instances": len(episodes),
        "n_distribution_comparisons": len(all_kls),
        "kl_mean": float(np.mean(kl_values)) if kl_values else None,
        "kl_p95": float(np.percentile(all_kls, 95)) if all_kls else None,
        "kl_max": float(
            np.max([e["distribution"]["kl_max"] for e in episodes
                    if e["distribution"]["kl_max"] is not None])
        ) if kl_values else None,
        "first_action_agreement_rate": (
            float(np.mean(agreement)) if agreement else None
        ),
        "delivered_std_mean": float(
            np.mean([e["closed_loop"]["delivered_std"] for e in episodes])
        ),
        "delivered_range_max": float(
            np.max([e["closed_loop"]["delivered_range"] for e in episodes])
        ),
        "time_std_mean_sec": float(
            np.mean([e["closed_loop"]["time_std_sec"] for e in episodes])
        ),
        "time_range_max_sec": float(
            np.max([e["closed_loop"]["time_range_sec"] for e in episodes])
        ),
    }
    output["complete"] = True
    _write_json_atomic(output_path, output)
    print("[PERM] summary:", json.dumps(output["summary"], indent=2))
    print(f"[PERM] wrote {output_path}")


if __name__ == "__main__":
    main()
