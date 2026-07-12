#!/usr/bin/env python3
"""Generic-hindsight test (blueprint SCHEDULED #2, §3.11 genericity).

Question: is trajectory-hindsight inflation a property of the learned
sampler, or a generic consequence of selecting among stochastic dynamic
trajectories after realization?

Design (predeclared): best-of-K over K=8 randomized classical candidates —
randomized repair restarts. Each candidate is the paper's lightweight-repair
controller planning on a fixed multiplicative cost perturbation
(i.i.d. U[0.8, 1.2] per arc, drawn once per candidate per instance); the
environment, masks, and executed travel times are untouched, so every
candidate is a legitimate online controller. All K candidates run against
the SAME realized pre-sampled schedule, mirroring the recorded oracle-8
protocol exactly, including its selection granularity: the winner is chosen
by episode-level (delivered_mean, -time_mean) over the episode's instances,
identical to ``best_of_k``.

Reported per bucket:
  * online estimate  — mean outcome across the K candidates (the expected
    value of deploying one randomized-repair restart chosen a priori);
  * hindsight bound  — best-of-K after realization;
  * paired Δ(hindsight − online) with episode-bootstrap 95% CI;
  * cross-run paired Δ vs the recorded canonical repair row
    (``repair_nn2opt`` in the reference suite; identical instances and
    schedules regenerate from the seed).

No wall-clock-budgeted component exists in repair (2-opt runs to
convergence), so this evaluation is robust to machine load. CPU only.
"""
from __future__ import annotations

import argparse
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


class RandomizedRepairController(stage2.RepairControllerV2):
    """RepairControllerV2 planning on a fixed random cost perturbation.

    The perturbation only biases the PLANNING costs (initial tour, 2-opt,
    reinsertion); execution, masks, and the fallback all see the true
    effective matrix through the unchanged harness.
    """

    def __init__(self, state: stage2.SimStateV2, noise_seed: int,
                 noise_low: float = 0.8, noise_high: float = 1.2):
        rng = np.random.RandomState(noise_seed)
        self._noise = rng.uniform(
            noise_low, noise_high, size=(state.n_nodes, state.n_nodes)
        )
        super().__init__(state)

    def _cost(self, state: stage2.SimStateV2) -> np.ndarray:
        return np.where(
            np.isfinite(state.eff_dist),
            state.eff_dist * self._noise,
            stage2.BIG,
        )


def episode_key(delivered: float, elapsed: float) -> tuple[float, float]:
    return (delivered, -elapsed)


def run_candidates(
    init_states: Sequence[stage2.SimStateV2],
    init_effs: Sequence[np.ndarray],
    schedules: Sequence[Sequence[Any]],
    max_steps: int,
    episode_seed: int,
    k_candidates: int,
) -> list[dict[str, float]]:
    outcomes = []
    for k in range(k_candidates):
        counter = {"i": 0}

        def act_fn(mode, state, ctrl=None, k=k, counter=counter):
            if mode == "init":
                noise_seed = episode_seed * 1000 + 703 + 17 * k + counter["i"]
                counter["i"] += 1
                return RandomizedRepairController(state, noise_seed)
            return ctrl.act(state)

        result = stage2.run_rollout_v2(
            list(init_states), list(init_effs), list(schedules), max_steps, act_fn
        )
        outcomes.append(
            {
                "candidate": k,
                "delivered_mean": float(result["delivered_mean"]),
                "time_mean": float(result["time_mean"]),
            }
        )
    return outcomes


def bootstrap_ci(values: np.ndarray, n_resamples: int = 10000,
                 seed: int = 0) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    n = len(values)
    means = values[rng.randint(0, n, size=(n_resamples, n))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_bucket(
    episodes: list[dict[str, Any]],
    reference_episodes: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    online_del = np.array([e["online_mean"]["delivered_mean"] for e in episodes])
    hind_del = np.array([e["hindsight_best"]["delivered_mean"] for e in episodes])
    online_time = np.array([e["online_mean"]["time_mean"] for e in episodes])
    hind_time = np.array([e["hindsight_best"]["time_mean"] for e in episodes])
    delta_del = hind_del - online_del
    delta_time = hind_time - online_time
    summary: dict[str, Any] = {
        "n_episodes": len(episodes),
        "online_mean_delivered": float(online_del.mean()),
        "hindsight_best_delivered": float(hind_del.mean()),
        "paired_hindsight_minus_online_delivered": {
            "mean": float(delta_del.mean()),
            "ci95": bootstrap_ci(delta_del),
        },
        "paired_hindsight_minus_online_time_sec": {
            "mean": float(delta_time.mean()),
            "ci95": bootstrap_ci(delta_time),
        },
    }
    if reference_episodes is not None and len(reference_episodes) == len(episodes):
        ref_del = np.array(
            [e["repair_nn2opt"]["delivered_mean"] for e in reference_episodes]
        )
        summary["recorded_canonical_repair_delivered"] = float(ref_del.mean())
        summary["paired_hindsight_minus_canonical_repair_delivered"] = {
            "mean": float((hind_del - ref_del).mean()),
            "ci95": bootstrap_ci(hind_del - ref_del, seed=1),
        }
        summary["paired_online_minus_canonical_repair_delivered"] = {
            "mean": float((online_del - ref_del).mean()),
            "ci95": bootstrap_ci(online_del - ref_del, seed=2),
        }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Best-of-K hindsight selection over randomized repair restarts"
    )
    parser.add_argument(
        "--instance-pool",
        default="results/osrm_instance_pool/pool.npz",
        help="cached pool; pass an empty string for synthetic smoke instances",
    )
    parser.add_argument("--n-episodes", type=int, default=40)
    parser.add_argument("--n-nodes", type=int, default=20)
    parser.add_argument("--num-instances", type=int, default=4)
    parser.add_argument("--k-candidates", type=int, default=8)
    parser.add_argument(
        "--buckets", nargs="+", choices=("low", "medium", "high"),
        default=["low", "medium", "high"],
    )
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument(
        "--reference",
        default="results/scenario_bucket_v2_osrm_s5_seed_12345.json",
        help="recorded control suite for cross-run pairing; empty to skip",
    )
    parser.add_argument("--output", default="results/generic_hindsight_repair.json")
    return parser


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
        "schema_version": "generic_hindsight_repair.v1",
        "complete": False,
        "config": {**vars(args), "max_steps_effective": max_steps},
        "provenance": {
            "candidate_generator": (
                "RepairControllerV2 with fixed multiplicative planning-cost "
                "noise U[0.8,1.2] per arc per candidate (execution untouched)"
            ),
            "selection_granularity": (
                "episode-level (delivered_mean, -time_mean), identical to best_of_k"
            ),
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

            candidates = run_candidates(
                init_states, init_effs, schedules, max_steps,
                episode_seed, args.k_candidates,
            )
            best = max(
                candidates,
                key=lambda c: episode_key(c["delivered_mean"], c["time_mean"]),
            )
            bucket_records.append(
                {
                    "episode_index": episode_index,
                    "episode_seed": episode_seed,
                    "candidates": candidates,
                    "online_mean": {
                        "delivered_mean": float(
                            np.mean([c["delivered_mean"] for c in candidates])
                        ),
                        "time_mean": float(
                            np.mean([c["time_mean"] for c in candidates])
                        ),
                    },
                    "hindsight_best": {
                        "candidate": best["candidate"],
                        "delivered_mean": best["delivered_mean"],
                        "time_mean": best["time_mean"],
                    },
                }
            )
            if (episode_index + 1) % 10 == 0:
                print(
                    f"[GH] {bucket} {episode_index + 1}/{args.n_episodes} "
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
            f"[GH] {bucket} summary: "
            f"{json.dumps(output['buckets'][bucket]['summary'], indent=2)}",
            flush=True,
        )

    output["complete"] = True
    _write_json_atomic(output_path, output)
    print(f"[GH] wrote {output_path}")


if __name__ == "__main__":
    main()
