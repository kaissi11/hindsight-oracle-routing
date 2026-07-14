#!/usr/bin/env python3
"""Retrospective oracle-K sweep: how does episode-level hindsight-selected
best-of-K grow with K in {1, 2, 4, 8}?

Answers the referee question "why only K=8?" WITHOUT new online runs: the
recorded oracle-8 arm (``best_of_k`` in scenario_bucket_eval_v2) draws its K
sampled rollouts with per-candidate seeds ``seed0 + s`` and keeps the
lexicographic best (delivered_mean, -time_mean). Prefixes of that candidate
stream are therefore EXACTLY the K=1/2/4 protocols: rerunning the same seeds
and recording the incremental best after candidates 1, 2, 4, 8 reproduces
best-of-K for each K in one pass. K=1 is a single sampled rollout chosen a
priori -- a deployable policy with no selection -- so the K-curve reads as
selection inflation directly.

Pairing/validation:
  * instances + pre-sampled schedules regenerate from the recorded suite's
    seed formula (asserted against the reference JSON's config echo);
  * the K=8 point must equal the recorded ``policy_*_samplexN`` value per
    episode (bit-identity check, reported per suite; mismatches are counted,
    not silently ignored);
  * per-episode paired deltas vs the recorded online look-8 arm are computed
    from the reference JSON (cross-run pairing, same convention as
    generic_hindsight_eval.py).

No wall-clock-budgeted component runs here (GPU sampling arm only), so this
evaluation is robust to machine load and safe to run in parallel with other
non-rolling-OR jobs.

Run from 01_paper1/code (exercised suites: Damascus H=8 "osrm_s5" and
Damascus H=4 "hstress_h4", seeds 12345/13345/14345):

    python oracle_k_sweep.py
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
from matched_information_eval import _resolve_from_root, _write_json_atomic

ROOT = Path(__file__).resolve().parent

K_GRID = (1, 2, 4, 8)


def episode_key(delivered: float, elapsed: float) -> tuple[float, float]:
    return (delivered, -elapsed)


def load_pool(path: str):
    pool = np.load(path)
    lonlats, durations = pool["lonlats"], pool["durations"]
    bbox = stage2.DAMASCUS_BBOX
    meta_path = Path(path).parent / "pool_meta.json"
    if meta_path.exists():
        bb = json.loads(meta_path.read_text(encoding="utf-8")).get("bbox")
        if bb:
            from osrm_client import BBox
            bbox = BBox(min_lon=bb[0], min_lat=bb[1], max_lon=bb[2], max_lat=bb[3])
    return lonlats, durations, bbox


def make_pool_states(pool_lonlats, pool_durations, pool_bbox, cfg, seed: int,
                     num_instances: int) -> list:
    idx = np.random.RandomState(seed).choice(
        len(pool_lonlats), num_instances, replace=False)
    states = []
    for k in idx:
        n = pool_lonlats.shape[1]
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        states.append(stage2.SimStateV2(
            coords=stage2.normalize_lonlat(
                pool_lonlats[k], pool_bbox).astype(np.float64),
            base_dist=pool_durations[k].astype(np.float64).copy(),
            eff_dist=pool_durations[k].astype(np.float64).copy(),
            visited=visited, node_blocked=np.zeros(n, dtype=np.float32),
            current_node=0, elapsed_time=0.0,
            horizon_sec=float(cfg.time_horizon_sec), n_nodes=n,
        ))
    return states


def best_of_k_incremental(init_states, init_effs, schedules, max_steps,
                          policy, device, k_max: int, seed0: int) -> dict[int, dict]:
    """Reproduces stage2.best_of_k for every K prefix in one pass."""
    best, key = None, None
    out: dict[int, dict] = {}
    for s in range(k_max):
        np.random.seed(seed0 + s)
        torch.manual_seed(seed0 + s)
        r = stage2.run_rollout_v2(
            init_states, init_effs, schedules, max_steps,
            stage2.make_act_fn_policy(policy, device, sampling=True))
        rk = (r["delivered_mean"], -r["time_mean"])
        if key is None or rk > key:
            best, key = r, rk
        if (s + 1) in K_GRID:
            out[s + 1] = {"delivered_mean": float(best["delivered_mean"]),
                          "time_mean": float(best["time_mean"])}
    return out


def bootstrap_ci(values: np.ndarray, n_resamples: int = 10000,
                 seed: int = 0) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    n = len(values)
    means = values[rng.randint(0, n, size=(n_resamples, n))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def assert_config_matches(reference: dict, *, n_nodes: int, num_instances: int,
                          base_seed: int, horizon_hours: float,
                          n_episodes: int) -> None:
    cfg = reference.get("config", {})
    checks = {
        "n_nodes": (cfg.get("n_nodes"), n_nodes),
        "num_instances": (cfg.get("num_instances"), num_instances),
        "base_seed": (cfg.get("base_seed"), base_seed),
        "horizon_hours": (cfg.get("horizon_hours", 8.0), horizon_hours),
        "policy_n_samples": (cfg.get("policy_n_samples"), max(K_GRID)),
    }
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if cfg.get("n_episodes", 0) < n_episodes:
        bad["n_episodes"] = (cfg.get("n_episodes"), f"<= required {n_episodes}")
    if bad:
        raise SystemExit(f"[KSWEEP] reference config mismatch, aborting: {bad}")


def run_suite(tag: str, *, pool_path: str, horizon_hours: float, base_seed: int,
              n_episodes: int, n_nodes: int, num_instances: int,
              reference_path: str, buckets: Sequence[str],
              policies: dict, device, output_path: Path) -> None:
    reference = json.loads(
        _resolve_from_root(reference_path).read_text(encoding="utf-8"))
    assert_config_matches(reference, n_nodes=n_nodes,
                          num_instances=num_instances, base_seed=base_seed,
                          horizon_hours=horizon_hours, n_episodes=n_episodes)

    pool_lonlats, pool_durations, pool_bbox = load_pool(pool_path)
    max_steps = n_nodes * 8 + 64

    output: dict[str, Any] = {
        "schema_version": "oracle_k_sweep.v1",
        "complete": False,
        "config": {
            "tag": tag, "pool": pool_path, "horizon_hours": horizon_hours,
            "base_seed": base_seed, "n_episodes": n_episodes,
            "n_nodes": n_nodes, "num_instances": num_instances,
            "k_grid": list(K_GRID), "reference": reference_path,
        },
        "provenance": {
            "candidate_stream": (
                "identical to best_of_k: per-candidate seeds seed*1000+100+s, "
                "lexicographic (delivered_mean, -time_mean); prefix bests "
                "recorded at K=1/2/4/8"
            ),
            "k8_identity": (
                "K=8 must reproduce the recorded policy_*_samplexN per episode"
            ),
        },
        "buckets": {},
    }
    _write_json_atomic(output_path, output)

    for bidx, bucket in enumerate(buckets):
        cfg = stage2.apply_bucket_v2(stage2.ResearchEnvV2Config(
            n_nodes=n_nodes, num_instances=num_instances,
            device=device.type, auto_reset=False, use_augmentation=True,
        ), bucket)
        cfg.time_horizon_sec = horizon_hours * 3600.0

        ref_eps = reference["buckets"][bucket]["episodes"]
        records: list[dict[str, Any]] = []
        output["buckets"][bucket] = {"episodes": records}
        mismatches = {pol: 0 for pol in policies}
        started = time.perf_counter()

        for ep in range(n_episodes):
            seed = base_seed + ep + 10000 * bidx
            np.random.seed(seed)
            torch.manual_seed(seed)
            init_states = make_pool_states(
                pool_lonlats, pool_durations, pool_bbox, cfg, seed,
                num_instances)
            init_effs, schedules = [], []
            for i, st in enumerate(init_states):
                eff0, sched = stage2.presample_schedule_v2(
                    st, cfg, max_steps, seed + 999 + i)
                init_effs.append(eff0)
                schedules.append(sched)

            curves = {
                pol: best_of_k_incremental(
                    init_states, init_effs, schedules, max_steps,
                    policy, device, max(K_GRID), seed * 1000 + 100)
                for pol, policy in policies.items()
            }
            ref_ep = ref_eps[ep]
            rec = {
                "episode": ep + 1,
                "episode_seed": seed,
                "recorded": {
                    "policy_v2_samplexN": ref_ep["policy_v2_samplexN"]["delivered_mean"],
                    "policy_v1_samplexN": ref_ep["policy_v1_samplexN"]["delivered_mean"],
                    "policy_v2_lookahead": ref_ep.get(
                        "policy_v2_lookahead", {}).get("delivered_mean"),
                    "policy_v1_lookahead": ref_ep.get(
                        "policy_v1_lookahead", {}).get("delivered_mean"),
                    "repair_nn2opt": ref_ep["repair_nn2opt"]["delivered_mean"],
                },
            }
            for pol in policies:
                rec[f"oracle_k_{pol}"] = {str(k): v for k, v in curves[pol].items()}
                got = curves[pol][max(K_GRID)]["delivered_mean"]
                want = rec["recorded"][f"policy_{pol}_samplexN"]
                if abs(got - want) > 1e-9:
                    mismatches[pol] += 1
            records.append(rec)
            if (ep + 1) % 10 == 0:
                print(f"[KSWEEP] {tag} seed {base_seed} {bucket} "
                      f"{ep + 1}/{n_episodes} "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)
                _write_json_atomic(output_path, output)

        summary: dict[str, Any] = {
            "n_episodes": len(records),
            "k8_identity_mismatch_episodes": mismatches,
        }
        for pol in policies:
            for k in K_GRID:
                vals = np.array([r[f"oracle_k_{pol}"][str(k)]["delivered_mean"]
                                 for r in records])
                summary[f"oracle{k}_{pol}_delivered_mean"] = float(vals.mean())
            k1 = np.array([r[f"oracle_k_{pol}"]["1"]["delivered_mean"]
                           for r in records])
            for k in K_GRID[1:]:
                vk = np.array([r[f"oracle_k_{pol}"][str(k)]["delivered_mean"]
                               for r in records])
                summary[f"paired_oracle{k}_minus_oracle1_{pol}"] = {
                    "mean": float((vk - k1).mean()),
                    "ci95": bootstrap_ci(vk - k1, seed=k),
                }
            look = [r["recorded"][f"policy_{pol}_lookahead"] for r in records]
            if all(v is not None for v in look):
                look_arr = np.array(look, dtype=float)
                for k in K_GRID:
                    vk = np.array([r[f"oracle_k_{pol}"][str(k)]["delivered_mean"]
                                   for r in records])
                    summary[f"paired_oracle{k}_minus_look8_{pol}"] = {
                        "mean": float((vk - look_arr).mean()),
                        "ci95": bootstrap_ci(vk - look_arr, seed=100 + k),
                    }
        output["buckets"][bucket]["summary"] = summary
        _write_json_atomic(output_path, output)
        print(f"[KSWEEP] {tag} seed {base_seed} {bucket} summary: "
              f"{json.dumps(summary, indent=2)}", flush=True)

    output["complete"] = True
    _write_json_atomic(output_path, output)
    print(f"[KSWEEP] wrote {output_path}", flush=True)


SUITES = {
    "osrm_h8": dict(
        pool_path="results/osrm_instance_pool/pool.npz", horizon_hours=8.0,
        reference_fmt="results/scenario_bucket_v2_osrm_s5_seed_{seed}.json"),
    "hstress_h4": dict(
        pool_path="results/osrm_instance_pool/pool.npz", horizon_hours=4.0,
        reference_fmt="results/scenario_bucket_v2_hstress_h4_seed_{seed}.json"),
}


def main(argv: Sequence[str] | None = None) -> None:
    import sys
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        print("[KSWEEP] keep-awake armed", flush=True)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suites", nargs="+", default=["osrm_h8", "hstress_h4"],
                        choices=sorted(SUITES))
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[12345, 13345, 14345])
    parser.add_argument("--n-episodes", type=int, default=40)
    parser.add_argument("--n-nodes", type=int, default=20)
    parser.add_argument("--num-instances", type=int, default=4)
    parser.add_argument("--buckets", nargs="+",
                        default=["low", "medium", "high"])
    parser.add_argument("--policy-v2-checkpoint",
                        default="checkpoints_research_v2_pomo/research_v2_best.pt")
    parser.add_argument("--policy-v1-checkpoint",
                        default=str(ROOT / "checkpoints_research_pomo" / "research_best.pt"))
    parser.add_argument("--policies", nargs="+", default=["v1"],
                        choices=["v1", "v2"],
                        help="which checkpoints to sweep (v1 = the paper's "
                             "headline zero-shot policy; v2 adds the fine-tune)")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policies = {}
    if "v2" in args.policies:
        policy_v2, ckpt_v2 = stage2.load_policy(args.policy_v2_checkpoint, device)
        policies["v2"] = policy_v2
        print(f"[KSWEEP] v2_epoch={ckpt_v2.get('epoch')}", flush=True)
    if "v1" in args.policies:
        policy_v1, ckpt_v1 = stage2.load_policy(args.policy_v1_checkpoint, device)
        policies["v1"] = policy_v1
        print(f"[KSWEEP] v1_epoch={ckpt_v1.get('epoch')}", flush=True)
    print(f"[KSWEEP] device={device} policies={sorted(policies)}", flush=True)

    for tag in args.suites:
        spec = SUITES[tag]
        for seed in args.seeds:
            out = ROOT / "results" / f"oracle_k_sweep_{tag}_seed_{seed}.json"
            try:
                done = json.loads(out.read_text(encoding="utf-8"))
                if done.get("complete"):
                    print(f"[KSWEEP] {tag} seed {seed}: already complete - skip",
                          flush=True)
                    continue
            except (OSError, json.JSONDecodeError):
                pass
            run_suite(
                tag, pool_path=spec["pool_path"],
                horizon_hours=spec["horizon_hours"], base_seed=seed,
                n_episodes=args.n_episodes, n_nodes=args.n_nodes,
                num_instances=args.num_instances,
                reference_path=spec["reference_fmt"].format(seed=seed),
                buckets=args.buckets, policies=policies,
                device=device, output_path=out)
    print("[KSWEEP] all suites done", flush=True)


if __name__ == "__main__":
    main()
