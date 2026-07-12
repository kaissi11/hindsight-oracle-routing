#!/usr/bin/env python3
"""Pool the three-seed "P3" ablation extensions (observability + horizon stress).

This script never changes the experiment JSONs. It reads the completed
seed-12345/13345/14345 runs, computes paired hierarchical-bootstrap summaries,
prints the paper-facing values, and writes one compact aggregate JSON.

Pairing:
* obs_*: within-run method comparisons, plus cross-run comparisons against the
  same-seed ``osrm_s5`` live-matrix control (identical schedules).
* hstress_h4: within-run comparisons only because the horizon differs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import stats


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SEEDS = (12345, 13345, 14345)
BUCKETS = ("low", "medium", "high")
OBS_JOBS = ("obs_base", "obs_mask", "obs_traffic")
MARGIN_N20 = 0.05


def load_complete(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing required result: {path.name}")
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = [b for b in BUCKETS if b not in data.get("buckets", {})]
    if missing:
        raise RuntimeError(f"incomplete result {path.name}: missing buckets {missing}")
    return data


def metric_values(data: dict, bucket: str, method: str, metric: str) -> np.ndarray:
    values = [
        float(ep[method][metric])
        for ep in data["buckets"][bucket]["episodes"]
        if method in ep and metric in ep[method]
    ]
    if not values:
        raise KeyError(f"no {method}.{metric} values in bucket {bucket}")
    return np.asarray(values, dtype=float)


def hierarchical_cis(
    per_seed: dict[int, np.ndarray],
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Resample seeds, then paired episodes within each sampled seed."""
    arrays = list(per_seed.values())
    n_seed = len(arrays)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        chosen = rng.integers(0, n_seed, n_seed)
        chunks = []
        for idx in chosen:
            arr = arrays[int(idx)]
            chunks.append(arr[rng.integers(0, len(arr), len(arr))])
        boot[i] = float(np.concatenate(chunks).mean())
    lo95, hi95 = np.percentile(boot, (2.5, 97.5))
    lo90, hi90 = np.percentile(boot, (5.0, 95.0))
    return (float(lo95), float(hi95)), (float(lo90), float(hi90))


def paired_tost_p(values: np.ndarray, margin: float) -> float:
    n = len(values)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(n))
    if se == 0:
        return 0.0 if abs(mean) < margin else 1.0
    p_lower = stats.t.sf((mean + margin) / se, n - 1)
    p_upper = stats.t.cdf((mean - margin) / se, n - 1)
    return float(max(p_lower, p_upper))


def summarize(
    per_seed: dict[int, np.ndarray],
    *,
    n_boot: int,
    rng_seed: int,
    margin: float = MARGIN_N20,
) -> dict:
    if set(per_seed) != set(SEEDS):
        raise ValueError(f"expected seeds {SEEDS}, got {sorted(per_seed)}")
    lengths = {seed: len(values) for seed, values in per_seed.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"episode-count mismatch across seeds: {lengths}")

    pooled = np.concatenate([per_seed[s] for s in SEEDS])
    ci95, ci90 = hierarchical_cis(
        per_seed, n_boot=n_boot, rng=np.random.default_rng(rng_seed)
    )
    try:
        wilcoxon_p = float(stats.wilcoxon(pooled, zero_method="pratt").pvalue)
    except ValueError:
        wilcoxon_p = 1.0

    if -margin < ci90[0] and ci90[1] < margin:
        verdict = "equivalent"
    elif ci95[0] > 0 or ci95[1] < 0:
        verdict = "different"
    else:
        verdict = "inconclusive"

    return {
        "mean": float(pooled.mean()),
        "ci95_cluster": list(ci95),
        "ci90_cluster": list(ci90),
        "wilcoxon_p": wilcoxon_p,
        "tost_p": paired_tost_p(pooled, margin),
        "margin": margin,
        "verdict": verdict,
        "n_seeds": len(SEEDS),
        "n_episodes": int(pooled.size),
        "per_seed_means": {
            str(seed): float(per_seed[seed].mean()) for seed in SEEDS
        },
    }


def within_run_deltas(
    runs: dict[int, dict],
    bucket: str,
    method_a: str,
    method_b: str,
    metric: str,
) -> dict[int, np.ndarray]:
    out = {}
    for seed, data in runs.items():
        a = metric_values(data, bucket, method_a, metric)
        b = metric_values(data, bucket, method_b, metric)
        if len(a) != len(b):
            raise ValueError(f"within-run pairing mismatch for seed {seed}")
        out[seed] = a - b
    return out


def cross_run_deltas(
    runs: dict[int, dict],
    controls: dict[int, dict],
    bucket: str,
    method: str,
    metric: str,
) -> dict[int, np.ndarray]:
    out = {}
    for seed in SEEDS:
        candidate = metric_values(runs[seed], bucket, method, metric)
        control = metric_values(controls[seed], bucket, method, metric)
        if len(candidate) != len(control):
            raise ValueError(f"cross-run pairing mismatch for seed {seed}")
        out[seed] = candidate - control
    return out


def fmt(summary: dict) -> str:
    lo, hi = summary["ci95_cluster"]
    return (
        f"{summary['mean']:+.3f} [{lo:+.3f},{hi:+.3f}] "
        f"{summary['verdict']} (3 seeds, n={summary['n_episodes']})"
    )


def build_aggregate(n_boot: int) -> dict:
    controls = {
        seed: load_complete(RESULTS / f"scenario_bucket_v2_osrm_s5_seed_{seed}.json")
        for seed in SEEDS
    }
    obs_runs = {
        job: {
            seed: load_complete(RESULTS / f"scenario_bucket_v2_{job}_seed_{seed}.json")
            for seed in SEEDS
        }
        for job in OBS_JOBS
    }
    h4_runs = {
        seed: load_complete(RESULTS / f"scenario_bucket_v2_hstress_h4_seed_{seed}.json")
        for seed in SEEDS
    }

    result: dict = {
        "seeds": list(SEEDS),
        "buckets": list(BUCKETS),
        "n_boot": n_boot,
        "margin_n20": MARGIN_N20,
        "observability": {},
        "hstress_h4": {},
    }
    rng_seed = 1000

    for job in OBS_JOBS:
        result["observability"][job] = {}
        for bucket in BUCKETS:
            cells = {}
            definitions: Iterable[tuple[str, dict[int, np.ndarray]]] = (
                (
                    "oracle_minus_repair_delivered",
                    within_run_deltas(
                        obs_runs[job],
                        bucket,
                        "policy_v1_samplexN",
                        "repair_nn2opt",
                        "delivered_mean",
                    ),
                ),
                (
                    "look_minus_repair_delivered",
                    within_run_deltas(
                        obs_runs[job],
                        bucket,
                        "policy_v1_lookahead",
                        "repair_nn2opt",
                        "delivered_mean",
                    ),
                ),
                (
                    "oracle_minus_live_delivered",
                    cross_run_deltas(
                        obs_runs[job],
                        controls,
                        bucket,
                        "policy_v1_samplexN",
                        "delivered_mean",
                    ),
                ),
                (
                    "look_minus_live_delivered",
                    cross_run_deltas(
                        obs_runs[job],
                        controls,
                        bucket,
                        "policy_v1_lookahead",
                        "delivered_mean",
                    ),
                ),
                (
                    "oracle_minus_live_time",
                    cross_run_deltas(
                        obs_runs[job],
                        controls,
                        bucket,
                        "policy_v1_samplexN",
                        "time_mean",
                    ),
                ),
                (
                    "look_minus_live_time",
                    cross_run_deltas(
                        obs_runs[job],
                        controls,
                        bucket,
                        "policy_v1_lookahead",
                        "time_mean",
                    ),
                ),
            )
            for name, deltas in definitions:
                cells[name] = summarize(
                    deltas, n_boot=n_boot, rng_seed=rng_seed
                )
                rng_seed += 1
            result["observability"][job][bucket] = cells

    for bucket in BUCKETS:
        repair_values = {
            seed: metric_values(
                h4_runs[seed], bucket, "repair_nn2opt", "delivered_mean"
            )
            for seed in SEEDS
        }
        cells = {
            "repair_mean": float(
                np.concatenate([repair_values[s] for s in SEEDS]).mean()
            )
        }
        definitions = (
            (
                "look_minus_repair_delivered",
                "policy_v1_lookahead",
                "repair_nn2opt",
            ),
            (
                "oracle_minus_repair_delivered",
                "policy_v1_samplexN",
                "repair_nn2opt",
            ),
            (
                "look_minus_oracle_delivered",
                "policy_v1_lookahead",
                "policy_v1_samplexN",
            ),
            (
                "look_minus_rolling_or_delivered",
                "policy_v1_lookahead",
                "rolling_or",
            ),
        )
        for name, method_a, method_b in definitions:
            deltas = within_run_deltas(
                h4_runs, bucket, method_a, method_b, "delivered_mean"
            )
            cells[name] = summarize(deltas, n_boot=n_boot, rng_seed=rng_seed)
            rng_seed += 1
        result["hstress_h4"][bucket] = cells

    return result


def print_report(result: dict) -> None:
    print("\n=== P3 observability (matrix mode minus live control) ===")
    for job in OBS_JOBS:
        print(f"\n{job}")
        for bucket in BUCKETS:
            cells = result["observability"][job][bucket]
            print(
                f"  {bucket:6s} oracle-live delivered "
                f"{fmt(cells['oracle_minus_live_delivered'])}; "
                f"time {fmt(cells['oracle_minus_live_time'])}"
            )
            print(
                f"         look-live delivered "
                f"{fmt(cells['look_minus_live_delivered'])}; "
                f"time {fmt(cells['look_minus_live_time'])}"
            )

    print("\n=== P3 horizon stress H=4 h ===")
    for bucket in BUCKETS:
        cells = result["hstress_h4"][bucket]
        print(f"  {bucket:6s} repair mean {cells['repair_mean']:.3f}/19")
        for name in (
            "look_minus_repair_delivered",
            "oracle_minus_repair_delivered",
            "look_minus_oracle_delivered",
            "look_minus_rolling_or_delivered",
        ):
            print(f"         {name}: {fmt(cells[name])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=20_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "p3_aggregate_3seeds.json",
    )
    args = parser.parse_args()

    result = build_aggregate(args.n_boot)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print_report(result)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
