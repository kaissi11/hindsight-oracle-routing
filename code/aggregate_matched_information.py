#!/usr/bin/env python3
"""Aggregate the predeclared matched-information experiment.

Outputs the closed-loop decomposition

    oracle - online = (clairvoyant - online) + (oracle - clairvoyant)

and same-state selector diagnostics with seed-clustered uncertainty.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from aggregate_p3_seeds import SEEDS, summarize


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ASSETS = HERE / "paper" / "assets"
HORIZONS = (8, 4)
BUCKETS = ("low", "medium", "high")
MARGIN = 0.05


def load_complete(path: Path, *, matched: bool = False) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if matched and not data.get("complete"):
        raise RuntimeError(f"incomplete matched result: {path.name}")
    missing = [bucket for bucket in BUCKETS if bucket not in data.get("buckets", {})]
    if missing:
        raise RuntimeError(f"incomplete result {path.name}: {missing}")
    return data


def reference_path(horizon: int, seed: int) -> Path:
    variant = "osrm_s5" if horizon == 8 else "hstress_h4"
    return RESULTS / f"scenario_bucket_v2_{variant}_seed_{seed}.json"


def matched_path(horizon: int, seed: int) -> Path:
    return RESULTS / f"matched_information_h{horizon}_seed_{seed}.json"


def episode_arrays(
    matched_runs: dict[int, dict],
    references: dict[int, dict],
    bucket: str,
) -> dict[str, dict[int, np.ndarray]]:
    metrics: dict[str, dict[int, np.ndarray]] = {
        "clair_minus_online_delivered": {},
        "oracle_minus_clair_delivered": {},
        "oracle_minus_online_delivered": {},
        "clair_minus_online_time": {},
    }
    for seed in SEEDS:
        matched_eps = matched_runs[seed]["buckets"][bucket]["episodes"]
        reference_eps = references[seed]["buckets"][bucket]["episodes"]
        if len(matched_eps) != len(reference_eps):
            raise ValueError(f"{bucket} seed {seed}: episode-count mismatch")

        clair_minus_online = []
        oracle_minus_clair = []
        oracle_minus_online = []
        clair_minus_online_time = []
        for matched_ep, reference_ep in zip(matched_eps, reference_eps):
            aggregate = matched_ep["aggregate"]
            online = aggregate["online_frozen"]
            clair = aggregate["clairvoyant_realized"]
            oracle = reference_ep["policy_v1_samplexN"]
            recorded_online = reference_ep["policy_v1_lookahead"]

            if abs(
                float(online["delivered_mean"])
                - float(recorded_online["delivered_mean"])
            ) > 1e-8:
                raise AssertionError("frozen arm differs from recorded v1 look-8")
            if abs(
                float(online["elapsed_time_mean_sec"])
                - float(recorded_online["time_mean"])
            ) > 1e-8:
                raise AssertionError("frozen-arm time differs from recorded v1 look-8")

            online_delivered = float(online["delivered_mean"])
            clair_delivered = float(clair["delivered_mean"])
            oracle_delivered = float(oracle["delivered_mean"])
            clair_minus_online.append(clair_delivered - online_delivered)
            oracle_minus_clair.append(oracle_delivered - clair_delivered)
            oracle_minus_online.append(oracle_delivered - online_delivered)
            clair_minus_online_time.append(
                float(clair["elapsed_time_mean_sec"])
                - float(online["elapsed_time_mean_sec"])
            )

        metrics["clair_minus_online_delivered"][seed] = np.asarray(
            clair_minus_online, dtype=float
        )
        metrics["oracle_minus_clair_delivered"][seed] = np.asarray(
            oracle_minus_clair, dtype=float
        )
        metrics["oracle_minus_online_delivered"][seed] = np.asarray(
            oracle_minus_online, dtype=float
        )
        metrics["clair_minus_online_time"][seed] = np.asarray(
            clair_minus_online_time, dtype=float
        )
    return metrics


def episode_diagnostic(
    episode: dict,
    extractor: Callable[[dict[str, Any]], float],
    *,
    nested_k: int | None = None,
) -> float:
    values = []
    for instance in episode["instances"]:
        decisions = instance["outcomes"]["same_state_diagnostic"]["decisions"]
        for decision in decisions:
            if nested_k is None:
                item = decision["primary"]
            else:
                nested = decision.get("nested", {}).get("by_k", {})
                if str(nested_k) not in nested:
                    continue
                item = nested[str(nested_k)]
            values.append(extractor(item))
    return float(np.mean(values)) if values else float("nan")


def diagnostic_arrays(
    matched_runs: dict[int, dict],
    bucket: str,
    nested_k_values: tuple[int, ...],
) -> dict[str, dict[int, np.ndarray]]:
    extractors: dict[str, Callable[[dict[str, Any]], float]] = {
        "first_action_disagreement_rate": lambda item: float(
            not item["first_action_agreement"]
        ),
        "candidate_disagreement_rate": lambda item: float(
            not item["selected_candidate_agreement"]
        ),
        "realized_delivered_gain": lambda item: float(
            item["realized_value_difference"]["delivered_gain"]
        ),
        "realized_time_saving_sec": lambda item: float(
            item["realized_value_difference"]["time_saving_sec"]
        ),
        "positive_realized_value_rate": lambda item: float(
            item["realized_value_difference"]["lexicographic_gain"] > 0
        ),
    }
    output: dict[str, dict[int, np.ndarray]] = {}
    for name, extractor in extractors.items():
        output[name] = {}
        for seed in SEEDS:
            values = [
                episode_diagnostic(episode, extractor)
                for episode in matched_runs[seed]["buckets"][bucket]["episodes"]
            ]
            output[name][seed] = np.asarray(values, dtype=float)

    for k in nested_k_values:
        for base_name, extractor in extractors.items():
            name = f"k{k}_{base_name}"
            output[name] = {}
            for seed in SEEDS:
                values = [
                    episode_diagnostic(episode, extractor, nested_k=k)
                    for episode in matched_runs[seed]["buckets"][bucket]["episodes"]
                ]
                output[name][seed] = np.asarray(values, dtype=float)
    return output


def summarize_all(per_seed_metrics: dict[str, dict[int, np.ndarray]], n_boot: int) -> dict:
    output = {}
    for index, (name, per_seed) in enumerate(per_seed_metrics.items()):
        if any(np.isnan(values).any() for values in per_seed.values()):
            raise ValueError(f"missing diagnostic values for {name}")
        output[name] = summarize(
            per_seed,
            n_boot=n_boot,
            rng_seed=50_000 + index,
            margin=MARGIN,
        )
    return output


def write_markdown(result: dict) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    lines = [
        "**Table — Matched selector attribution (3 seeds, n=120/bucket; "
        "paired seed-clustered 95% CIs).**",
        "",
        "| Horizon | Bucket | clair−online | oracle−clair | oracle−online | "
        "same-state action disagreement |",
        "|---|---|---|---|---|---|",
    ]
    for horizon in HORIZONS:
        for bucket in BUCKETS:
            cell = result[f"h{horizon}"][bucket]

            def value(name: str) -> str:
                summary = cell["closed_loop"][name]
                lo, hi = summary["ci95_cluster"]
                return f"{summary['mean']:+.3f} [{lo:+.3f}, {hi:+.3f}]"

            disagreement = cell["same_state"]["first_action_disagreement_rate"]
            dlo, dhi = disagreement["ci95_cluster"]
            lines.append(
                f"| H={horizon} h | {bucket} | "
                f"{value('clair_minus_online_delivered')} | "
                f"{value('oracle_minus_clair_delivered')} | "
                f"{value('oracle_minus_online_delivered')} | "
                f"{disagreement['mean']:.3f} [{dlo:.3f}, {dhi:.3f}] |"
            )
    (ASSETS / "table_matched_information.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def build(n_boot: int) -> dict:
    result: dict[str, Any] = {
        "schema_version": "matched_information_aggregate.v1",
        "seeds": list(SEEDS),
        "n_boot": n_boot,
        "equivalence_margin": MARGIN,
    }
    nested_k_values = (1, 2, 4, 8, 16)

    for horizon in HORIZONS:
        matched_runs = {
            seed: load_complete(matched_path(horizon, seed), matched=True)
            for seed in SEEDS
        }
        references = {
            seed: load_complete(reference_path(horizon, seed))
            for seed in SEEDS
        }
        horizon_result = {}
        for bucket in BUCKETS:
            decomposed_arrays = episode_arrays(matched_runs, references, bucket)
            closed_loop = summarize_all(
                decomposed_arrays, n_boot
            )
            diagnostics = summarize_all(
                diagnostic_arrays(matched_runs, bucket, nested_k_values), n_boot
            )
            # Numerical identity gate for the predeclared decomposition.
            for seed in SEEDS:
                lhs = decomposed_arrays["oracle_minus_online_delivered"][seed]
                rhs = (
                    decomposed_arrays["clair_minus_online_delivered"][seed]
                    + decomposed_arrays["oracle_minus_clair_delivered"][seed]
                )
                if not np.allclose(lhs, rhs, atol=1e-12):
                    raise AssertionError("matched decomposition identity failed")
            horizon_result[bucket] = {
                "closed_loop": closed_loop,
                "same_state": diagnostics,
            }
        result[f"h{horizon}"] = horizon_result
    return result


def print_report(result: dict) -> None:
    for horizon in HORIZONS:
        print(f"\n=== Matched information H={horizon} h ===")
        for bucket in BUCKETS:
            cell = result[f"h{horizon}"][bucket]
            print(f"  {bucket}")
            for name in (
                "clair_minus_online_delivered",
                "oracle_minus_clair_delivered",
                "oracle_minus_online_delivered",
            ):
                summary = cell["closed_loop"][name]
                lo, hi = summary["ci95_cluster"]
                print(
                    f"    {name}: {summary['mean']:+.3f} "
                    f"[{lo:+.3f},{hi:+.3f}] {summary['verdict']}"
                )
            disagreement = cell["same_state"]["first_action_disagreement_rate"]
            lo, hi = disagreement["ci95_cluster"]
            print(
                f"    action disagreement: {disagreement['mean']:.3f} "
                f"[{lo:.3f},{hi:.3f}]"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=20_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "matched_information_aggregate.json",
    )
    args = parser.parse_args()

    result = build(args.n_boot)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_markdown(result)
    print_report(result)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
