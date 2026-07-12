#!/usr/bin/env python3
"""Validation gate for a matched-information evaluation run.

The frozen arm must reproduce the recorded v1 look-8 result episode by episode.
This check is intentionally strict: any mismatch stops the queued extra seeds
before more GPU time is spent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BUCKETS = ("low", "medium", "high")


def load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def validate(matched_path: Path, reference_path: Path, atol: float) -> None:
    matched = load(matched_path)
    reference = load(reference_path)
    if not matched.get("complete"):
        raise RuntimeError(f"matched run is incomplete: {matched_path}")

    max_delivered_error = 0.0
    max_time_error = 0.0
    checked = 0
    for bucket in BUCKETS:
        matched_eps = matched.get("buckets", {}).get(bucket, {}).get("episodes", [])
        reference_eps = reference.get("buckets", {}).get(bucket, {}).get("episodes", [])
        if len(matched_eps) != len(reference_eps):
            raise AssertionError(
                f"{bucket}: episode count {len(matched_eps)} != {len(reference_eps)}"
            )

        for matched_ep, reference_ep in zip(matched_eps, reference_eps):
            frozen = matched_ep["aggregate"]["online_frozen"]
            recorded = reference_ep["policy_v1_lookahead"]
            delivered_error = abs(
                float(frozen["delivered_mean"])
                - float(recorded["delivered_mean"])
            )
            time_error = abs(
                float(frozen["elapsed_time_mean_sec"])
                - float(recorded["time_mean"])
            )
            max_delivered_error = max(max_delivered_error, delivered_error)
            max_time_error = max(max_time_error, time_error)
            checked += 1

    if not np.isfinite(max_delivered_error + max_time_error):
        raise AssertionError("non-finite validation error")
    if max_delivered_error > atol or max_time_error > atol:
        raise AssertionError(
            "frozen arm does not reproduce recorded v1 look-8: "
            f"max delivered error={max_delivered_error:.12g}, "
            f"max time error={max_time_error:.12g}, atol={atol}"
        )

    print(
        f"[MATCHED-VALIDATE] PASS {checked} paired episodes; "
        f"max delivered error={max_delivered_error:.3g}; "
        f"max time error={max_time_error:.3g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matched", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--atol", type=float, default=1e-8)
    args = parser.parse_args()
    validate(args.matched, args.reference, args.atol)


if __name__ == "__main__":
    main()
