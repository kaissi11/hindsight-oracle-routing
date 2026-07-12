#!/usr/bin/env python3
"""Run the predeclared matched-information suites after the "P3" ablation
queue (run_p3_extra_seeds.py / run_p3_parallel.py) drains.

The process may be launched while P3 is active: it only waits and does not load
the model or touch the GPU until every P3 result gate is complete. Runs are
strictly sequential and resumable. Each completed run must reproduce the
recorded v1 look-8 arm before the queue proceeds.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PYTHON = sys.executable
SEEDS = (12345, 13345, 14345)
BUCKETS = ("low", "medium", "high")
P3_JOBS = ("obs_base", "obs_mask", "obs_traffic", "hstress_h4")


def buckets_done(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    return sum(bucket in data.get("buckets", {}) for bucket in BUCKETS)


def matched_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(data.get("complete")) and all(
        bucket in data.get("buckets", {}) for bucket in BUCKETS
    )


def wait_for_p3() -> None:
    gates = [
        RESULTS / f"scenario_bucket_v2_{job}_seed_{seed}.json"
        for job in P3_JOBS
        for seed in SEEDS
    ]
    while True:
        incomplete = [path for path in gates if buckets_done(path) < 3]
        if not incomplete:
            break
        summary = ", ".join(
            f"{path.stem.replace('scenario_bucket_v2_', '')}:"
            f"{buckets_done(path)}/3"
            for path in incomplete
        )
        print(f"[MATCHED-QUEUE] waiting for P3: {summary}", flush=True)
        time.sleep(300)
    print("[MATCHED-QUEUE] P3 result gates complete; waiting 120 s for GPU release")
    time.sleep(120)


def output_path(horizon: int, seed: int) -> Path:
    return RESULTS / f"matched_information_h{horizon}_seed_{seed}.json"


def reference_path(horizon: int, seed: int) -> Path:
    variant = "osrm_s5" if horizon == 8 else "hstress_h4"
    return RESULTS / f"scenario_bucket_v2_{variant}_seed_{seed}.json"


def validate(horizon: int, seed: int) -> bool:
    cmd = [
        PYTHON,
        str(HERE / "validate_matched_information.py"),
        str(output_path(horizon, seed)),
        str(reference_path(horizon, seed)),
    ]
    return subprocess.run(cmd, cwd=HERE).returncode == 0


def run_one(horizon: int, seed: int) -> bool:
    destination = output_path(horizon, seed)
    if matched_complete(destination):
        print(f"[MATCHED-QUEUE] H={horizon} seed={seed}: complete; validating")
        return validate(horizon, seed)

    command = [
        PYTHON,
        "-u",
        str(HERE / "matched_information_eval.py"),
        "--base-seed",
        str(seed),
        "--horizon-hours",
        str(horizon),
        "--output",
        str(destination),
    ]
    log_path = RESULTS / f"matched_information_h{horizon}_seed_{seed}.log"
    print(
        f"[MATCHED-QUEUE] H={horizon} seed={seed}: starting "
        f"(log: {log_path.name})"
    )
    started = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(command)}\n")
        log.flush()
        return_code = subprocess.run(
            command,
            cwd=HERE,
            stdout=log,
            stderr=subprocess.STDOUT,
        ).returncode
    elapsed_minutes = (time.time() - started) / 60
    if return_code != 0 or not matched_complete(destination):
        print(
            f"[MATCHED-QUEUE] H={horizon} seed={seed}: FAILED "
            f"rc={return_code} after {elapsed_minutes:.0f} min"
        )
        return False
    if not validate(horizon, seed):
        print(
            f"[MATCHED-QUEUE] H={horizon} seed={seed}: "
            "VALIDATION FAILED; queue stopped"
        )
        return False
    print(
        f"[MATCHED-QUEUE] H={horizon} seed={seed}: OK "
        f"after {elapsed_minutes:.0f} min"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start-now",
        action="store_true",
        help="skip the P3 gate only when the GPU is already confirmed free",
    )
    args = parser.parse_args()

    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        print("[MATCHED-QUEUE] keep-awake armed")

    if not args.start_now:
        wait_for_p3()

    # Validate both seed-12345 controls before spending the extra-seed budget.
    order = (
        (8, 12345),
        (4, 12345),
        (8, 13345),
        (8, 14345),
        (4, 13345),
        (4, 14345),
    )
    for horizon, seed in order:
        if not run_one(horizon, seed):
            raise SystemExit(1)

    aggregate_command = [
        PYTHON,
        str(HERE / "aggregate_matched_information.py"),
    ]
    if subprocess.run(aggregate_command, cwd=HERE).returncode != 0:
        print("[MATCHED-QUEUE] aggregation FAILED")
        raise SystemExit(1)
    print(
        "[MATCHED-QUEUE] all six suites complete; frozen-arm validation and "
        "matched-information aggregation passed."
    )


if __name__ == "__main__":
    main()
