#!/usr/bin/env python3
"""Progress watcher for the six matched-information suites (read-only).

Shows per-suite state, bucket/episode progress, log freshness, and overall
completion. Deliberately prints NO result values: the predeclared
interpretation rules are applied only after all six suites are complete
AND validated.

  python watch_matched_suites.py             # one snapshot
  python watch_matched_suites.py --watch 60  # refresh every 60 s until 6/6
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SUITES = [(8, 12345), (4, 12345), (8, 13345), (8, 14345), (4, 13345), (4, 14345)]
BUCKETS = ("low", "medium", "high")
EPISODE_RE = re.compile(r"\[MATCHED\] (low|medium|high) episode (\d+)/(\d+)")
EPISODES_PER_BUCKET = 40


def suite_status(horizon: int, seed: int, now: float) -> tuple[bool, str]:
    json_path = RESULTS / f"matched_information_h{horizon}_seed_{seed}.json"
    log_file = RESULTS / f"matched_information_h{horizon}_seed_{seed}.log"

    complete = False
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            complete = bool(data.get("complete")) and all(
                b in data.get("buckets", {}) for b in BUCKETS
            )
        except (OSError, json.JSONDecodeError):
            pass
    if complete:
        return True, f"  h{horizon} seed {seed}: COMPLETE"

    if not log_file.exists():
        return False, f"  h{horizon} seed {seed}: PENDING (not started)"

    age_min = (now - log_file.stat().st_mtime) / 60
    last = None
    for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines()[::-1]:
        match = EPISODE_RE.search(line)
        if match:
            last = match
            break
    if last is None:
        state = "STARTING" if age_min < 10 else f"STALLED? ({age_min:.0f} min silent)"
        return False, f"  h{horizon} seed {seed}: {state}"

    bucket, episode, total = last.group(1), int(last.group(2)), int(last.group(3))
    done_eps = BUCKETS.index(bucket) * EPISODES_PER_BUCKET + episode
    pct = 100.0 * done_eps / (EPISODES_PER_BUCKET * len(BUCKETS))
    state = "RUNNING" if age_min < 10 else f"STALLED? ({age_min:.0f} min silent)"
    return False, (f"  h{horizon} seed {seed}: {state} - {bucket} {episode}/{total} "
                   f"(~{pct:.0f}% of suite; log {age_min:.0f} min ago)")


def snapshot() -> int:
    now = time.time()
    done = 0
    lines = []
    for horizon, seed in SUITES:
        is_done, line = suite_status(horizon, seed, now)
        done += int(is_done)
        lines.append(line)
    print(f"[{time.strftime('%H:%M:%S')}] matched suites: {done}/6 complete")
    for line in lines:
        print(line)
    orchestrator_log = RESULTS / "matched_parallel.log"
    if orchestrator_log.exists():
        tail = orchestrator_log.read_text(encoding="utf-8", errors="replace").splitlines()
        if tail:
            print(f"  orchestrator: {tail[-1]}")
    print("  (result values withheld by design until 6/6 complete + validated; then:")
    print("   aggregate_matched_information.py -> the paper's matched-information tables)")
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", type=int, default=0, metavar="SECONDS",
                        help="refresh every N seconds until all six complete")
    args = parser.parse_args()
    while True:
        done = snapshot()
        if not args.watch or done == len(SUITES):
            break
        time.sleep(args.watch)
        print()


if __name__ == "__main__":
    main()
