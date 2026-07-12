#!/usr/bin/env python3
"""Build a cached pool of connected OSRM instances for the Damascus OSRM suites.

Why: the live builder (sample 20 -> snap 20 -> table -> accept/reject)
runs at ~4 s/try with a very low acceptance rate (~20 min/episode observed).
This builder amortizes the work:

1. snap a large batch of uniform bbox points to the road network once,
2. one big /table query over a candidate core,
3. iteratively drop the worst-connected nodes until the core has NO
   unreachable pairs (stricter than the live builder's 10% tolerance),
4. assemble instances by slicing the core matrix - zero extra OSRM calls.

Durations are identical to per-instance /table queries (point-to-point).

Output: results/osrm_instance_pool/pool.npz  (lonlats [I,N,2], durations [I,N,N])
        + pool_meta.json

Run from the code directory (OSRM docker must be up):
    python build_osrm_instance_pool.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

V62_DIR = Path(__file__).resolve().parent  # merged layout: frozen deps live alongside
sys.path.insert(0, str(V62_DIR))

from osrm_client import OSRMClient, DAMASCUS_BBOX, BBox  # noqa: E402  (frozen)


def snap_candidates(client: OSRMClient, rng: np.random.Generator, n: int, radius_m: int,
                    bbox: BBox) -> np.ndarray:
    snapped = []
    t0 = time.time()
    for i in range(n):
        lon = rng.uniform(bbox.min_lon, bbox.max_lon)
        lat = rng.uniform(bbox.min_lat, bbox.max_lat)
        try:
            snapped.append(client.nearest(lon, lat, radius_m=radius_m))
        except Exception:
            continue
        if (i + 1) % 200 == 0:
            print(f"  snapped {i + 1}/{n} ({time.time() - t0:.0f}s)")
    pts = np.asarray(snapped, dtype=np.float64)
    pts = np.unique(np.round(pts, 6), axis=0)
    print(f"  {len(pts)} unique snapped points")
    return pts


def connected_core(client: OSRMClient, pts: np.ndarray, core_size: int, rng: np.random.Generator):
    if len(pts) > core_size:
        idx = rng.choice(len(pts), core_size, replace=False)
        pts = pts[idx]
    print(f"  querying {len(pts)}x{len(pts)} table...")
    D = client.table_durations(pts, timeout=300)

    keep = np.arange(len(pts))
    while True:
        sub = D[np.ix_(keep, keep)]
        inf_mask = ~np.isfinite(sub)
        np.fill_diagonal(inf_mask, False)
        n_inf = inf_mask.sum()
        if n_inf == 0:
            break
        badness = inf_mask.sum(axis=0) + inf_mask.sum(axis=1)
        keep = np.delete(keep, int(np.argmax(badness)))
        if len(keep) < 40:
            raise RuntimeError("Core collapsed below 40 nodes; OSRM extract too fragmented")
    print(f"  fully-connected core: {len(keep)}/{len(pts)} nodes")
    return pts[keep], D[np.ix_(keep, keep)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-candidates", type=int, default=1200)
    parser.add_argument("--core-size", type=int, default=250)
    parser.add_argument("--n-instances", type=int, default=160)
    parser.add_argument("--n-nodes", type=int, default=20)
    parser.add_argument("--snap-radius-m", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--osrm-url", default="http://localhost:5000")
    parser.add_argument("--out-dir", default="results/osrm_instance_pool")
    parser.add_argument("--bbox", type=float, nargs=4, default=None,
                        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
                        help="sampling bbox; default = Damascus")
    args = parser.parse_args()

    bbox = BBox(*args.bbox) if args.bbox else DAMASCUS_BBOX
    client = OSRMClient(base_url=args.osrm_url)
    rng = np.random.default_rng(args.seed)

    print(f"[POOL] bbox: {[bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat]}")
    print("[POOL] snapping candidates...")
    pts = snap_candidates(client, rng, args.n_candidates, args.snap_radius_m, bbox)

    print("[POOL] extracting connected core...")
    core_pts, core_D = connected_core(client, pts, args.core_size, rng)

    print(f"[POOL] assembling {args.n_instances} instances of N={args.n_nodes}...")
    lonlats = np.empty((args.n_instances, args.n_nodes, 2), dtype=np.float64)
    durations = np.empty((args.n_instances, args.n_nodes, args.n_nodes), dtype=np.float32)
    for i in range(args.n_instances):
        sel = rng.choice(len(core_pts), args.n_nodes, replace=False)
        lonlats[i] = core_pts[sel]
        durations[i] = core_D[np.ix_(sel, sel)].astype(np.float32)
    assert np.isfinite(durations).all()

    out_dir = Path(__file__).resolve().parent / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "pool.npz", lonlats=lonlats, durations=durations)
    meta = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": vars(args),
        "bbox": [bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat],
        "core_nodes": int(len(core_pts)),
        "duration_stats_sec": {
            "mean": float(durations.mean()),
            "p50": float(np.percentile(durations, 50)),
            "p95": float(np.percentile(durations, 95)),
            "max": float(durations.max()),
        },
    }
    (out_dir / "pool_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[POOL] saved {args.n_instances} instances -> {out_dir / 'pool.npz'}")
    print(json.dumps(meta["duration_stats_sec"], indent=2))


if __name__ == "__main__":
    main()
