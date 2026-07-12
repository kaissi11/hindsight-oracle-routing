# osrm_connected_sampler.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from osrm_client import OSRMClient, BBox, DAMASCUS_BBOX


@dataclass
class SamplerConfig:
    n_nodes: int = 20
    bbox: BBox = DAMASCUS_BBOX
    seed: int = 0

    # nearest snapping
    snap_radius_m: int = 50000
    snap_radius_growth: float = 2.0
    snap_max_radius_m: int = 200000

    # connectivity acceptance
    max_inf_frac: float = 0.10   # table inf entries ratio threshold
    max_tries: int = 200


def _bfs_reachable(adj: np.ndarray, start: int) -> np.ndarray:
    """
    adj: [N,N] bool directed adjacency
    returns visited bool [N]
    """
    N = adj.shape[0]
    seen = np.zeros((N,), dtype=bool)
    q = [start]
    seen[start] = True
    while q:
        u = q.pop()
        nbrs = np.where(adj[u])[0]
        for v in nbrs:
            if not seen[v]:
                seen[v] = True
                q.append(v)
    return seen


def sample_uniform_lonlat(rng: np.random.Generator, bbox: BBox, n: int) -> np.ndarray:
    lon = rng.uniform(bbox.min_lon, bbox.max_lon, size=(n,))
    lat = rng.uniform(bbox.min_lat, bbox.max_lat, size=(n,))
    return np.stack([lon, lat], axis=1).astype(np.float64)


def snap_all_adaptive(client: OSRMClient, lonlats: np.ndarray, cfg: SamplerConfig) -> np.ndarray:
    snapped = np.zeros_like(lonlats, dtype=np.float64)
    for i in range(lonlats.shape[0]):
        lon, lat = float(lonlats[i, 0]), float(lonlats[i, 1])
        r = cfg.snap_radius_m
        ok = False
        while r <= cfg.snap_max_radius_m:
            try:
                snapped[i] = client.nearest(lon, lat, radius_m=int(r))
                ok = True
                break
            except Exception:
                r = int(r * cfg.snap_radius_growth)

        if not ok:
            raise RuntimeError(f"Failed to snap point {i} within max radius")
    return snapped


def build_connected_instance(
    client: OSRMClient,
    cfg: SamplerConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      snapped_lonlats: [N,2]
      D: [N,N] durations seconds (inf where unreachable)
    Connectivity condition used:
      - depot (0) can reach all nodes (finite path)
      - all nodes can reach depot (finite path)
    This implies strong connectivity via i->depot->j when both edges exist in directed graph.
    """
    rng = np.random.default_rng(cfg.seed)

    for attempt in range(cfg.max_tries):
        lonlats = sample_uniform_lonlat(rng, cfg.bbox, cfg.n_nodes)

        # snap to road
        try:
            snapped = snap_all_adaptive(client, lonlats, cfg)
        except Exception:
            continue

        # table durations
        try:
            D = client.table_durations(snapped)
        except Exception:
            continue

        # inf fraction
        inf_frac = float(np.isinf(D).mean())
        if inf_frac > cfg.max_inf_frac:
            continue

        # directed adjacency (finite edges)
        adj = np.isfinite(D) & (D > 0.0)

        # reachability checks with depot=0
        reach_from_depot = _bfs_reachable(adj, 0)
        reach_to_depot = _bfs_reachable(adj.T, 0)

        if reach_from_depot.all() and reach_to_depot.all():
            return snapped, D

    raise RuntimeError("Could not build connected OSRM instance within max_tries")
