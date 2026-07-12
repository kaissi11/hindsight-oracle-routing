# connected_instance_builder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

from osrm_client import OSRMClient, BBox, DAMASCUS_BBOX
from osrm_connected_sampler import SamplerConfig, build_connected_instance


@dataclass
class ConnectedBuilderConfig:
    n_nodes: int = 20
    bbox: BBox = DAMASCUS_BBOX
    seed: int = 0
    max_inf_frac: float = 0.10
    snap_radius_m: int = 50000
    max_tries: int = 200


def build_osrm_connected(
    client: OSRMClient,
    cfg: ConnectedBuilderConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    scfg = SamplerConfig(
        n_nodes=cfg.n_nodes,
        bbox=cfg.bbox,
        seed=cfg.seed,
        snap_radius_m=cfg.snap_radius_m,
        max_inf_frac=cfg.max_inf_frac,
        max_tries=cfg.max_tries,
    )
    lonlats, D = build_connected_instance(client, scfg)
    return lonlats, D


def normalize_lonlat(lonlats: np.ndarray, bbox: BBox) -> np.ndarray:
    """
    Map lonlat to [0..1] for model inputs, but keep original lonlat for plotting.
    """
    lon = lonlats[:, 0]
    lat = lonlats[:, 1]
    x = (lon - bbox.min_lon) / (bbox.max_lon - bbox.min_lon + 1e-12)
    y = (lat - bbox.min_lat) / (bbox.max_lat - bbox.min_lat + 1e-12)
    xy = np.stack([x, y], axis=1).astype(np.float32)
    xy = np.clip(xy, 0.0, 1.0)
    return xy
