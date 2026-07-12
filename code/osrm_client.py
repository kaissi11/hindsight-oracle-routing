# osrm_client.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import requests


@dataclass(frozen=True)
class BBox:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def clamp(self, lon: float, lat: float) -> Tuple[float, float]:
        lon2 = min(max(lon, self.min_lon), self.max_lon)
        lat2 = min(max(lat, self.min_lat), self.max_lat)
        return lon2, lat2


# Syria-ish bbox (adjust if you want tighter/looser)
# SYRIA_BBOX = BBox(
#     min_lon=35.70,
#     min_lat=32.30,
#     max_lon=42.35,
#     max_lat=37.35,
# )

DAMASCUS_BBOX = BBox(
    min_lon=36.15,
    min_lat=33.40,
    max_lon=36.40,
    max_lat=33.65,
)


class OSRMClient:
    def __init__(
        self,
        base_url: str = "http://localhost:5000",
        profile: str = "driving",
        user_agent: str = "rl-osrm-client/1.0",
    ):
        self.base_url = base_url.rstrip("/")
        self.profile = profile
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 60) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        r = self.session.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 60) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        r = self.session.post(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def route_with_steps(self, coords):
        """
        coords: list/array of (lon, lat) pairs. Length >= 2.
        Returns raw OSRM JSON with steps.
        """
        if len(coords) < 2:
            raise ValueError("Need at least 2 coordinates")

        coord_str = ";".join(f"{lon},{lat}" for (lon, lat) in coords)
        url = f"{self.base_url}/route/v1/{self.profile}/{coord_str}"
        params = {
            "overview": "full",
            "steps": "true",
            "geometries": "geojson",
        }
        r = self.session.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()


    # ---------- sanity ----------
    def sanity_nearest(self, lon: float, lat: float, radius_m: int = 100000) -> bool:
        try:
            _ = self.nearest(lon, lat, radius_m=radius_m)
            return True
        except Exception:
            return False

    # ---------- services ----------
    def nearest(self, lon: float, lat: float, radius_m: int = 50000) -> Tuple[float, float]:
        js = self._get(
            f"/nearest/v1/{self.profile}/{lon:.6f},{lat:.6f}",
            params={"radiuses": str(int(radius_m))},
            timeout=30,
        )
        if js.get("code") != "Ok" or not js.get("waypoints"):
            raise RuntimeError(f"OSRM nearest failed: {js.get('code')}")

        loc = js["waypoints"][0]["location"]  # [lon,lat]
        return float(loc[0]), float(loc[1])

    def table_durations(self, lonlats: np.ndarray, timeout: int = 120) -> np.ndarray:
        """
        lonlats: [N,2] with columns [lon,lat]
        returns: [N,N] float64 seconds, inf where unreachable
        """
        assert lonlats.ndim == 2 and lonlats.shape[1] == 2
        coords = ";".join([f"{lon:.6f},{lat:.6f}" for lon, lat in lonlats])

        # OSRM uses GET for /table; POST is also accepted in some builds.
        js = self._get(
            f"/table/v1/{self.profile}/{coords}",
            params={"annotations": "duration"},
            timeout=timeout,
        )

        if js.get("code") != "Ok":
            raise RuntimeError(f"OSRM table failed: {js.get('code')}")

        durations = js.get("durations", None)
        if durations is None:
            raise RuntimeError("OSRM table returned no durations")

        D = np.array(durations, dtype=np.float64)
        D[~np.isfinite(D)] = np.inf
        return D

    def route_geojson(self, lonlats: np.ndarray, timeout: int = 120) -> np.ndarray:
        """
        Full route through waypoints in given order.
        returns polyline coords [M,2] lonlat
        """
        coords = ";".join([f"{lon:.6f},{lat:.6f}" for lon, lat in lonlats])
        js = self._get(
            f"/route/v1/{self.profile}/{coords}",
            params={
                "overview": "full",
                "geometries": "geojson",
                "steps": "false",
            },
            timeout=timeout,
        )
        if js.get("code") != "Ok" or not js.get("routes"):
            raise RuntimeError(f"OSRM route failed: {js.get('code')}")
        geom = js["routes"][0]["geometry"]
        line = np.array(geom["coordinates"], dtype=np.float64)  # [M,2] lonlat
        return line

    def route_leg_with_meta(self, lonlats: np.ndarray, timeout: int = 120):
        """
        Single leg (A -> B) with distance/duration + geometry.

        lonlats: [2,2] array [[lonA, latA], [lonB, latB]]
        Returns:
          distance_m: float
          duration_s: float
          geometry: np.ndarray [M,2] (lon,lat)
        """
        assert lonlats.shape == (2, 2), "route_leg_with_meta expects exactly 2 points"

        coords = ";".join([f"{lon:.6f},{lat:.6f}" for lon, lat in lonlats])
        js = self._get(
            f"/route/v1/{self.profile}/{coords}",
            params={
                "overview": "full",
                "geometries": "geojson",
                "steps": "false",
            },
            timeout=timeout,
        )

        if js.get("code") != "Ok" or not js.get("routes"):
            raise RuntimeError(f"OSRM route_leg_with_meta failed: {js.get('code')}")

        route = js["routes"][0]
        distance_m = float(route.get("distance", 0.0))
        duration_s = float(route.get("duration", 0.0))

        geom = route["geometry"]
        line = np.array(geom["coordinates"], dtype=np.float64)  # [M,2] lonlat

        return distance_m, duration_s, line
  
