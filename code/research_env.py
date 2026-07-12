# research_env.py — Research Environment (Dynamic TSP)
# 6 features: x, y, visited, is_current, blocked, time_remaining_frac
# Dynamic: road blocking events + traffic fluctuation (this is the research contribution)
# Reward: -travel_cost (dominant) with small terminal penalty
# Purpose: Prove RL handles dynamic disruptions better than classical re-solving

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
from torch import Tensor

from osrm_client import OSRMClient, DAMASCUS_BBOX, BBox
from connected_instance_builder import ConnectedBuilderConfig, build_osrm_connected, normalize_lonlat
from search_utils import augment_coords_8fold


@dataclass
class ResearchEnvConfig:
    n_nodes: int = 20
    num_instances: int = 64
    device: str = "cuda"

    time_horizon_sec: float = 8 * 60 * 60
    travel_scale_sec: float = 3600.0

    terminal_penalty_weight: float = 0.5

    auto_reset: bool = True
    min_visits: int = 3

    # Dynamic events (THIS is the research differentiator)
    block_prob_per_step: float = 0.03
    unblock_prob_per_step: float = 0.02
    traffic_init: float = 1.0
    traffic_min: float = 0.7
    traffic_max: float = 1.8
    traffic_rw_std: float = 0.03

    # Synthetic distance generation
    noise_strength: float = 0.10
    asymmetry_strength: float = 0.20

    # OSRM
    use_osrm_instances: bool = False
    osrm_base_url: str = "http://localhost:5000"
    osrm_profile: str = "driving"
    osrm_bbox: BBox = DAMASCUS_BBOX
    osrm_snap_radius_m: int = 50000
    osrm_max_inf_frac: float = 0.10
    osrm_max_tries: int = 200

    # Augmentation
    use_augmentation: bool = True


class ResearchEnv:
    """
    Dynamic TSP environment for research benchmarking.
    Observation: [x, y, visited, is_current, blocked, time_remaining_frac] = 6 features.
    Dynamic: blocking events + traffic random walk (RL's advantage over classical solvers).
    Reward: -travel_cost with terminal penalty. No visit/priority bonus.
    """

    NODE_DIM = 6

    def __init__(self, cfg: ResearchEnvConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.num_instances = cfg.num_instances
        self.n_nodes = cfg.n_nodes

        # POMO expansion
        self.pomo_size = cfg.n_nodes
        self.num_envs = self.num_instances * self.pomo_size
        self.batch = torch.arange(self.num_envs, device=self.device)

        # Static tensors
        self.coords: Tensor = torch.empty(0, device=self.device)
        self.coords_lonlat: Optional[Tensor] = None
        self.base_dist: Tensor = torch.empty(0, device=self.device)

        # Dynamic tensors
        self.current_node = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.visited = torch.zeros(self.num_envs, self.n_nodes, dtype=torch.bool, device=self.device)
        self.elapsed_time = torch.zeros(self.num_envs, device=self.device)
        self.blocked: Tensor = torch.zeros(self.num_envs, self.n_nodes, device=self.device)
        self.traffic: Tensor = torch.full((self.num_envs,), cfg.traffic_init, device=self.device)

        # OSRM client
        self._osrm: Optional[OSRMClient] = None
        if cfg.use_osrm_instances:
            self._osrm = OSRMClient(base_url=cfg.osrm_base_url, profile=cfg.osrm_profile)

        self._reset_all()

    @property
    def node_dim(self) -> int:
        return self.NODE_DIM

    @property
    def obs_dim(self) -> int:
        return self.n_nodes * self.node_dim

    @property
    def action_dim(self) -> int:
        return self.n_nodes

    def get_pomo_starting_actions(self) -> Tensor:
        return torch.arange(self.pomo_size, device=self.device).repeat(self.num_instances)

    def reset(self) -> Tuple[Tensor, Dict[str, Any]]:
        self._reset_all()
        return self._build_obs(), {"action_mask": self._build_action_mask()}

    def step(self, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Any]]:
        actions = actions.to(self.device).long()

        mask = self._build_action_mask()
        chosen = actions.clamp(0, self.n_nodes - 1)
        invalid = ~mask[self.batch, chosen]

        safe_actions = chosen.clone()
        safe_actions[invalid] = self.current_node[invalid]

        prev = self.current_node
        base_travel = self.base_dist[self.batch, prev, safe_actions]

        # Traffic multiplier affects travel time (dynamic!)
        travel_sec = base_travel * self.traffic

        big = 0.25 * self.cfg.time_horizon_sec
        travel_sec = torch.where(
            torch.isfinite(travel_sec), travel_sec,
            torch.full_like(travel_sec, big),
        )
        travel_sec = travel_sec + invalid.float() * big

        self.elapsed_time += travel_sec

        # === REWARD: Travel cost dominant, small invalid penalty ===
        travel_hours = travel_sec / max(1e-6, self.cfg.travel_scale_sec)
        reward = -travel_hours
        reward -= invalid.float() * 0.5

        # Track visits
        self.visited[self.batch, safe_actions] = True
        self.current_node = safe_actions

        visited_count = self.visited.sum(dim=1)
        can_terminate = visited_count >= self.cfg.min_visits

        done_time = self.elapsed_time >= self.cfg.time_horizon_sec
        done_all = self.visited.all(dim=1)
        # Always end when the horizon is hit, even if min_visits not reached. Otherwise an env can
        # never set done=True (e.g. heavy blocking + invalid moves → time runs out with <3 visits),
        # and the training loop spins forever on one epoch.
        done = done_time | (done_all & can_terminate)

        # Terminal penalty for unvisited nodes
        horizon_fail = done_time & ~done_all
        if horizon_fail.any() and self.cfg.terminal_penalty_weight > 0:
            idx = horizon_fail.nonzero(as_tuple=False).squeeze(-1)
            unvisited_count = (~self.visited[idx]).float().sum(dim=1)
            reward[idx] -= self.cfg.terminal_penalty_weight * unvisited_count

        # Dynamic events: traffic + blocking
        self._update_traffic()
        self._update_blocking()

        info: Dict[str, Any] = {"action_mask": self._build_action_mask()}

        if done.any() and self.cfg.auto_reset:
            idx = done.nonzero(as_tuple=False).squeeze(-1)
            info["episode_nodes_visited"] = self.visited[idx].sum(dim=1).long()
            info["episode_time_used"] = self.elapsed_time[idx]
            self._reset_indices(idx)

        return self._build_obs(), reward, done, info

    def _update_traffic(self) -> None:
        self.traffic += self.cfg.traffic_rw_std * torch.randn_like(self.traffic)
        self.traffic = self.traffic.clamp(self.cfg.traffic_min, self.cfg.traffic_max)

    def _update_blocking(self) -> None:
        BN, N = self.num_envs, self.n_nodes
        block = torch.rand((BN,), device=self.device) < self.cfg.block_prob_per_step
        if block.any():
            idx = block.nonzero(as_tuple=False).squeeze(-1)
            nodes = torch.randint(1, N, (idx.numel(),), device=self.device)
            self.blocked[idx, nodes] = 1.0
        unblock = torch.rand((BN,), device=self.device) < self.cfg.unblock_prob_per_step
        if unblock.any():
            idx = unblock.nonzero(as_tuple=False).squeeze(-1)
            nodes = torch.randint(1, N, (idx.numel(),), device=self.device)
            self.blocked[idx, nodes] = 0.0

    # ---- Reset ----

    def _reset_all(self) -> None:
        B, N = self.num_instances, self.n_nodes

        if self.cfg.use_osrm_instances:
            coords_xy = torch.empty((B, N, 2), device=self.device)
            coords_ll = torch.empty((B, N, 2), device=self.device, dtype=torch.float64)
            base = torch.empty((B, N, N), device=self.device)

            for b in range(B):
                build_cfg = ConnectedBuilderConfig(
                    n_nodes=N,
                    bbox=self.cfg.osrm_bbox,
                    seed=int(torch.randint(0, 2**31 - 1, (1,)).item()),
                    max_inf_frac=self.cfg.osrm_max_inf_frac,
                    snap_radius_m=self.cfg.osrm_snap_radius_m,
                    max_tries=self.cfg.osrm_max_tries,
                )
                lonlats_np, D_np = build_osrm_connected(self._osrm, build_cfg)
                xy_np = normalize_lonlat(lonlats_np, self.cfg.osrm_bbox)
                coords_xy[b] = torch.from_numpy(xy_np).to(self.device)
                coords_ll[b] = torch.from_numpy(lonlats_np).to(self.device)
                base[b] = torch.from_numpy(D_np.astype(np.float32)).to(self.device)

            self.coords_lonlat = coords_ll.repeat_interleave(self.pomo_size, dim=0)
        else:
            coords_xy = torch.rand((B, N, 2), device=self.device)
            dist = torch.cdist(coords_xy, coords_xy, p=2) + 1e-6
            noise = 1.0 + self.cfg.noise_strength * torch.randn((B, N, N), device=self.device)
            dist = dist * noise.clamp(0.2, 3.0)
            bias = self.cfg.asymmetry_strength * torch.randn((B, N, N), device=self.device)
            dist = dist * (1.0 + bias).clamp(0.2, 3.0)
            base = dist * (12 * 60.0)
            self.coords_lonlat = None

        # Instance augmentation: randomly pick one of 8 transformations
        if self.cfg.use_augmentation and not self.cfg.use_osrm_instances:
            aug_variants = augment_coords_8fold(coords_xy)
            aug_idx = torch.randint(0, 8, (B,))
            for b in range(B):
                coords_xy[b] = aug_variants[aug_idx[b].item()][b]
                # recompute distances for augmented coords
                d = torch.cdist(coords_xy[b:b+1], coords_xy[b:b+1], p=2) + 1e-6
                noise_b = 1.0 + self.cfg.noise_strength * torch.randn((1, N, N), device=self.device)
                d = d * noise_b.clamp(0.2, 3.0)
                bias_b = self.cfg.asymmetry_strength * torch.randn((1, N, N), device=self.device)
                d = d * (1.0 + bias_b).clamp(0.2, 3.0)
                base[b] = (d * (12 * 60.0)).squeeze(0)

        self.coords = coords_xy.repeat_interleave(self.pomo_size, dim=0)
        self.base_dist = base.repeat_interleave(self.pomo_size, dim=0)

        self.blocked = torch.zeros((self.num_envs, self.n_nodes), device=self.device)
        self.current_node.zero_()
        self.visited.zero_()
        self.visited[:, 0] = True
        self.elapsed_time.zero_()
        self.traffic.fill_(self.cfg.traffic_init)

    def _reset_indices(self, idx: Tensor) -> None:
        if self.cfg.use_osrm_instances:
            self._reset_all()
            return

        num, N = idx.numel(), self.n_nodes
        coords = torch.rand((num, N, 2), device=self.device)

        if self.cfg.use_augmentation:
            aug_variants = augment_coords_8fold(coords)
            aug_idx = torch.randint(0, 8, (num,))
            for b in range(num):
                coords[b] = aug_variants[aug_idx[b].item()][b]

        dist = torch.cdist(coords, coords, p=2) + 1e-6
        noise = 1.0 + self.cfg.noise_strength * torch.randn((num, N, N), device=self.device)
        dist = dist * noise.clamp(0.2, 3.0)
        bias = self.cfg.asymmetry_strength * torch.randn((num, N, N), device=self.device)
        dist = dist * (1.0 + bias).clamp(0.2, 3.0)
        dist = dist * (12 * 60.0)

        self.coords[idx] = coords
        self.base_dist[idx] = dist
        self.blocked[idx] = 0.0
        self.current_node[idx] = 0
        self.visited[idx] = False
        self.visited[idx, 0] = True
        self.elapsed_time[idx] = 0.0
        self.traffic[idx] = self.cfg.traffic_init

    # ---- Obs / Mask ----

    def _build_obs(self) -> Tensor:
        BN, N = self.num_envs, self.n_nodes

        x = self.coords[:, :, 0]
        y = self.coords[:, :, 1]
        visited = self.visited.float()

        is_current = torch.zeros((BN, N), device=self.device)
        is_current[self.batch, self.current_node] = 1.0

        blocked = self.blocked

        t_rem = (self.cfg.time_horizon_sec - self.elapsed_time).clamp(min=0)
        t_frac = (t_rem / self.cfg.time_horizon_sec).clamp(0, 1)
        t_feat = t_frac.unsqueeze(1).expand(BN, N)

        feats = torch.stack([x, y, visited, is_current, blocked, t_feat], dim=-1)
        return feats.reshape(BN, N * self.NODE_DIM)

    def _build_action_mask(self) -> Tensor:
        mask = (~self.visited) & (self.blocked < 0.5)
        mask[:, 0] = False
        return mask
