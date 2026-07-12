# research_env_v2.py — Stage 2: decision-relevant dynamics.
# Derived from frozen v6.2/research_env.py. Deltas (see STAGE2_PLAN.md):
#   1. Scalar traffic -> spatially correlated zonal OU traffic field;
#      per-edge multiplier = mean of endpoint zone factors.
#   2. Edge blocking via correlated incident disks (clears as a unit);
#      node blocking kept unchanged.
#   3. effective_dist() exposes the current effective duration matrix
#      (inf on blocked edges) for the model's edge-attention input.
#   4. Action mask requires a finite direct edge from the current node.
# OSRM instance generation intentionally dropped here: training is synthetic;
# OSRM evals inject cached pool instances directly into the eval harness.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

_V62 = Path(__file__).resolve().parent  # merged layout: frozen deps live alongside
if str(_V62) not in sys.path:
    sys.path.insert(0, str(_V62))

from search_utils import augment_coords_8fold  # noqa: E402  (v6.2, frozen)


@dataclass
class ResearchEnvV2Config:
    n_nodes: int = 20
    num_instances: int = 64
    device: str = "cuda"

    time_horizon_sec: float = 8 * 60 * 60
    travel_scale_sec: float = 3600.0
    terminal_penalty_weight: float = 0.5

    auto_reset: bool = True
    min_visits: int = 3

    # Node blocking (unchanged from v6.2)
    block_prob_per_step: float = 0.03
    unblock_prob_per_step: float = 0.02

    # Zonal traffic field (replaces v6.2 scalar traffic)
    zone_grid: int = 4
    zone_ou_theta: float = 0.05      # mean reversion toward traffic_init
    zone_ou_sigma: float = 0.15      # innovation std per step (pre-smoothing)
    traffic_init: float = 1.0
    traffic_min: float = 0.7
    traffic_max: float = 1.8

    # Edge blocking via incidents
    edge_block_prob_per_step: float = 0.02
    incident_clear_prob: float = 0.10
    incident_radius: float = 0.15

    # Optional deterministic rush-hour cycle (Stage 5+ falsification test):
    # zone-phased sinusoid overlaid on the OU field. amp=0 disables (default,
    # preserves all recorded behavior). Makes the dynamics non-Markovian in
    # the observed matrix: near-future costs differ predictably from current.
    traffic_cycle_amp: float = 0.0
    traffic_cycle_period: int = 40

    # Synthetic distance generation (unchanged)
    noise_strength: float = 0.10
    asymmetry_strength: float = 0.20
    use_augmentation: bool = True


class ResearchEnvV2:
    """Dynamic TSP env with spatially correlated traffic + edge blocking.

    Observation: [x, y, visited, is_current, blocked, time_remaining_frac]
    (6 features, identical to v6.2 — dynamics are observed through the
    effective distance matrix fed to the model's edge attention).
    """

    NODE_DIM = 6

    def __init__(self, cfg: ResearchEnvV2Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.num_instances = cfg.num_instances
        self.n_nodes = cfg.n_nodes

        self.pomo_size = cfg.n_nodes
        self.num_envs = self.num_instances * self.pomo_size
        self.batch = torch.arange(self.num_envs, device=self.device)

        # Static per-episode tensors
        self.coords: Tensor = torch.empty(0, device=self.device)
        self.base_dist: Tensor = torch.empty(0, device=self.device)

        # Dynamic tensors
        self.current_node = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.visited = torch.zeros(self.num_envs, self.n_nodes, dtype=torch.bool, device=self.device)
        self.elapsed_time = torch.zeros(self.num_envs, device=self.device)
        self.blocked: Tensor = torch.zeros(self.num_envs, self.n_nodes, device=self.device)

        K = cfg.zone_grid
        self.zone_factors = torch.full((self.num_envs, K, K), cfg.traffic_init, device=self.device)
        gx, gy = torch.meshgrid(torch.arange(K, device=self.device),
                                torch.arange(K, device=self.device), indexing="ij")
        self._zone_phase = 2 * torch.pi * (gx + gy).float() / (2.0 * K)   # [K,K]
        self._cycle_step = 0
        self.edge_blocked = torch.zeros(self.num_envs, self.n_nodes, self.n_nodes,
                                        dtype=torch.bool, device=self.device)
        self._node_zone = torch.zeros(self.num_envs, self.n_nodes, dtype=torch.long, device=self.device)
        self._edge_mid = torch.zeros(self.num_envs, self.n_nodes, self.n_nodes, 2, device=self.device)
        self._eff_dist = torch.empty(0, device=self.device)

        self._reset_all()

    # ---- Properties ----

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

    def effective_dist(self) -> Tensor:
        """Current effective duration matrix [BN,N,N]; inf on blocked edges."""
        return self._eff_dist

    # ---- Core API ----

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
        travel_sec = self._eff_dist[self.batch, prev, safe_actions]

        big = 0.25 * self.cfg.time_horizon_sec
        travel_sec = torch.where(
            torch.isfinite(travel_sec), travel_sec,
            torch.full_like(travel_sec, big),
        )
        travel_sec = travel_sec + invalid.float() * big

        self.elapsed_time += travel_sec

        travel_hours = travel_sec / max(1e-6, self.cfg.travel_scale_sec)
        reward = -travel_hours
        reward -= invalid.float() * 0.5

        self.visited[self.batch, safe_actions] = True
        self.current_node = safe_actions

        visited_count = self.visited.sum(dim=1)
        can_terminate = visited_count >= self.cfg.min_visits

        done_time = self.elapsed_time >= self.cfg.time_horizon_sec
        done_all = self.visited.all(dim=1)
        done = done_time | (done_all & can_terminate)

        horizon_fail = done_time & ~done_all
        if horizon_fail.any() and self.cfg.terminal_penalty_weight > 0:
            idx = horizon_fail.nonzero(as_tuple=False).squeeze(-1)
            unvisited_count = (~self.visited[idx]).float().sum(dim=1)
            reward[idx] -= self.cfg.terminal_penalty_weight * unvisited_count

        # Dynamic events
        self._cycle_step += 1
        self._update_zonal_traffic()
        self._update_node_blocking()
        self._update_edge_blocking()
        self._refresh_effective_dist()

        info: Dict[str, Any] = {"action_mask": self._build_action_mask()}

        if done.any() and self.cfg.auto_reset:
            idx = done.nonzero(as_tuple=False).squeeze(-1)
            info["episode_nodes_visited"] = self.visited[idx].sum(dim=1).long()
            info["episode_time_used"] = self.elapsed_time[idx]
            self._reset_indices(idx)

        return self._build_obs(), reward, done, info

    # ---- Dynamics ----

    def _smoothed_noise(self, like: Tensor) -> Tensor:
        """Spatially correlated innovations: 3x3 neighbor smoothing, replicate padding."""
        eps = self.cfg.zone_ou_sigma * torch.randn_like(like)
        return F.avg_pool2d(
            F.pad(eps.unsqueeze(1), (1, 1, 1, 1), mode="replicate"),
            kernel_size=3, stride=1,
        ).squeeze(1)

    def _init_zone_factors(self, idx: Tensor) -> None:
        """Sample from the OU stationary distribution so episodes start with a
        heterogeneous (realistic) traffic field rather than a flat one."""
        theta = self.cfg.zone_ou_theta
        stationary_scale = 1.0 / max(1e-6, (1.0 - (1.0 - theta) ** 2)) ** 0.5
        noise = self._smoothed_noise(self.zone_factors[idx]) * stationary_scale
        self.zone_factors[idx] = (self.cfg.traffic_init + noise).clamp(
            self.cfg.traffic_min, self.cfg.traffic_max)

    def _update_zonal_traffic(self) -> None:
        cfg = self.cfg
        eps = self._smoothed_noise(self.zone_factors)
        self.zone_factors = self.zone_factors + cfg.zone_ou_theta * (cfg.traffic_init - self.zone_factors) + eps
        self.zone_factors = self.zone_factors.clamp(cfg.traffic_min, cfg.traffic_max)

    def _update_node_blocking(self) -> None:
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

    def _update_edge_blocking(self) -> None:
        BN = self.num_envs
        clear = torch.rand((BN,), device=self.device) < self.cfg.incident_clear_prob
        if clear.any():
            self.edge_blocked[clear.nonzero(as_tuple=False).squeeze(-1)] = False

        spawn = torch.rand((BN,), device=self.device) < self.cfg.edge_block_prob_per_step
        if spawn.any():
            idx = spawn.nonzero(as_tuple=False).squeeze(-1)
            centers = torch.rand((idx.numel(), 1, 1, 2), device=self.device)
            d2 = ((self._edge_mid[idx] - centers) ** 2).sum(dim=-1)
            hit = d2 < (self.cfg.incident_radius ** 2)
            self.edge_blocked[idx] |= hit

    def _refresh_effective_dist(self) -> None:
        K = self.cfg.zone_grid
        zf = self.zone_factors
        if self.cfg.traffic_cycle_amp > 0:
            mod = self.cfg.traffic_cycle_amp * torch.sin(
                2 * torch.pi * self._cycle_step / max(1, self.cfg.traffic_cycle_period)
                + self._zone_phase)
            zf = (zf + mod).clamp(self.cfg.traffic_min, self.cfg.traffic_max)
        zf_flat = zf.view(self.num_envs, K * K)
        node_zf = zf_flat.gather(1, self._node_zone)                      # [BN,N]
        edge_mult = 0.5 * (node_zf.unsqueeze(2) + node_zf.unsqueeze(1))   # [BN,N,N]
        eff = self.base_dist * edge_mult
        self._eff_dist = torch.where(
            self.edge_blocked, torch.full_like(eff, float("inf")), eff,
        )

    def _assign_zones(self, idx: Tensor) -> None:
        K = self.cfg.zone_grid
        zx = (self.coords[idx, :, 0] * K).long().clamp(0, K - 1)
        zy = (self.coords[idx, :, 1] * K).long().clamp(0, K - 1)
        self._node_zone[idx] = zx * K + zy
        c = self.coords[idx]
        self._edge_mid[idx] = 0.5 * (c.unsqueeze(2) + c.unsqueeze(1))

    # ---- Reset ----

    def _make_instances(self, num: int):
        N = self.n_nodes
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
        return coords, dist

    def _reset_all(self) -> None:
        B = self.num_instances
        coords, dist = self._make_instances(B)
        self.coords = coords.repeat_interleave(self.pomo_size, dim=0)
        self.base_dist = dist.repeat_interleave(self.pomo_size, dim=0)

        self.blocked = torch.zeros((self.num_envs, self.n_nodes), device=self.device)
        self.current_node.zero_()
        self.visited.zero_()
        self.visited[:, 0] = True
        self.elapsed_time.zero_()
        self._init_zone_factors(self.batch)
        self.edge_blocked.zero_()
        self._assign_zones(self.batch)
        self._refresh_effective_dist()

    def _reset_indices(self, idx: Tensor) -> None:
        num = idx.numel()
        coords, dist = self._make_instances(num)
        self.coords[idx] = coords
        self.base_dist[idx] = dist
        self.blocked[idx] = 0.0
        self.current_node[idx] = 0
        self.visited[idx] = False
        self.visited[idx, 0] = True
        self.elapsed_time[idx] = 0.0
        self._init_zone_factors(idx)
        self.edge_blocked[idx] = False
        self._assign_zones(idx)
        self._refresh_effective_dist()

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
        traversable = ~self.edge_blocked[self.batch, self.current_node]  # [BN,N]
        mask = (~self.visited) & (self.blocked < 0.5) & traversable
        mask[:, 0] = False
        return mask
