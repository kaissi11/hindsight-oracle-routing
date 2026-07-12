#!/usr/bin/env python3
"""Main paired evaluation under decision-relevant dynamics (v2 env).

Methods (all on IDENTICAL instances + IDENTICAL presampled disruption schedules):
- policy_v2_greedy   : v2-finetuned policy, greedy decoding
- policy_v2_samplexN : v2-finetuned policy, best-of-K sampling
- policy_v1_samplexN : FROZEN v1 policy under v2 dynamics (ablation:
                       does retraining under the new dynamics help?)
- rolling_or         : OR-Tools re-solve every step on the CURRENT effective
                       matrix (full observability, 30 ms budget)
- repair_nn2opt      : event-triggered repair — replans patches only when the
                       blocking pattern changes, using the effective matrix
                       at repair time; follows a stale plan between events
- reactive_nn        : myopic nearest feasible neighbor on current effective
- policy_*_lookahead : ONLINE test-time search (look-K) — per-step best-of-K
                       completions sampled under the FROZEN current effective
                       matrix (certainty-equivalent); commits the first action
                       of the best completion. Deployable, unlike samplexN
                       whose episode-level selection sees the realized future
                       (oracle upper bound).

Observability design: policies, rolling-OR and reactive-NN see the current
effective matrix every step; repair sees it only at event-triggered repairs.
That isolates "continuous adaptation" vs "event-triggered repair".

Buckets scale zone_ou_sigma, edge_block_prob, node block/unblock by
0.5/1.0/2.0 (low/medium/high), mirroring the v1 harness bucket convention.

Run from the code directory:
    python scenario_bucket_eval_v2.py
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

V62_DIR = Path(__file__).resolve().parent  # merged layout: frozen deps live alongside
sys.path.insert(0, str(V62_DIR))

from scenario_bucket_eval import load_policy, solve_path_with_ortools  # noqa: E402
from rolling_horizon_or_baseline import two_opt_path  # noqa: E402
from connected_instance_builder import normalize_lonlat  # noqa: E402
from osrm_client import DAMASCUS_BBOX  # noqa: E402

from research_env_v2 import ResearchEnvV2, ResearchEnvV2Config  # noqa: E402
from scenario_bucket_eval_with_repair import (  # noqa: E402
    bootstrap_ci,
    cheapest_insertion,
    nearest_neighbor_order,
)

BIG = 1e9


# ---------------------------------------------------------------------------
# Simulation state + schedule presampling (numpy mirror of ResearchEnvV2)
# ---------------------------------------------------------------------------

@dataclass
class SimStateV2:
    coords: np.ndarray          # [N,2] in [0,1]
    base_dist: np.ndarray       # [N,N]
    eff_dist: np.ndarray        # [N,N] current effective (inf = blocked edge)
    visited: np.ndarray         # [N] bool
    node_blocked: np.ndarray    # [N] float
    current_node: int
    elapsed_time: float
    horizon_sec: float
    n_nodes: int


def smooth3x3(eps: np.ndarray) -> np.ndarray:
    K = eps.shape[0]
    p = np.pad(eps, 1, mode="edge")
    out = np.zeros_like(eps)
    for di in range(3):
        for dj in range(3):
            out += p[di:di + K, dj:dj + K]
    return out / 9.0


def presample_schedule_v2(state: SimStateV2, cfg: ResearchEnvV2Config,
                          max_steps: int, seed: int):
    """Per-step (eff_dist, node_blocked) sequence; also returns the initial field."""
    rng = np.random.RandomState(seed)
    K = cfg.zone_grid
    N = state.n_nodes

    zx = np.clip((state.coords[:, 0] * K).astype(int), 0, K - 1)
    zy = np.clip((state.coords[:, 1] * K).astype(int), 0, K - 1)
    node_zone = (zx, zy)
    mid = 0.5 * (state.coords[:, None, :] + state.coords[None, :, :])  # [N,N,2]

    theta, sigma = cfg.zone_ou_theta, cfg.zone_ou_sigma
    stationary = 1.0 / max(1e-6, (1.0 - (1.0 - theta) ** 2)) ** 0.5
    zf = cfg.traffic_init + smooth3x3(sigma * rng.randn(K, K)) * stationary
    zf = np.clip(zf, cfg.traffic_min, cfg.traffic_max)

    edge_blocked = np.zeros((N, N), dtype=bool)
    node_blocked = state.node_blocked.copy()

    # Optional deterministic rush-hour cycle (makes dynamics non-Markovian in
    # the observed matrix): a zone-phased sinusoid overlaid on the OU field.
    amp = getattr(cfg, "traffic_cycle_amp", 0.0)
    period = max(1, getattr(cfg, "traffic_cycle_period", 40))
    gx, gy = np.meshgrid(np.arange(K), np.arange(K), indexing="ij")
    zphase = 2 * np.pi * (gx + gy) / (2.0 * K)

    def effective(zf, edge_blocked, t):
        z = zf
        if amp > 0:
            z = np.clip(zf + amp * np.sin(2 * np.pi * t / period + zphase),
                        cfg.traffic_min, cfg.traffic_max)
        nzf = z[node_zone]                                   # [N]
        mult = 0.5 * (nzf[:, None] + nzf[None, :])
        eff = state.base_dist * mult
        return np.where(edge_blocked, np.inf, eff).astype(np.float64)

    initial_eff = effective(zf, edge_blocked, 0)
    sched = []
    for t in range(max_steps):
        eps = smooth3x3(sigma * rng.randn(K, K))
        zf = np.clip(zf + theta * (cfg.traffic_init - zf) + eps,
                     cfg.traffic_min, cfg.traffic_max)
        if rng.rand() < cfg.block_prob_per_step:
            node_blocked[rng.randint(1, N)] = 1.0
        if rng.rand() < cfg.unblock_prob_per_step:
            node_blocked[rng.randint(1, N)] = 0.0
        if rng.rand() < cfg.incident_clear_prob:
            edge_blocked[:] = False
        if rng.rand() < cfg.edge_block_prob_per_step:
            c = rng.rand(2)
            d2 = ((mid - c) ** 2).sum(axis=-1)
            edge_blocked |= d2 < cfg.incident_radius ** 2
        sched.append((effective(zf, edge_blocked, t + 1), node_blocked.copy()))
    return initial_eff, sched


def valid_mask_v2(state: SimStateV2) -> np.ndarray:
    traversable = np.isfinite(state.eff_dist[state.current_node])
    mask = (~state.visited) & (state.node_blocked < 0.5) & traversable
    mask[0] = False
    return mask


# Observability ablation (paper §6.4a): what matrix the LEARNED policy observes.
# The environment, the action mask, and every classical baseline always use the
# true effective matrix — this changes only the policy's observation channel
# (node features, incl. the blocked flag, are unchanged in every mode).
POLICY_MATRIX_MODE = "live"


def policy_view_matrix(state: SimStateV2) -> np.ndarray:
    mode = POLICY_MATRIX_MODE
    if mode == "live":            # full effective matrix (default; the paper's policy)
        return state.eff_dist
    if mode == "base":            # neither traffic nor blocking visible
        return state.base_dist
    if mode == "mask_only":       # blocking visible, traffic hidden
        m = state.base_dist.copy()
        m[~np.isfinite(state.eff_dist)] = np.inf
        return m
    if mode == "traffic_only":    # traffic visible, blocking hidden
        blocked = ~np.isfinite(state.eff_dist)
        m = state.eff_dist.copy()
        m[blocked] = state.base_dist[blocked]
        return m
    raise ValueError(f"unknown POLICY_MATRIX_MODE {mode!r}")


def apply_action_and_advance_v2(state: SimStateV2, action: int, event) -> None:
    mask = valid_mask_v2(state)
    if 0 <= action < state.n_nodes and mask[action]:
        travel = float(state.eff_dist[state.current_node, action])
        state.elapsed_time += travel if np.isfinite(travel) else 0.25 * state.horizon_sec
        state.current_node = int(action)
        state.visited[action] = True
    state.eff_dist = event[0]
    state.node_blocked = event[1].copy()


# ---------------------------------------------------------------------------
# Method controllers
# ---------------------------------------------------------------------------

def choose_policy_action_v2(state: SimStateV2, policy, device, sampling: bool) -> int:
    n = state.n_nodes
    x = torch.tensor(state.coords[:, 0], device=device, dtype=torch.float32).unsqueeze(0)
    y = torch.tensor(state.coords[:, 1], device=device, dtype=torch.float32).unsqueeze(0)
    visited = torch.tensor(state.visited.astype(np.float32), device=device).unsqueeze(0)
    blocked = torch.tensor(state.node_blocked.astype(np.float32), device=device).unsqueeze(0)
    is_current = torch.zeros((1, n), device=device)
    is_current[0, state.current_node] = 1.0
    t_rem = max(0.0, state.horizon_sec - state.elapsed_time)
    t_frac = torch.full((1, n), float(t_rem / state.horizon_sec), device=device)
    obs = torch.stack([x, y, visited, is_current, blocked, t_frac], dim=-1).reshape(1, n * 6)

    eff = torch.tensor(policy_view_matrix(state), device=device, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = policy(obs, n, dist_matrix=eff).float().squeeze(0)
    mask = torch.tensor(valid_mask_v2(state), device=device)
    if mask.sum().item() == 0:
        return state.current_node
    logits[~mask] = -1e9
    if sampling:
        return int(Categorical(logits=logits).sample().item())
    return int(torch.argmax(logits).item())


def choose_rolling_or_action_v2(state: SimStateV2, time_limit_ms: int) -> int:
    mask = valid_mask_v2(state)
    # Solve over all unvisited+unblocked nodes; unreachable arcs get BIG cost.
    feasible = [j for j in range(1, state.n_nodes)
                if (not state.visited[j]) and state.node_blocked[j] < 0.5]
    if not feasible:
        return state.current_node
    cost = np.where(np.isfinite(state.eff_dist), state.eff_dist, BIG)
    route = solve_path_with_ortools(state.current_node, feasible, cost, time_limit_ms)
    for nxt in route:
        if mask[nxt]:
            return int(nxt)
    feas_arr = np.flatnonzero(mask)
    if feas_arr.size == 0:
        return state.current_node
    return int(feas_arr[np.argmin(cost[state.current_node, feas_arr])])


class ReactiveNNControllerV2:
    def __init__(self, state: SimStateV2):
        pass

    def act(self, state: SimStateV2) -> int:
        mask = valid_mask_v2(state)
        feasible = np.flatnonzero(mask)
        if feasible.size == 0:
            return state.current_node
        d = state.eff_dist[state.current_node, feasible]
        return int(feasible[int(np.argmin(d))])


class RepairControllerV2:
    """Event-triggered repair. Plans on the effective matrix AT EVENT TIME,
    then follows the (stale) plan until the blocking pattern changes."""

    def __init__(self, state: SimStateV2, two_opt_passes: int = 5):
        self.two_opt_passes = two_opt_passes
        self.deferred: list[int] = []
        self.last_node_blocked = (state.node_blocked > 0.5).copy()
        self.last_edge_sig = ~np.isfinite(state.eff_dist)
        cost = self._cost(state)
        nodes = [j for j in range(1, state.n_nodes) if not state.visited[j]]
        route = nearest_neighbor_order(state.current_node, nodes, cost)
        self.route = two_opt_path(route, state.current_node, cost, max_passes=20)
        self._repair(state)

    @staticmethod
    def _cost(state: SimStateV2) -> np.ndarray:
        return np.where(np.isfinite(state.eff_dist), state.eff_dist, BIG)

    def _repair(self, state: SimStateV2) -> None:
        blocked = state.node_blocked > 0.5
        cost = self._cost(state)
        cur = state.current_node

        self.route = [j for j in self.route if not state.visited[j]]
        self.deferred = [j for j in self.deferred if not state.visited[j]]

        newly_blocked = [j for j in self.route if blocked[j]]
        if newly_blocked:
            self.route = [j for j in self.route if not blocked[j]]
            self.deferred.extend(j for j in newly_blocked if j not in self.deferred)
        for j in [j for j in self.deferred if not blocked[j]]:
            self.deferred.remove(j)
            self.route = cheapest_insertion(self.route, j, cur, cost)
        if len(self.route) >= 4:
            self.route = two_opt_path(self.route, cur, cost, max_passes=self.two_opt_passes)

        self.last_node_blocked = blocked.copy()
        self.last_edge_sig = ~np.isfinite(state.eff_dist)

    def act(self, state: SimStateV2) -> int:
        node_blocked = state.node_blocked > 0.5
        edge_sig = ~np.isfinite(state.eff_dist)
        if (not np.array_equal(node_blocked, self.last_node_blocked)
                or not np.array_equal(edge_sig, self.last_edge_sig)):
            self._repair(state)
        else:
            self.route = [j for j in self.route if not state.visited[j]]

        mask = valid_mask_v2(state)
        while self.route and not mask[self.route[0]]:
            j = self.route.pop(0)
            if node_blocked[j] and not state.visited[j] and j not in self.deferred:
                self.deferred.append(j)
        if self.route:
            return self.route.pop(0)

        feasible = np.flatnonzero(mask)
        if feasible.size == 0:
            return state.current_node
        d = state.eff_dist[state.current_node, feasible]
        return int(feasible[int(np.argmin(d))])


class LookaheadControllerV2:
    """ONLINE test-time search (look-K, the "s5" suites) — the deployable counterpart of
    episode-level best-of-K.

    Episode-level best-of-K (``policy_*_samplexN``) runs K complete
    closed-loop episodes against the SAME realized disruption schedule and
    keeps the winner — i.e. the selection step sees the future. That is an
    oracle upper bound, not an implementable policy.

    This controller uses only information available at decision time: at
    each step it simulates K completions of the remaining route from the
    policy under the FROZEN current effective matrix (certainty-equivalent
    lookahead: traffic stays at current values, blocked stays blocked),
    scores each completion by (stops delivered within the remaining
    horizon, then time), and commits the first action of the best
    completion. Re-planned every step, like the sampling policy.

    Candidate portfolio (all online-legitimate):
      row 0        greedy completion (argmax) — guards against sampling
                   noise degrading the controller below the greedy policy;
      row 1        warm start — re-simulates the previous step's best plan
                   under the current matrix (prevents route thrash from
                   per-step resampling);
      rows 2..K-1  stochastic completions sampled at ``temperature``.

    Optional upgrades (still online: they use the disruption *process*, never
    the realized future):
      use_2opt     patch each completion with classical 2-opt under the
                   frozen matrix before scoring (learned sampler + classical
                   polish — the symmetric counterpart of repair's NN+2-opt);
      n_scenarios  score each completion under S sampled futures of the
                   disruption process (approximate per-node OU factors +
                   block/unblock + incident events at the bucket's rates)
                   and rank by mean delivered, then mean time — a learned-
                   policy instantiation of the Multiple Scenario Approach
                   [Bent & Van Hentenryck 2004] with consensus-by-best."""

    def __init__(self, policy, device, k: int, seed: int, temperature: float = 1.0,
                 use_2opt: bool = False, n_scenarios: int = 0,
                 scen_cfg: dict | None = None):
        self.policy = policy
        self.device = device
        self.k = max(2, k)
        self.temperature = max(1e-3, temperature)
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.prev_plan: list[int] = []
        self.use_2opt = use_2opt
        self.n_scenarios = n_scenarios
        self.scen_cfg = scen_cfg or {}
        self.np_rng = np.random.RandomState((seed * 2654435761) % (2**31))
        self._t = 0  # decision-step counter (drives the optional cycle model)

    def act(self, state: SimStateV2) -> int:
        mask0 = valid_mask_v2(state)
        feas0 = np.flatnonzero(mask0)
        if feas0.size == 0:
            return state.current_node
        if feas0.size == 1:
            return int(feas0[0])

        n, K, dev = state.n_nodes, self.k, self.device
        horizon = float(state.horizon_sec)

        eff = torch.tensor(policy_view_matrix(state), device=dev, dtype=torch.float32)  # inf = blocked edge (policy's view)
        eff_b = eff.unsqueeze(0).expand(K, n, n)
        finite = torch.isfinite(eff)                                          # [N,N]
        coords = torch.tensor(state.coords, device=dev, dtype=torch.float32)  # [N,2]
        x = coords[:, 0].unsqueeze(0).expand(K, n)
        y = coords[:, 1].unsqueeze(0).expand(K, n)
        node_free = torch.tensor(state.node_blocked < 0.5, device=dev)        # [N]
        blocked_feat = torch.tensor(state.node_blocked.astype(np.float32),
                                    device=dev).unsqueeze(0).expand(K, n)

        visited = torch.tensor(state.visited, device=dev).unsqueeze(0).repeat(K, 1)
        cur = torch.full((K,), state.current_node, dtype=torch.long, device=dev)
        elapsed = torch.full((K,), float(state.elapsed_time), device=dev)
        rows = torch.arange(K, device=dev)
        max_inner = n + 4
        act_hist = torch.full((K, max_inner), -1, dtype=torch.long, device=dev)

        # Warm-start queue: previous best plan, minus anything already served.
        plan_queue = [j for j in self.prev_plan if not state.visited[j]]

        for inner in range(max_inner):
            m = (~visited) & node_free.unsqueeze(0) & finite[cur]
            m[:, 0] = False
            can_act = m.any(dim=1) & (elapsed < horizon)
            if not can_act.any():
                break
            t_frac = ((horizon - elapsed).clamp(min=0) / horizon)
            is_cur = torch.zeros((K, n), device=dev)
            is_cur[rows, cur] = 1.0
            obs = torch.stack([x, y, visited.float(), is_cur, blocked_feat,
                               t_frac.unsqueeze(1).expand(K, n)], dim=-1).reshape(K, n * 6)
            with torch.no_grad():
                logits = self.policy(obs, n, dist_matrix=eff_b).float()
            logits = logits.masked_fill(~m, -1e9)
            probs = torch.softmax(logits / self.temperature, dim=-1)
            acts = torch.multinomial(probs, 1, generator=self.gen).squeeze(1)
            # Row 0: greedy completion.
            acts[0] = torch.argmax(logits[0])
            # Row 1: follow the previous best plan while it stays feasible.
            while plan_queue and not bool(m[1, plan_queue[0]]):
                plan_queue.pop(0)
            if plan_queue:
                acts[1] = plan_queue.pop(0)

            travel = eff[cur, acts]
            travel = torch.where(torch.isfinite(travel), travel,
                                 torch.full_like(travel, 0.25 * horizon))
            elapsed = torch.where(can_act, elapsed + travel, elapsed)
            upd = rows[can_act]
            visited[upd, acts[upd]] = True
            cur = torch.where(can_act, acts, cur)
            act_hist[upd, inner] = acts[upd]

        self._t += 1
        seqs = [[int(v) for v in act_hist[i].tolist() if v >= 0] for i in range(K)]

        if self.use_2opt or self.n_scenarios > 0:
            best_seq = self._rescore(state, seqs)
        else:
            delivered = visited[:, 1:].sum(dim=1).float()
            score = delivered * 1e9 - elapsed      # lexicographic (delivered, -time)
            score[act_hist[:, 0] < 0] = -float("inf")
            best_seq = seqs[int(torch.argmax(score).item())]

        a = best_seq[0] if best_seq else -1
        if a >= 0 and mask0[a]:
            self.prev_plan = best_seq[1:]
            return a
        self.prev_plan = []
        d = policy_view_matrix(state)[state.current_node, feas0]
        return int(feas0[int(np.argmin(np.where(np.isfinite(d), d, BIG)))])

    # -- numpy rescoring path (2-opt patch and/or scenario scoring) --

    def _rescore(self, state: SimStateV2, seqs: list[list[int]]) -> list[int]:
        view = policy_view_matrix(state)
        cost0 = np.where(np.isfinite(view), view, BIG)
        if self.use_2opt:
            seqs = [two_opt_path(s, state.current_node, cost0, max_passes=5)
                    if len(s) >= 4 else s for s in seqs]

        if self.n_scenarios > 0:
            mats = [self._sample_scenario(state) for _ in range(self.n_scenarios)]
        else:
            mats = [[ ]]  # sentinel: frozen scoring

        best_seq, best_key = None, None
        for s in seqs:
            if not s:
                continue
            dels, tims = [], []
            for mat_seq in mats:
                d, t = self._walk(state, s, mat_seq if mat_seq else None, cost0)
                dels.append(d)
                tims.append(t)
            key = (float(np.mean(dels)), -float(np.mean(tims)))
            if best_key is None or key > best_key:
                best_seq, best_key = s, key
        return best_seq or []

    def _walk(self, state: SimStateV2, seq: list[int], mat_seq, cost0) -> tuple[int, float]:
        """Deliveries/time of a fixed node sequence, under the frozen matrix
        (mat_seq=None) or one sampled scenario (list of per-step matrices)."""
        elapsed = float(state.elapsed_time)
        cur, delivered = state.current_node, 0
        for step, j in enumerate(seq):
            if elapsed >= state.horizon_sec:
                break
            c = cost0 if mat_seq is None else mat_seq[min(step, len(mat_seq) - 1)]
            leg = float(c[cur, j])
            if leg >= BIG:      # blocked in this scenario: skip the stop
                continue
            elapsed += leg
            delivered += 1
            cur = j
        return delivered, elapsed

    def _sample_scenario(self, state: SimStateV2) -> list[np.ndarray]:
        """Approximate future of the disruption process from the current state:
        per-node traffic factors (row means of the current edge multipliers)
        evolved by OU at the bucket's rates, plus node-block/unblock and
        incident-disk edge blocking, plus the (known) deterministic cycle if
        one is configured. Uses the process model only — never the realized
        schedule."""
        cfg = self.scen_cfg
        rng = self.np_rng
        N = state.n_nodes
        base = state.base_dist
        finite0 = np.isfinite(state.eff_dist)
        with np.errstate(invalid="ignore"):
            mult = np.where(finite0 & (base > 0), state.eff_dist / np.maximum(base, 1e-9), 1.0)
        f = mult.mean(axis=1)                                   # per-node factor estimate
        blocked = state.node_blocked > 0.5
        edge_blocked = ~finite0
        mid = 0.5 * (state.coords[:, None, :] + state.coords[None, :, :])
        theta = cfg.get("theta", 0.05)
        sigma = cfg.get("sigma", 0.15) / 3.0                    # 3x3 smoothing shrinks std ~3x
        amp = cfg.get("cycle_amp", 0.0)
        period = max(1, cfg.get("cycle_period", 40))
        phase = cfg.get("node_phase")
        if phase is None and amp > 0:
            K = cfg.get("zone_grid", 4)
            zx = np.clip((state.coords[:, 0] * K).astype(int), 0, K - 1)
            zy = np.clip((state.coords[:, 1] * K).astype(int), 0, K - 1)
            phase = 2 * np.pi * (zx + zy) / (2.0 * K)
            cfg["node_phase"] = phase

        horizon_steps = min(N + 4, 64)
        mats = []
        for h in range(horizon_steps):
            f = np.clip(f + theta * (1.0 - f) + sigma * rng.randn(N),
                        cfg.get("tmin", 0.7), cfg.get("tmax", 1.8))
            if rng.rand() < cfg.get("block_p", 0.03):
                blocked[rng.randint(1, N)] = True
            if rng.rand() < cfg.get("unblock_p", 0.02):
                blocked[rng.randint(1, N)] = False
            if rng.rand() < cfg.get("clear_p", 0.10):
                edge_blocked[:] = False
            if rng.rand() < cfg.get("edge_p", 0.02):
                c = rng.rand(2)
                edge_blocked |= ((mid - c) ** 2).sum(axis=-1) < cfg.get("radius", 0.15) ** 2
            fe = f
            if amp > 0:
                fe = f + amp * np.sin(2 * np.pi * (self._t + h) / period + phase)
            m = 0.5 * (fe[:, None] + fe[None, :])
            eff = base * m
            eff = np.where(edge_blocked | blocked[None, :], BIG, eff)
            mats.append(eff)
        return mats


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------

def run_rollout_v2(init_states, init_effs, schedules, max_steps: int, act_fn):
    states = []
    for s, eff in zip(init_states, init_effs):
        st = copy.deepcopy(s)
        st.eff_dist = eff.copy()
        states.append(st)
    ctx = [act_fn("init", st) for st in states]  # controller instances or None
    for step_idx in range(max_steps):
        all_done = True
        for i, st in enumerate(states):
            if st.elapsed_time >= st.horizon_sec or st.visited[1:].all():
                continue
            all_done = False
            action = act_fn("act", st, ctx[i])
            apply_action_and_advance_v2(st, action, schedules[i][step_idx])
        if all_done:
            break
    return {
        "time_mean": float(np.mean([s.elapsed_time for s in states])),
        "delivered_mean": float(np.mean([int(s.visited[1:].sum()) for s in states])),
    }


def make_act_fn_policy(policy, device, sampling: bool):
    def fn(mode, state, ctrl=None):
        if mode == "init":
            return None
        return choose_policy_action_v2(state, policy, device, sampling)
    return fn


def make_act_fn_rolling(time_limit_ms: int):
    def fn(mode, state, ctrl=None):
        if mode == "init":
            return None
        return choose_rolling_or_action_v2(state, time_limit_ms)
    return fn


def make_act_fn_controller(ctor):
    def fn(mode, state, ctrl=None):
        if mode == "init":
            return ctor(state)
        return ctrl.act(state)
    return fn


def make_act_fn_lookahead(policy, device, k: int, seed_base: int, temperature: float = 1.0,
                          use_2opt: bool = False, n_scenarios: int = 0,
                          scen_cfg: dict | None = None):
    counter = {"i": 0}

    def fn(mode, state, ctrl=None):
        if mode == "init":
            counter["i"] += 1
            return LookaheadControllerV2(policy, device, k, seed_base + counter["i"],
                                         temperature=temperature, use_2opt=use_2opt,
                                         n_scenarios=n_scenarios,
                                         scen_cfg=dict(scen_cfg) if scen_cfg else None)
        return ctrl.act(state)
    return fn


def best_of_k(init_states, init_effs, schedules, max_steps, policy, device, k, seed0):
    best = None
    key = None
    for s in range(k):
        np.random.seed(seed0 + s)
        torch.manual_seed(seed0 + s)
        r = run_rollout_v2(init_states, init_effs, schedules, max_steps,
                           make_act_fn_policy(policy, device, sampling=True))
        rk = (r["delivered_mean"], -r["time_mean"])
        if key is None or rk > key:
            best, key = r, rk
    return best


def paired_summary(a, b, seed=0):
    d_del = [x["delivered_mean"] - y["delivered_mean"] for x, y in zip(a, b)]
    d_time = [x["time_mean"] - y["time_mean"] for x, y in zip(a, b)]
    dm, dlo, dhi = bootstrap_ci(d_del, seed=seed)
    tm, tlo, thi = bootstrap_ci(d_time, seed=seed + 1)
    wins = sum(1 for d in d_del if d > 1e-9)
    losses = sum(1 for d in d_del if d < -1e-9)
    return {
        "delivered_delta_mean": dm, "delivered_delta_ci95": [dlo, dhi],
        "time_delta_mean": tm, "time_delta_ci95": [tlo, thi],
        "delivered_win_tie_loss": [wins, len(d_del) - wins - losses, losses],
    }


def mean_key(xs, key):
    vals = [x[key] for x in xs if key in x]
    return float(np.mean(vals)) if vals else float("nan")


def apply_bucket_v2(cfg: ResearchEnvV2Config, bucket: str) -> ResearchEnvV2Config:
    mult = {"low": 0.5, "medium": 1.0, "high": 2.0}[bucket.lower()]
    cfg.zone_ou_sigma *= mult
    cfg.edge_block_prob_per_step = min(1.0, cfg.edge_block_prob_per_step * mult)
    cfg.block_prob_per_step = min(1.0, cfg.block_prob_per_step * mult)
    cfg.unblock_prob_per_step = min(1.0, cfg.unblock_prob_per_step * mult)
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-v2-checkpoint",
                        default="checkpoints_research_v2_pomo/research_v2_best.pt")
    parser.add_argument("--policy-v1-checkpoint",
                        default=str(V62_DIR / "checkpoints_research_pomo" / "research_best.pt"))
    parser.add_argument("--instance-pool",
                        default="results/osrm_instance_pool/pool.npz",
                        help="cached OSRM pool.npz (default; pass '' for synthetic instances)")
    parser.add_argument("--n-episodes", type=int, default=40)
    parser.add_argument("--n-nodes", type=int, default=20)
    parser.add_argument("--num-instances", type=int, default=4)
    parser.add_argument("--policy-n-samples", type=int, default=8)
    parser.add_argument("--lookahead-samples", type=int, default=8,
                        help="K for the ONLINE lookahead policy (look-K); 0 disables")
    parser.add_argument("--lookahead-temp", type=float, default=1.0,
                        help="sampling temperature for lookahead completions (rows 2..K-1)")
    parser.add_argument("--lookahead-2opt", action="store_true",
                        help="2-opt-patch each lookahead completion before scoring (online)")
    parser.add_argument("--lookahead-scenarios", type=int, default=0,
                        help="score lookahead completions under S sampled process futures "
                             "(MSA-style, online); 0 = frozen-matrix scoring")
    parser.add_argument("--traffic-cycle-amp", type=float, default=0.0,
                        help="deterministic rush-hour cycle amplitude added to zone factors "
                             "(makes dynamics non-Markovian in the observed matrix)")
    parser.add_argument("--traffic-cycle-period", type=int, default=40,
                        help="cycle period in decision steps")
    parser.add_argument("--ortools-time-limit-ms", type=int, default=30)
    parser.add_argument("--policy-matrix-mode", default="live",
                        choices=["live", "base", "mask_only", "traffic_only"],
                        help="observability ablation (§6.4a): matrix the LEARNED policy "
                             "observes; env, action mask, and classical baselines always "
                             "use the true effective matrix")
    parser.add_argument("--horizon-hours", type=float, default=8.0,
                        help="mission horizon H in hours (horizon-stress / ceiling ablation)")
    parser.add_argument("--buckets", nargs="+", default=["low", "medium", "high"])
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--save-json", default="results/scenario_bucket_v2.json")
    args = parser.parse_args()

    global POLICY_MATRIX_MODE
    POLICY_MATRIX_MODE = args.policy_matrix_mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_v2, ckpt_v2 = load_policy(args.policy_v2_checkpoint, device)
    policy_v1, ckpt_v1 = load_policy(args.policy_v1_checkpoint, device)

    pool_lonlats = pool_durations = None
    pool_bbox = DAMASCUS_BBOX
    if args.instance_pool:
        pool = np.load(args.instance_pool)
        pool_lonlats, pool_durations = pool["lonlats"], pool["durations"]
        # Normalize with the bbox the pool was built in (cross-city support).
        meta_path = Path(args.instance_pool).parent / "pool_meta.json"
        if meta_path.exists():
            bb = json.loads(meta_path.read_text(encoding="utf-8")).get("bbox")
            if bb:
                from osrm_client import BBox
                pool_bbox = BBox(min_lon=bb[0], min_lat=bb[1], max_lon=bb[2], max_lat=bb[3])
        print(f"[STAGE2] Instance pool: {args.instance_pool} ({len(pool_lonlats)} instances, "
              f"bbox={[pool_bbox.min_lon, pool_bbox.min_lat, pool_bbox.max_lon, pool_bbox.max_lat]})")

    print(f"[STAGE2] Device: {device}")
    print(f"[STAGE2] v2 policy: {args.policy_v2_checkpoint} (epoch {ckpt_v2.get('epoch')})")
    print(f"[STAGE2] v1 policy: {args.policy_v1_checkpoint} (epoch {ckpt_v1.get('epoch')})")

    out = {"config": vars(args), "buckets": {}}
    max_steps = args.n_nodes * 8 + 64
    methods = ["policy_v2_greedy", "policy_v2_samplexN", "policy_v1_samplexN",
               "rolling_or", "repair_nn2opt", "reactive_nn"]
    if args.lookahead_samples > 0:
        methods += ["policy_v2_lookahead", "policy_v1_lookahead"]

    for bidx, bucket in enumerate(args.buckets):
        cfg = apply_bucket_v2(ResearchEnvV2Config(
            n_nodes=args.n_nodes, num_instances=args.num_instances,
            device=device.type, auto_reset=False, use_augmentation=True,
        ), bucket)
        cfg.traffic_cycle_amp = args.traffic_cycle_amp
        cfg.traffic_cycle_period = args.traffic_cycle_period
        cfg.time_horizon_sec = args.horizon_hours * 3600.0
        scen_cfg = dict(theta=cfg.zone_ou_theta, sigma=cfg.zone_ou_sigma,
                        block_p=cfg.block_prob_per_step, unblock_p=cfg.unblock_prob_per_step,
                        edge_p=cfg.edge_block_prob_per_step, clear_p=cfg.incident_clear_prob,
                        radius=cfg.incident_radius, tmin=cfg.traffic_min, tmax=cfg.traffic_max,
                        zone_grid=cfg.zone_grid, cycle_amp=args.traffic_cycle_amp,
                        cycle_period=args.traffic_cycle_period)

        per_method = {m: [] for m in methods}
        episodes = []

        for ep in range(args.n_episodes):
            seed = args.base_seed + ep + 10000 * bidx
            np.random.seed(seed)
            torch.manual_seed(seed)

            # --- instances ---
            init_states = []
            if pool_lonlats is not None:
                idx = np.random.RandomState(seed).choice(
                    len(pool_lonlats), args.num_instances, replace=False)
                for k in idx:
                    n = pool_lonlats.shape[1]
                    visited = np.zeros(n, dtype=bool); visited[0] = True
                    init_states.append(SimStateV2(
                        coords=normalize_lonlat(pool_lonlats[k], pool_bbox).astype(np.float64),
                        base_dist=pool_durations[k].astype(np.float64).copy(),
                        eff_dist=pool_durations[k].astype(np.float64).copy(),
                        visited=visited, node_blocked=np.zeros(n, dtype=np.float32),
                        current_node=0, elapsed_time=0.0,
                        horizon_sec=float(cfg.time_horizon_sec), n_nodes=n,
                    ))
            else:
                env = ResearchEnvV2(cfg)
                env.reset()
                p = env.pomo_size
                for b in range(cfg.num_instances):
                    row = b * p
                    n = env.n_nodes
                    visited = np.zeros(n, dtype=bool); visited[0] = True
                    init_states.append(SimStateV2(
                        coords=env.coords[row].cpu().numpy().astype(np.float64),
                        base_dist=env.base_dist[row].cpu().numpy().astype(np.float64),
                        eff_dist=env.base_dist[row].cpu().numpy().astype(np.float64),
                        visited=visited, node_blocked=np.zeros(n, dtype=np.float32),
                        current_node=0, elapsed_time=0.0,
                        horizon_sec=float(cfg.time_horizon_sec), n_nodes=n,
                    ))

            init_effs, schedules = [], []
            for i, st in enumerate(init_states):
                eff0, sched = presample_schedule_v2(st, cfg, max_steps, seed + 999 + i)
                init_effs.append(eff0)
                schedules.append(sched)

            def run(act_fn):
                t0 = time.perf_counter()
                r = run_rollout_v2(init_states, init_effs, schedules, max_steps, act_fn)
                r["wall_sec"] = time.perf_counter() - t0
                return r

            np.random.seed(seed); torch.manual_seed(seed)
            results = {
                "policy_v2_greedy": run(make_act_fn_policy(policy_v2, device, sampling=False)),
                "rolling_or": run(make_act_fn_rolling(args.ortools_time_limit_ms)),
                "repair_nn2opt": run(make_act_fn_controller(RepairControllerV2)),
                "reactive_nn": run(make_act_fn_controller(ReactiveNNControllerV2)),
            }
            t0 = time.perf_counter()
            results["policy_v2_samplexN"] = best_of_k(
                init_states, init_effs, schedules, max_steps, policy_v2, device,
                args.policy_n_samples, seed * 1000 + 100)
            results["policy_v2_samplexN"]["wall_sec"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            results["policy_v1_samplexN"] = best_of_k(
                init_states, init_effs, schedules, max_steps, policy_v1, device,
                args.policy_n_samples, seed * 1000 + 100)
            results["policy_v1_samplexN"]["wall_sec"] = time.perf_counter() - t0
            if args.lookahead_samples > 0:
                la_kw = dict(temperature=args.lookahead_temp, use_2opt=args.lookahead_2opt,
                             n_scenarios=args.lookahead_scenarios, scen_cfg=scen_cfg)
                results["policy_v2_lookahead"] = run(make_act_fn_lookahead(
                    policy_v2, device, args.lookahead_samples, seed * 1000 + 500, **la_kw))
                results["policy_v1_lookahead"] = run(make_act_fn_lookahead(
                    policy_v1, device, args.lookahead_samples, seed * 1000 + 600, **la_kw))

            for m in methods:
                per_method[m].append(results[m])
            episodes.append({"episode": ep + 1, **results})
            look_str = ""
            if args.lookahead_samples > 0:
                look_str = (f"v2look del={results['policy_v2_lookahead']['delivered_mean']:.2f} | "
                            f"v1look del={results['policy_v1_lookahead']['delivered_mean']:.2f} | ")
            print(
                f"[STAGE2] {bucket} ep {ep+1}/{args.n_episodes} | "
                f"v2x8 del={results['policy_v2_samplexN']['delivered_mean']:.2f} | "
                f"v1x8 del={results['policy_v1_samplexN']['delivered_mean']:.2f} | "
                f"{look_str}"
                f"repair del={results['repair_nn2opt']['delivered_mean']:.2f} | "
                f"rollOR del={results['rolling_or']['delivered_mean']:.2f} | "
                f"reactNN del={results['reactive_nn']['delivered_mean']:.2f}"
            )

        summary = {m: {
            "time_mean": mean_key(per_method[m], "time_mean"),
            "delivered_mean": mean_key(per_method[m], "delivered_mean"),
            "episode_wall_sec_mean": mean_key(per_method[m], "wall_sec"),
        } for m in methods}
        pairs = {
            "v2x8_minus_repair": ("policy_v2_samplexN", "repair_nn2opt"),
            "v2x8_minus_rolling_or": ("policy_v2_samplexN", "rolling_or"),
            "v2x8_minus_v1x8": ("policy_v2_samplexN", "policy_v1_samplexN"),
            "v1x8_minus_repair": ("policy_v1_samplexN", "repair_nn2opt"),
            "repair_minus_reactive_nn": ("repair_nn2opt", "reactive_nn"),
            "repair_minus_rolling_or": ("repair_nn2opt", "rolling_or"),
        }
        if args.lookahead_samples > 0:
            pairs.update({
                "v2look_minus_repair": ("policy_v2_lookahead", "repair_nn2opt"),
                "v2look_minus_rolling_or": ("policy_v2_lookahead", "rolling_or"),
                "v2look_minus_v2x8": ("policy_v2_lookahead", "policy_v2_samplexN"),
                "v2look_minus_v1look": ("policy_v2_lookahead", "policy_v1_lookahead"),
                "v1look_minus_repair": ("policy_v1_lookahead", "repair_nn2opt"),
            })
        for name, (ma, mb) in pairs.items():
            summary[f"paired_{name}"] = paired_summary(per_method[ma], per_method[mb], seed=bidx * 100)

        out["buckets"][bucket] = {"summary": summary, "episodes": episodes}

        save_path = Path(__file__).resolve().parent / args.save_json
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

        p = summary["paired_v2x8_minus_repair"]
        q = summary["paired_v2x8_minus_v1x8"]
        print(f"[STAGE2] === {bucket} === v2x8-repair: {p['delivered_delta_mean']:+.3f} "
              f"CI{p['delivered_delta_ci95']} | v2x8-v1x8: {q['delivered_delta_mean']:+.3f} "
              f"CI{q['delivered_delta_ci95']}")
        if args.lookahead_samples > 0:
            lr = summary["paired_v2look_minus_repair"]
            lo = summary["paired_v2look_minus_v2x8"]
            print(f"[STAGE2] === {bucket} === v2look-repair: {lr['delivered_delta_mean']:+.3f} "
                  f"CI{lr['delivered_delta_ci95']} | v2look-v2x8(oracle): "
                  f"{lo['delivered_delta_mean']:+.3f} CI{lo['delivered_delta_ci95']}")

    print(f"[STAGE2] Saved JSON: {Path(__file__).resolve().parent / args.save_json}")


if __name__ == "__main__":
    main()
