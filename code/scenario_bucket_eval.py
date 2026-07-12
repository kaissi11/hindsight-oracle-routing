#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    HAS_ORTOOLS = True
except Exception:
    HAS_ORTOOLS = False

from tsp_model_v2 import TSPActorV2
from research_env import ResearchEnv, ResearchEnvConfig

LARGE_COST = 10**9
PREDICTOR_FEATURE_COLUMNS = [
    "cand_x","cand_y","cur_x","cur_y","base_dist_cur_to_cand","base_dist_depot_to_cand",
    "dist_ratio_to_mean","time_remaining_frac","elapsed_time_frac","traffic_scalar",
    "visited_flag","blocked_flag","current_flag",
]

class RiskMLP(nn.Module):
    def __init__(self, in_dim:int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU(),
            nn.Linear(128,64), nn.ReLU(),
        )
        self.feasible_head = nn.Linear(64,1)
        self.cost_head = nn.Linear(64,1)
    def forward(self,x):
        h = self.backbone(x)
        return self.feasible_head(h).squeeze(-1), self.cost_head(h).squeeze(-1)

@dataclass
class SimState:
    coords: np.ndarray
    base_dist: np.ndarray
    visited: np.ndarray
    blocked: np.ndarray
    current_node: int
    traffic: float
    elapsed_time: float
    horizon_sec: float
    n_nodes: int

def load_policy(path:str, device:torch.device):
    ckpt = torch.load(path, map_location=device)
    model = TSPActorV2(node_dim=6, embed_dim=128, num_heads=8, num_layers=6, ff_dim=512, dropout=0.0).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, ckpt

def load_predictor(path:str, device:torch.device):
    ckpt = torch.load(path, map_location=device)
    model = RiskMLP(len(PREDICTOR_FEATURE_COLUMNS)).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    mean = torch.tensor(np.asarray(ckpt["mean"], dtype=np.float32), device=device)
    std = torch.tensor(np.asarray(ckpt["std"], dtype=np.float32), device=device)
    return model, mean, std, ckpt

def valid_mask(state:SimState):
    mask = (~state.visited) & (state.blocked < 0.5)
    mask[0] = False
    return mask

def build_obs(state:SimState, device):
    n = state.n_nodes
    x = torch.tensor(state.coords[:,0], device=device, dtype=torch.float32).unsqueeze(0)
    y = torch.tensor(state.coords[:,1], device=device, dtype=torch.float32).unsqueeze(0)
    visited = torch.tensor(state.visited.astype(np.float32), device=device).unsqueeze(0)
    blocked = torch.tensor(state.blocked.astype(np.float32), device=device).unsqueeze(0)
    is_current = torch.zeros((1,n), device=device); is_current[0, state.current_node] = 1.0
    t_rem = max(0.0, state.horizon_sec - state.elapsed_time)
    t_frac = torch.full((1,n), float(t_rem/state.horizon_sec), device=device)
    feats = torch.stack([x,y,visited,is_current,blocked,t_frac], dim=-1)
    return feats.reshape(1, n*6)

def predictor_features(state:SimState):
    cur = state.current_node
    coords = state.coords; base = state.base_dist; blocked = state.blocked; visited = state.visited
    cur_x, cur_y = coords[cur]
    mean_base = float(base.mean()) + 1e-6
    tr = max(0.0, min(1.0, (state.horizon_sec - state.elapsed_time)/state.horizon_sec))
    ef = max(0.0, min(1.0, state.elapsed_time/state.horizon_sec))
    rows = []
    for cand in range(state.n_nodes):
        cx, cy = coords[cand]
        dcur = float(base[cur,cand]); ddep = float(base[0,cand]); ratio = dcur/mean_base
        rows.append([cx,cy,cur_x,cur_y,dcur,ddep,ratio,tr,ef,state.traffic,float(bool(visited[cand])),float(bool(blocked[cand]>0.5)),float(cand==cur)])
    return np.asarray(rows, dtype=np.float32)

def predictor_outputs(predictor, mean, std, state:SimState):
    xt = torch.from_numpy(predictor_features(state)).to(mean.device)
    xt = (xt - mean) / std
    with torch.no_grad():
        flogit, cpred = predictor(xt)
        fp = torch.sigmoid(flogit).cpu().numpy()
        pc = torch.expm1(cpred).clamp_min(0.0).cpu().numpy()
    return fp, pc

def choose_policy_action(state:SimState, policy, device, sampling, predictor=None, pred_mean=None, pred_std=None, alpha=0.0, beta=0.0):
    obs = build_obs(state, device)
    dist_matrix = torch.tensor(state.base_dist, device=device, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = policy(obs, state.n_nodes, dist_matrix=dist_matrix).float().squeeze(0)
    mask_np = valid_mask(state)
    mask = torch.tensor(mask_np, device=device)
    if predictor is not None and pred_mean is not None and pred_std is not None and (alpha!=0.0 or beta!=0.0):
        fp, pc = predictor_outputs(predictor, pred_mean, pred_std, state)
        fp = torch.tensor(fp, device=device, dtype=logits.dtype)
        pc = torch.tensor(pc, device=device, dtype=logits.dtype)
        pc = (pc - pc.mean()) / pc.std().clamp_min(1e-6)
        logits = logits + alpha*torch.log(fp.clamp_min(1e-6)) - beta*pc
    if mask.sum().item() == 0:
        return state.current_node
    logits[~mask] = -1e9
    return int(Categorical(logits=logits).sample().item()) if sampling else int(torch.argmax(logits).item())

def solve_path_with_ortools(current:int, feasible_nodes, travel_cost, time_limit_ms:int):
    if not feasible_nodes:
        return []
    real_nodes = [current] + list(feasible_nodes)
    dummy_end = len(real_nodes)
    num_nodes = len(real_nodes)+1
    mat = np.full((num_nodes,num_nodes), LARGE_COST, dtype=np.int64)
    for i, ni in enumerate(real_nodes):
        for j, nj in enumerate(real_nodes):
            if i == j: mat[i,j] = 0
            else:
                c = float(travel_cost[ni,nj])
                mat[i,j] = max(1, int(round(c))) if np.isfinite(c) else LARGE_COST
    for i in range(len(real_nodes)): mat[i,dummy_end] = 0
    mat[dummy_end,dummy_end] = 0
    mgr = pywrapcp.RoutingIndexManager(num_nodes, 1, [0], [dummy_end])
    routing = pywrapcp.RoutingModel(mgr)
    def cb(fi,ti):
        return int(mat[mgr.IndexToNode(fi), mgr.IndexToNode(ti)])
    tidx = routing.RegisterTransitCallback(cb)
    routing.SetArcCostEvaluatorOfAllVehicles(tidx)
    sp = pywrapcp.DefaultRoutingSearchParameters()
    sp.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    sp.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    sp.time_limit.FromMilliseconds(time_limit_ms)
    sol = routing.SolveWithParameters(sp)
    if sol is None:
        return list(feasible_nodes)
    route = []
    idx = routing.Start(0)
    while not routing.IsEnd(idx):
        node = mgr.IndexToNode(idx)
        if node != 0 and node != dummy_end:
            route.append(real_nodes[node])
        idx = sol.Value(routing.NextVar(idx))
    return route

def choose_rolling_or_action(state:SimState, time_limit_ms:int):
    feasible = [j for j in range(1, state.n_nodes) if (not state.visited[j]) and state.blocked[j] < 0.5]
    if not feasible:
        return state.current_node
    route = solve_path_with_ortools(state.current_node, feasible, state.base_dist * state.traffic, time_limit_ms)
    return int(route[0]) if route else state.current_node

def init_states_from_env(env):
    p = getattr(env, "pomo_size", 1)
    xs = []
    for inst in range(env.cfg.num_instances):
        row = inst * p
        xs.append(SimState(
            coords=env.coords[row].detach().cpu().numpy().copy(),
            base_dist=env.base_dist[row].detach().cpu().numpy().copy(),
            visited=env.visited[row].detach().cpu().numpy().astype(bool).copy(),
            blocked=env.blocked[row].detach().cpu().numpy().copy(),
            current_node=int(env.current_node[row].item()),
            traffic=float(env.traffic[row].item()),
            elapsed_time=float(env.elapsed_time[row].item()),
            horizon_sec=float(env.cfg.time_horizon_sec),
            n_nodes=env.n_nodes,
        ))
    return xs

def apply_bucket(cfg, bucket:str):
    bucket = bucket.lower()
    mult = {"low":0.5, "medium":1.0, "high":2.0}[bucket]
    if hasattr(cfg, "traffic_rw_std"): cfg.traffic_rw_std = float(cfg.traffic_rw_std) * mult
    if hasattr(cfg, "block_prob_per_step"): cfg.block_prob_per_step = min(1.0, float(cfg.block_prob_per_step) * mult)
    if hasattr(cfg, "unblock_prob_per_step"): cfg.unblock_prob_per_step = min(1.0, float(cfg.unblock_prob_per_step) * mult)
    return cfg

def presample_events(init_states, cfg, max_steps:int, base_seed:int):
    rng = np.random.RandomState(base_seed)
    out = []
    for state in init_states:
        traffic = float(state.traffic); blocked = state.blocked.copy(); sched = []
        for _ in range(max_steps):
            if hasattr(cfg, "traffic_rw_std"):
                traffic = float(np.clip(traffic + cfg.traffic_rw_std * rng.randn(), getattr(cfg, "traffic_min", 0.5), getattr(cfg, "traffic_max", 2.0)))
            if hasattr(cfg, "block_prob_per_step") and rng.rand() < cfg.block_prob_per_step:
                blocked[int(rng.randint(1, state.n_nodes))] = 1.0
            if hasattr(cfg, "unblock_prob_per_step") and rng.rand() < cfg.unblock_prob_per_step:
                blocked[int(rng.randint(1, state.n_nodes))] = 0.0
            sched.append((traffic, blocked.copy()))
        out.append(sched)
    return out

def apply_action_and_advance(state:SimState, action:int, event):
    mask = valid_mask(state)
    if action >= 0 and action < state.n_nodes and mask[action]:
        travel = float(state.base_dist[state.current_node, action]) * float(state.traffic)
        state.elapsed_time += travel if np.isfinite(travel) else 0.25*state.horizon_sec
        state.current_node = int(action)
        state.visited[action] = True
    state.traffic = float(event[0]); state.blocked = event[1].copy()

def run_single_rollout(init_states, schedule, method_name:str, max_steps:int, policy, device, predictor=None, pred_mean=None, pred_std=None, alpha=0.0, beta=0.0, sampling=True, ortools_time_limit_ms=30):
    states = [copy.deepcopy(s) for s in init_states]
    for step_idx in range(max_steps):
        all_done = True
        for i, state in enumerate(states):
            if state.elapsed_time >= state.horizon_sec or state.visited[1:].all():
                continue
            all_done = False
            if method_name == "policy":
                action = choose_policy_action(state, policy, device, sampling, predictor, pred_mean, pred_std, alpha, beta)
            elif method_name == "rolling_or":
                action = choose_rolling_or_action(state, ortools_time_limit_ms)
            else:
                raise ValueError(method_name)
            apply_action_and_advance(state, action, schedule[i][step_idx])
        if all_done:
            break
    times = [s.elapsed_time for s in states]
    delivered = [int(s.visited[1:].sum()) for s in states]
    return {"time_mean": float(np.mean(times)), "delivered_mean": float(np.mean(delivered))}

def select_best_of_k(results):
    best = results[0]; best_key = (best["delivered_mean"], -best["time_mean"])
    for r in results[1:]:
        key = (r["delivered_mean"], -r["time_mean"])
        if key > best_key:
            best, best_key = r, key
    return best

def run_policy_best_of_k(init_states, schedule, max_steps, policy, device, k:int, predictor=None, pred_mean=None, pred_std=None, alpha=0.0, beta=0.0, ortools_time_limit_ms=30, sample_seed_base=0):
    trials = []
    for s in range(k):
        np.random.seed(sample_seed_base+s); torch.manual_seed(sample_seed_base+s)
        trials.append(run_single_rollout(init_states, schedule, "policy", max_steps, policy, device, predictor, pred_mean, pred_std, alpha, beta, True, ortools_time_limit_ms))
    return select_best_of_k(trials)

def mean_key(xs, key):
    return float(np.mean([x[key] for x in xs])) if xs else float("nan")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--predictor-checkpoint", default="")
    parser.add_argument("--n-episodes", type=int, default=40)
    parser.add_argument("--n-nodes", type=int, default=20)
    parser.add_argument("--num-instances", type=int, default=4)
    parser.add_argument("--use-osrm", action="store_true")
    parser.add_argument("--policy-n-samples", type=int, default=8)
    parser.add_argument("--guided-alpha", type=float, default=0.0)
    parser.add_argument("--guided-beta", type=float, default=0.6)
    parser.add_argument("--ortools-time-limit-ms", type=int, default=30)
    parser.add_argument("--buckets", nargs="+", default=["low","medium","high"])
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--save-json", default="results/scenario_bucket_eval.json")
    args = parser.parse_args()
    if not HAS_ORTOOLS:
        raise RuntimeError("OR-Tools is not available in this Python environment.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, policy_ckpt = load_policy(args.policy_checkpoint, device)
    predictor = pred_mean = pred_std = predictor_ckpt = None
    if args.predictor_checkpoint:
        predictor, pred_mean, pred_std, predictor_ckpt = load_predictor(args.predictor_checkpoint, device)

    print(f"[SCENARIO BUCKET] Device: {device}")
    print(f"[SCENARIO BUCKET] Policy: {args.policy_checkpoint}")
    print(f"[SCENARIO BUCKET] Predictor: {args.predictor_checkpoint or 'none'}")
    print(f"[SCENARIO BUCKET] Buckets={args.buckets}, episodes={args.n_episodes}, samples={args.policy_n_samples}")

    out = {"config": vars(args), "policy_meta": {"epoch": policy_ckpt.get("epoch"), "best_of_pomo": policy_ckpt.get("best_of_pomo")}, "buckets": {}}
    if predictor_ckpt is not None:
        out["predictor_meta"] = {"epoch": predictor_ckpt.get("epoch"), "val_loss": predictor_ckpt.get("val_loss"), "val_acc": predictor_ckpt.get("val_acc"), "val_future_cost_mae": predictor_ckpt.get("val_future_cost_mae")}

    max_steps = args.n_nodes*8 + 64

    for bidx, bucket in enumerate(args.buckets):
        cfg = ResearchEnvConfig(n_nodes=args.n_nodes, num_instances=args.num_instances, device=device.type, use_osrm_instances=args.use_osrm, auto_reset=False, use_augmentation=not args.use_osrm)
        cfg = apply_bucket(cfg, bucket)

        greedy_results=[]; sample_results=[]; guided_results=[]; rolling_results=[]; episodes=[]

        for ep in range(args.n_episodes):
            seed = args.base_seed + ep + 10000*bidx
            np.random.seed(seed); torch.manual_seed(seed)
            env = ResearchEnv(cfg); env.reset()
            init_states = init_states_from_env(env)
            schedule = presample_events(init_states, cfg, max_steps, seed+999)

            greedy = run_single_rollout(init_states, schedule, "policy", max_steps, policy, device, sampling=False, ortools_time_limit_ms=args.ortools_time_limit_ms)
            samplexN = run_policy_best_of_k(init_states, schedule, max_steps, policy, device, args.policy_n_samples, None, None, None, 0.0, 0.0, args.ortools_time_limit_ms, seed*1000+100)
            rolling = run_single_rollout(init_states, schedule, "rolling_or", max_steps, None, device, sampling=False, ortools_time_limit_ms=args.ortools_time_limit_ms)

            entry = {"episode": ep+1, "baseline_greedy": greedy, "baseline_samplexN": samplexN, "rolling_or": rolling}
            greedy_results.append(greedy); sample_results.append(samplexN); rolling_results.append(rolling)

            if predictor is not None:
                guided = run_policy_best_of_k(init_states, schedule, max_steps, policy, device, args.policy_n_samples, predictor, pred_mean, pred_std, args.guided_alpha, args.guided_beta, args.ortools_time_limit_ms, seed*1000+500)
                entry["guided_samplexN"] = guided
                guided_results.append(guided)

            episodes.append(entry)
            print(f"[SCENARIO BUCKET] {bucket} episode {ep+1}/{args.n_episodes} complete | greedy_time={greedy['time_mean']:.2f} | samplexN_time={samplexN['time_mean']:.2f} | rolling_time={rolling['time_mean']:.2f}")

        summary = {
            "baseline_greedy": {"time_mean": mean_key(greedy_results,"time_mean"), "delivered_mean": mean_key(greedy_results,"delivered_mean")},
            "baseline_samplexN": {"time_mean": mean_key(sample_results,"time_mean"), "delivered_mean": mean_key(sample_results,"delivered_mean")},
            "rolling_or": {"time_mean": mean_key(rolling_results,"time_mean"), "delivered_mean": mean_key(rolling_results,"delivered_mean")},
            "baseline_samplexN_vs_rolling_or": {
                "time_gap_mean": mean_key(sample_results,"time_mean") - mean_key(rolling_results,"time_mean"),
                "delivered_gap_mean": mean_key(sample_results,"delivered_mean") - mean_key(rolling_results,"delivered_mean"),
            },
        }
        if guided_results:
            summary["guided_samplexN"] = {"time_mean": mean_key(guided_results,"time_mean"), "delivered_mean": mean_key(guided_results,"delivered_mean")}
            summary["guided_samplexN_vs_rolling_or"] = {
                "time_gap_mean": mean_key(guided_results,"time_mean") - mean_key(rolling_results,"time_mean"),
                "delivered_gap_mean": mean_key(guided_results,"delivered_mean") - mean_key(rolling_results,"delivered_mean"),
            }
            summary["guided_samplexN_vs_baseline_samplexN"] = {
                "time_gain_mean": mean_key(sample_results,"time_mean") - mean_key(guided_results,"time_mean"),
                "delivered_gain_mean": mean_key(guided_results,"delivered_mean") - mean_key(sample_results,"delivered_mean"),
            }

        out["buckets"][bucket] = {"summary": summary, "episodes": episodes}

    save_path = Path(args.save_json)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[SCENARIO BUCKET] Saved JSON: {save_path}")

if __name__ == "__main__":
    main()
