"""MATCHED_SELECTOR_PLAN gates (a)+(b) on CPU: the matched frozen arm (with
shadow diagnostics enabled, as run by the queue) must reproduce plain look-8
bit-identically on the same episodes."""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # merged layout: modules live alongside

import numpy as np
import torch

import scenario_bucket_eval_v2 as stage2
from matched_information_eval import MatchedInformationController

device = torch.device("cpu")
stage2.POLICY_MATRIX_MODE = "live"
policy, _ = stage2.load_policy(
    str(stage2.V62_DIR / "checkpoints_research_pomo" / "research_best.pt"), device
)
policy.eval()

N_EPISODES = 2
MAX_STEPS = 24
failures = 0

for episode_index in range(N_EPISODES):
    episode_seed = 12345 + episode_index
    np.random.seed(episode_seed)
    torch.manual_seed(episode_seed)
    cfg = stage2.apply_bucket_v2(
        stage2.ResearchEnvV2Config(
            n_nodes=20, num_instances=1, device="cpu",
            auto_reset=False, use_augmentation=True,
        ),
        "high",
    )
    env = stage2.ResearchEnvV2(cfg)
    env.reset()
    n_nodes = env.n_nodes
    visited = np.zeros(n_nodes, dtype=bool)
    visited[0] = True
    state = stage2.SimStateV2(
        coords=env.coords[0].cpu().numpy().astype(np.float64),
        base_dist=env.base_dist[0].cpu().numpy().astype(np.float64),
        eff_dist=env.base_dist[0].cpu().numpy().astype(np.float64),
        visited=visited,
        node_blocked=np.zeros(n_nodes, dtype=np.float32),
        current_node=0,
        elapsed_time=0.0,
        horizon_sec=float(cfg.time_horizon_sec),
        n_nodes=n_nodes,
    )
    schedule_seed = episode_seed + 999
    initial_eff, schedule = stage2.presample_schedule_v2(
        state, cfg, MAX_STEPS, schedule_seed
    )
    controller_seed = episode_seed * 1000 + 601

    # Plain look-8 (recorded-arm construction).
    np.random.seed(controller_seed)
    torch.manual_seed(controller_seed)
    plain = stage2.LookaheadControllerV2(
        policy, device, 8, controller_seed, temperature=1.0,
        use_2opt=False, n_scenarios=0,
    )

    def plain_fn(mode, st, ctrl=None, c=plain):
        return c if mode == "init" else ctrl.act(st)

    plain_result = stage2.run_rollout_v2(
        [state], [initial_eff], [schedule], MAX_STEPS, plain_fn
    )

    # Matched frozen arm exactly as run_matched_instance constructs it.
    np.random.seed(controller_seed)
    torch.manual_seed(controller_seed)
    frozen = MatchedInformationController(
        policy, device, portfolio_k=8, selection_k=8, seed=controller_seed,
        schedule=schedule, scoring_mode="frozen", temperature=1.0,
        diagnostic_k_values=(8,), nested_k_values=(1, 2, 4, 8, 16),
        collect_diagnostics=True,
    )

    def frozen_fn(mode, st, ctrl=None, c=frozen):
        return c if mode == "init" else ctrl.act(st)

    frozen_result = stage2.run_rollout_v2(
        [state], [initial_eff], [schedule], MAX_STEPS, frozen_fn
    )

    delivered_match = plain_result["delivered_mean"] == frozen_result["delivered_mean"]
    time_match = plain_result["time_mean"] == frozen_result["time_mean"]
    status = "PASS" if delivered_match and time_match else "FAIL"
    if status == "FAIL":
        failures += 1
    print(
        f"[GATE] episode {episode_index}: {status} "
        f"plain=({plain_result['delivered_mean']:.6f}, {plain_result['time_mean']:.3f}) "
        f"frozen=({frozen_result['delivered_mean']:.6f}, {frozen_result['time_mean']:.3f})"
    )

print("[GATE] overall:", "PASS" if failures == 0 else f"FAIL ({failures})")
sys.exit(1 if failures else 0)
