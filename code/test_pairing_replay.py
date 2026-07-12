"""Pairing/replay validation gates for the paired-episode contract.

The paper's paired evidence relies on: (a) schedules regenerating
bit-identically from their seed, (b) the environment stream being
unaffected by decoder/policy RNG consumption, (c) rollouts neither
mutating the schedule nor depending on run order, and (d) the matched
shadow probe regenerating the identical candidate portfolio it diagnoses.

Tests 1-4 are CPU-only and model-free. Test 5 loads the policy checkpoint
(CPU) and is skipped when unavailable.
"""
from __future__ import annotations

import copy
import hashlib
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

import scenario_bucket_eval_v2 as stage2


def make_state(seed: int, n_nodes: int = 12) -> stage2.SimStateV2:
    rng = np.random.RandomState(seed)
    coords = rng.rand(n_nodes, 2)
    base = rng.uniform(60.0, 600.0, size=(n_nodes, n_nodes))
    np.fill_diagonal(base, 0.0)
    visited = np.zeros(n_nodes, dtype=bool)
    visited[0] = True
    return stage2.SimStateV2(
        coords=coords,
        base_dist=base,
        eff_dist=base.copy(),
        visited=visited,
        node_blocked=np.zeros(n_nodes, dtype=np.float32),
        current_node=0,
        elapsed_time=0.0,
        horizon_sec=8 * 3600.0,
        n_nodes=n_nodes,
    )


def make_cfg(bucket: str = "high") -> stage2.ResearchEnvV2Config:
    return stage2.apply_bucket_v2(
        stage2.ResearchEnvV2Config(
            n_nodes=12, num_instances=1, device="cpu",
            auto_reset=False, use_augmentation=True,
        ),
        bucket,
    )


def schedule_digest(initial_eff: np.ndarray, schedule) -> str:
    hasher = hashlib.sha256()
    hasher.update(np.ascontiguousarray(initial_eff).tobytes())
    for matrix, blocked in schedule:
        hasher.update(np.ascontiguousarray(matrix).tobytes())
        hasher.update(np.ascontiguousarray(blocked).tobytes())
    return hasher.hexdigest()


class PairingReplayTests(unittest.TestCase):
    MAX_STEPS = 30

    def test_schedule_regenerates_bit_identically_from_seed(self) -> None:
        state = make_state(7)
        cfg = make_cfg()
        a = stage2.presample_schedule_v2(state, cfg, self.MAX_STEPS, 4242)
        b = stage2.presample_schedule_v2(state, cfg, self.MAX_STEPS, 4242)
        self.assertEqual(schedule_digest(*a), schedule_digest(*b))
        c = stage2.presample_schedule_v2(state, cfg, self.MAX_STEPS, 4243)
        self.assertNotEqual(schedule_digest(*a), schedule_digest(*c))

    def test_environment_stream_independent_of_decoder_rng(self) -> None:
        # Consuming arbitrary amounts of global numpy/torch randomness (as a
        # policy/decoder with different K or seed would) must not change the
        # regenerated exogenous schedule.
        state = make_state(7)
        cfg = make_cfg()
        reference = schedule_digest(
            *stage2.presample_schedule_v2(state, cfg, self.MAX_STEPS, 4242)
        )
        np.random.seed(999)
        np.random.rand(1000)
        torch.manual_seed(123)
        torch.rand(1000)
        replayed = schedule_digest(
            *stage2.presample_schedule_v2(state, cfg, self.MAX_STEPS, 4242)
        )
        self.assertEqual(reference, replayed)

    def test_rollout_replay_reproduces_outcome_and_preserves_schedule(self) -> None:
        state = make_state(11)
        cfg = make_cfg()
        initial_eff, schedule = stage2.presample_schedule_v2(
            state, cfg, self.MAX_STEPS, 5555
        )
        before = schedule_digest(initial_eff, schedule)

        def nearest_fn(mode, st, ctrl=None):
            if mode == "init":
                return None
            mask = stage2.valid_mask_v2(st)
            feasible = np.flatnonzero(mask)
            if feasible.size == 0:
                return st.current_node
            return int(
                feasible[np.argmin(st.eff_dist[st.current_node, feasible])]
            )

        first = stage2.run_rollout_v2(
            [state], [initial_eff], [schedule], self.MAX_STEPS, nearest_fn
        )
        second = stage2.run_rollout_v2(
            [state], [initial_eff], [schedule], self.MAX_STEPS, nearest_fn
        )
        self.assertEqual(first, second)
        self.assertEqual(before, schedule_digest(initial_eff, schedule))

    def test_run_order_does_not_change_results(self) -> None:
        # Running a second (different) method before replaying the first must
        # not change the first method's outcome: methods share the world only
        # through the immutable pre-sampled schedule.
        state = make_state(13)
        cfg = make_cfg()
        initial_eff, schedule = stage2.presample_schedule_v2(
            state, cfg, self.MAX_STEPS, 7777
        )

        def nearest_fn(mode, st, ctrl=None):
            if mode == "init":
                return None
            mask = stage2.valid_mask_v2(st)
            feasible = np.flatnonzero(mask)
            if feasible.size == 0:
                return st.current_node
            return int(
                feasible[np.argmin(st.eff_dist[st.current_node, feasible])]
            )

        def farthest_fn(mode, st, ctrl=None):
            if mode == "init":
                return None
            mask = stage2.valid_mask_v2(st)
            feasible = np.flatnonzero(mask)
            if feasible.size == 0:
                return st.current_node
            finite = st.eff_dist[st.current_node, feasible]
            finite = np.where(np.isfinite(finite), finite, -1.0)
            return int(feasible[np.argmax(finite)])

        alone = stage2.run_rollout_v2(
            [copy.deepcopy(state)], [initial_eff], [schedule],
            self.MAX_STEPS, nearest_fn,
        )
        stage2.run_rollout_v2(
            [copy.deepcopy(state)], [initial_eff], [schedule],
            self.MAX_STEPS, farthest_fn,
        )
        after_other = stage2.run_rollout_v2(
            [copy.deepcopy(state)], [initial_eff], [schedule],
            self.MAX_STEPS, nearest_fn,
        )
        self.assertEqual(alone, after_other)

    def test_probe_regenerates_identical_candidate_portfolio(self) -> None:
        checkpoint = (
            stage2.V62_DIR / "checkpoints_research_pomo" / "research_best.pt"
        )
        if not checkpoint.exists():
            self.skipTest("policy checkpoint unavailable")
        from matched_information_eval import MatchedInformationController

        stage2.POLICY_MATRIX_MODE = "live"
        device = torch.device("cpu")
        policy, _ = stage2.load_policy(str(checkpoint), device)
        policy.eval()

        state = make_state(17, n_nodes=12)
        cfg = make_cfg()
        _, schedule = stage2.presample_schedule_v2(state, cfg, 20, 8888)
        controller = MatchedInformationController(
            policy, device, portfolio_k=4, selection_k=4, seed=31337,
            schedule=schedule, scoring_mode="frozen",
            diagnostic_k_values=(4,), nested_k_values=(),
            collect_diagnostics=True,
        )
        # Two probes from the same pre-decision snapshot must regenerate the
        # same portfolio (checksum equality = shared-state candidate identity).
        first = controller._run_probe(state, 4, (4,))
        second = controller._run_probe(state, 4, (4,))
        self.assertEqual(first[2], second[2])
        self.assertEqual(first[1], second[1])


if __name__ == "__main__":
    unittest.main()
