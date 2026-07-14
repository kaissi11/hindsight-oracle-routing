"""Deadline-accounting and waiting-semantics tests (review P0.1).

Documents and locks the transition semantics the paper reports:

* recorded KPI ("delivered"): a customer counts when the vehicle DEPARTS
  toward it before H (the rollout loop guard), even if arrival crosses H --
  a departure-cutoff estimand;
* strict KPI ("delivered_strict"): arrival <= H, the operational
  service-completion reading;
* at most ONE stop per route can differ between the two (only the final
  leg can straddle H);
* waiting (no feasible action) consumes zero elapsed time while the world
  advances by one event -- the simulator is decision-indexed, not
  clock-indexed (paper limitation; §3 wording);
* the dual-count bookkeeping is pure accounting: it never changes actions,
  elapsed time, masks, or the delivered/time outputs.
"""
from __future__ import annotations

import copy
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

import scenario_bucket_eval_v2 as stage2


def tiny_state(horizon: float = 1000.0, travel: float = 400.0,
               n_nodes: int = 3) -> stage2.SimStateV2:
    coords = np.linspace(0.1, 0.9, n_nodes * 2).reshape(n_nodes, 2)
    base = np.full((n_nodes, n_nodes), travel, dtype=np.float64)
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
        horizon_sec=horizon,
        n_nodes=n_nodes,
    )


def neutral_event(state: stage2.SimStateV2):
    return (state.eff_dist.copy(), state.node_blocked.copy())


class DeadlineEdgeTests(unittest.TestCase):
    def test_departure_before_H_arrival_after_H_counts_recorded_not_strict(self):
        # elapsed = H - epsilon, travel > epsilon: the recorded KPI counts the
        # stop, the strict KPI does not.
        st = tiny_state(horizon=1000.0, travel=400.0)
        st.elapsed_time = 999.0  # H - 1, travel 400 >> 1
        stage2.apply_action_and_advance_v2(st, 1, neutral_event(st))
        self.assertTrue(st.visited[1])                    # recorded: delivered
        self.assertEqual(st.strict_delivered, 0)          # strict: NOT delivered
        self.assertEqual(st.late_arrivals, 1)
        self.assertGreater(st.elapsed_time, st.horizon_sec)

    def test_arrival_exactly_at_H_counts_in_both(self):
        st = tiny_state(horizon=1000.0, travel=400.0)
        st.elapsed_time = 600.0  # arrival lands exactly on H
        stage2.apply_action_and_advance_v2(st, 1, neutral_event(st))
        self.assertTrue(st.visited[1])
        self.assertEqual(st.strict_delivered, 1)
        self.assertEqual(st.late_arrivals, 0)

    def test_arrival_strictly_before_H_counts_in_both(self):
        st = tiny_state(horizon=1000.0, travel=400.0)
        stage2.apply_action_and_advance_v2(st, 1, neutral_event(st))
        self.assertTrue(st.visited[1])
        self.assertEqual(st.strict_delivered, 1)
        self.assertEqual(st.late_arrivals, 0)

    def test_at_most_one_straddling_stop_per_route(self):
        # Rollout loop guard: once elapsed >= H no further action is taken, so
        # recorded - strict is 0 or 1 for any trajectory.
        st = tiny_state(horizon=1000.0, travel=700.0, n_nodes=5)

        def nearest_fn(mode, s, ctrl=None):
            if mode == "init":
                return None
            mask = stage2.valid_mask_v2(s)
            feas = np.flatnonzero(mask)
            if feas.size == 0:
                return s.current_node
            return int(feas[np.argmin(s.eff_dist[s.current_node, feas])])

        cfg = stage2.apply_bucket_v2(
            stage2.ResearchEnvV2Config(n_nodes=5, num_instances=1,
                                       device="cpu", auto_reset=False,
                                       use_augmentation=True), "low")
        initial_eff, schedule = stage2.presample_schedule_v2(st, cfg, 30, 123)
        out = stage2.run_rollout_v2([st], [initial_eff], [schedule], 30,
                                    nearest_fn)
        gap = out["delivered_mean"] - out["delivered_strict_mean"]
        self.assertIn(gap, (0.0, 1.0))
        self.assertEqual(gap, out["late_arrival_mean"])


class WaitingSemanticsTests(unittest.TestCase):
    def test_wait_consumes_no_time_but_world_advances(self):
        # All customers infeasible -> the vehicle waits: elapsed time is
        # unchanged (decision-indexed simulator), the event still applies.
        st = tiny_state(horizon=1000.0, travel=400.0)
        st.node_blocked[1:] = 1.0  # nothing feasible
        new_eff = st.eff_dist * 2.0
        unblocked = np.zeros_like(st.node_blocked)
        stage2.apply_action_and_advance_v2(st, st.current_node,
                                           (new_eff, unblocked))
        self.assertEqual(st.elapsed_time, 0.0)            # no time consumed
        self.assertEqual(st.wait_steps, 1)                # logged
        self.assertTrue((st.eff_dist == new_eff).all())   # world advanced
        self.assertTrue((st.node_blocked == 0.0).all())

    def test_wait_never_counts_a_delivery(self):
        st = tiny_state()
        st.node_blocked[1:] = 1.0
        stage2.apply_action_and_advance_v2(st, 1, neutral_event(st))
        self.assertFalse(st.visited[1:].any())
        self.assertEqual(st.strict_delivered, 0)
        self.assertEqual(st.late_arrivals, 0)
        self.assertEqual(st.wait_steps, 1)

    def test_wait_cost_probe_charges_elapsed_time(self):
        # --wait-cost-sec > 0 makes waiting consume horizon (exploratory probe).
        prev = stage2.WAIT_COST_SEC
        try:
            stage2.WAIT_COST_SEC = 60.0
            st = tiny_state(horizon=1000.0, travel=400.0)
            st.node_blocked[1:] = 1.0
            stage2.apply_action_and_advance_v2(st, st.current_node,
                                               neutral_event(st))
            self.assertEqual(st.wait_steps, 1)
            self.assertEqual(st.elapsed_time, 60.0)
        finally:
            stage2.WAIT_COST_SEC = prev


class BookkeepingNeutralityTests(unittest.TestCase):
    def test_counters_do_not_change_actions_or_outcomes(self):
        # A rollout must produce identical delivered/time values whether or
        # not the counters start at zero (they are write-only bookkeeping).
        st = tiny_state(horizon=4000.0, travel=400.0, n_nodes=6)
        cfg = stage2.apply_bucket_v2(
            stage2.ResearchEnvV2Config(n_nodes=6, num_instances=1,
                                       device="cpu", auto_reset=False,
                                       use_augmentation=True), "medium")
        initial_eff, schedule = stage2.presample_schedule_v2(st, cfg, 40, 777)

        def nearest_fn(mode, s, ctrl=None):
            if mode == "init":
                return None
            mask = stage2.valid_mask_v2(s)
            feas = np.flatnonzero(mask)
            if feas.size == 0:
                return s.current_node
            return int(feas[np.argmin(s.eff_dist[s.current_node, feas])])

        clean = stage2.run_rollout_v2([copy.deepcopy(st)], [initial_eff],
                                      [schedule], 40, nearest_fn)
        poisoned = copy.deepcopy(st)
        poisoned.strict_delivered = 100
        poisoned.late_arrivals = 100
        poisoned.wait_steps = 100
        dirty = stage2.run_rollout_v2([poisoned], [initial_eff], [schedule],
                                      40, nearest_fn)
        self.assertEqual(clean["delivered_mean"], dirty["delivered_mean"])
        self.assertEqual(clean["time_mean"], dirty["time_mean"])


if __name__ == "__main__":
    unittest.main()
