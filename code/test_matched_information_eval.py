"""CPU-only synthetic tests for matched candidate scoring."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

from matched_information_eval import (
    compare_candidate_portfolio,
    score_candidate_frozen,
    score_candidate_realized,
    select_candidate_index,
)
from scenario_bucket_eval_v2 import SimStateV2


def make_state(matrix: np.ndarray, horizon_sec: float = 100.0) -> SimStateV2:
    n_nodes = matrix.shape[0]
    visited = np.zeros(n_nodes, dtype=bool)
    visited[0] = True
    return SimStateV2(
        coords=np.column_stack(
            [np.linspace(0.0, 1.0, n_nodes), np.zeros(n_nodes)]
        ),
        base_dist=matrix.copy(),
        eff_dist=matrix.copy(),
        visited=visited,
        node_blocked=np.zeros(n_nodes, dtype=np.float32),
        current_node=0,
        elapsed_time=0.0,
        horizon_sec=horizon_sec,
        n_nodes=n_nodes,
    )


class CandidateScoringTests(unittest.TestCase):
    def test_constant_realized_schedule_matches_frozen_scoring(self) -> None:
        matrix = np.array(
            [
                [0.0, 2.0, 9.0],
                [2.0, 0.0, 3.0],
                [9.0, 3.0, 0.0],
            ]
        )
        state = make_state(matrix)
        event = (matrix.copy(), np.zeros(3, dtype=np.float32))

        frozen = score_candidate_frozen(state, [1, 2])
        realized = score_candidate_realized(state, [1, 2], [event, event])

        self.assertEqual(frozen, realized)
        self.assertEqual(realized.delivered, 2)
        self.assertEqual(realized.elapsed_time_sec, 5.0)

    def test_realized_event_is_applied_after_current_action(self) -> None:
        current = np.array(
            [
                [0.0, 2.0, 9.0],
                [2.0, 0.0, 3.0],
                [9.0, 3.0, 0.0],
            ]
        )
        future = current.copy()
        future[1, 2] = 7.0
        state = make_state(current)
        event = (future, np.zeros(3, dtype=np.float32))

        realized = score_candidate_realized(state, [1, 2], [event, event])

        # First leg uses current[0,1]=2; only the second uses future[1,2]=7.
        self.assertEqual(realized.delivered, 2)
        self.assertEqual(realized.elapsed_time_sec, 9.0)

    def test_realized_node_block_can_invalidate_later_candidate_action(self) -> None:
        matrix = np.array(
            [
                [0.0, 2.0, 9.0],
                [2.0, 0.0, 3.0],
                [9.0, 3.0, 0.0],
            ]
        )
        state = make_state(matrix)
        blocked = np.zeros(3, dtype=np.float32)
        blocked[2] = 1.0
        clear = np.zeros(3, dtype=np.float32)
        schedule = [(matrix.copy(), blocked), (matrix.copy(), clear)]

        frozen = score_candidate_frozen(state, [1, 2])
        realized = score_candidate_realized(state, [1, 2], schedule)

        self.assertEqual(frozen.delivered, 2)
        self.assertEqual(realized.delivered, 1)
        self.assertEqual(realized.elapsed_time_sec, 2.0)

    def test_realized_scoring_matches_harness_rollout(self) -> None:
        # Gate (c): executing the scored sequence through run_rollout_v2 with
        # the same schedule must land on the scorer's predicted terminal value.
        from scenario_bucket_eval_v2 import run_rollout_v2

        current = np.array(
            [
                [0.0, 2.0, 9.0, 4.0],
                [2.0, 0.0, 3.0, 8.0],
                [9.0, 3.0, 0.0, 5.0],
                [4.0, 8.0, 5.0, 0.0],
            ]
        )
        step1 = current * 1.5
        step2 = current * 0.5
        blocked = np.zeros(4, dtype=np.float32)
        schedule = [
            (step1, blocked.copy()),
            (step2, blocked.copy()),
            (current.copy(), blocked.copy()),
        ]
        state = make_state(current, horizon_sec=1000.0)
        sequence = [1, 2, 3]

        predicted = score_candidate_realized(state, sequence, schedule)

        cursor = {"i": 0}

        def act_fn(mode, st, ctrl=None):
            if mode == "init":
                return None
            action = sequence[cursor["i"]]
            cursor["i"] += 1
            return action

        result = run_rollout_v2(
            [state], [current.copy()], [schedule], len(sequence), act_fn
        )

        self.assertEqual(predicted.delivered, int(result["delivered_mean"]))
        self.assertEqual(predicted.elapsed_time_sec, result["time_mean"])

    def test_same_portfolio_can_select_different_candidates(self) -> None:
        current = np.full((4, 4), 50.0)
        np.fill_diagonal(current, 0.0)
        current[0, 1] = 1.0
        current[1, 2] = 1.0
        current[0, 3] = 2.0
        current[3, 2] = 2.0
        state = make_state(current)

        future = current.copy()
        future[1, 2] = np.inf
        event = (future, np.zeros(4, dtype=np.float32))
        sequences = [[1, 2], [3, 2]]
        frozen = [
            score_candidate_frozen(state, sequence) for sequence in sequences
        ]
        realized = [
            score_candidate_realized(state, sequence, [event, event])
            for sequence in sequences
        ]

        self.assertEqual(select_candidate_index(sequences, frozen, 2), 0)
        self.assertEqual(select_candidate_index(sequences, realized, 2), 1)

        comparisons = compare_candidate_portfolio(
            state, sequences, frozen, realized, [1, 2]
        )
        self.assertTrue(comparisons["1"]["first_action_agreement"])
        self.assertFalse(comparisons["2"]["selected_candidate_agreement"])
        self.assertFalse(comparisons["2"]["first_action_agreement"])
        self.assertEqual(
            comparisons["2"]["realized_value_difference"]["delivered_gain"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
