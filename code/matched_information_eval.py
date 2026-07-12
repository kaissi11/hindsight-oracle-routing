"""Matched-information online-versus-clairvoyant lookahead evaluator.

This module leaves ``scenario_bucket_eval_v2.py`` unchanged and reuses its
state, schedule, policy, candidate-generation, and rollout machinery.  The
only treatment in the primary comparison is how a shared candidate-generation
algorithm is scored:

* ``frozen``: hold the current effective matrix and node availability fixed;
* ``realized``: execute the same fixed candidate sequence against the
  remaining pre-sampled schedule, applying schedule[t] after action t exactly
  as the rollout harness does.

The committed online trajectory also carries a same-state shadow diagnostic.
Its primary K diagnostic scores the online controller's actual portfolio both
ways.  A separate, non-committing Kmax portfolio is generated from a copy of
the online controller's pre-decision RNG and warm-start state, then evaluated
through nested prefixes K={1,2,4,8,16}.

Do not launch the checkpoint-backed evaluation while another GPU queue is
active.  The pure scoring helpers in this file are intentionally model-free
and are covered by CPU-only synthetic tests.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

import scenario_bucket_eval_v2 as stage2


ROOT = Path(__file__).resolve().parent
DEFAULT_SENSITIVITY_K = (1, 2, 4, 8, 16)
ScheduleEvent = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class CandidateValue:
    """Lexicographic route value at the end of a fixed candidate sequence."""

    delivered: int
    elapsed_time_sec: float

    @property
    def key(self) -> tuple[int, float]:
        return self.delivered, -self.elapsed_time_sec

    def to_json(self) -> dict[str, int | float]:
        return {
            "delivered": int(self.delivered),
            "elapsed_time_sec": float(self.elapsed_time_sec),
        }


def _terminal_value(state: stage2.SimStateV2) -> CandidateValue:
    return CandidateValue(
        delivered=int(state.visited[1:].sum()),
        elapsed_time_sec=float(state.elapsed_time),
    )


def score_candidate_frozen(
    state: stage2.SimStateV2,
    sequence: Sequence[int],
) -> CandidateValue:
    """Score a fixed sequence with the current matrix frozen.

    The action/clock semantics deliberately go through the v2-harness transition
    function.  Thus a stop reached by the action that crosses the horizon is
    counted, matching the committed rollout harness.
    """

    sim = copy.deepcopy(state)
    frozen_event = (state.eff_dist, state.node_blocked)
    for action in sequence:
        if sim.elapsed_time >= sim.horizon_sec:
            break
        stage2.apply_action_and_advance_v2(sim, int(action), frozen_event)
    return _terminal_value(sim)


def score_candidate_realized(
    state: stage2.SimStateV2,
    sequence: Sequence[int],
    remaining_schedule: Sequence[ScheduleEvent],
) -> CandidateValue:
    """Score a fixed sequence against the exact remaining realized schedule.

    ``remaining_schedule[0]`` is the event applied *after* the candidate's
    first action, exactly matching ``run_rollout_v2``.  If the presampled
    harness horizon is exhausted, no actions beyond it are assigned invented
    future events.
    """

    sim = copy.deepcopy(state)
    for action, event in zip(sequence, remaining_schedule):
        if sim.elapsed_time >= sim.horizon_sec:
            break
        stage2.apply_action_and_advance_v2(sim, int(action), event)
    return _terminal_value(sim)


def select_candidate_index(
    sequences: Sequence[Sequence[int]],
    values: Sequence[CandidateValue],
    candidate_limit: int,
) -> int | None:
    """Return the first lexicographic maximizer in a portfolio prefix."""

    limit = min(int(candidate_limit), len(sequences), len(values))
    best_index: int | None = None
    best_key: tuple[int, float] | None = None
    for index in range(limit):
        if not sequences[index]:
            continue
        key = values[index].key
        if best_key is None or key > best_key:
            best_index, best_key = index, key
    return best_index


def _selected_value(
    values: Sequence[CandidateValue],
    index: int | None,
    fallback: CandidateValue,
) -> CandidateValue:
    return fallback if index is None else values[index]


def _first_action(sequences: Sequence[Sequence[int]], index: int | None) -> int | None:
    if index is None or not sequences[index]:
        return None
    return int(sequences[index][0])


def _average_ranks(keys: Sequence[tuple]) -> np.ndarray:
    order = sorted(range(len(keys)), key=lambda index: keys[index])
    ranks = np.empty(len(keys))
    start = 0
    while start < len(keys):
        stop = start
        while (
            stop + 1 < len(keys)
            and keys[order[stop + 1]] == keys[order[start]]
        ):
            stop += 1
        for position in range(start, stop + 1):
            ranks[order[position]] = (start + stop) / 2.0
        start = stop + 1
    return ranks


def spearman_correlation(
    a_keys: Sequence[tuple], b_keys: Sequence[tuple]
) -> float | None:
    """Spearman rank correlation over lexicographic score keys (ties averaged)."""
    if len(a_keys) < 2:
        return None
    rank_a = _average_ranks(a_keys)
    rank_b = _average_ranks(b_keys)
    if rank_a.std() == 0 or rank_b.std() == 0:
        return None
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def compare_candidate_portfolio(
    state: stage2.SimStateV2,
    sequences: Sequence[Sequence[int]],
    frozen_values: Sequence[CandidateValue],
    realized_values: Sequence[CandidateValue],
    k_values: Sequence[int],
) -> dict[str, dict[str, Any]]:
    """Build same-state online/clairvoyant comparisons for nested prefixes."""

    fallback = _terminal_value(state)
    comparisons: dict[str, dict[str, Any]] = {}
    for k in sorted(set(int(v) for v in k_values)):
        frozen_index = select_candidate_index(sequences, frozen_values, k)
        realized_index = select_candidate_index(sequences, realized_values, k)
        prefix = [
            index
            for index in range(min(int(k), len(sequences)))
            if sequences[index]
        ]
        rank_correlation = spearman_correlation(
            [frozen_values[index].key for index in prefix],
            [realized_values[index].key for index in prefix],
        )
        unique_first_actions = len(
            {int(sequences[index][0]) for index in prefix}
        )
        unique_sequences = len({tuple(sequences[index]) for index in prefix})
        duplicate_sequence_rate = (
            1.0 - unique_sequences / len(prefix) if prefix else None
        )
        online_realized = _selected_value(realized_values, frozen_index, fallback)
        clair_realized = _selected_value(realized_values, realized_index, fallback)
        frozen_first = _first_action(sequences, frozen_index)
        realized_first = _first_action(sequences, realized_index)
        if clair_realized.key > online_realized.key:
            lexicographic_gain = 1
        elif clair_realized.key < online_realized.key:
            lexicographic_gain = -1
        else:
            lexicographic_gain = 0

        frozen_sequence = [] if frozen_index is None else list(sequences[frozen_index])
        realized_sequence = [] if realized_index is None else list(sequences[realized_index])
        comparisons[str(k)] = {
            "candidate_limit": int(k),
            "available_candidates": min(int(k), len(sequences)),
            "online_vs_realized_rank_correlation": rank_correlation,
            "unique_first_actions": unique_first_actions,
            "duplicate_sequence_rate": duplicate_sequence_rate,
            "frozen_selected_candidate_index": frozen_index,
            "clairvoyant_selected_candidate_index": realized_index,
            "selected_candidate_agreement": frozen_index == realized_index,
            "selected_sequence_agreement": frozen_sequence == realized_sequence,
            "frozen_selected_first_action": frozen_first,
            "clairvoyant_selected_first_action": realized_first,
            "first_action_agreement": frozen_first == realized_first,
            "frozen_selected_sequence": frozen_sequence,
            "clairvoyant_selected_sequence": realized_sequence,
            "frozen_selected_frozen_value": _selected_value(
                frozen_values, frozen_index, fallback
            ).to_json(),
            "frozen_selected_realized_value": online_realized.to_json(),
            "clairvoyant_selected_realized_value": clair_realized.to_json(),
            "realized_value_difference": {
                "delivered_gain": int(
                    clair_realized.delivered - online_realized.delivered
                ),
                "time_saving_sec": float(
                    online_realized.elapsed_time_sec - clair_realized.elapsed_time_sec
                ),
                "lexicographic_gain": lexicographic_gain,
            },
        }
    return comparisons


def _candidate_scores(
    sequences: Sequence[Sequence[int]],
    frozen_values: Sequence[CandidateValue],
    realized_values: Sequence[CandidateValue],
) -> list[dict[str, Any]]:
    """Per-candidate online (frozen) and realized scores for shadow logging."""
    return [
        {
            "first_action": int(sequence[0]) if sequence else None,
            "frozen": frozen.to_json(),
            "realized": realized.to_json(),
        }
        for sequence, frozen, realized in zip(
            sequences, frozen_values, realized_values
        )
    ]


def _portfolio_digest(sequences: Sequence[Sequence[int]]) -> str:
    payload = json.dumps(
        [list(sequence) for sequence in sequences],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def schedule_digest(
    initial_eff: np.ndarray, schedule: Sequence[ScheduleEvent]
) -> str:
    """Checksum of one pre-sampled exogenous world (replay contract)."""
    hasher = hashlib.sha256()
    hasher.update(np.ascontiguousarray(initial_eff).tobytes())
    for matrix, blocked in schedule:
        hasher.update(np.ascontiguousarray(matrix).tobytes())
        hasher.update(np.ascontiguousarray(blocked).tobytes())
    return hasher.hexdigest()[:16]


class MatchedInformationController(stage2.LookaheadControllerV2):
    """v2-harness lookahead with an isolated candidate-scoring intervention.

    Candidate construction remains in the inherited ``act`` method.

    Commitment semantics differ by arm:

    * ``scoring_mode="frozen"`` (the online arm): the committed action MUST be
      bit-identical to the recorded plain look-8, so the inherited internal
      selection path runs unmodified (``use_2opt=False``) — including its
      float32 ``delivered*1e9 - elapsed`` score, whose resolution at
      delivered≈19 is ≈2048 s, i.e. sub-2048-s time differences act as ties
      broken by candidate index.  Shadow diagnostics are computed on a
      bit-identical regeneration of the candidate portfolio from a copied RNG
      probe, scored exactly (both frozen and realized), and never influence
      the committed trajectory.
    * ``scoring_mode="realized"`` (the clairvoyant arm): the ``_rescore`` hook
      commits the exact-lexicographic realized-future selection (the
      predeclared look-8-clair definition).

    ``probe_only=True`` builds a non-committing portfolio-capture instance
    (hook active regardless of scoring mode); used internally.
    """

    def __init__(
        self,
        policy,
        device: torch.device,
        *,
        portfolio_k: int,
        selection_k: int,
        seed: int,
        schedule: Sequence[ScheduleEvent],
        scoring_mode: str,
        temperature: float = 1.0,
        diagnostic_k_values: Sequence[int] = (),
        nested_k_values: Sequence[int] = (),
        collect_diagnostics: bool = False,
        probe_only: bool = False,
    ) -> None:
        if scoring_mode not in {"frozen", "realized"}:
            raise ValueError(f"unknown scoring mode {scoring_mode!r}")
        if selection_k < 1:
            raise ValueError("selection_k must be positive")
        if portfolio_k < selection_k:
            raise ValueError("portfolio_k must be at least selection_k")

        super().__init__(
            policy,
            device,
            portfolio_k,
            seed,
            temperature=temperature,
            # The hook must stay CLOSED for the committed frozen arm so the
            # inherited selection (and thus the committed trajectory) is
            # bit-identical to the recorded look-8.  It opens for the
            # clairvoyant arm (hook commits) and for capture probes.
            use_2opt=(scoring_mode == "realized") or probe_only,
            n_scenarios=0,
        )
        self.probe_only = bool(probe_only)
        self.selection_k = int(selection_k)
        self.schedule = schedule
        self.scoring_mode = scoring_mode
        self.nested_k_values = tuple(
            sorted({int(k) for k in nested_k_values if k > 0})
        )
        self.diagnostic_k_values = tuple(
            sorted(
                {
                    self.selection_k,
                    *(
                        int(k)
                        for k in diagnostic_k_values
                        if 0 < int(k) <= self.k
                    ),
                    *(
                        int(k)
                        for k in self.nested_k_values
                        if int(k) <= self.k
                    ),
                }
            )
        )
        self.collect_diagnostics = bool(collect_diagnostics)
        self.schedule_step = 0
        self._active_schedule_step = 0
        self._last_sequences: list[list[int]] = []
        self._last_comparisons: dict[str, dict[str, Any]] = {}
        self._last_values: tuple[list[CandidateValue], list[CandidateValue]] = ([], [])
        self._last_record: dict[str, Any] | None = None
        self.diagnostics: list[dict[str, Any]] = []

    def _score_portfolio(
        self,
        state: stage2.SimStateV2,
        sequences: Sequence[Sequence[int]],
    ) -> tuple[list[CandidateValue], list[CandidateValue]]:
        remaining = self.schedule[self._active_schedule_step :]
        frozen = [score_candidate_frozen(state, sequence) for sequence in sequences]
        realized = [
            score_candidate_realized(state, sequence, remaining)
            for sequence in sequences
        ]
        return frozen, realized

    def _rescore(
        self,
        state: stage2.SimStateV2,
        seqs: list[list[int]],
    ) -> list[int]:
        sequences = [list(sequence) for sequence in seqs]
        frozen, realized = self._score_portfolio(state, sequences)
        comparisons = compare_candidate_portfolio(
            state,
            sequences,
            frozen,
            realized,
            self.diagnostic_k_values,
        )
        record = {
            "decision_step": int(self._active_schedule_step),
            "feasible_action_count": int(stage2.valid_mask_v2(state).sum()),
            "candidate_portfolio_size": len(sequences),
            "candidate_portfolio_digest": _portfolio_digest(sequences),
            "primary_k": self.selection_k,
            "primary": comparisons[str(self.selection_k)],
            "candidate_scores": _candidate_scores(sequences, frozen, realized),
        }
        self._last_sequences = sequences
        self._last_comparisons = comparisons
        self._last_values = (list(frozen), list(realized))
        self._last_record = record
        if self.collect_diagnostics:
            self.diagnostics.append(record)

        selected_field = (
            "frozen_selected_candidate_index"
            if self.scoring_mode == "frozen"
            else "clairvoyant_selected_candidate_index"
        )
        selected_index = comparisons[str(self.selection_k)][selected_field]
        return [] if selected_index is None else sequences[int(selected_index)]

    def _run_probe(
        self,
        state: stage2.SimStateV2,
        portfolio_k: int,
        k_values: Sequence[int],
    ) -> tuple[
        dict[str, dict[str, Any]],
        list[list[int]],
        str,
        tuple[list[CandidateValue], list[CandidateValue]],
    ]:
        """Generate one non-committing portfolio from copied RNG state.

        With ``portfolio_k == self.k`` the probe regenerates bit-identically
        the portfolio the inherited ``act`` is about to construct (same
        pre-decision RNG state, warm-start plan, and state), so its exact
        frozen/realized scorings diagnose the committed decision without
        touching it.
        """

        probe = MatchedInformationController(
            self.policy,
            self.device,
            portfolio_k=portfolio_k,
            selection_k=min(self.selection_k, portfolio_k),
            seed=0,
            schedule=self.schedule,
            scoring_mode="frozen",
            temperature=self.temperature,
            diagnostic_k_values=k_values,
            nested_k_values=(),
            collect_diagnostics=False,
            probe_only=True,
        )
        probe.gen.set_state(self.gen.get_state())
        probe.prev_plan = list(self.prev_plan)
        probe._t = self._t
        probe.schedule_step = self.schedule_step
        probe.act(copy.deepcopy(state))
        return (
            copy.deepcopy(probe._last_comparisons),
            copy.deepcopy(probe._last_sequences),
            _portfolio_digest(probe._last_sequences),
            copy.deepcopy(probe._last_values),
        )

    def _record_trivial_decision(
        self,
        state: stage2.SimStateV2,
        action: int,
        feasible_count: int,
    ) -> None:
        sequences = [] if feasible_count == 0 else [[int(action)]]
        frozen, realized = self._score_portfolio(state, sequences)
        all_k = sorted(
            {
                self.selection_k,
                *self.diagnostic_k_values,
                *self.nested_k_values,
            }
        )
        comparisons = compare_candidate_portfolio(
            state, sequences, frozen, realized, all_k
        )
        record = {
            "decision_step": int(self._active_schedule_step),
            "feasible_action_count": int(feasible_count),
            "candidate_portfolio_size": len(sequences),
            "candidate_portfolio_digest": _portfolio_digest(sequences),
            "primary_k": self.selection_k,
            "primary": comparisons[str(self.selection_k)],
            "nested": {
                "portfolio_k_max": max(self.nested_k_values, default=self.selection_k),
                "portfolio_digest": _portfolio_digest(sequences),
                "primary_prefix_matches": True,
                "trivial_decision": True,
                "by_k": {
                    str(k): comparisons[str(k)]
                    for k in self.nested_k_values
                },
            },
        }
        self._last_sequences = sequences
        self._last_comparisons = comparisons
        self._last_record = record
        if self.collect_diagnostics:
            self.diagnostics.append(record)

    def act(self, state: stage2.SimStateV2) -> int:
        self._active_schedule_step = self.schedule_step
        feasible = np.flatnonzero(stage2.valid_mask_v2(state))

        if feasible.size <= 1:
            action = (
                int(feasible[0])
                if feasible.size == 1
                else int(state.current_node)
            )
            if self.collect_diagnostics:
                self._record_trivial_decision(
                    state, action, int(feasible.size)
                )
            # The inherited method returns before incrementing _t in this case.
            self._t += 1
            self.schedule_step += 1
            return action

        ProbeResult = tuple[
            dict[str, dict[str, Any]],
            list[list[int]],
            str,
            tuple[list[CandidateValue], list[CandidateValue]],
        ]
        primary: ProbeResult | None = None
        nested: ProbeResult | None = None
        if self.collect_diagnostics:
            if self.scoring_mode == "frozen" and not self.use_2opt:
                # Committed selection is inherited (bit-identical look-8);
                # diagnostics come from a bit-identical portfolio probe.
                primary = self._run_probe(state, self.k, self.diagnostic_k_values)
            if self.nested_k_values and max(self.nested_k_values) > self.k:
                nested = self._run_probe(
                    state, max(self.nested_k_values), self.nested_k_values
                )

        action = int(super().act(state))

        if primary is not None:
            comparisons, sequences, digest, (frozen, realized) = primary
            selected = comparisons[str(self.selection_k)]
            record = {
                "decision_step": int(self._active_schedule_step),
                "feasible_action_count": int(feasible.size),
                "candidate_portfolio_size": len(sequences),
                "candidate_portfolio_digest": digest,
                "primary_k": self.selection_k,
                "primary": selected,
                "candidate_scores": _candidate_scores(sequences, frozen, realized),
                "committed_first_action": int(action),
                "committed_matches_exact_frozen_selected": (
                    selected["frozen_selected_first_action"] == int(action)
                ),
            }
            self._last_sequences = sequences
            self._last_comparisons = comparisons
            self._last_record = record
            self.diagnostics.append(record)

        if self.collect_diagnostics and self._last_record is not None:
            if nested is not None:
                nested_comparisons, nested_sequences, nested_digest, _ = nested
                prefix = nested_sequences[: len(self._last_sequences)]
                self._last_record["nested"] = {
                    "portfolio_k_max": max(self.nested_k_values),
                    "portfolio_digest": nested_digest,
                    "primary_prefix_matches": prefix == self._last_sequences,
                    "trivial_decision": False,
                    "by_k": {
                        str(k): nested_comparisons[str(k)]
                        for k in self.nested_k_values
                    },
                }
            elif self.nested_k_values:
                self._last_record["nested"] = {
                    "portfolio_k_max": self.k,
                    "portfolio_digest": self._last_record[
                        "candidate_portfolio_digest"
                    ],
                    "primary_prefix_matches": True,
                    "trivial_decision": False,
                    "by_k": {
                        str(k): self._last_comparisons[str(k)]
                        for k in self.nested_k_values
                    },
                }

        self.schedule_step += 1
        return action


def _comparison_summary(comparisons: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not comparisons:
        return {
            "n_decisions": 0,
            "first_action_agreement_rate": None,
            "selected_candidate_agreement_rate": None,
            "mean_realized_delivered_gain": None,
            "mean_realized_time_saving_sec": None,
            "positive_realized_value_rate": None,
        }

    gains = [item["realized_value_difference"] for item in comparisons]
    rank_correlations = [
        item["online_vs_realized_rank_correlation"]
        for item in comparisons
        if item.get("online_vs_realized_rank_correlation") is not None
    ]
    duplicate_rates = [
        item["duplicate_sequence_rate"]
        for item in comparisons
        if item.get("duplicate_sequence_rate") is not None
    ]
    return {
        "n_decisions": len(comparisons),
        "first_action_agreement_rate": float(
            np.mean([item["first_action_agreement"] for item in comparisons])
        ),
        "selected_candidate_agreement_rate": float(
            np.mean([item["selected_candidate_agreement"] for item in comparisons])
        ),
        "mean_realized_delivered_gain": float(
            np.mean([item["delivered_gain"] for item in gains])
        ),
        "mean_realized_time_saving_sec": float(
            np.mean([item["time_saving_sec"] for item in gains])
        ),
        "positive_realized_value_rate": float(
            np.mean([item["lexicographic_gain"] > 0 for item in gains])
        ),
        "mean_online_vs_realized_rank_correlation": (
            float(np.mean(rank_correlations)) if rank_correlations else None
        ),
        "mean_unique_first_actions": float(
            np.mean([item.get("unique_first_actions", 0) for item in comparisons])
        ),
        "mean_duplicate_sequence_rate": (
            float(np.mean(duplicate_rates)) if duplicate_rates else None
        ),
        "greedy_selected_rate": float(
            np.mean(
                [
                    item["frozen_selected_candidate_index"] == 0
                    for item in comparisons
                ]
            )
        ),
        "incumbent_selected_rate": float(
            np.mean(
                [
                    item["frozen_selected_candidate_index"] == 1
                    for item in comparisons
                ]
            )
        ),
    }


def summarize_diagnostics(
    records: Sequence[dict[str, Any]],
    sensitivity_k: Sequence[int],
) -> dict[str, Any]:
    primary = _comparison_summary([record["primary"] for record in records])
    nested = {
        str(k): _comparison_summary(
            [
                record["nested"]["by_k"][str(k)]
                for record in records
                if "nested" in record and str(k) in record["nested"]["by_k"]
            ]
        )
        for k in sensitivity_k
    }
    prefix_flags = [
        record["nested"]["primary_prefix_matches"]
        for record in records
        if "nested" in record
    ]
    committed_flags = [
        record["committed_matches_exact_frozen_selected"]
        for record in records
        if "committed_matches_exact_frozen_selected" in record
    ]
    return {
        "primary": primary,
        "nested_k_sensitivity": nested,
        "nested_primary_prefix_match_rate": (
            float(np.mean(prefix_flags)) if prefix_flags else None
        ),
        "committed_matches_exact_frozen_selected_rate": (
            float(np.mean(committed_flags)) if committed_flags else None
        ),
    }


def _controller_rollout(
    init_state: stage2.SimStateV2,
    initial_eff: np.ndarray,
    schedule: Sequence[ScheduleEvent],
    max_steps: int,
    controller: MatchedInformationController,
) -> tuple[dict[str, float], float]:
    def act_fn(mode, state, ctrl=None):
        if mode == "init":
            return controller
        return ctrl.act(state)

    started = time.perf_counter()
    result = stage2.run_rollout_v2(
        [init_state],
        [initial_eff],
        [schedule],
        max_steps,
        act_fn,
    )
    return result, time.perf_counter() - started


def run_matched_instance(
    init_state: stage2.SimStateV2,
    initial_eff: np.ndarray,
    schedule: Sequence[ScheduleEvent],
    *,
    max_steps: int,
    policy,
    device: torch.device,
    primary_k: int,
    sensitivity_k: Sequence[int],
    controller_seed: int,
    temperature: float,
) -> dict[str, Any]:
    """Run the paired controllers and the online-trajectory diagnostic."""

    online = MatchedInformationController(
        policy,
        device,
        portfolio_k=primary_k,
        selection_k=primary_k,
        seed=controller_seed,
        schedule=schedule,
        scoring_mode="frozen",
        temperature=temperature,
        diagnostic_k_values=(primary_k,),
        nested_k_values=sensitivity_k,
        collect_diagnostics=True,
    )
    clairvoyant = MatchedInformationController(
        policy,
        device,
        portfolio_k=primary_k,
        selection_k=primary_k,
        seed=controller_seed,
        schedule=schedule,
        scoring_mode="realized",
        temperature=temperature,
        diagnostic_k_values=(primary_k,),
        nested_k_values=(),
        collect_diagnostics=False,
    )

    np.random.seed(controller_seed)
    torch.manual_seed(controller_seed)
    online_result, online_wall = _controller_rollout(
        init_state, initial_eff, schedule, max_steps, online
    )

    np.random.seed(controller_seed)
    torch.manual_seed(controller_seed)
    clair_result, clair_wall = _controller_rollout(
        init_state, initial_eff, schedule, max_steps, clairvoyant
    )

    online_delivered = float(online_result["delivered_mean"])
    clair_delivered = float(clair_result["delivered_mean"])
    online_time = float(online_result["time_mean"])
    clair_time = float(clair_result["time_mean"])
    return {
        "online_frozen": {
            "delivered": online_delivered,
            "elapsed_time_sec": online_time,
            "wall_sec_with_shadow_diagnostic": online_wall,
        },
        "clairvoyant_realized": {
            "delivered": clair_delivered,
            "elapsed_time_sec": clair_time,
            "wall_sec": clair_wall,
        },
        "paired_clairvoyant_minus_online": {
            "delivered_delta": clair_delivered - online_delivered,
            "elapsed_time_delta_sec": clair_time - online_time,
        },
        "same_state_diagnostic": {
            "summary": summarize_diagnostics(
                online.diagnostics, sensitivity_k
            ),
            "decisions": online.diagnostics,
        },
    }


@dataclass
class InstancePool:
    lonlats: np.ndarray
    durations: np.ndarray
    bbox: Any


def _resolve_from_root(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _load_instance_pool(path_text: str) -> InstancePool | None:
    if not path_text:
        return None
    pool_path = _resolve_from_root(path_text)
    with np.load(pool_path) as raw:
        lonlats = raw["lonlats"].copy()
        durations = raw["durations"].copy()

    bbox = stage2.DAMASCUS_BBOX
    meta_path = pool_path.parent / "pool_meta.json"
    if meta_path.exists():
        values = json.loads(meta_path.read_text(encoding="utf-8")).get("bbox")
        if values:
            from osrm_client import BBox

            bbox = BBox(
                min_lon=values[0],
                min_lat=values[1],
                max_lon=values[2],
                max_lat=values[3],
            )
    return InstancePool(lonlats=lonlats, durations=durations, bbox=bbox)


def _make_initial_states(
    cfg,
    episode_seed: int,
    pool: InstancePool | None,
) -> list[tuple[stage2.SimStateV2, dict[str, Any]]]:
    states: list[tuple[stage2.SimStateV2, dict[str, Any]]] = []
    if pool is not None:
        if cfg.num_instances > len(pool.lonlats):
            raise ValueError("num_instances exceeds the cached pool size")
        indices = np.random.RandomState(episode_seed).choice(
            len(pool.lonlats), cfg.num_instances, replace=False
        )
        for pool_index in indices:
            n_nodes = pool.lonlats.shape[1]
            visited = np.zeros(n_nodes, dtype=bool)
            visited[0] = True
            states.append(
                (
                    stage2.SimStateV2(
                        coords=stage2.normalize_lonlat(
                            pool.lonlats[pool_index], pool.bbox
                        ).astype(np.float64),
                        base_dist=pool.durations[pool_index].astype(
                            np.float64
                        ).copy(),
                        eff_dist=pool.durations[pool_index].astype(
                            np.float64
                        ).copy(),
                        visited=visited,
                        node_blocked=np.zeros(n_nodes, dtype=np.float32),
                        current_node=0,
                        elapsed_time=0.0,
                        horizon_sec=float(cfg.time_horizon_sec),
                        n_nodes=n_nodes,
                    ),
                    {
                        "source": "cached_pool",
                        "pool_index": int(pool_index),
                    },
                )
            )
        return states

    env = stage2.ResearchEnvV2(cfg)
    env.reset()
    pomo_size = env.pomo_size
    for instance_index in range(cfg.num_instances):
        row = instance_index * pomo_size
        n_nodes = env.n_nodes
        visited = np.zeros(n_nodes, dtype=bool)
        visited[0] = True
        states.append(
            (
                stage2.SimStateV2(
                    coords=env.coords[row].cpu().numpy().astype(np.float64),
                    base_dist=env.base_dist[row].cpu().numpy().astype(np.float64),
                    eff_dist=env.base_dist[row].cpu().numpy().astype(np.float64),
                    visited=visited,
                    node_blocked=np.zeros(n_nodes, dtype=np.float32),
                    current_node=0,
                    elapsed_time=0.0,
                    horizon_sec=float(cfg.time_horizon_sec),
                    n_nodes=n_nodes,
                ),
                {
                    "source": "synthetic",
                    "synthetic_instance_index": instance_index,
                },
            )
        )
    return states


def _episode_aggregate(instances: Sequence[dict[str, Any]]) -> dict[str, Any]:
    online_delivered = [
        item["outcomes"]["online_frozen"]["delivered"] for item in instances
    ]
    clair_delivered = [
        item["outcomes"]["clairvoyant_realized"]["delivered"]
        for item in instances
    ]
    online_time = [
        item["outcomes"]["online_frozen"]["elapsed_time_sec"]
        for item in instances
    ]
    clair_time = [
        item["outcomes"]["clairvoyant_realized"]["elapsed_time_sec"]
        for item in instances
    ]
    return {
        "online_frozen": {
            "delivered_mean": float(np.mean(online_delivered)),
            "elapsed_time_mean_sec": float(np.mean(online_time)),
        },
        "clairvoyant_realized": {
            "delivered_mean": float(np.mean(clair_delivered)),
            "elapsed_time_mean_sec": float(np.mean(clair_time)),
        },
        "paired_clairvoyant_minus_online": {
            "delivered_delta": float(
                np.mean(np.asarray(clair_delivered) - np.asarray(online_delivered))
            ),
            "elapsed_time_delta_sec": float(
                np.mean(np.asarray(clair_time) - np.asarray(online_time))
            ),
        },
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Windows: replace() fails with a sharing violation while a reader (e.g.
    # watch_paper_progress.py) briefly holds the destination open — retry.
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.5 * (attempt + 1))


def _device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Matched frozen-vs-realized candidate-scoring evaluation"
    )
    parser.add_argument(
        "--policy-checkpoint",
        default=str(
            stage2.V62_DIR
            / "checkpoints_research_pomo"
            / "research_best.pt"
        ),
        help="headline zero-shot v1 checkpoint unless explicitly overridden",
    )
    parser.add_argument(
        "--instance-pool",
        default="results/osrm_instance_pool/pool.npz",
        help="cached pool; pass an empty string for synthetic smoke instances",
    )
    parser.add_argument("--n-episodes", type=int, default=40)
    parser.add_argument("--n-nodes", type=int, default=20)
    parser.add_argument("--num-instances", type=int, default=4)
    parser.add_argument(
        "--horizon-hours",
        type=float,
        choices=(4.0, 8.0),
        default=8.0,
    )
    parser.add_argument(
        "--buckets",
        nargs="+",
        choices=("low", "medium", "high"),
        default=["low", "medium", "high"],
    )
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--primary-k", type=int, default=8)
    parser.add_argument(
        "--sensitivity-k",
        nargs="+",
        type=int,
        default=list(DEFAULT_SENSITIVITY_K),
        help="nested same-state prefixes from one max-K portfolio",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="0 uses n_nodes*8+64; lower values enable non-scientific smoke runs",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--output",
        default="results/matched_information_eval.json",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.n_episodes < 1:
        raise ValueError("n_episodes must be positive")
    if args.n_nodes < 2:
        raise ValueError("n_nodes must be at least 2")
    if args.num_instances < 1:
        raise ValueError("num_instances must be positive")
    if args.primary_k < 1:
        raise ValueError("primary_k must be positive")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if args.max_steps < 0:
        raise ValueError("max_steps cannot be negative")
    if any(k < 1 for k in args.sensitivity_k):
        raise ValueError("all sensitivity K values must be positive")
    args.sensitivity_k = sorted(set(args.sensitivity_k))
    args.buckets = list(dict.fromkeys(args.buckets))


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _validate_args(args)

    # This experiment is explicitly the live, frozen-current-matrix controller.
    stage2.POLICY_MATRIX_MODE = "live"
    device = _device_from_arg(args.device)
    checkpoint_path = _resolve_from_root(args.policy_checkpoint)
    policy, checkpoint = stage2.load_policy(str(checkpoint_path), device)
    policy.eval()
    pool = _load_instance_pool(args.instance_pool)
    output_path = _resolve_from_root(args.output)

    max_steps = args.max_steps or (args.n_nodes * 8 + 64)
    output: dict[str, Any] = {
        "schema_version": "matched_information_eval.v1",
        "complete": False,
        "config": {
            **vars(args),
            "max_steps_effective": max_steps,
            "policy_matrix_mode": "live",
            "device_effective": device.type,
        },
        "provenance": {
            "candidate_generator": (
                "scenario_bucket_eval_v2.LookaheadControllerV2.act"
            ),
            "rollout_harness": "scenario_bucket_eval_v2.run_rollout_v2",
            "transition": (
                "scenario_bucket_eval_v2.apply_action_and_advance_v2"
            ),
            "schedule_generator": (
                "scenario_bucket_eval_v2.presample_schedule_v2"
            ),
            "schedule_alignment": (
                "schedule[t] is applied after candidate action t"
            ),
            "primary_intervention": "candidate scoring only",
            "frozen_arm_commitment": (
                "inherited LookaheadControllerV2 internal selection "
                "(bit-identical to recorded look-8, including its float32 "
                "delivered*1e9-elapsed score quantization); shadow "
                "diagnostics use exact lexicographic scoring on a "
                "probe-regenerated identical portfolio and never affect "
                "the committed trajectory"
            ),
            "policy_checkpoint_epoch": checkpoint.get("epoch"),
        },
        "buckets": {},
    }
    _write_json_atomic(output_path, output)

    print(
        f"[MATCHED] device={device} H={args.horizon_hours:g} "
        f"episodes={args.n_episodes} primary_K={args.primary_k}"
    )
    for bucket_index, bucket in enumerate(args.buckets):
        cfg = stage2.apply_bucket_v2(
            stage2.ResearchEnvV2Config(
                n_nodes=args.n_nodes,
                num_instances=args.num_instances,
                device=device.type,
                auto_reset=False,
                use_augmentation=True,
            ),
            bucket,
        )
        cfg.time_horizon_sec = args.horizon_hours * 3600.0
        bucket_output: dict[str, Any] = {
            "bucket_index": bucket_index,
            "episodes": [],
        }
        output["buckets"][bucket] = bucket_output

        for episode_index in range(args.n_episodes):
            episode_seed = args.base_seed + episode_index + 10000 * bucket_index
            np.random.seed(episode_seed)
            torch.manual_seed(episode_seed)
            initial_states = _make_initial_states(cfg, episode_seed, pool)
            instance_outputs: list[dict[str, Any]] = []

            for instance_index, (initial_state, source) in enumerate(initial_states):
                schedule_seed = episode_seed + 999 + instance_index
                initial_eff, schedule = stage2.presample_schedule_v2(
                    initial_state,
                    cfg,
                    max_steps,
                    schedule_seed,
                )
                # Match scenario_bucket_eval_v2.py's v1-lookahead factory:
                # seed_base=episode_seed*1000+600, then counter starts at 1.
                controller_seed = episode_seed * 1000 + 601 + instance_index
                outcomes = run_matched_instance(
                    initial_state,
                    initial_eff,
                    schedule,
                    max_steps=max_steps,
                    policy=policy,
                    device=device,
                    primary_k=args.primary_k,
                    sensitivity_k=args.sensitivity_k,
                    controller_seed=controller_seed,
                    temperature=args.temperature,
                )
                instance_outputs.append(
                    {
                        "instance_index": instance_index,
                        **source,
                        "schedule_seed": schedule_seed,
                        "schedule_digest": schedule_digest(initial_eff, schedule),
                        "candidate_rng_seed": controller_seed,
                        "outcomes": outcomes,
                    }
                )

            episode_output = {
                "episode_index": episode_index,
                "episode_number": episode_index + 1,
                "episode_seed": episode_seed,
                "pairing_key": (
                    f"seed={episode_seed};bucket={bucket};H={args.horizon_hours:g}"
                ),
                "instances": instance_outputs,
                "aggregate": _episode_aggregate(instance_outputs),
            }
            bucket_output["episodes"].append(episode_output)
            _write_json_atomic(output_path, output)
            delta = episode_output["aggregate"][
                "paired_clairvoyant_minus_online"
            ]["delivered_delta"]
            print(
                f"[MATCHED] {bucket} episode {episode_index + 1}/"
                f"{args.n_episodes} delta_delivered={delta:+.3f}"
            )

    output["complete"] = True
    _write_json_atomic(output_path, output)
    print(f"[MATCHED] wrote {output_path}")


if __name__ == "__main__":
    main()
