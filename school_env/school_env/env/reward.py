"""
Dense Reward Function for the School Timetable Scheduling Environment.
Provides granular, shaped rewards that guide the agent toward valid, efficient schedules.
"""

from __future__ import annotations
from typing import List

from .models import (
    Action,
    ActionType,
    TimetableEntry,
    SchoolConfig,
    RewardBreakdown,
    ConstraintViolation,
)
from .constraints import ConstraintsEngine, ValidationResult


# ─────────────────────────────────────────────
# Reward Constants
# ─────────────────────────────────────────────

VALID_ASSIGNMENT_REWARD = 0.20
INVALID_CONFLICT_PENALTY = -0.30
EFFICIENCY_BONUS = 0.10
REDUNDANCY_PENALTY = -0.10
REMOVE_NEUTRAL = 0.00
RESCHEDULE_BONUS = 0.05      # small bonus for fixing conflicts


class RewardCalculator:
    """
    Calculates step-level and episode-level rewards.

    Step rewards:
        +0.20  valid assignment
        -0.30  invalid / conflict
        +0.10  efficient decision (advances underserved areas)
        -0.10  redundant / wasteful action

    Final score (0.0 – 1.0):
        weighted sum of:
            - timetable completeness   (50%)
            - zero-conflict rate       (30%)
            - resource efficiency      (20%)
    """

    def __init__(self, config: SchoolConfig, engine: ConstraintsEngine):
        self.config = config
        self.engine = engine

        self._subjects = {s.subject_id: s for s in config.subjects}
        self._divisions = {d.division_id: d for d in config.divisions}
        self._faculty = {f.faculty_id: f for f in config.faculty}

    # ─────────────────────────────────────────
    # Step Reward
    # ─────────────────────────────────────────

    def compute_step_reward(
        self,
        action: Action,
        validation: ValidationResult,
        entries_before: List[TimetableEntry],
        entries_after: List[TimetableEntry],
        is_redundant: bool = False,
    ) -> RewardBreakdown:

        base = 0.0
        validity_bonus = 0.0
        efficiency_bonus = 0.0
        conflict_penalty = 0.0
        redundancy_penalty = 0.0

        if action.action_type == ActionType.REMOVE_ASSIGNMENT:
            # Neutral unless it resolves a known bad state
            base = REMOVE_NEUTRAL
            breakdown = RewardBreakdown(
                base_reward=base,
                validity_bonus=0.0,
                efficiency_bonus=0.0,
                conflict_penalty=0.0,
                redundancy_penalty=0.0,
                total=base,
            )
            return breakdown

        if action.action_type == ActionType.RESCHEDULE_CLASS:
            if validation.is_valid:
                base = RESCHEDULE_BONUS
                validity_bonus = VALID_ASSIGNMENT_REWARD * 0.5
            else:
                conflict_penalty = INVALID_CONFLICT_PENALTY
        else:
            # ASSIGN_CLASS
            if not validation.is_valid:
                conflict_penalty = INVALID_CONFLICT_PENALTY
            elif is_redundant:
                redundancy_penalty = REDUNDANCY_PENALTY
            else:
                validity_bonus = VALID_ASSIGNMENT_REWARD
                efficiency_bonus = self._compute_efficiency_bonus(
                    action, entries_before
                )

        total = base + validity_bonus + efficiency_bonus + conflict_penalty + redundancy_penalty
        # Clamp to reasonable range
        total = max(-1.0, min(1.0, total))

        return RewardBreakdown(
            base_reward=base,
            validity_bonus=validity_bonus,
            efficiency_bonus=efficiency_bonus,
            conflict_penalty=conflict_penalty,
            redundancy_penalty=redundancy_penalty,
            total=total,
        )

    def _compute_efficiency_bonus(
        self,
        action: Action,
        entries_before: List[TimetableEntry],
    ) -> float:
        """
        Grant efficiency bonus when:
        - The assigned subject is one of the most under-scheduled for this division
        - Faculty is one of the most underutilized
        """
        if action.action_type != ActionType.ASSIGN_CLASS or not action.assign_class:
            return 0.0

        a = action.assign_class
        unscheduled = self.engine.get_unscheduled_slots(a.division_id, entries_before)
        if not unscheduled:
            return 0.0

        total_remaining = sum(unscheduled.values())
        if total_remaining == 0:
            return 0.0

        # Higher bonus if assigning a subject with more remaining sessions
        subject_remaining = unscheduled.get(a.subject_id, 0)
        urgency_ratio = subject_remaining / total_remaining

        # Bonus scales with urgency (0.0 – 0.10)
        return round(EFFICIENCY_BONUS * urgency_ratio, 4)

    # ─────────────────────────────────────────
    # Episode Final Score
    # ─────────────────────────────────────────

    def compute_final_score(self, entries: List[TimetableEntry]) -> float:
        """
        Returns a normalized score in [0.0, 1.0] based on:
            - Completeness  (50%)
            - Conflict-free (30%)
            - Efficiency    (20%)
        """
        completeness = self._score_completeness(entries)
        conflict_score = self._score_no_conflicts(entries)
        efficiency = self._score_efficiency(entries)

        score = (
            0.50 * completeness
            + 0.30 * conflict_score
            + 0.20 * efficiency
        )
        return round(min(1.0, max(0.0, score)), 4)

    def _score_completeness(self, entries: List[TimetableEntry]) -> float:
        total_required = 0
        total_assigned = 0
        for div in self.config.divisions:
            assigned, required = self.engine.compute_division_completion(
                div.division_id, entries
            )
            total_required += required
            total_assigned += min(assigned, required)  # don't reward over-scheduling

        if total_required == 0:
            return 1.0
        return total_assigned / total_required

    def _score_no_conflicts(self, entries: List[TimetableEntry]) -> float:
        """
        Simulate assigning all entries and count violations.
        Score = 1.0 if zero conflicts, decreases with conflict count.
        """
        seen_faculty_slots: set = set()
        seen_room_slots: set = set()
        seen_division_slots: set = set()
        conflict_count = 0

        for e in entries:
            fs_key = (e.faculty_id, e.slot_id)
            rs_key = (e.room_id, e.slot_id)
            ds_key = (e.division_id, e.slot_id)

            if fs_key in seen_faculty_slots:
                conflict_count += 1
            seen_faculty_slots.add(fs_key)

            if rs_key in seen_room_slots:
                conflict_count += 1
            seen_room_slots.add(rs_key)

            if ds_key in seen_division_slots:
                conflict_count += 1
            seen_division_slots.add(ds_key)

        if not entries:
            return 0.0
        conflict_rate = conflict_count / len(entries)
        return max(0.0, 1.0 - conflict_rate)

    def _score_efficiency(self, entries: List[TimetableEntry]) -> float:
        """
        Efficiency = average faculty utilization within workload limits.
        Penalizes both under-use and (especially) over-load.
        """
        if not self.config.faculty:
            return 1.0

        scores = []
        for f in self.config.faculty:
            current = self.engine.compute_faculty_workload(f.faculty_id, entries)
            utilization = current / f.max_workload
            if utilization > 1.0:
                scores.append(0.0)          # heavily penalize overload
            else:
                scores.append(utilization)  # reward higher utilization

        return sum(scores) / len(scores) if scores else 0.0
