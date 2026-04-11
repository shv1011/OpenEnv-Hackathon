"""
School Admin Timetable Scheduling Environment
=============================================
OpenEnv-compatible environment implementing reset() / step() / state().
"""

from __future__ import annotations
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple

from .models import (
    Action,
    ActionType,
    AssignClassAction,
    RescheduleClassAction,
    RemoveAssignmentAction,
    SchoolConfig,
    TimetableEntry,
    Observation,
    TimetableProgress,
    ResourceUtilization,
    StepResult,
    TerminationReason,
    ConstraintViolation,
)
from .constraints import ConstraintsEngine
from .reward import RewardCalculator

logger = logging.getLogger("school_env")


class SchoolTimetableEnv:
    """
    Simulates a school administrator constructing a weekly timetable.

    Interface
    ---------
    reset()         → Observation
    step(action)    → StepResult
    state()         → Observation

    The agent receives a rich observation at every step, including:
    - current timetable entries
    - progress toward completion
    - resource utilization
    - recent constraint violations
    - hints about next valid moves
    """

    def __init__(self, config: SchoolConfig, debug: bool = False):
        self.config = config
        self.debug = debug

        if debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        self._engine = ConstraintsEngine(config)
        self._reward_calc = RewardCalculator(config, self._engine)

        # Runtime state (initialized on reset)
        self._entries: List[TimetableEntry] = []
        self._step_count: int = 0
        self._invalid_action_count: int = 0
        self._recent_violations: List[ConstraintViolation] = []
        self._done: bool = False
        self._termination_reason: Optional[str] = None
        self._episode_rewards: List[float] = []

        # Lookup maps
        self._subjects = {s.subject_id: s for s in config.subjects}
        self._faculty = {f.faculty_id: f for f in config.faculty}
        self._rooms = {r.room_id: r for r in config.rooms}
        self._divisions = {d.division_id: d for d in config.divisions}
        self._slots = {s.slot_id: s for s in config.time_slots}

    # ─────────────────────────────────────────
    # OpenEnv Interface
    # ─────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset environment to initial empty state."""
        self._entries = []
        self._step_count = 0
        self._invalid_action_count = 0
        self._recent_violations = []
        self._done = False
        self._termination_reason = None
        self._episode_rewards = []

        logger.info("[START] Timetable environment reset. Episode begins.")
        obs = self._build_observation()
        return obs

    def step(self, action: Action) -> StepResult:
        """
        Execute one action and return (observation, reward, done, info).

        Actions:
            assign_class      → validate & add entry
            reschedule_class  → validate & move existing entry
            remove_assignment → remove entry
        """
        if self._done:
            raise RuntimeError("Episode is over. Call reset() to start a new episode.")

        self._step_count += 1
        logger.debug(f"[STEP {self._step_count}] Processing action: {action.action_type}")

        entries_before = list(self._entries)
        validation = None
        is_redundant = False

        # ── Route by action type ──────────────────
        if action.action_type == ActionType.ASSIGN_CLASS:
            result_tuple = self._handle_assign(action.assign_class, entries_before)
        elif action.action_type == ActionType.RESCHEDULE_CLASS:
            result_tuple = self._handle_reschedule(action.reschedule_class)
        elif action.action_type == ActionType.REMOVE_ASSIGNMENT:
            result_tuple = self._handle_remove(action.remove_assignment)
        else:
            raise ValueError(f"Unknown action type: {action.action_type}")

        validation, is_redundant = result_tuple

        # ── Compute reward ────────────────────────
        reward_breakdown = self._reward_calc.compute_step_reward(
            action=action,
            validation=validation,
            entries_before=entries_before,
            entries_after=self._entries,
            is_redundant=is_redundant,
        )
        reward = reward_breakdown.total
        self._episode_rewards.append(reward)

        if validation and not validation.is_valid:
            self._invalid_action_count += 1
            self._recent_violations = validation.violations
        else:
            self._recent_violations = []

        # ── Check termination ─────────────────────
        self._check_termination()

        obs = self._build_observation()

        info = self._build_info(
            action=action,
            validation=validation,
            reward_breakdown=reward_breakdown,
            is_redundant=is_redundant,
        )

        if self._done:
            final_score = self._reward_calc.compute_final_score(self._entries)
            info["final_score"] = final_score
            info["episode_cumulative_reward"] = sum(self._episode_rewards)
            logger.info(
                f"[END] Episode finished. Reason: {self._termination_reason}. "
                f"Final score: {final_score:.4f}"
            )

        return StepResult(
            observation=obs,
            reward=reward,
            reward_breakdown=reward_breakdown,
            done=self._done,
            info=info,
        )

    def state(self) -> Observation:
        """Return current observation without advancing the episode."""
        return self._build_observation()

    # ─────────────────────────────────────────
    # Action Handlers
    # ─────────────────────────────────────────

    def _handle_assign(
        self,
        assign_action: AssignClassAction,
        entries_before: List[TimetableEntry],
    ) -> Tuple[Any, bool]:
        from .constraints import ValidationResult

        validation = self._engine.validate_assign(assign_action, entries_before)
        is_redundant = self._engine.check_is_redundant(assign_action, entries_before)

        if is_redundant:
            logger.debug(f"  Redundant assignment detected for {assign_action.subject_id}.")
            dummy_fail = ValidationResult.fail([
                ConstraintViolation(
                    violation_type="REDUNDANT_ASSIGNMENT",
                    description="This exact assignment already exists.",
                    involved_entities={},
                )
            ])
            return dummy_fail, True

        if validation.is_valid:
            entry = TimetableEntry(
                entry_id=str(uuid.uuid4())[:8],
                division_id=assign_action.division_id,
                subject_id=assign_action.subject_id,
                faculty_id=assign_action.faculty_id,
                room_id=assign_action.room_id,
                slot_id=assign_action.slot_id,
            )
            self._entries.append(entry)
            logger.debug(
                f"  ✓ Assigned {assign_action.subject_id} → "
                f"{assign_action.division_id} @ {assign_action.slot_id}"
            )
        else:
            logger.debug(
                f"  ✗ Invalid: {[v.violation_type for v in validation.violations]}"
            )

        return validation, False

    def _handle_reschedule(
        self,
        reschedule_action: RescheduleClassAction,
    ) -> Tuple[Any, bool]:
        from .constraints import ValidationResult

        entry = next(
            (e for e in self._entries if e.entry_id == reschedule_action.entry_id),
            None,
        )
        if entry is None:
            fail = ValidationResult.fail([
                ConstraintViolation(
                    violation_type="ENTRY_NOT_FOUND",
                    description=f"No entry with id '{reschedule_action.entry_id}'.",
                )
            ])
            return fail, False

        validation = self._engine.validate_reschedule(
            reschedule_action, self._entries, entry
        )

        if validation.is_valid:
            self._entries.remove(entry)
            updated = TimetableEntry(
                entry_id=entry.entry_id,
                division_id=entry.division_id,
                subject_id=entry.subject_id,
                faculty_id=reschedule_action.new_faculty_id or entry.faculty_id,
                room_id=reschedule_action.new_room_id or entry.room_id,
                slot_id=reschedule_action.new_slot_id,
            )
            self._entries.append(updated)
            logger.debug(
                f"  ↺ Rescheduled entry {entry.entry_id} → slot {reschedule_action.new_slot_id}"
            )

        return validation, False

    def _handle_remove(
        self,
        remove_action: RemoveAssignmentAction,
    ) -> Tuple[Any, bool]:
        from .constraints import ValidationResult

        entry = next(
            (e for e in self._entries if e.entry_id == remove_action.entry_id),
            None,
        )
        if entry is None:
            fail = ValidationResult.fail([
                ConstraintViolation(
                    violation_type="ENTRY_NOT_FOUND",
                    description=f"No entry with id '{remove_action.entry_id}'.",
                )
            ])
            return fail, False

        self._entries.remove(entry)
        logger.debug(f"  ✗ Removed entry {entry.entry_id}")
        return ValidationResult.ok(), False

    # ─────────────────────────────────────────
    # Termination Check
    # ─────────────────────────────────────────

    def _check_termination(self):
        # Max invalid actions
        if self._invalid_action_count >= self.config.max_invalid_actions:
            self._done = True
            self._termination_reason = TerminationReason.TOO_MANY_INVALID.value
            return

        # Max steps
        if self._step_count >= self.config.max_steps:
            self._done = True
            self._termination_reason = TerminationReason.MAX_STEPS.value
            return

        # Timetable complete
        if self._is_timetable_complete():
            self._done = True
            self._termination_reason = TerminationReason.COMPLETE.value

    def _is_timetable_complete(self) -> bool:
        for div in self.config.divisions:
            assigned, required = self._engine.compute_division_completion(
                div.division_id, self._entries
            )
            if assigned < required:
                return False
        return True

    # ─────────────────────────────────────────
    # Observation Builder
    # ─────────────────────────────────────────

    def _build_observation(self) -> Observation:
        progress = self._build_progress()
        utilization = self._build_utilization()
        hints = self._build_action_hints()

        return Observation(
            step_count=self._step_count,
            timetable_entries=list(self._entries),
            progress=progress,
            resource_utilization=utilization,
            recent_violations=list(self._recent_violations),
            available_actions_hint=hints,
            is_terminal=self._done,
            termination_reason=self._termination_reason,
        )

    def _build_progress(self) -> TimetableProgress:
        total_required = 0
        total_assigned = 0
        complete = []
        pending = []

        for div in self.config.divisions:
            assigned, required = self._engine.compute_division_completion(
                div.division_id, self._entries
            )
            total_required += required
            total_assigned += assigned
            if assigned >= required:
                complete.append(div.division_id)
            else:
                pending.append(div.division_id)

        pct = (total_assigned / total_required * 100) if total_required > 0 else 0.0
        return TimetableProgress(
            total_required_sessions=total_required,
            assigned_sessions=total_assigned,
            completion_percentage=round(pct, 2),
            divisions_complete=complete,
            divisions_pending=pending,
        )

    def _build_utilization(self) -> ResourceUtilization:
        faculty_util: Dict[str, float] = {}
        for f in self.config.faculty:
            load = self._engine.compute_faculty_workload(f.faculty_id, self._entries)
            faculty_util[f.faculty_id] = round(load / f.max_workload, 3)

        room_util: Dict[str, float] = {}
        total_slots = len(self.config.time_slots)
        for r in self.config.rooms:
            used = sum(1 for e in self._entries if e.room_id == r.room_id)
            room_util[r.room_id] = round(used / total_slots, 3) if total_slots else 0.0

        total_capacity = total_slots * len(self.config.rooms)
        total_used = len(self._entries)
        slot_util = round(total_used / total_capacity, 3) if total_capacity else 0.0

        return ResourceUtilization(
            faculty_utilization=faculty_util,
            room_utilization=room_util,
            slot_utilization=slot_util,
        )

    def _build_action_hints(self) -> Dict[str, Any]:
        """Provide structured hints to help the agent identify next valid actions."""
        hints: Dict[str, Any] = {}
        pending_work: List[Dict] = []

        for div in self.config.divisions:
            remaining = self._engine.get_unscheduled_slots(div.division_id, self._entries)
            if remaining:
                pending_work.append({
                    "division_id": div.division_id,
                    "division_name": div.name,
                    "subjects_needed": remaining,
                })

        hints["pending_work"] = pending_work

        # Faculty availability summary
        faculty_summary: List[Dict] = []
        for f in self.config.faculty:
            load = self._engine.compute_faculty_workload(f.faculty_id, self._entries)
            occupied_slots = {
                e.slot_id for e in self._entries if e.faculty_id == f.faculty_id
            }
            free_slots = [
                s for s in f.available_slots if s not in occupied_slots
            ]
            faculty_summary.append({
                "faculty_id": f.faculty_id,
                "name": f.name,
                "free_slots": free_slots,
                "current_load": load,
                "max_workload": f.max_workload,
                "can_teach": f.subjects_can_teach,
            })
        hints["faculty_status"] = faculty_summary

        # Room availability
        room_summary: List[Dict] = []
        for r in self.config.rooms:
            occupied = {e.slot_id for e in self._entries if e.room_id == r.room_id}
            all_slots = {s.slot_id for s in self.config.time_slots}
            free = list(all_slots - occupied)
            room_summary.append({
                "room_id": r.room_id,
                "room_type": r.room_type,
                "free_slots": sorted(free),
            })
        hints["room_status"] = room_summary

        return hints

    def _build_info(
        self,
        action: Action,
        validation: Any,
        reward_breakdown: Any,
        is_redundant: bool,
    ) -> Dict[str, Any]:
        return {
            "action_type": action.action_type,
            "valid": validation.is_valid if validation else True,
            "redundant": is_redundant,
            "violations": [
                v.dict() for v in (validation.violations if validation else [])
            ],
            "reward_breakdown": reward_breakdown.dict(),
            "step_count": self._step_count,
            "invalid_action_count": self._invalid_action_count,
            "total_entries": len(self._entries),
        }

    # ─────────────────────────────────────────
    # Utility / Debug
    # ─────────────────────────────────────────

    def get_entries(self) -> List[TimetableEntry]:
        return list(self._entries)

    def get_conflict_report(self) -> Dict[str, Any]:
        """Return a detailed report of all current conflicts in the timetable."""
        faculty_slots: Dict[str, List[str]] = {}
        room_slots: Dict[str, List[str]] = {}
        div_slots: Dict[str, List[str]] = {}

        conflicts = []

        for e in self._entries:
            fs = (e.faculty_id, e.slot_id)
            if fs in faculty_slots:
                conflicts.append({
                    "type": "TEACHER_DOUBLE_BOOKING",
                    "entities": {"faculty_id": e.faculty_id, "slot_id": e.slot_id},
                })
            faculty_slots.setdefault(e.faculty_id + e.slot_id, []).append(e.entry_id)

            rs = (e.room_id, e.slot_id)
            if rs in room_slots:
                conflicts.append({
                    "type": "ROOM_DOUBLE_BOOKING",
                    "entities": {"room_id": e.room_id, "slot_id": e.slot_id},
                })
            room_slots.setdefault(e.room_id + e.slot_id, []).append(e.entry_id)

        return {
            "total_conflicts": len(conflicts),
            "conflicts": conflicts,
        }

    def get_summary_metrics(self) -> Dict[str, Any]:
        progress = self._build_progress()
        utilization = self._build_utilization()
        conflict_report = self.get_conflict_report()
        final_score = self._reward_calc.compute_final_score(self._entries)

        return {
            "step_count": self._step_count,
            "total_entries": len(self._entries),
            "completion_percentage": progress.completion_percentage,
            "divisions_complete": progress.divisions_complete,
            "divisions_pending": progress.divisions_pending,
            "total_conflicts": conflict_report["total_conflicts"],
            "faculty_utilization": utilization.faculty_utilization,
            "room_utilization": utilization.room_utilization,
            "final_score": final_score,
            "cumulative_reward": round(sum(self._episode_rewards), 4),
        }
