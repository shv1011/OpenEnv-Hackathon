"""
OpenEnv-compliant Environment for School Timetable Scheduling.
Wraps the existing SchoolTimetableEnv with the openenv.core.Environment interface.
"""
from __future__ import annotations
import sys, os
from typing import Optional

# Ensure env package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openenv.core import Environment

from models import TimetableAction, TimetableObservation, TimetableState
from env import (
    SchoolTimetableEnv as _InnerEnv,
    Action as _Action,
    ActionType,
    AssignClassAction,
    RescheduleClassAction,
    RemoveAssignmentAction,
    get_task,
)


class SchoolTimetableEnvironment(
    Environment[TimetableAction, TimetableObservation, TimetableState]
):
    """
    School Timetable Scheduling — OpenEnv Environment.

    The agent must assign all required class sessions across divisions,
    respecting faculty availability, room types, workload limits, and
    no-double-booking constraints.

    Tasks: easy | medium | hard  (pass via reset(task=...))
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._env: Optional[_InnerEnv] = None
        self._task_id: str = "easy"
        self._task_cls = None
        self._cumulative_reward: float = 0.0

    # ── OpenEnv Interface ─────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: str = "easy",
        **kwargs,
    ) -> TimetableObservation:
        self._task_id = task if task in ("easy", "medium", "hard") else "easy"
        self._task_cls = get_task(self._task_id)
        config = self._task_cls.get_config()
        self._env = _InnerEnv(config)
        self._cumulative_reward = 0.0
        obs = self._env.reset()
        return self._to_obs(obs, reward=0.0, done=False)

    def step(
        self,
        action: TimetableAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> TimetableObservation:
        # Auto-reset for stateless HTTP usage (create_app calls step on fresh instances)
        if self._env is None:
            self.reset()

        inner_action = self._convert_action(action)
        result = self._env.step(inner_action)
        self._cumulative_reward += result.reward

        final_score = None
        if result.done:
            final_score = self._task_cls.grade(self._env.get_entries())

        return self._to_obs(
            result.observation,
            reward=result.reward,
            done=result.done,
            violations=result.info.get("violations", []),
            final_score=final_score,
        )

    @property
    def state(self) -> TimetableState:
        if self._env is None:
            return TimetableState(task=self._task_id)
        m = self._env.get_summary_metrics()
        return TimetableState(
            task=self._task_id,
            step_count=m["step_count"],
            total_required_sessions=m["total_entries"],
            completion_percentage=m["completion_percentage"],
            total_conflicts=m["total_conflicts"],
            cumulative_reward=round(self._cumulative_reward, 4),
            final_score=m.get("final_score"),
        )

    def close(self):
        self._env = None

    # ── Helpers ───────────────────────────────────────────────

    def _convert_action(self, action: TimetableAction) -> _Action:
        atype = action.action_type

        if atype == "assign_class":
            return _Action(
                action_type=ActionType.ASSIGN_CLASS,
                assign_class=AssignClassAction(
                    division_id=action.division_id,
                    subject_id=action.subject_id,
                    faculty_id=action.faculty_id,
                    room_id=action.room_id,
                    slot_id=action.slot_id,
                ),
            )
        elif atype == "reschedule_class":
            return _Action(
                action_type=ActionType.RESCHEDULE_CLASS,
                reschedule_class=RescheduleClassAction(
                    entry_id=action.entry_id,
                    new_slot_id=action.new_slot_id,
                    new_room_id=action.new_room_id,
                    new_faculty_id=action.new_faculty_id,
                ),
            )
        elif atype == "remove_assignment":
            return _Action(
                action_type=ActionType.REMOVE_ASSIGNMENT,
                remove_assignment=RemoveAssignmentAction(
                    entry_id=action.entry_id,
                ),
            )
        else:
            raise ValueError(f"Unknown action_type: {atype}")

    def _to_obs(
        self,
        obs,
        reward: float,
        done: bool,
        violations=None,
        final_score=None,
    ) -> TimetableObservation:
        hints = obs.available_actions_hint
        entries = [e.model_dump() for e in obs.timetable_entries]

        return TimetableObservation(
            done=done,
            reward=reward if final_score is None else final_score,
            step_count=obs.step_count,
            completion_percentage=obs.progress.completion_percentage,
            assigned_sessions=obs.progress.assigned_sessions,
            total_required_sessions=obs.progress.total_required_sessions,
            divisions_complete=obs.progress.divisions_complete,
            divisions_pending=obs.progress.divisions_pending,
            timetable_entries=entries,
            violations=[
            {k: str(v) for k, v in (viol.items() if isinstance(viol, dict) else viol.model_dump().items())}
            for viol in (violations or [])
        ],
            pending_work=hints.get("pending_work", []),
            faculty_status=hints.get("faculty_status", []),
            termination_reason=obs.termination_reason,
        )
