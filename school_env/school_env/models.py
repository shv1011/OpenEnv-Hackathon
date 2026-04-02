"""
OpenEnv-compliant Action, Observation, and State models
for the School Timetable Scheduling Environment.
"""
from __future__ import annotations
from typing import Optional, Dict, List, Any
from openenv.core import Action, Observation, State


# ── Action ────────────────────────────────────────────────────

class TimetableAction(Action):
    """
    A single scheduling action.

    action_type: "assign_class" | "reschedule_class" | "remove_assignment"

    For assign_class:
        division_id, subject_id, faculty_id, room_id, slot_id

    For reschedule_class:
        entry_id, new_slot_id, new_room_id (optional), new_faculty_id (optional)

    For remove_assignment:
        entry_id
    """
    action_type: str = ""

    # assign_class fields
    division_id: Optional[str] = None
    subject_id: Optional[str] = None
    faculty_id: Optional[str] = None
    room_id: Optional[str] = None
    slot_id: Optional[str] = None

    # reschedule_class fields
    entry_id: Optional[str] = None
    new_slot_id: Optional[str] = None
    new_room_id: Optional[str] = None
    new_faculty_id: Optional[str] = None


# ── Observation ───────────────────────────────────────────────

class TimetableObservation(Observation):
    """Full environment observation returned after every step."""

    # Inherited: done, reward, metadata

    step_count: int = 0
    completion_percentage: float = 0.0
    assigned_sessions: int = 0
    total_required_sessions: int = 0
    divisions_complete: List[str] = []
    divisions_pending: List[str] = []

    # Current timetable entries
    timetable_entries: List[Dict[str, str]] = []

    # Constraint violations from last action
    violations: List[Dict[str, Any]] = []

    # Hints for the agent
    pending_work: List[Dict[str, Any]] = []
    faculty_status: List[Dict[str, Any]] = []

    # Termination
    termination_reason: Optional[str] = None


# ── State ─────────────────────────────────────────────────────

class TimetableState(State):
    """Episode-level metadata."""

    # Inherited: episode_id, step_count

    task: str = "easy"
    total_required_sessions: int = 0
    completion_percentage: float = 0.0
    total_conflicts: int = 0
    cumulative_reward: float = 0.0
    final_score: Optional[float] = None
