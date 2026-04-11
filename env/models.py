"""
Pydantic models for the School Timetable Scheduling Environment.
Defines all data structures: actions, observations, rewards, and domain entities.
"""

from __future__ import annotations
from typing import Optional, Dict, List, Any
from enum import Enum
from pydantic import BaseModel, Field, model_validator


# ─────────────────────────────────────────────
# Domain Enums
# ─────────────────────────────────────────────

class RoomType(str, Enum):
    CLASSROOM = "classroom"
    LAB = "lab"


class ActionType(str, Enum):
    ASSIGN_CLASS = "assign_class"
    RESCHEDULE_CLASS = "reschedule_class"
    REMOVE_ASSIGNMENT = "remove_assignment"


class TerminationReason(str, Enum):
    COMPLETE = "complete"
    MAX_STEPS = "max_steps"
    TOO_MANY_INVALID = "too_many_invalid"
    MANUAL = "manual"


# ─────────────────────────────────────────────
# Domain Entities
# ─────────────────────────────────────────────

class Room(BaseModel):
    room_id: str
    room_type: RoomType
    capacity: int = 40

    class Config:
        use_enum_values = True


class Subject(BaseModel):
    subject_id: str
    name: str
    requires_lab: bool = False
    sessions_per_week: int = Field(ge=1, le=6)

    class Config:
        use_enum_values = True


class Faculty(BaseModel):
    faculty_id: str
    name: str
    email: str = ""
    subjects_can_teach: List[str]          # list of subject_ids
    available_slots: List[str]              # e.g. ["Mon-1", "Mon-2", ...]
    max_workload: int = Field(ge=1, le=40)  # max sessions per week

    class Config:
        use_enum_values = True


class Division(BaseModel):
    division_id: str
    name: str
    subjects: List[str]                     # list of subject_ids
    student_count: int = 40


class TimeSlot(BaseModel):
    slot_id: str                            # e.g. "Mon-1"
    day: str                                # e.g. "Monday"
    period: int                             # 1-based period number


class SchoolConfig(BaseModel):
    rooms: List[Room]
    subjects: List[Subject]
    faculty: List[Faculty]
    divisions: List[Division]
    time_slots: List[TimeSlot]
    max_steps: int = 200
    max_invalid_actions: int = 20


# ─────────────────────────────────────────────
# Timetable Entries
# ─────────────────────────────────────────────

class TimetableEntry(BaseModel):
    entry_id: str
    division_id: str
    subject_id: str
    faculty_id: str
    room_id: str
    slot_id: str


# ─────────────────────────────────────────────
# Actions
# ─────────────────────────────────────────────

class AssignClassAction(BaseModel):
    division_id: str
    subject_id: str
    faculty_id: str
    room_id: str
    slot_id: str


class RescheduleClassAction(BaseModel):
    entry_id: str                           # existing entry to move
    new_slot_id: str
    new_room_id: Optional[str] = None
    new_faculty_id: Optional[str] = None


class RemoveAssignmentAction(BaseModel):
    entry_id: str


class Action(BaseModel):
    action_type: ActionType
    assign_class: Optional[AssignClassAction] = None
    reschedule_class: Optional[RescheduleClassAction] = None
    remove_assignment: Optional[RemoveAssignmentAction] = None

    class Config:
        use_enum_values = True

    @model_validator(mode="after")
    def _check_payload(self):
        at = self.action_type
        if at == ActionType.ASSIGN_CLASS.value and self.assign_class is None:
            raise ValueError("assign_class payload required for ASSIGN_CLASS action")
        if at == ActionType.RESCHEDULE_CLASS.value and self.reschedule_class is None:
            raise ValueError("reschedule_class payload required for RESCHEDULE_CLASS action")
        if at == ActionType.REMOVE_ASSIGNMENT.value and self.remove_assignment is None:
            raise ValueError("remove_assignment payload required for REMOVE_ASSIGNMENT action")
        return self


# ─────────────────────────────────────────────
# Observation
# ─────────────────────────────────────────────

class ConstraintViolation(BaseModel):
    violation_type: str
    description: str
    involved_entities: Dict[str, str] = Field(default_factory=dict)


class TimetableProgress(BaseModel):
    total_required_sessions: int
    assigned_sessions: int
    completion_percentage: float
    divisions_complete: List[str]
    divisions_pending: List[str]


class ResourceUtilization(BaseModel):
    faculty_utilization: Dict[str, float]   # faculty_id -> % of max_workload used
    room_utilization: Dict[str, float]      # room_id -> % of slots used
    slot_utilization: float                 # % of total slots filled


class Observation(BaseModel):
    step_count: int
    timetable_entries: List[TimetableEntry]
    progress: TimetableProgress
    resource_utilization: ResourceUtilization
    recent_violations: List[ConstraintViolation]
    available_actions_hint: Dict[str, Any]   # hints for agent
    is_terminal: bool
    termination_reason: Optional[str] = None


# ─────────────────────────────────────────────
# Reward
# ─────────────────────────────────────────────

class RewardBreakdown(BaseModel):
    base_reward: float
    validity_bonus: float
    efficiency_bonus: float
    conflict_penalty: float
    redundancy_penalty: float
    total: float


class StepResult(BaseModel):
    observation: Observation
    reward: float
    reward_breakdown: RewardBreakdown
    done: bool
    info: Dict[str, Any]
