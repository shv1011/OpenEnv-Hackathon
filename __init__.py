"""
School Admin Timetable Scheduling Environment
=============================================
OpenEnv-compatible multi-step scheduling simulation.
"""

from .environment import SchoolTimetableEnv
from .models import (
    Action,
    ActionType,
    AssignClassAction,
    RescheduleClassAction,
    RemoveAssignmentAction,
    SchoolConfig,
    TimetableEntry,
    Observation,
    StepResult,
    Room,
    Subject,
    Faculty,
    Division,
    TimeSlot,
    RoomType,
)
from .tasks import EasyTask, MediumTask, HardTask, get_task, TASK_REGISTRY
from .constraints import ConstraintsEngine
from .reward import RewardCalculator
from .export import (
    get_faculty_timetable,
    export_faculty_timetable_csv,
    export_all_faculty_timetables_csv,
    export_master_timetable_csv,
    format_timetable_text,
)

__all__ = [
    "SchoolTimetableEnv",
    "Action", "ActionType",
    "AssignClassAction", "RescheduleClassAction", "RemoveAssignmentAction",
    "SchoolConfig", "TimetableEntry", "Observation", "StepResult",
    "Room", "Subject", "Faculty", "Division", "TimeSlot", "RoomType",
    "EasyTask", "MediumTask", "HardTask",
    "get_task", "TASK_REGISTRY",
    "ConstraintsEngine", "RewardCalculator",
    "get_faculty_timetable",
    "export_faculty_timetable_csv",
    "export_all_faculty_timetables_csv",
    "export_master_timetable_csv",
    "format_timetable_text",
]
