"""
Constraints Engine for the School Timetable Scheduling Environment.
Validates every proposed action against a rich set of scheduling rules.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from .models import (
    AssignClassAction,
    RescheduleClassAction,
    TimetableEntry,
    SchoolConfig,
    RoomType,
    ConstraintViolation,
)


@dataclass
class ValidationResult:
    is_valid: bool
    violations: List[ConstraintViolation]
    warnings: List[str]

    @classmethod
    def ok(cls) -> "ValidationResult":
        return cls(is_valid=True, violations=[], warnings=[])

    @classmethod
    def fail(cls, violations: List[ConstraintViolation]) -> "ValidationResult":
        return cls(is_valid=False, violations=violations, warnings=[])


class ConstraintsEngine:
    """
    Validates scheduling actions against:
    - Teacher double-booking
    - Room double-booking
    - Faculty availability
    - Subject-teacher qualification
    - Lab requirement enforcement
    - Workload limits
    - Redundant assignments
    """

    def __init__(self, config: SchoolConfig):
        self.config = config

        # Build lookup dicts for O(1) access
        self._rooms: Dict[str, any] = {r.room_id: r for r in config.rooms}
        self._subjects: Dict[str, any] = {s.subject_id: s for s in config.subjects}
        self._faculty: Dict[str, any] = {f.faculty_id: f for f in config.faculty}
        self._divisions: Dict[str, any] = {d.division_id: d for d in config.divisions}
        self._slots: Dict[str, any] = {s.slot_id: s for s in config.time_slots}

    # ─────────────────────────────────────────
    # Public Interface
    # ─────────────────────────────────────────

    def validate_assign(
        self,
        action: AssignClassAction,
        existing_entries: List[TimetableEntry],
    ) -> ValidationResult:
        """Full constraint check for an assign_class action."""
        violations: List[ConstraintViolation] = []

        violations += self._check_entities_exist(action)
        if violations:
            return ValidationResult.fail(violations)

        violations += self._check_teacher_double_booking(action, existing_entries)
        violations += self._check_room_double_booking(action, existing_entries)
        violations += self._check_faculty_availability(action)
        violations += self._check_subject_faculty_match(action)
        violations += self._check_lab_requirement(action)
        violations += self._check_workload_limit(action, existing_entries)
        violations += self._check_division_subject_valid(action)
        violations += self._check_duplicate_assignment(action, existing_entries)

        if violations:
            return ValidationResult.fail(violations)
        return ValidationResult.ok()

    def validate_reschedule(
        self,
        action: RescheduleClassAction,
        existing_entries: List[TimetableEntry],
        entry_to_move: TimetableEntry,
    ) -> ValidationResult:
        """Validate a reschedule by constructing the resulting assign action."""
        simulate = AssignClassAction(
            division_id=entry_to_move.division_id,
            subject_id=entry_to_move.subject_id,
            faculty_id=action.new_faculty_id or entry_to_move.faculty_id,
            room_id=action.new_room_id or entry_to_move.room_id,
            slot_id=action.new_slot_id,
        )
        # Exclude the original entry from clash detection
        remaining = [e for e in existing_entries if e.entry_id != entry_to_move.entry_id]
        return self.validate_assign(simulate, remaining)

    def check_is_redundant(
        self,
        action: AssignClassAction,
        existing_entries: List[TimetableEntry],
    ) -> bool:
        """True if exactly the same slot is already assigned for this division+subject."""
        for e in existing_entries:
            if (
                e.division_id == action.division_id
                and e.subject_id == action.subject_id
                and e.slot_id == action.slot_id
            ):
                return True
        return False

    def compute_faculty_workload(
        self, faculty_id: str, entries: List[TimetableEntry]
    ) -> int:
        return sum(1 for e in entries if e.faculty_id == faculty_id)

    def compute_division_completion(
        self, division_id: str, entries: List[TimetableEntry]
    ) -> Tuple[int, int]:
        """Returns (assigned, required) session counts for a division."""
        division = self._divisions.get(division_id)
        if not division:
            return 0, 0

        required = sum(
            self._subjects[s].sessions_per_week
            for s in division.subjects
            if s in self._subjects
        )
        assigned = sum(1 for e in entries if e.division_id == division_id)
        return assigned, required

    def get_unscheduled_slots(
        self, division_id: str, entries: List[TimetableEntry]
    ) -> Dict[str, int]:
        """Returns dict of subject_id -> remaining sessions needed."""
        division = self._divisions.get(division_id)
        if not division:
            return {}

        assigned_counts: Dict[str, int] = {}
        for e in entries:
            if e.division_id == division_id:
                assigned_counts[e.subject_id] = assigned_counts.get(e.subject_id, 0) + 1

        remaining = {}
        for subj_id in division.subjects:
            subj = self._subjects.get(subj_id)
            if subj:
                needed = subj.sessions_per_week - assigned_counts.get(subj_id, 0)
                if needed > 0:
                    remaining[subj_id] = needed
        return remaining

    # ─────────────────────────────────────────
    # Individual Constraint Checks
    # ─────────────────────────────────────────

    def _check_entities_exist(self, action: AssignClassAction) -> List[ConstraintViolation]:
        violations = []

        if action.division_id not in self._divisions:
            violations.append(ConstraintViolation(
                violation_type="UNKNOWN_DIVISION",
                description=f"Division '{action.division_id}' does not exist.",
                involved_entities={"division_id": action.division_id},
            ))

        if action.subject_id not in self._subjects:
            violations.append(ConstraintViolation(
                violation_type="UNKNOWN_SUBJECT",
                description=f"Subject '{action.subject_id}' does not exist.",
                involved_entities={"subject_id": action.subject_id},
            ))

        if action.faculty_id not in self._faculty:
            violations.append(ConstraintViolation(
                violation_type="UNKNOWN_FACULTY",
                description=f"Faculty '{action.faculty_id}' does not exist.",
                involved_entities={"faculty_id": action.faculty_id},
            ))

        if action.room_id not in self._rooms:
            violations.append(ConstraintViolation(
                violation_type="UNKNOWN_ROOM",
                description=f"Room '{action.room_id}' does not exist.",
                involved_entities={"room_id": action.room_id},
            ))

        if action.slot_id not in self._slots:
            violations.append(ConstraintViolation(
                violation_type="UNKNOWN_SLOT",
                description=f"Time slot '{action.slot_id}' does not exist.",
                involved_entities={"slot_id": action.slot_id},
            ))

        return violations

    def _check_teacher_double_booking(
        self, action: AssignClassAction, entries: List[TimetableEntry]
    ) -> List[ConstraintViolation]:
        for e in entries:
            if e.faculty_id == action.faculty_id and e.slot_id == action.slot_id:
                faculty = self._faculty[action.faculty_id]
                return [ConstraintViolation(
                    violation_type="TEACHER_DOUBLE_BOOKING",
                    description=(
                        f"Faculty '{faculty.name}' is already teaching "
                        f"in slot '{action.slot_id}' (entry: {e.entry_id})."
                    ),
                    involved_entities={
                        "faculty_id": action.faculty_id,
                        "slot_id": action.slot_id,
                        "conflicting_entry": e.entry_id,
                    },
                )]
        return []

    def _check_room_double_booking(
        self, action: AssignClassAction, entries: List[TimetableEntry]
    ) -> List[ConstraintViolation]:
        for e in entries:
            if e.room_id == action.room_id and e.slot_id == action.slot_id:
                return [ConstraintViolation(
                    violation_type="ROOM_DOUBLE_BOOKING",
                    description=(
                        f"Room '{action.room_id}' is already occupied "
                        f"in slot '{action.slot_id}' (entry: {e.entry_id})."
                    ),
                    involved_entities={
                        "room_id": action.room_id,
                        "slot_id": action.slot_id,
                        "conflicting_entry": e.entry_id,
                    },
                )]
        return []

    def _check_faculty_availability(
        self, action: AssignClassAction
    ) -> List[ConstraintViolation]:
        faculty = self._faculty[action.faculty_id]
        if action.slot_id not in faculty.available_slots:
            return [ConstraintViolation(
                violation_type="FACULTY_NOT_AVAILABLE",
                description=(
                    f"Faculty '{faculty.name}' is not available in slot '{action.slot_id}'. "
                    f"Available: {faculty.available_slots}"
                ),
                involved_entities={
                    "faculty_id": action.faculty_id,
                    "slot_id": action.slot_id,
                },
            )]
        return []

    def _check_subject_faculty_match(
        self, action: AssignClassAction
    ) -> List[ConstraintViolation]:
        faculty = self._faculty[action.faculty_id]
        if action.subject_id not in faculty.subjects_can_teach:
            subject = self._subjects[action.subject_id]
            return [ConstraintViolation(
                violation_type="SUBJECT_FACULTY_MISMATCH",
                description=(
                    f"Faculty '{faculty.name}' is not qualified to teach "
                    f"'{subject.name}'. Qualified subjects: {faculty.subjects_can_teach}"
                ),
                involved_entities={
                    "faculty_id": action.faculty_id,
                    "subject_id": action.subject_id,
                },
            )]
        return []

    def _check_lab_requirement(
        self, action: AssignClassAction
    ) -> List[ConstraintViolation]:
        subject = self._subjects[action.subject_id]
        room = self._rooms[action.room_id]
        if subject.requires_lab and room.room_type != RoomType.LAB.value:
            return [ConstraintViolation(
                violation_type="LAB_REQUIRED",
                description=(
                    f"Subject '{subject.name}' requires a lab, "
                    f"but room '{action.room_id}' is a {room.room_type}."
                ),
                involved_entities={
                    "subject_id": action.subject_id,
                    "room_id": action.room_id,
                },
            )]
        # Prevent theory classes in labs (optional but realistic)
        if not subject.requires_lab and room.room_type == RoomType.LAB.value:
            return [ConstraintViolation(
                violation_type="LAB_MISUSE",
                description=(
                    f"Subject '{subject.name}' does not need a lab. "
                    f"Room '{action.room_id}' is a lab and should be reserved for lab subjects."
                ),
                involved_entities={
                    "subject_id": action.subject_id,
                    "room_id": action.room_id,
                },
            )]
        return []

    def _check_workload_limit(
        self, action: AssignClassAction, entries: List[TimetableEntry]
    ) -> List[ConstraintViolation]:
        faculty = self._faculty[action.faculty_id]
        current_load = self.compute_faculty_workload(action.faculty_id, entries)
        if current_load >= faculty.max_workload:
            return [ConstraintViolation(
                violation_type="WORKLOAD_EXCEEDED",
                description=(
                    f"Faculty '{faculty.name}' has reached max workload "
                    f"({current_load}/{faculty.max_workload} sessions)."
                ),
                involved_entities={
                    "faculty_id": action.faculty_id,
                    "current_load": str(current_load),
                    "max_workload": str(faculty.max_workload),
                },
            )]
        return []

    def _check_division_subject_valid(
        self, action: AssignClassAction
    ) -> List[ConstraintViolation]:
        division = self._divisions[action.division_id]
        if action.subject_id not in division.subjects:
            subject = self._subjects.get(action.subject_id)
            name = subject.name if subject else action.subject_id
            return [ConstraintViolation(
                violation_type="SUBJECT_NOT_IN_DIVISION",
                description=(
                    f"Subject '{name}' is not part of division '{division.name}'s curriculum."
                ),
                involved_entities={
                    "division_id": action.division_id,
                    "subject_id": action.subject_id,
                },
            )]
        return []

    def _check_duplicate_assignment(
        self, action: AssignClassAction, entries: List[TimetableEntry]
    ) -> List[ConstraintViolation]:
        """Prevent assigning the same division+subject to the same slot twice."""
        for e in entries:
            if (
                e.division_id == action.division_id
                and e.slot_id == action.slot_id
            ):
                return [ConstraintViolation(
                    violation_type="DIVISION_SLOT_CONFLICT",
                    description=(
                        f"Division '{action.division_id}' already has a class "
                        f"scheduled in slot '{action.slot_id}'."
                    ),
                    involved_entities={
                        "division_id": action.division_id,
                        "slot_id": action.slot_id,
                        "conflicting_entry": e.entry_id,
                    },
                )]
        return []
