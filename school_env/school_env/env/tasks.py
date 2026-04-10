"""
Task Definitions for the School Timetable Scheduling Environment.

Provides three tasks:
    🟢 EASY    – Small dataset, straightforward scheduling
    🟡 MEDIUM  – Limited availability, requires conflict resolution
    🔴 HARD    – Many constraints, optimization required

Each task has:
    - get_config()          → SchoolConfig
    - grade(entries)        → float [0.0–1.0]
"""

from __future__ import annotations
import random
from typing import List, Dict

from .models import (
    SchoolConfig,
    Room,
    Subject,
    Faculty,
    Division,
    TimeSlot,
    TimetableEntry,
    RoomType,
)
from .constraints import ConstraintsEngine
from .reward import RewardCalculator


# ═══════════════════════════════════════════════════════════════
# Helper: Build standard Mon–Fri time slots
# ═══════════════════════════════════════════════════════════════

def _make_slots(days: List[str], periods: int) -> List[TimeSlot]:
    slots = []
    for day in days:
        for p in range(1, periods + 1):
            slot_id = f"{day[:3]}-{p}"
            slots.append(TimeSlot(slot_id=slot_id, day=day, period=p))
    return slots


DAYS_FULL = ["Mon", "Tue", "Wed", "Thu", "Fri"]


# ═══════════════════════════════════════════════════════════════
# 🟢 TASK 1 — EASY
# ═══════════════════════════════════════════════════════════════

class EasyTask:
    """
    Single division, 4 subjects, 3 faculty, 2 classrooms.
    No lab subjects. Simple availability. Should be solvable in ~15 steps.
    """

    TASK_ID = "easy"
    DESCRIPTION = "Single-division scheduling with no lab constraints."

    @staticmethod
    def get_config() -> SchoolConfig:
        rooms = [
            Room(room_id="CR101", room_type=RoomType.CLASSROOM, capacity=40),
            Room(room_id="CR102", room_type=RoomType.CLASSROOM, capacity=40),
        ]

        subjects = [
            Subject(subject_id="MATH", name="Mathematics", requires_lab=False, sessions_per_week=3),
            Subject(subject_id="ENG", name="English", requires_lab=False, sessions_per_week=2),
            Subject(subject_id="SCI", name="Science", requires_lab=False, sessions_per_week=2),
            Subject(subject_id="HIST", name="History", requires_lab=False, sessions_per_week=2),
        ]

        all_slots = [f"{d}-{p}" for d in DAYS_FULL for p in range(1, 5)]

        faculty = [
            Faculty(
                faculty_id="F001",
                name="Dr. Sharma",
                email="sharma@school.edu",
                subjects_can_teach=["MATH"],
                available_slots=all_slots,
                max_workload=10,
            ),
            Faculty(
                faculty_id="F002",
                name="Prof. Mehta",
                email="mehta@school.edu",
                subjects_can_teach=["ENG", "HIST"],
                available_slots=all_slots,
                max_workload=10,
            ),
            Faculty(
                faculty_id="F003",
                name="Ms. Patel",
                email="patel@school.edu",
                subjects_can_teach=["SCI"],
                available_slots=all_slots,
                max_workload=10,
            ),
        ]

        divisions = [
            Division(
                division_id="Sem1-A",
                name="Semester 1 – Division A",
                subjects=["MATH", "ENG", "SCI", "HIST"],
                student_count=35,
            )
        ]

        time_slots = _make_slots(DAYS_FULL, 4)

        return SchoolConfig(
            rooms=rooms,
            subjects=subjects,
            faculty=faculty,
            divisions=divisions,
            time_slots=time_slots,
            max_steps=60,
            max_invalid_actions=15,
        )

    @staticmethod
    def grade(entries: List[TimetableEntry]) -> float:
        config = EasyTask.get_config()
        engine = ConstraintsEngine(config)
        calc = RewardCalculator(config, engine)
        score = calc.compute_final_score(entries)
        return round(max(0.01, min(0.99, score)), 4)


# ═══════════════════════════════════════════════════════════════
# 🟡 TASK 2 — MEDIUM
# ═══════════════════════════════════════════════════════════════

class MediumTask:
    """
    Two divisions, 6 subjects (1 lab), 5 faculty with limited availability.
    Agent must resolve conflicts and schedule around constraints.
    """

    TASK_ID = "medium"
    DESCRIPTION = "Two divisions, lab subject, limited faculty availability."

    @staticmethod
    def get_config() -> SchoolConfig:
        rooms = [
            Room(room_id="CR101", room_type=RoomType.CLASSROOM, capacity=45),
            Room(room_id="CR102", room_type=RoomType.CLASSROOM, capacity=45),
            Room(room_id="CR103", room_type=RoomType.CLASSROOM, capacity=45),
            Room(room_id="Lab1",  room_type=RoomType.LAB,       capacity=30),
        ]

        subjects = [
            Subject(subject_id="MATH",  name="Mathematics",      requires_lab=False, sessions_per_week=3),
            Subject(subject_id="PHY",   name="Physics",          requires_lab=False, sessions_per_week=2),
            Subject(subject_id="CHEM",  name="Chemistry",        requires_lab=False, sessions_per_week=2),
            Subject(subject_id="CHEMLAB", name="Chemistry Lab",  requires_lab=True,  sessions_per_week=1),
            Subject(subject_id="ENG",   name="English",          requires_lab=False, sessions_per_week=2),
            Subject(subject_id="CS",    name="Computer Science", requires_lab=False, sessions_per_week=2),
        ]

        morning = [f"{d}-{p}" for d in DAYS_FULL for p in range(1, 4)]
        afternoon = [f"{d}-{p}" for d in DAYS_FULL for p in range(3, 6)]
        all_slots = [f"{d}-{p}" for d in DAYS_FULL for p in range(1, 6)]
        mwf = [f"{d}-{p}" for d in ["Mon", "Wed", "Fri"] for p in range(1, 6)]
        tth = [f"{d}-{p}" for d in ["Tue", "Thu"] for p in range(1, 6)]

        faculty = [
            Faculty(
                faculty_id="F001",
                name="Dr. Sharma",
                email="sharma@school.edu",
                subjects_can_teach=["MATH"],
                available_slots=all_slots,
                max_workload=8,
            ),
            Faculty(
                faculty_id="F002",
                name="Prof. Rao",
                email="rao@school.edu",
                subjects_can_teach=["PHY"],
                available_slots=morning,
                max_workload=6,
            ),
            Faculty(
                faculty_id="F003",
                name="Dr. Kapoor",
                email="kapoor@school.edu",
                subjects_can_teach=["CHEM", "CHEMLAB"],
                available_slots=afternoon,
                max_workload=7,
            ),
            Faculty(
                faculty_id="F004",
                name="Ms. Singh",
                email="singh@school.edu",
                subjects_can_teach=["ENG"],
                available_slots=mwf,
                max_workload=6,
            ),
            Faculty(
                faculty_id="F005",
                name="Mr. Verma",
                email="verma@school.edu",
                subjects_can_teach=["CS", "MATH"],
                available_slots=tth,
                max_workload=6,
            ),
        ]

        divisions = [
            Division(
                division_id="Sem2-A",
                name="Semester 2 – Division A",
                subjects=["MATH", "PHY", "CHEM", "CHEMLAB", "ENG"],
                student_count=40,
            ),
            Division(
                division_id="Sem2-B",
                name="Semester 2 – Division B",
                subjects=["MATH", "CS", "ENG", "PHY"],
                student_count=38,
            ),
        ]

        time_slots = _make_slots(DAYS_FULL, 5)

        return SchoolConfig(
            rooms=rooms,
            subjects=subjects,
            faculty=faculty,
            divisions=divisions,
            time_slots=time_slots,
            max_steps=150,
            max_invalid_actions=20,
        )

    @staticmethod
    def grade(entries: List[TimetableEntry]) -> float:
        config = MediumTask.get_config()
        engine = ConstraintsEngine(config)
        calc = RewardCalculator(config, engine)
        base = calc.compute_final_score(entries)

        # Extra penalty if lab subject in wrong room
        lab_subjects = {"CHEMLAB"}
        lab_rooms = {"Lab1"}
        lab_violations = sum(
            1 for e in entries
            if e.subject_id in lab_subjects and e.room_id not in lab_rooms
        )
        penalty = lab_violations * 0.05
        return round(max(0.01, min(0.99, base - penalty)), 4)


# ═══════════════════════════════════════════════════════════════
# 🔴 TASK 3 — HARD
# ═══════════════════════════════════════════════════════════════

class HardTask:
    """
    Three divisions, 8 subjects (2 lab), 8 faculty with uneven availability
    and tight workload constraints. Optimization required.
    """

    TASK_ID = "hard"
    DESCRIPTION = "Three divisions, multiple labs, tight workloads, optimization required."

    @staticmethod
    def get_config() -> SchoolConfig:
        rooms = [
            Room(room_id="CR101", room_type=RoomType.CLASSROOM, capacity=50),
            Room(room_id="CR102", room_type=RoomType.CLASSROOM, capacity=50),
            Room(room_id="CR103", room_type=RoomType.CLASSROOM, capacity=45),
            Room(room_id="CR104", room_type=RoomType.CLASSROOM, capacity=45),
            Room(room_id="Lab1",  room_type=RoomType.LAB,       capacity=25),
            Room(room_id="Lab2",  room_type=RoomType.LAB,       capacity=25),
        ]

        subjects = [
            Subject(subject_id="MATH",   name="Mathematics",       requires_lab=False, sessions_per_week=4),
            Subject(subject_id="PHY",    name="Physics",            requires_lab=False, sessions_per_week=3),
            Subject(subject_id="PHYLAB", name="Physics Lab",        requires_lab=True,  sessions_per_week=1),
            Subject(subject_id="CHEM",   name="Chemistry",          requires_lab=False, sessions_per_week=3),
            Subject(subject_id="CHEMLAB",name="Chemistry Lab",      requires_lab=True,  sessions_per_week=1),
            Subject(subject_id="ENG",    name="English",            requires_lab=False, sessions_per_week=2),
            Subject(subject_id="CS",     name="Computer Science",   requires_lab=False, sessions_per_week=2),
            Subject(subject_id="ECON",   name="Economics",          requires_lab=False, sessions_per_week=2),
        ]

        # Deliberately uneven availability
        all_slots    = [f"{d}-{p}" for d in DAYS_FULL for p in range(1, 6)]
        mon_wed_fri  = [f"{d}-{p}" for d in ["Mon", "Wed", "Fri"] for p in range(1, 6)]
        tue_thu      = [f"{d}-{p}" for d in ["Tue", "Thu"] for p in range(1, 6)]
        morning_only = [f"{d}-{p}" for d in DAYS_FULL for p in range(1, 4)]
        afternoon    = [f"{d}-{p}" for d in DAYS_FULL for p in range(3, 6)]
        limited      = [f"{d}-{p}" for d in ["Mon", "Tue"] for p in range(1, 4)]

        faculty = [
            Faculty(faculty_id="F001", name="Dr. Sharma",   email="sharma@edu.in",
                    subjects_can_teach=["MATH"],
                    available_slots=all_slots,       max_workload=12),
            Faculty(faculty_id="F002", name="Prof. Rao",    email="rao@edu.in",
                    subjects_can_teach=["PHY", "PHYLAB"],
                    available_slots=morning_only,    max_workload=8),
            Faculty(faculty_id="F003", name="Dr. Kapoor",   email="kapoor@edu.in",
                    subjects_can_teach=["CHEM", "CHEMLAB"],
                    available_slots=afternoon,       max_workload=8),
            Faculty(faculty_id="F004", name="Ms. Singh",    email="singh@edu.in",
                    subjects_can_teach=["ENG"],
                    available_slots=mon_wed_fri,     max_workload=6),
            Faculty(faculty_id="F005", name="Mr. Verma",    email="verma@edu.in",
                    subjects_can_teach=["CS"],
                    available_slots=tue_thu,         max_workload=6),
            Faculty(faculty_id="F006", name="Dr. Gupta",    email="gupta@edu.in",
                    subjects_can_teach=["ECON", "ENG"],
                    available_slots=all_slots,       max_workload=8),
            Faculty(faculty_id="F007", name="Prof. Nair",   email="nair@edu.in",
                    subjects_can_teach=["MATH", "PHY"],
                    available_slots=limited,         max_workload=5),  # very limited
            Faculty(faculty_id="F008", name="Ms. Joshi",    email="joshi@edu.in",
                    subjects_can_teach=["CS", "MATH"],
                    available_slots=all_slots,       max_workload=10),
        ]

        divisions = [
            Division(division_id="Sem3-A", name="Semester 3 – Division A",
                     subjects=["MATH", "PHY", "PHYLAB", "CHEM", "CHEMLAB", "ENG"],
                     student_count=45),
            Division(division_id="Sem3-B", name="Semester 3 – Division B",
                     subjects=["MATH", "CS", "ENG", "ECON", "PHY"],
                     student_count=42),
            Division(division_id="Sem3-C", name="Semester 3 – Division C",
                     subjects=["MATH", "CHEM", "CHEMLAB", "CS", "ECON"],
                     student_count=40),
        ]

        time_slots = _make_slots(DAYS_FULL, 5)

        return SchoolConfig(
            rooms=rooms,
            subjects=subjects,
            faculty=faculty,
            divisions=divisions,
            time_slots=time_slots,
            max_steps=300,
            max_invalid_actions=25,
        )

    @staticmethod
    def grade(entries: List[TimetableEntry]) -> float:
        config = HardTask.get_config()
        engine = ConstraintsEngine(config)
        calc = RewardCalculator(config, engine)
        base = calc.compute_final_score(entries)

        # Extra credit: workload balance bonus
        workload_scores = []
        for f in config.faculty:
            load = engine.compute_faculty_workload(f.faculty_id, entries)
            ideal = f.max_workload * 0.75
            diff = abs(load - ideal)
            workload_scores.append(max(0.0, 1.0 - diff / f.max_workload))
        balance_bonus = (sum(workload_scores) / len(workload_scores)) * 0.05

        return round(max(0.01, min(0.99, base + balance_bonus)), 4)


# ═══════════════════════════════════════════════════════════════
# Random Scenario Generator (Bonus)
# ═══════════════════════════════════════════════════════════════

def generate_random_scenario(
    num_divisions: int = 2,
    num_subjects: int = 5,
    num_faculty: int = 4,
    num_classrooms: int = 3,
    num_labs: int = 1,
    seed: int = 42,
) -> SchoolConfig:
    """Generate a random but valid school configuration for testing."""
    random.seed(seed)

    rooms = [
        Room(room_id=f"CR{100+i}", room_type=RoomType.CLASSROOM, capacity=40)
        for i in range(num_classrooms)
    ] + [
        Room(room_id=f"Lab{i+1}", room_type=RoomType.LAB, capacity=25)
        for i in range(num_labs)
    ]

    subjects = []
    for i in range(num_subjects):
        is_lab = (i == num_subjects - 1) and num_labs > 0
        subjects.append(Subject(
            subject_id=f"SUB{i+1}",
            name=f"Subject {i+1}",
            requires_lab=is_lab,
            sessions_per_week=random.randint(1, 3),
        ))

    all_slots = [f"{d}-{p}" for d in DAYS_FULL for p in range(1, 5)]

    faculty = []
    subj_ids = [s.subject_id for s in subjects]
    for i in range(num_faculty):
        can_teach = random.sample(subj_ids, k=random.randint(1, min(3, len(subj_ids))))
        avail = random.sample(all_slots, k=random.randint(10, len(all_slots)))
        faculty.append(Faculty(
            faculty_id=f"F{i+1:03}",
            name=f"Faculty {i+1}",
            email=f"f{i+1}@school.edu",
            subjects_can_teach=can_teach,
            available_slots=avail,
            max_workload=random.randint(5, 12),
        ))

    divisions = []
    for i in range(num_divisions):
        div_subjs = random.sample(subj_ids, k=random.randint(3, min(5, len(subj_ids))))
        divisions.append(Division(
            division_id=f"Div{i+1}",
            name=f"Division {i+1}",
            subjects=div_subjs,
            student_count=random.randint(30, 50),
        ))

    time_slots = _make_slots(DAYS_FULL, 4)

    return SchoolConfig(
        rooms=rooms,
        subjects=subjects,
        faculty=faculty,
        divisions=divisions,
        time_slots=time_slots,
        max_steps=200,
        max_invalid_actions=20,
    )


# ═══════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════

TASK_REGISTRY = {
    "easy":   EasyTask,
    "medium": MediumTask,
    "hard":   HardTask,
}


def get_task(task_id: str):
    if task_id not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_id}. Available: {list(TASK_REGISTRY)}")
    return TASK_REGISTRY[task_id]
