"""
OpenEnv client for the School Timetable Scheduling Environment.

Usage:
    from client import SchoolTimetableEnvClient, TimetableAction

    # Async (recommended for TRL training)
    async with SchoolTimetableEnvClient(base_url="https://your-space.hf.space") as env:
        obs = await env.reset(task="easy")
        obs = await env.step(TimetableAction(
            action_type="assign_class",
            division_id="Sem1-A",
            subject_id="MATH",
            faculty_id="F001",
            room_id="CR101",
            slot_id="Mon-1",
        ))
        print(obs.completion_percentage)

    # Sync (for quick testing)
    with SchoolTimetableEnvClient(base_url="https://your-space.hf.space").sync() as env:
        obs = env.reset(task="easy")
        obs = env.step(TimetableAction(action_type="assign_class", ...))
"""
from __future__ import annotations
from openenv.core import EnvClient
from models import TimetableAction, TimetableObservation, TimetableState


class SchoolTimetableEnvClient(
    EnvClient[TimetableAction, TimetableObservation, TimetableState]
):
    """
    WebSocket client for the School Timetable Scheduling Environment.

    Connects to a running HF Space or local server.
    Compatible with TRL's GRPOTrainer environment_factory.
    """
    pass


# Re-export models for convenience
__all__ = [
    "SchoolTimetableEnvClient",
    "TimetableAction",
    "TimetableObservation",
    "TimetableState",
]
