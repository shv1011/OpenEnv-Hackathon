from __future__ import annotations
from typing import Any, Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from models import TimetableAction, TimetableObservation, TimetableState


class SchoolTimetableEnvClient(
    EnvClient[TimetableAction, TimetableObservation, TimetableState]
):
    def _step_payload(self, action: TimetableAction) -> Dict[str, Any]:
        return {k: v for k, v in action.model_dump().items() if v is not None}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[TimetableObservation]:
        obs_data = payload.get("observation", payload)
        obs = TimetableObservation(**{
            k: v for k, v in obs_data.items()
            if k in TimetableObservation.model_fields
        })
        return StepResult(
            observation=obs,
            reward=payload.get("reward", obs.reward),
            done=payload.get("done", obs.done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> TimetableState:
        return TimetableState(**{
            k: v for k, v in payload.items()
            if k in TimetableState.model_fields
        })


__all__ = ["SchoolTimetableEnvClient", "TimetableAction", "TimetableObservation", "TimetableState"]
