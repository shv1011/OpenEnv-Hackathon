from __future__ import annotations
from typing import Any, Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from models import TimetableAction, TimetableObservation, TimetableState

_OBS_FIELDS = set(TimetableObservation.model_fields.keys())
_STATE_FIELDS = set(TimetableState.model_fields.keys())


def _build_obs(data: Dict[str, Any]) -> TimetableObservation:
    if "observation" in data and isinstance(data["observation"], dict):
        data = data["observation"]
    return TimetableObservation(**{k: v for k, v in data.items() if k in _OBS_FIELDS})


class SchoolTimetableEnvClient(
    EnvClient[TimetableAction, TimetableObservation, TimetableState]
):
    def _step_payload(self, action: TimetableAction) -> Dict[str, Any]:
        return {k: v for k, v in action.model_dump().items() if v is not None}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[TimetableObservation]:
        obs = _build_obs(payload)
        return StepResult(
            observation=obs,
            reward=float(payload.get("reward") or obs.reward or 0.0),
            done=bool(payload.get("done", obs.done)),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> TimetableState:
        return TimetableState(**{k: v for k, v in payload.items() if k in _STATE_FIELDS})


__all__ = ["SchoolTimetableEnvClient", "TimetableAction", "TimetableObservation", "TimetableState"]
