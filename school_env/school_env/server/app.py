"""
OpenEnv server for School Timetable Scheduling.

Exposes:
  WebSocket /ws          — persistent session (used by EnvClient / TRL)
  POST     /reset        — stateless reset
  POST     /step         — stateless step
  GET      /state        — current state
  GET      /health       — health check
  GET      /web          — interactive web UI
  GET      /docs         — OpenAPI docs

Run locally:
    uvicorn server.app:app --host 0.0.0.0 --port 7860

On HF Spaces:
    Dockerfile sets ENTRYPOINT to this.
"""
from __future__ import annotations
import sys, os

# Make root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core import create_app
from environment import SchoolTimetableEnvironment
from models import TimetableAction, TimetableObservation

# ── Build the FastAPI app via openenv-core ────────────────────
app = create_app(
    env=SchoolTimetableEnvironment,          # factory callable
    action_cls=TimetableAction,
    observation_cls=TimetableObservation,
    env_name="school-timetable-env",
    max_concurrent_envs=64,
)

# ── Mount the existing demo UI at /ui ─────────────────────────
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import Request

# ── Tasks endpoint ────────────────────────────────────────────
@app.get("/tasks")
def list_tasks():
    """Return all tasks with grader information — required by validator."""
    from env.tasks import EasyTask, MediumTask, HardTask
    tasks = []
    for task_id, cls in [("easy", EasyTask), ("medium", MediumTask), ("hard", HardTask)]:
        score = cls.grade([])
        tasks.append({
            "id": task_id,
            "description": cls.DESCRIPTION,
            "grader": {
                "enabled": True,
                "function": f"env.tasks.{cls.__name__}.grade",
                "callable": True,
            },
            "score": score,
            "target_score": {"easy": 0.90, "medium": 0.80, "hard": 0.70}[task_id],
        })
    return {
        "tasks": tasks,
        "total_tasks": len(tasks),
        "tasks_with_graders": len(tasks),
    }


@app.post("/grade")
async def grade_task(request: Request):
    """Grade a timetable for a given task."""
    body = await request.json()
    task_id = body.get("task", "easy")
    entries = body.get("entries", [])
    from env.tasks import EasyTask, MediumTask, HardTask
    from env.models import TimetableEntry
    cls = {"easy": EasyTask, "medium": MediumTask, "hard": HardTask}.get(task_id, EasyTask)
    try:
        timetable = [TimetableEntry(**e) for e in entries]
        score = cls.grade(timetable)
    except Exception:
        score = cls.grade([])
    return {"task": task_id, "score": score, "grader_enabled": True}


@app.post("/grader")
async def grader_endpoint(request: Request):
    """Alternative grader endpoint for OpenEnv validator compatibility."""
    body = await request.json()
    task_id = body.get("task", "easy")
    entries = body.get("entries", [])
    session_id = body.get("session_id")
    
    from env.tasks import EasyTask, MediumTask, HardTask
    from env.models import TimetableEntry
    cls = {"easy": EasyTask, "medium": MediumTask, "hard": HardTask}.get(task_id, EasyTask)
    try:
        timetable = [TimetableEntry(**e) for e in entries]
        score = cls.grade(timetable)
    except Exception:
        score = cls.grade([])
    
    return {
        "task": task_id,
        "score": score,
        "session_id": session_id,
        "grader_enabled": True,
        "grader_function": f"env.tasks.{cls.__name__}.grade",
    }

static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/ui/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/ui")
    def ui_root():
        return FileResponse(str(static_dir / "index.html"))

    @app.get("/ui/generate")
    def ui_generate():
        return FileResponse(str(static_dir / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
