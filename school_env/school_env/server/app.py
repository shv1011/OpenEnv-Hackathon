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

static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/ui/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/ui")
    def ui_root():
        return FileResponse(str(static_dir / "index.html"))

    @app.get("/ui/generate")
    def ui_generate():
        return FileResponse(str(static_dir / "generator.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)
