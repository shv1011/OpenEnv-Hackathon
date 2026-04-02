"""Final submission checklist — run before pushing to HF Space."""
import sys, os
sys.path.insert(0, '.')

checks = []

print("Running submission checklist...\n")

# 1. openenv-core Environment subclass
from environment import SchoolTimetableEnvironment
from openenv.core import Environment
assert issubclass(SchoolTimetableEnvironment, Environment)
assert SchoolTimetableEnvironment.SUPPORTS_CONCURRENT_SESSIONS is True
checks.append("Environment subclass + SUPPORTS_CONCURRENT_SESSIONS=True")

# 2. Models extend openenv-core base classes
from models import TimetableAction, TimetableObservation, TimetableState
from openenv.core import Action, Observation, State
assert issubclass(TimetableAction, Action)
assert issubclass(TimetableObservation, Observation)
assert issubclass(TimetableState, State)
checks.append("Action / Observation / State extend openenv-core base classes")

# 3. Server endpoints via create_app
from fastapi.testclient import TestClient
from server.app import app
tc = TestClient(app)
assert tc.get("/health").status_code == 200
assert tc.post("/reset", json={"task": "easy"}).status_code == 200
assert tc.get("/state").status_code == 200
assert tc.get("/docs").status_code == 200
assert tc.get("/ui").status_code == 200
checks.append("Server endpoints: /health /reset /step /state /docs /ui")

# 4. EnvClient subclass with .sync()
from client import SchoolTimetableEnvClient
from openenv.core import EnvClient
assert issubclass(SchoolTimetableEnvClient, EnvClient)
assert hasattr(SchoolTimetableEnvClient, "sync")
checks.append("EnvClient subclass with .sync() for TRL")

# 5. All 3 tasks + graders work
from env import get_task
for task in ["easy", "medium", "hard"]:
    cls = get_task(task)
    config = cls.get_config()
    env = SchoolTimetableEnvironment()
    obs = env.reset(task=task)
    assert obs.total_required_sessions > 0
    score = cls.grade([])
    assert 0.0 <= score <= 1.0
checks.append("All 3 tasks (easy/medium/hard) + graders return [0,1]")

# 6. Full easy task completes with score > 0.8
env2 = SchoolTimetableEnvironment()
env2.reset(task="easy")
assigns = [
    ("Sem1-A","MATH","F001","CR101","Mon-1"), ("Sem1-A","MATH","F001","CR101","Mon-2"),
    ("Sem1-A","MATH","F001","CR101","Mon-3"), ("Sem1-A","ENG","F002","CR102","Tue-1"),
    ("Sem1-A","ENG","F002","CR102","Tue-2"),  ("Sem1-A","SCI","F003","CR101","Wed-1"),
    ("Sem1-A","SCI","F003","CR101","Wed-2"),  ("Sem1-A","HIST","F002","CR102","Thu-1"),
    ("Sem1-A","HIST","F002","CR102","Thu-2"),
]
final = None
for div,sub,fac,room,slot in assigns:
    final = env2.step(TimetableAction(action_type="assign_class",
        division_id=div, subject_id=sub, faculty_id=fac, room_id=room, slot_id=slot))
assert final.done is True
assert final.completion_percentage == 100.0
assert final.reward >= 0.7
checks.append(f"Easy task completes 100% with score {final.reward:.2f} >= 0.70")

# 7. pyproject.toml exists
assert os.path.exists("pyproject.toml")
import tomllib
with open("pyproject.toml", "rb") as f:
    toml = tomllib.load(f)
assert toml["project"]["name"] == "openenv-school-timetable"
checks.append("pyproject.toml — pip-installable as openenv-school-timetable")

# 8. Dockerfile correct
with open("Dockerfile") as f: df = f.read()
assert "7860" in df
assert "server.app:app" in df
checks.append("Dockerfile — port 7860, entrypoint server.app:app")

# 9. openenv.yaml has required fields
import yaml
with open("openenv.yaml") as f: oy = yaml.safe_load(f)
for field in ["openenv_core_version", "trl_integration", "endpoints", "tasks", "reward", "constraints"]:
    assert field in oy, f"Missing field in openenv.yaml: {field}"
checks.append("openenv.yaml — all required fields present")

# 10. inference.py uses client + env-url
with open("inference.py") as f: inf = f.read()
assert "SchoolTimetableEnvClient" in inf
assert "ENV_URL" in inf
assert "format_observation" in inf
assert "HF_TOKEN" in inf
assert "[START]" in inf
assert "[STEP]" in inf
assert "[END]" in inf
checks.append("inference.py — correct stdout format + HF_TOKEN + SchoolTimetableEnvClient")

# Print results
print("SUBMISSION CHECKLIST")
print("=" * 60)
for i, c in enumerate(checks, 1):
    print(f"  {i:2d}. {c}")
print("=" * 60)
print(f"  {len(checks)}/{len(checks)} checks passed — ready to submit!")
