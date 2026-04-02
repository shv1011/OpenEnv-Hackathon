"""Full OpenEnv compliance test."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=== 1. Models ===")
from models import TimetableAction, TimetableObservation, TimetableState
a = TimetableAction(action_type="assign_class", division_id="Sem1-A",
                    subject_id="MATH", faculty_id="F001", room_id="CR101", slot_id="Mon-1")
assert a.action_type == "assign_class"
print("Action OK:", a.action_type, a.division_id)
o = TimetableObservation(done=False, reward=0.2, step_count=1, completion_percentage=11.1,
                         assigned_sessions=1, total_required_sessions=9)
assert o.completion_percentage == 11.1
print("Observation OK:", o.done, o.reward)
s = TimetableState(task="easy", step_count=1)
assert s.task == "easy"
print("State OK:", s.task)
print("MODELS PASS\n")

print("=== 2. Environment ===")
from environment import SchoolTimetableEnvironment
env = SchoolTimetableEnvironment()
assert env.SUPPORTS_CONCURRENT_SESSIONS is True

obs = env.reset(task="easy")
assert obs.completion_percentage == 0.0
assert obs.total_required_sessions == 9
assert len(obs.pending_work) == 1
print("reset() OK - pending:", obs.pending_work[0]["division_id"])

obs2 = env.step(TimetableAction(action_type="assign_class",
    division_id="Sem1-A", subject_id="MATH", faculty_id="F001", room_id="CR101", slot_id="Mon-1"))
assert obs2.reward > 0
assert obs2.completion_percentage > 0
print("step() assign OK - reward:", round(obs2.reward, 3), "completion:", obs2.completion_percentage)

obs3 = env.step(TimetableAction(action_type="assign_class",
    division_id="Sem1-A", subject_id="MATH", faculty_id="F001", room_id="CR101", slot_id="Mon-1"))
assert len(obs3.violations) > 0
print("step() conflict OK - violation:", obs3.violations[0].get("violation_type"))

state = env.state
assert state.task == "easy"
print("state() OK - task:", state.task, "completion:", state.completion_percentage)
print("ENVIRONMENT PASS\n")

print("=== 3. Full easy task ===")
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
assert final.termination_reason == "complete"
assert final.reward > 0.5   # final score
print("completion:", final.completion_percentage, "done:", final.done, "score:", final.reward)
print("FULL TASK PASS\n")

print("=== 4. Server (create_app) ===")
from fastapi.testclient import TestClient
from server.app import app
client = TestClient(app)

r = client.get("/health")
assert r.status_code == 200
assert r.json()["status"] == "healthy"
print("GET /health OK:", r.json())

r = client.post("/reset", json={"task": "easy"})
assert r.status_code == 200
obs_data = r.json()["observation"]
assert obs_data["total_required_sessions"] == 9
print("POST /reset OK - required sessions:", obs_data["total_required_sessions"])

r = client.post("/step", json={"action": {
    "action_type": "assign_class",
    "division_id": "Sem1-A", "subject_id": "MATH",
    "faculty_id": "F001", "room_id": "CR101", "slot_id": "Mon-1"
}})
assert r.status_code == 200
step_data = r.json()
assert step_data["reward"] > 0
print("POST /step OK - reward:", step_data["reward"])

r = client.get("/state")
assert r.status_code == 200
print("GET /state OK:", r.json())

r = client.get("/docs")
assert r.status_code == 200
print("GET /docs OK")

r = client.get("/ui")
assert r.status_code == 200
print("GET /ui OK")

print("SERVER PASS\n")

print("=== 5. Client ===")
from client import SchoolTimetableEnvClient, TimetableAction
assert hasattr(SchoolTimetableEnvClient, "reset")
assert hasattr(SchoolTimetableEnvClient, "step")
assert hasattr(SchoolTimetableEnvClient, "sync")
print("Client class OK:", SchoolTimetableEnvClient.__name__)
print("CLIENT PASS\n")

print("=== 6. All 3 tasks ===")
for task in ["easy", "medium", "hard"]:
    e = SchoolTimetableEnvironment()
    o = e.reset(task=task)
    assert o.total_required_sessions > 0
    print(f"  {task}: required={o.total_required_sessions} pending={len(o.pending_work)}")
print("ALL TASKS PASS\n")

print("ALL OPENENV COMPLIANCE TESTS PASSED")
