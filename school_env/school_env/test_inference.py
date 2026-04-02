import sys
sys.path.insert(0, '.')

print("=== 1. Imports ===")
from client import SchoolTimetableEnvClient, TimetableAction
from inference import TimetableAgent, format_observation
from models import TimetableObservation
print("Imports OK")

print("\n=== 2. format_observation ===")
obs = TimetableObservation(
    done=False, reward=0.0, step_count=3,
    completion_percentage=33.3, assigned_sessions=3,
    total_required_sessions=9,
    divisions_pending=["Sem1-A"],
    pending_work=[{"division_id":"Sem1-A","subjects_needed":{"MATH":2,"ENG":2}}],
    faculty_status=[{
        "faculty_id":"F001","name":"Dr. Sharma",
        "current_load":1,"max_workload":10,
        "can_teach":["MATH"],"free_slots":["Mon-2","Tue-1","Wed-3"]
    }],
    timetable_entries=[{
        "entry_id":"abc123","division_id":"Sem1-A",
        "subject_id":"MATH","faculty_id":"F001",
        "room_id":"CR101","slot_id":"Mon-1"
    }],
)
text = format_observation(obs)
assert "PENDING WORK" in text
assert "FACULTY STATUS" in text
assert "RECENT ASSIGNMENTS" in text
assert "33.3" in text
print(text)
print("format_observation OK")

print("\n=== 3. TimetableAction flat JSON (new format) ===")
a = TimetableAction(
    action_type="assign_class",
    division_id="Sem1-A", subject_id="MATH",
    faculty_id="F001", room_id="CR101", slot_id="Mon-1"
)
assert a.action_type == "assign_class"
assert a.division_id == "Sem1-A"
print("assign_class OK:", a.action_type, a.slot_id)

a2 = TimetableAction(action_type="reschedule_class",
                     entry_id="abc123", new_slot_id="Fri-4")
assert a2.action_type == "reschedule_class"
print("reschedule_class OK:", a2.entry_id, a2.new_slot_id)

a3 = TimetableAction(action_type="remove_assignment", entry_id="abc123")
assert a3.action_type == "remove_assignment"
print("remove_assignment OK:", a3.entry_id)

print("\n=== 4. Client connects to local server ===")
import subprocess, sys as _sys
# Start server in background, test connection
from fastapi.testclient import TestClient
from server.app import app
tc = TestClient(app)

# Simulate what inference.py does via the sync client
# (WebSocket test via TestClient)
r = tc.get("/health")
assert r.status_code == 200
print("Server health OK:", r.json())

r2 = tc.post("/reset", json={"task": "easy"})
assert r2.status_code == 200
obs_data = r2.json()["observation"]
assert obs_data["total_required_sessions"] == 9
print("Server reset OK - sessions:", obs_data["total_required_sessions"])

print("\n=== 5. CLI args ===")
import subprocess
result = subprocess.run(
    [_sys.executable, "inference.py", "--help"],
    capture_output=True, text=True
)
assert "--task" in result.stdout
assert "--env-url" in result.stdout
assert "--export-csv" in result.stdout
print(result.stdout.strip())
print("CLI args OK")

print("\nALL INFERENCE TESTS PASSED")
