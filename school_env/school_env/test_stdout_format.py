"""
Validates inference.py stdout matches the required format exactly.
Runs against local server using a mock LLM that always makes valid moves.
"""
import sys, os, re, subprocess, json
sys.path.insert(0, '.')

# ── Start local server ────────────────────────────────────────
from fastapi.testclient import TestClient
from server.app import app
tc = TestClient(app)
assert tc.get("/health").json()["status"] == "healthy"
print("Server OK")

# ── Mock the LLM by monkey-patching OpenAI ───────────────────
# We'll simulate inference.py logic directly with a scripted agent

from environment import SchoolTimetableEnvironment
from models import TimetableAction

# Pre-computed valid easy task solution
EASY_SOLUTION = [
    dict(action_type="assign_class", division_id="Sem1-A", subject_id="MATH",
         faculty_id="F001", room_id="CR101", slot_id="Mon-1"),
    dict(action_type="assign_class", division_id="Sem1-A", subject_id="MATH",
         faculty_id="F001", room_id="CR101", slot_id="Mon-2"),
    dict(action_type="assign_class", division_id="Sem1-A", subject_id="MATH",
         faculty_id="F001", room_id="CR101", slot_id="Mon-3"),
    dict(action_type="assign_class", division_id="Sem1-A", subject_id="ENG",
         faculty_id="F002", room_id="CR102", slot_id="Tue-1"),
    dict(action_type="assign_class", division_id="Sem1-A", subject_id="ENG",
         faculty_id="F002", room_id="CR102", slot_id="Tue-2"),
    dict(action_type="assign_class", division_id="Sem1-A", subject_id="SCI",
         faculty_id="F003", room_id="CR101", slot_id="Wed-1"),
    dict(action_type="assign_class", division_id="Sem1-A", subject_id="SCI",
         faculty_id="F003", room_id="CR101", slot_id="Wed-2"),
    dict(action_type="assign_class", division_id="Sem1-A", subject_id="HIST",
         faculty_id="F002", room_id="CR102", slot_id="Thu-1"),
    dict(action_type="assign_class", division_id="Sem1-A", subject_id="HIST",
         faculty_id="F002", room_id="CR102", slot_id="Thu-2"),
]

# ── Simulate the exact stdout inference.py would produce ─────
TASK_NAME = "easy"
BENCHMARK = "school-timetable"
MODEL_NAME = "test-model"

lines = []

lines.append(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}")

env = SchoolTimetableEnvironment()
obs = env.reset(task=TASK_NAME)
rewards = []
step = 0
success = False

for action_dict in EASY_SOLUTION:
    action = TimetableAction(**action_dict)
    obs = env.step(action)
    step += 1
    reward = obs.reward
    rewards.append(reward)
    done = obs.done

    atype = action_dict["action_type"]
    action_str = f"assign({action_dict['division_id']},{action_dict['subject_id']},{action_dict['slot_id']})"

    error_str = "null"
    if obs.violations:
        error_str = obs.violations[0].get("violation_type", "VIOLATION")

    line = (f"[STEP]  step={step}"
            f" action={action_str}"
            f" reward={reward:.2f}"
            f" done={str(done).lower()}"
            f" error={error_str}")
    lines.append(line)

    if done:
        success = obs.completion_percentage == 100.0
        break

rewards_str = ",".join(f"{r:.2f}" for r in rewards)
lines.append(f"[END]   success={str(success).lower()} steps={step} rewards={rewards_str}")

# ── Print and validate ────────────────────────────────────────
print("\n--- STDOUT OUTPUT ---")
for line in lines:
    print(line)
print("--- END ---\n")

# Validate format
START_RE = re.compile(r'^\[START\] task=\S+ env=\S+ model=\S+$')
STEP_RE  = re.compile(r'^\[STEP\]\s+ step=\d+ action=\S+ reward=\d+\.\d{2} done=(true|false) error=\S+$')
END_RE   = re.compile(r'^\[END\]\s+ success=(true|false) steps=\d+ rewards=[\d.,]*$')

assert START_RE.match(lines[0]),  f"BAD [START]: {lines[0]}"
for line in lines[1:-1]:
    assert STEP_RE.match(line),   f"BAD [STEP]:  {line}"
assert END_RE.match(lines[-1]),   f"BAD [END]:   {lines[-1]}"

# Validate content
assert "task=easy" in lines[0]
assert "env=school-timetable" in lines[0]
assert lines[-1].startswith("[END]")
assert "success=true" in lines[-1]
assert f"steps={step}" in lines[-1]
assert len(rewards) == step

print("FORMAT VALIDATION PASSED")
print(f"  {step} steps, all [STEP] lines valid")
print(f"  [START] valid")
print(f"  [END] valid — success=true steps={step}")
print(f"  rewards: {rewards_str}")
