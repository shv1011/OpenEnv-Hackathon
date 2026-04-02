# 🏫 School Admin Timetable Scheduling Environment

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-blue)]()
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green)]()
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-orange)]()

A production-grade, multi-step AI environment that simulates the real-world
workflow of a school administrator constructing a weekly class timetable.

---

## 🎯 Motivation

Building a school timetable is one of the hardest combinatorial optimization
problems in educational administration. A typical administrator must:

- Schedule **dozens of subjects** across **multiple divisions**
- Assign **qualified faculty** without double-booking them
- Allocate **limited rooms and labs** efficiently
- Respect **faculty availability** windows and workload limits
- Deliver a **conflict-free** schedule under tight time pressure

This environment makes that problem accessible to LLM agents, rewarding
efficient, constraint-satisfying, multi-step decision making.

---

## 🏗️ Project Structure

```
school_env/
├── env/
│   ├── __init__.py         — Package exports
│   ├── environment.py      — Core environment (reset/step/state)
│   ├── models.py           — Pydantic models (Action, Observation, Reward…)
│   ├── constraints.py      — Full constraint validation engine
│   ├── reward.py           — Dense reward calculator
│   ├── tasks.py            — Easy / Medium / Hard tasks + graders
│   ├── export.py           — CSV & PDF timetable export
│   └── emailer.py          — SMTP email delivery
│
├── inference.py            — LLM agent loop (OpenAI-compatible)
├── openenv.yaml            — OpenEnv specification
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🔁 Environment Interface

### `reset() → Observation`
Clears the timetable and starts a new episode.

### `step(action: Action) → StepResult`
Executes one scheduling action and returns:
- `observation` — full current state
- `reward` — step reward (+/−)
- `done` — episode termination flag
- `info` — action metadata, violations, reward breakdown

### `state() → Observation`
Returns the current observation without advancing the episode.

---

## 🧩 Action Space

### `assign_class`
Assign a subject session to a division.
```json
{
  "action_type": "assign_class",
  "assign_class": {
    "division_id": "Sem1-A",
    "subject_id": "MATH",
    "faculty_id": "F001",
    "room_id": "CR101",
    "slot_id": "Mon-1"
  }
}
```

### `reschedule_class`
Move an existing entry to a new slot.
```json
{
  "action_type": "reschedule_class",
  "reschedule_class": {
    "entry_id": "ab12cd34",
    "new_slot_id": "Tue-3",
    "new_room_id": "CR102"
  }
}
```

### `remove_assignment`
Remove a timetable entry.
```json
{
  "action_type": "remove_assignment",
  "remove_assignment": {
    "entry_id": "ab12cd34"
  }
}
```

---

## 👁️ Observation Space

```python
class Observation:
    step_count: int
    timetable_entries: List[TimetableEntry]
    progress: TimetableProgress          # completion %, pending divisions
    resource_utilization: ResourceUtilization   # faculty/room utilization
    recent_violations: List[ConstraintViolation]
    available_actions_hint: Dict         # pending work, faculty/room availability
    is_terminal: bool
    termination_reason: Optional[str]
```

---

## ⚙️ Constraints Engine

All actions are validated against seven hard constraints:

| # | Constraint | Penalty |
|---|-----------|---------|
| 1 | ❌ Teacher double-booking | −0.30 |
| 2 | ❌ Room double-booking | −0.30 |
| 3 | ❌ Faculty outside availability | −0.30 |
| 4 | ❌ Subject–faculty mismatch | −0.30 |
| 5 | ❌ Lab subject in wrong room | −0.30 |
| 6 | ❌ Workload limit exceeded | −0.30 |
| 7 | ❌ Division double-booked in slot | −0.30 |

---

## 💰 Reward Function

### Step Rewards
| Event | Reward |
|-------|--------|
| Valid assignment | +0.20 |
| Conflict / invalid action | −0.30 |
| Efficient assignment (high urgency) | +0.10 |
| Redundant action | −0.10 |
| Successful reschedule | +0.05 |

### Final Score (0.0 – 1.0)
```
Score = 0.50 × completeness + 0.30 × conflict_free + 0.20 × efficiency
```

---

## 🎮 Tasks

### 🟢 Easy
- 1 division · 4 subjects · 3 faculty · 2 classrooms · no labs
- Expected solution: ~15 steps
- Target score: 0.90+

### 🟡 Medium
- 2 divisions · 6 subjects (1 lab) · 5 faculty with limited availability
- Requires conflict resolution and lab scheduling
- Target score: 0.80+

### 🔴 Hard
- 3 divisions · 8 subjects (2 labs) · 8 faculty with uneven availability
- Tight workloads, optimization required
- Target score: 0.70+

---

## 🚀 Setup & Usage

### 1. Install Dependencies

```bash
cd school_env
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export OPENAI_API_KEY=sk-your-key-here
export MODEL_NAME=gpt-4o          # optional, default: gpt-4o
export API_BASE_URL=https://api.openai.com/v1  # optional
```

### 3. Run Inference

```bash
# Easy task
python inference.py --task easy

# Medium task with CSV export
python inference.py --task medium --export-csv

# Hard task with debug logging
python inference.py --task hard --debug --export-csv

# Log to file
python inference.py --task hard --log-file logs/run_001.txt
```

### 4. Use as a Library

```python
from env import SchoolTimetableEnv, Action, ActionType, AssignClassAction, EasyTask

config = EasyTask.get_config()
env = SchoolTimetableEnv(config)

obs = env.reset()

action = Action(
    action_type=ActionType.ASSIGN_CLASS,
    assign_class=AssignClassAction(
        division_id="Sem1-A",
        subject_id="MATH",
        faculty_id="F001",
        room_id="CR101",
        slot_id="Mon-1",
    )
)

result = env.step(action)
print(f"Reward: {result.reward}")
print(f"Valid: {result.info['valid']}")
print(f"Completion: {result.observation.progress.completion_percentage}%")

# Score the final timetable
score = EasyTask.grade(env.get_entries())
print(f"Final score: {score:.4f}")
```

### 5. Export Timetables

```python
from env import export_all_faculty_timetables_csv, export_master_timetable_csv

# Per-faculty CSVs
export_all_faculty_timetables_csv(env.get_entries(), config, "timetables/")

# Master timetable
export_master_timetable_csv(env.get_entries(), config, "timetables/master.csv")
```

### 6. Email Timetables (SMTP)

```python
from env.emailer import TimetableMailer

mailer = TimetableMailer(
    smtp_user="admin@school.edu",
    smtp_password="app_password",
    school_name="Springfield High",
)

# Send to one faculty
mailer.send_faculty_email("Dr. Sharma", env.get_entries(), config)

# Send to all faculty
results = mailer.send_all_faculty_emails(env.get_entries(), config)
```

---

## 🐳 Docker

```bash
# Build
docker build -t school-timetable-env .

# Run easy task
docker run --rm \
  -e OPENAI_API_KEY=sk-... \
  -v $(pwd)/timetables:/app/timetables \
  school-timetable-env --task easy --export-csv

# Run hard task with custom API (e.g. Groq)
docker run --rm \
  -e OPENAI_API_KEY=... \
  -e API_BASE_URL=https://api.groq.com/openai/v1 \
  -e MODEL_NAME=llama-3.1-70b-versatile \
  school-timetable-env --task hard --export-csv
```

---

## 📊 Sample Output

```
=================================================================
  School Timetable Scheduling — OpenEnv Agent
  Task:  EASY
  Model: gpt-4o
  Time:  2024-01-15 10:32:01
=================================================================

[START]
Task: easy | Divisions: 1 | Faculty: 3 | Max steps: 60

[STEP 1] action=assign_class | valid=True | reward=+0.300 | completion=11.1%
[STEP 2] action=assign_class | valid=True | reward=+0.267 | completion=22.2%
[STEP 3] action=assign_class | valid=True | reward=+0.250 | completion=33.3%
...
[STEP 9] action=assign_class | valid=True | reward=+0.200 | completion=100.0%

[END]
=================================================================
  Episode complete. Reason: complete
  Steps taken       : 9
  Valid actions     : 9
  Invalid actions   : 0
  Sessions assigned : 9
  Completion        : 100.0%
  Total conflicts   : 0
  Cumul. reward     : +2.4900
  ★ FINAL SCORE     : 0.9750 / 1.0000
=================================================================
```

---

## 🔁 Episode Termination

| Reason | Condition |
|--------|-----------|
| `complete` | All required sessions scheduled |
| `max_steps` | Step limit reached |
| `too_many_invalid` | Invalid action threshold exceeded |

---

## 🧪 Custom Scenarios

```python
from env.tasks import generate_random_scenario

config = generate_random_scenario(
    num_divisions=3,
    num_subjects=6,
    num_faculty=5,
    num_classrooms=4,
    num_labs=2,
    seed=42,
)
env = SchoolTimetableEnv(config)
```

---

## 📄 License

MIT License — Built for the OpenEnv Hackathon.
