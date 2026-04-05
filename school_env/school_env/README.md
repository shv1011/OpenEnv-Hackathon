---
title: School Timetable Scheduling
emoji: 🏫
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: OpenEnv RL environment for school timetable scheduling
---

# School Timetable Scheduling

An OpenEnv-compatible reinforcement learning environment where an agent learns to build a conflict-free weekly class timetable for a school. The problem is real — scheduling classes across multiple divisions, faculty, and rooms without conflicts is NP-hard and genuinely difficult to solve optimally.

The agent takes one scheduling action per step, receives a shaped reward, and must complete the timetable within a step budget while satisfying all constraints.

---

## The Problem

A school administrator needs to assign every subject session for every division to a time slot, faculty member, and room — without:

- Double-booking a teacher in the same slot
- Double-booking a room in the same slot  
- Scheduling a faculty member outside their available hours
- Assigning a subject to a teacher who isn't qualified for it
- Putting a lab subject in a regular classroom (or vice versa)
- Exceeding a faculty member's weekly workload limit
- Giving a division two classes in the same slot

Every constraint violation gives a negative reward. Every valid assignment gives a positive one. The episode ends when the timetable is complete, the step limit is hit, or too many invalid actions are taken.

---

## Environment Interface

```python
from client import SchoolTimetableEnvClient, TimetableAction

with SchoolTimetableEnvClient(base_url="https://shv1011-school-timetable-env.hf.space").sync() as env:
    obs = env.reset(task="easy")
    
    obs = env.step(TimetableAction(
        action_type="assign_class",
        division_id="Sem1-A",
        subject_id="MATH",
        faculty_id="F001",
        room_id="CR101",
        slot_id="Mon-1",
    ))
    
    print(obs.completion_percentage)  # 11.1
    print(obs.reward)                 # 0.233
    print(obs.violations)             # []
```

### Actions

| Type | Fields |
|------|--------|
| `assign_class` | `division_id`, `subject_id`, `faculty_id`, `room_id`, `slot_id` |
| `reschedule_class` | `entry_id`, `new_slot_id`, `new_room_id` (opt), `new_faculty_id` (opt) |
| `remove_assignment` | `entry_id` |

### Observation

Each step returns a `TimetableObservation` with:

- `completion_percentage` — how much of the timetable is filled
- `assigned_sessions` / `total_required_sessions`
- `timetable_entries` — current scheduled entries
- `violations` — constraint violations from the last action
- `pending_work` — what still needs to be scheduled, per division
- `faculty_status` — each faculty member's free slots and current load
- `done`, `reward`, `termination_reason`

### Reward

```
valid assignment      +0.20
efficiency bonus      +0.10  (scheduling high-urgency subjects)
constraint violation  -0.30
redundant action      -0.10
reschedule            +0.05
```

Final score (0.0 – 1.0):
```
0.50 × completeness + 0.30 × conflict-free rate + 0.20 × faculty utilization
```

---

## Tasks

| Task | Divisions | Faculty | Subjects | Sessions | Labs | Target |
|------|-----------|---------|----------|----------|------|--------|
| `easy` | 1 | 3 | 4 | 9 | 0 | 0.90 |
| `medium` | 2 | 5 | 6 | 19 | 1 | 0.80 |
| `hard` | 3 | 8 | 8 | 39 | 2 | 0.70 |

Difficulty increases through limited faculty availability windows, lab room requirements, tight workload constraints, and more divisions competing for the same teachers.

---

## Setup

**Install dependencies**

```bash
pip install openenv-core>=0.2.1 fastapi uvicorn openai pydantic
```

**Run the server locally**

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

**Or with Docker**

```bash
docker build -t school-timetable-env .
docker run -p 7860:7860 school-timetable-env
```

**Run inference**

```bash
export HF_TOKEN=your_hf_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export ENV_URL=https://shv1011-school-timetable-env.hf.space

python inference.py
```

Output:
```
[START] task=easy env=school-timetable model=Qwen/Qwen2.5-72B-Instruct
[STEP]  step=1 action=assign(Sem1-A,MATH,Mon-1) reward=0.23 done=false error=null
[STEP]  step=2 action=assign(Sem1-A,MATH,Mon-2) reward=0.23 done=false error=null
...
[END]   success=true steps=9 score=0.86 rewards=0.23,0.23,0.21,...
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Start new episode (`{"task": "easy"}`) |
| `/step` | POST | Execute action |
| `/state` | GET | Current episode state |
| `/ws` | WebSocket | Persistent session (used by client) |
| `/docs` | GET | OpenAPI documentation |
| `/ui` | GET | Timetable generator UI |

---

## Project Structure

```
├── inference.py          # Inference script (run this)
├── client.py             # EnvClient for connecting to the server
├── environment.py        # OpenEnv Environment subclass
├── models.py             # Typed Action / Observation / State
├── server/
│   └── app.py            # FastAPI server via create_app()
├── env/
│   ├── constraints.py    # 7 hard constraint checks
│   ├── reward.py         # Dense reward calculator
│   ├── tasks.py          # Easy / Medium / Hard configs + graders
│   └── environment.py    # Core scheduling logic
├── static/
│   └── index.html        # Timetable generator UI
├── openenv.yaml          # OpenEnv spec
├── Dockerfile
└── pyproject.toml
```

---

## Live Demo

**Space:** https://huggingface.co/spaces/shv1011/school-timetable-env  
**Health:** https://shv1011-school-timetable-env.hf.space/health  
**UI:** https://shv1011-school-timetable-env.hf.space/ui  
**Docs:** https://shv1011-school-timetable-env.hf.space/docs
