---
title: School Timetable Scheduling
emoji: 🏫
colorFrom: indigo
colorTo: teal
sdk: docker
pinned: false
license: mit
short_description: OpenEnv RL environment for school timetable scheduling
---

# 🏫 School Timetable Scheduling — OpenEnv Environment

A production-grade multi-step RL environment where an AI agent builds a conflict-free weekly class timetable.

## Install

```bash
pip install "openenv-school-timetable @ git+https://huggingface.co/spaces/shv1011/school-timetable-env"
```

## Quick Start

```python
from client import SchoolTimetableEnvClient, TimetableAction

with SchoolTimetableEnvClient(base_url="https://shv1011-school-timetable-env.hf.space").sync() as env:
    obs = env.reset(task="easy")
    obs = env.step(TimetableAction(
        action_type="assign_class",
        division_id="Sem1-A", subject_id="MATH",
        faculty_id="F001", room_id="CR101", slot_id="Mon-1"
    ))
    print(obs.completion_percentage)
```

## Tasks

| Task | Divisions | Sessions | Target Score |
|------|-----------|----------|--------------|
| easy | 1 | 9 | 0.90 |
| medium | 2 | 19 | 0.80 |
| hard | 3 | 39 | 0.70 |

## Endpoints

- `GET /health` — health check
- `POST /reset` — start new episode
- `POST /step` — execute action
- `GET /state` — current state
- `WS /ws` — persistent WebSocket session (used by TRL)
- `GET /ui` — interactive demo
- `GET /docs` — OpenAPI docs
