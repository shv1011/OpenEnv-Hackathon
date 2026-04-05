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

# 🏫 School Timetable Scheduling — OpenEnv Environment

A production-grade reinforcement learning environment where an AI agent builds a conflict-free weekly class timetable.

## Quick Start

```bash
# Health check
curl https://shv1011-school-timetable-env.hf.space/health

# Install client
pip install "openenv-school-timetable @ git+https://huggingface.co/spaces/shv1011/school-timetable-env"

# Run inference
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export ENV_URL=https://shv1011-school-timetable-env.hf.space
python inference.py
```

## Tasks

| Task | Divisions | Sessions | Target Score |
|------|-----------|----------|--------------|
| easy | 1 | 9 | 0.90 |
| medium | 2 | 19 | 0.80 |
| hard | 3 | 39 | 0.70 |
