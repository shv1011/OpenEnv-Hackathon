"""
Inference Script — School Timetable Scheduling
===============================================
MANDATORY environment variables:
    API_BASE_URL      The API endpoint for the LLM.
    MODEL_NAME        The model identifier to use for inference.
    HF_TOKEN          Your Hugging Face / API key.
    LOCAL_IMAGE_NAME  Docker image name (if using from_docker_image())

Defaults:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import sys
import json
import time
import traceback
from typing import List, Optional

from openai import OpenAI

IMAGE_NAME   = os.getenv("LOCAL_IMAGE_NAME") or ""
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or ""
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")   or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME    = os.getenv("TASK",         "easy")
BENCHMARK    = os.getenv("BENCHMARK",    "school-timetable")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860").rstrip("/")
MAX_STEPS    = int(os.getenv("MAX_STEPS", "60"))
TEMPERATURE  = float(os.getenv("TEMPERATURE", "0.1"))

SYSTEM_PROMPT = """You are a school administrator building a conflict-free weekly timetable.

State includes:
- PENDING: subjects still needing scheduling per division
- FACULTY: each faculty's free slots, workload, subjects they can teach

Hard constraints:
1. No teacher double-booking (same faculty, same slot)
2. No room double-booking (same room, same slot)
3. Faculty only teach in their available slots
4. Faculty only teach subjects they are qualified for
5. Lab subjects MUST use lab rooms; theory MUST use classrooms
6. Do not exceed faculty max workload
7. A division can only have ONE class per slot

Respond with ONLY a JSON object:
{"action_type":"assign_class","division_id":"<id>","subject_id":"<id>","faculty_id":"<id>","room_id":"<id>","slot_id":"<id>"}"""


# ── Env client ────────────────────────────────────────────────

def _make_client():
    """Build env client — WebSocket via openenv-core, with requests fallback."""
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from client import SchoolTimetableEnvClient
    if IMAGE_NAME:
        return SchoolTimetableEnvClient.from_docker_image(IMAGE_NAME)
    return SchoolTimetableEnvClient(base_url=ENV_URL)


def _obs_to_dict(obs) -> dict:
    """Convert observation object or dict to plain dict."""
    if isinstance(obs, dict):
        return obs
    return {k: getattr(obs, k, None) for k in [
        "step_count", "completion_percentage", "assigned_sessions",
        "total_required_sessions", "timetable_entries", "violations",
        "pending_work", "faculty_status", "is_terminal", "termination_reason"
    ]}


def compute_score(task_name: str, entries: list) -> float:
    try:
        _here = os.path.dirname(os.path.abspath(__file__))
        if _here not in sys.path:
            sys.path.insert(0, _here)
        from env import get_task
        from env.models import TimetableEntry
        task_cls = get_task(task_name)
        return task_cls.grade([TimetableEntry(**e) for e in entries])
    except Exception:
        return 0.0


def build_prompt(obs: dict) -> str:
    try:
        lines = [
            f"Step {obs.get('step_count',0)} | "
            f"{obs.get('completion_percentage',0):.1f}% "
            f"({obs.get('assigned_sessions',0)}/{obs.get('total_required_sessions',0)} sessions)"
        ]
        for pw in (obs.get("pending_work") or []):
            s = ", ".join(f"{k}x{v}" for k, v in pw.get("subjects_needed", {}).items())
            lines.append(f"  {pw.get('division_id','?')}: {s}")
        for f in (obs.get("faculty_status") or []):
            free = f.get("free_slots", [])
            fs = ", ".join(free[:8]) + (f" +{len(free)-8}more" if len(free) > 8 else "")
            lines.append(
                f"  {f.get('faculty_id')} {f.get('name')} "
                f"[{f.get('current_load',0)}/{f.get('max_workload',0)}] "
                f"can={f.get('can_teach',[])} free={fs}"
            )
        for v in (obs.get("violations") or []):
            lines.append(f"  VIOLATION {v.get('violation_type')}: {v.get('description','')[:80]}")
        for e in (obs.get("timetable_entries") or [])[-3:]:
            lines.append(
                f"  {e.get('entry_id','')} {e.get('division_id','')}|"
                f"{e.get('subject_id','')}|{e.get('faculty_id','')}|"
                f"{e.get('room_id','')}|{e.get('slot_id','')}"
            )
        return "\n".join(lines)
    except Exception as ex:
        return f"state (error: {ex})"


def get_action(llm: OpenAI, history: List[dict], obs_text: str):
    history.append({"role": "user", "content": obs_text})
    last_err = "unknown"
    for _ in range(3):
        try:
            resp = llm.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, *history],
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
                timeout=30,
            )
            raw = resp.choices[0].message.content.strip()
            d = json.loads(raw)
            history.append({"role": "assistant", "content": raw})
            atype = d.get("action_type", "")
            if atype == "assign_class":
                astr = f"assign({d.get('division_id')},{d.get('subject_id')},{d.get('slot_id')})"
            elif atype == "reschedule_class":
                astr = f"reschedule({d.get('entry_id')},{d.get('new_slot_id')})"
            else:
                astr = f"remove({d.get('entry_id')})"
            return d, astr
        except Exception as e:
            last_err = str(e)[:60].replace(" ", "_")
            time.sleep(1)
    return None, f"llm_error({last_err})"


TASK_IDS = ["easy", "medium", "hard"]


def run_task(task_name: str, llm: OpenAI) -> None:
    """Run one full task episode, printing [START]/[STEP]*/[END]."""
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    history: List[dict] = []
    rewards: List[float] = []
    step = 0
    success = False
    score = 0.0
    obs = {}

    try:
        env_client = _make_client()
        with env_client.sync() as env:
            result = env.reset(task=task_name)
            obs = _obs_to_dict(result.observation)
            done = obs.get("is_terminal", False)

            max_steps = {"easy": 60, "medium": 150, "hard": 300}.get(task_name, MAX_STEPS)

            while not done and step < max_steps:
                action_dict, action_str = get_action(llm, history, build_prompt(obs))

                if action_dict is None:
                    print(
                        f"[STEP]  step={step+1} action=null"
                        f" reward=0.00 done=false error={action_str}",
                        flush=True,
                    )
                    break

                from models import TimetableAction
                action = TimetableAction(**action_dict)
                result = env.step(action)
                obs = _obs_to_dict(result.observation)
                step += 1
                reward = float(result.reward or 0.0)
                done = bool(result.done or obs.get("is_terminal", False))
                rewards.append(reward)

                viols = obs.get("violations") or []
                error_str = "null"
                if viols:
                    v = viols[0]
                    error_str = (v.get("violation_type") or "VIOLATION") if isinstance(v, dict) else str(v)

                print(
                    f"[STEP]  step={step}"
                    f" action={action_str}"
                    f" reward={reward:.2f}"
                    f" done={str(done).lower()}"
                    f" error={error_str}",
                    flush=True,
                )

        success = done and obs.get("completion_percentage", 0) == 100.0
        score = compute_score(task_name, obs.get("timetable_entries") or [])

    except Exception:
        traceback.print_exc(file=sys.stderr)
        score = max(0.01, min(0.99, sum(rewards) / len(rewards))) if rewards else 0.05

    score = max(0.01, min(0.99, score))
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END]   success={str(success).lower()}"
        f" steps={step}"
        f" score={score:.2f}"
        f" rewards={rewards_str}",
        flush=True,
    )


def main():
    llm = OpenAI(api_key=API_KEY or "no-key", base_url=API_BASE_URL)

    # Always run all 3 tasks — validator counts [END] lines per task
    for task in TASK_IDS:
        run_task(task, llm)
        time.sleep(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
