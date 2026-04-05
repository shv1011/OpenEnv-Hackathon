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

# ── Make project importable ───────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from client import SchoolTimetableEnvClient, TimetableAction

# ── Config (mirrors sample script pattern exactly) ────────────
IMAGE_NAME   = os.getenv("LOCAL_IMAGE_NAME")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")   or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME    = os.getenv("TASK",         "easy")
BENCHMARK    = os.getenv("BENCHMARK",    "school-timetable")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")
MAX_STEPS    = int(os.getenv("MAX_STEPS", "60"))
TEMPERATURE  = float(os.getenv("TEMPERATURE", "0.1"))

# ── System prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a school administrator building a conflict-free weekly timetable.

You receive:
- PENDING: subjects still needing scheduling per division (division_id: subject x sessions_left)
- FACULTY: each faculty's free slots, workload, subjects they can teach
- RECENT: last few assigned entries

Hard constraints (violations give negative reward):
1. No teacher double-booking (same faculty, same slot)
2. No room double-booking (same room, same slot)
3. Faculty only teach in their available slots
4. Faculty only teach subjects they are qualified for
5. Lab subjects MUST use lab rooms; theory MUST use classrooms
6. Do not exceed faculty max workload
7. A division can only have ONE class per slot

Strategy:
- Pick the most urgent subject from PENDING (most sessions remaining)
- Verify faculty is free in that slot AND division has no class there
- Pick a room matching the subject type (lab vs classroom)

Respond with ONLY a JSON object, no explanation:
{"action_type":"assign_class","division_id":"<id>","subject_id":"<id>","faculty_id":"<id>","room_id":"<id>","slot_id":"<id>"}

To reschedule an existing entry:
{"action_type":"reschedule_class","entry_id":"<id>","new_slot_id":"<id>"}

To remove a bad entry:
{"action_type":"remove_assignment","entry_id":"<id>"}"""


# ── Observation → prompt ──────────────────────────────────────
def build_prompt(obs) -> str:
    lines = [
        f"Step {obs.step_count} | {obs.completion_percentage:.1f}%"
        f" ({obs.assigned_sessions}/{obs.total_required_sessions} sessions)",
    ]
    if obs.pending_work:
        lines.append("PENDING:")
        for pw in obs.pending_work:
            s = ", ".join(f"{k}x{v}" for k, v in pw.get("subjects_needed", {}).items())
            lines.append(f"  {pw['division_id']}: {s}")
    if obs.faculty_status:
        lines.append("FACULTY:")
        for f in obs.faculty_status:
            free = f["free_slots"]
            fs = ", ".join(free[:8]) + (f" +{len(free)-8}more" if len(free) > 8 else "")
            lines.append(
                f"  {f['faculty_id']} {f['name']} [{f['current_load']}/{f['max_workload']}]"
                f" can={f['can_teach']} free={fs}"
            )
    if obs.violations:
        lines.append("LAST VIOLATIONS:")
        for v in obs.violations:
            lines.append(f"  {v.get('violation_type')}: {v.get('description','')[:80]}")
    if obs.timetable_entries:
        recent = obs.timetable_entries[-3:]
        lines.append(f"RECENT ({len(obs.timetable_entries)} total):")
        for e in recent:
            lines.append(
                f"  {e['entry_id']} {e['division_id']}|{e['subject_id']}"
                f"|{e['faculty_id']}|{e['room_id']}|{e['slot_id']}"
            )
    return "\n".join(lines)


# ── LLM → action ─────────────────────────────────────────────
def get_action(
    client: OpenAI,
    history: List[dict],
    obs_text: str,
) -> tuple[Optional[TimetableAction], str]:
    history.append({"role": "user", "content": obs_text})
    last_err = "unknown"

    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, *history],
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
                timeout=30,
            )
            raw = resp.choices[0].message.content.strip()
            d = json.loads(raw)
            action = TimetableAction(**d)
            history.append({"role": "assistant", "content": raw})

            atype = d.get("action_type", "")
            if atype == "assign_class":
                astr = f"assign({d.get('division_id')},{d.get('subject_id')},{d.get('slot_id')})"
            elif atype == "reschedule_class":
                astr = f"reschedule({d.get('entry_id')},{d.get('new_slot_id')})"
            else:
                astr = f"remove({d.get('entry_id')})"
            return action, astr

        except Exception as e:
            last_err = str(e)[:60].replace(" ", "_")
            time.sleep(1)

    return None, f"llm_error({last_err})"


# ── Grader ────────────────────────────────────────────────────
def compute_score(task_name: str, timetable_entries: list) -> float:
    try:
        from env import get_task
        from env.models import TimetableEntry
        task_cls = get_task(task_name)
        entries = [TimetableEntry(**e) for e in timetable_entries]
        return task_cls.grade(entries)
    except Exception:
        return 0.0


# ── Main ──────────────────────────────────────────────────────
def main():
    # [START] — always first, even before API key check
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    if not API_KEY:
        sys.stderr.write("ERROR: HF_TOKEN or API_KEY not set\n")
        print(f"[END]   success=false steps=0 score=0.00 rewards=", flush=True)
        sys.exit(1)

    llm = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    history: List[dict] = []
    rewards: List[float] = []
    step = 0
    success = False
    score = 0.0
    obs = None

    try:
        # Connect to environment
        if IMAGE_NAME:
            env_client = SchoolTimetableEnvClient.from_docker_image(IMAGE_NAME)
        else:
            env_client = SchoolTimetableEnvClient(base_url=ENV_URL)

        with env_client.sync() as env:
            obs = env.reset(task=TASK_NAME)
            done = False

            while not done and step < MAX_STEPS:
                action, action_str = get_action(llm, history, build_prompt(obs))

                if action is None:
                    # Failed to get action — emit step and stop
                    print(
                        f"[STEP]  step={step + 1} action=null"
                        f" reward=0.00 done=false error={action_str}",
                        flush=True,
                    )
                    break

                obs = env.step(action)
                step += 1
                reward = obs.reward
                rewards.append(reward)
                done = obs.done

                error_str = "null"
                if obs.violations:
                    error_str = obs.violations[0].get("violation_type", "VIOLATION")

                # [STEP] — one per step, immediately after env.step()
                print(
                    f"[STEP]  step={step}"
                    f" action={action_str}"
                    f" reward={reward:.2f}"
                    f" done={str(done).lower()}"
                    f" error={error_str}",
                    flush=True,
                )

            success = done and obs is not None and obs.completion_percentage == 100.0
            score = compute_score(TASK_NAME, obs.timetable_entries if obs else [])

    except Exception:
        traceback.print_exc(file=sys.stderr)

    # [END] — always emitted, even on exception
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END]   success={str(success).lower()}"
        f" steps={step}"
        f" score={score:.2f}"
        f" rewards={rewards_str}",
        flush=True,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
