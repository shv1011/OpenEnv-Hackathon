"""
inference.py — School Timetable Scheduling, OpenEnv Agent
==========================================================
Place this file in the ROOT of your project directory.

MANDATORY environment variables:
    API_BASE_URL     LLM API endpoint
    MODEL_NAME       Model identifier
    HF_TOKEN         HuggingFace / API key
    ENV_URL          OpenEnv server URL (default: http://localhost:7860)
    TASK             Task: easy | medium | hard (default: easy)
    LOCAL_IMAGE_NAME Docker image name (optional, for from_docker_image)

STDOUT FORMAT:
    [START] task=<task> env=school-timetable model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>
"""

from __future__ import annotations
import os, sys, json, time, traceback
from typing import List, Optional

# ── Make project importable from any working directory ────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from openai import OpenAI
from client import SchoolTimetableEnvClient, TimetableAction

# ── Config ────────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")
TASK_NAME    = os.getenv("TASK", "easy")
BENCHMARK    = "school-timetable"
MAX_STEPS    = int(os.getenv("MAX_STEPS", "60"))   # easy=60, medium=150, hard=300
TEMPERATURE  = float(os.getenv("TEMPERATURE", "0.1"))
LOCAL_IMAGE  = os.getenv("LOCAL_IMAGE_NAME", "")

# ── System prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a school administrator building a conflict-free weekly timetable.

State includes:
- PENDING WORK: subjects still needing scheduling per division
- FACULTY STATUS: each faculty's free slots, workload, subjects they can teach
- RECENT ASSIGNMENTS: last few entries

Hard constraints:
1. No teacher double-booking (same faculty, same slot)
2. No room double-booking (same room, same slot)
3. Faculty only teach in their available slots
4. Faculty only teach subjects they are qualified for
5. Lab subjects MUST use lab rooms; theory MUST use classrooms
6. Do not exceed faculty max workload
7. A division can only have ONE class per slot

Strategy: pick the most urgent subject from PENDING WORK, verify faculty is free in that slot and division has no class there, pick matching room type.

Respond with ONLY a JSON object:
{"action_type":"assign_class","division_id":"<id>","subject_id":"<id>","faculty_id":"<id>","room_id":"<id>","slot_id":"<id>"}

Or to reschedule:
{"action_type":"reschedule_class","entry_id":"<id>","new_slot_id":"<id>"}

Or to remove:
{"action_type":"remove_assignment","entry_id":"<id>"}"""


# ── Observation formatter ─────────────────────────────────────
def format_observation(obs) -> str:
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
                f"  {f['faculty_id']} {f['name']} "
                f"[{f['current_load']}/{f['max_workload']}] "
                f"can={f['can_teach']} free={fs}"
            )
    if obs.violations:
        lines.append("VIOLATIONS:")
        for v in obs.violations:
            lines.append(f"  {v.get('violation_type')}: {v.get('description','')[:80]}")
    if obs.timetable_entries:
        recent = obs.timetable_entries[-3:]
        lines.append(f"RECENT ({len(obs.timetable_entries)} total):")
        for e in recent:
            lines.append(
                f"  {e['entry_id']} {e['division_id']}|"
                f"{e['subject_id']}|{e['faculty_id']}|{e['room_id']}|{e['slot_id']}"
            )
    return "\n".join(lines)


# ── LLM call ─────────────────────────────────────────────────
def get_action(
    llm: OpenAI, history: List[dict], obs_text: str
) -> tuple[Optional[TimetableAction], str]:
    history.append({"role": "user", "content": obs_text})
    last_err = "unknown"

    for attempt in range(1, 4):
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


# ── Main ──────────────────────────────────────────────────────
def main():
    # Always emit [START] first so validator doesn't hang
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    if not API_KEY:
        print(f"[END]   success=false steps=0 rewards=", flush=True)
        sys.stderr.write("ERROR: HF_TOKEN / API_KEY / OPENAI_API_KEY not set\n")
        sys.exit(1)

    llm = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    history: List[dict] = []
    rewards: List[float] = []
    step = 0
    success = False
    obs = None

    try:
        if LOCAL_IMAGE:
            env_client = SchoolTimetableEnvClient.from_docker_image(LOCAL_IMAGE)
        else:
            env_client = SchoolTimetableEnvClient(base_url=ENV_URL)

        with env_client.sync() as env:
            obs = env.reset(task=TASK_NAME)
            done = False

            while not done and step < MAX_STEPS:
                action, astr = get_action(llm, history, format_observation(obs))

                if action is None:
                    print(
                        f"[STEP]  step={step+1} action=null"
                        f" reward=0.00 done=false error={astr}",
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

                print(
                    f"[STEP]  step={step}"
                    f" action={astr}"
                    f" reward={reward:.2f}"
                    f" done={str(done).lower()}"
                    f" error={error_str}",
                    flush=True,
                )

            success = done and obs is not None and obs.completion_percentage == 100.0

    except Exception as e:
        traceback.print_exc(file=sys.stderr)

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END]   success={str(success).lower()}"
        f" steps={step}"
        f" rewards={rewards_str}",
        flush=True,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
