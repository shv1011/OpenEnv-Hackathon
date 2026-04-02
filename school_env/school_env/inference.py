"""
inference.py — School Timetable Scheduling, OpenEnv Agent
==========================================================
MANDATORY environment variables:
    API_BASE_URL        LLM API endpoint
    MODEL_NAME          Model identifier
    HF_TOKEN            HuggingFace / API key
    LOCAL_IMAGE_NAME    Docker image name (if using from_docker_image)
    ENV_URL             OpenEnv server URL (default: http://localhost:7860)
    TASK                Task difficulty: easy | medium | hard (default: easy)

STDOUT FORMAT (strictly enforced):
    [START] task=<task> env=school-timetable model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>

Usage:
    # Local server
    python inference.py

    # Against HF Space
    ENV_URL=https://your-space.hf.space python inference.py

    # Hard task
    TASK=hard python inference.py
"""

from __future__ import annotations
import os, sys, json, time, traceback
from typing import List, Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client import SchoolTimetableEnvClient, TimetableAction

# ── Config from environment variables ────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")
TASK_NAME    = os.getenv("TASK", "easy")
BENCHMARK    = "school-timetable"
MAX_STEPS    = int(os.getenv("MAX_STEPS", "300"))
TEMPERATURE  = float(os.getenv("TEMPERATURE", "0.1"))
LOCAL_IMAGE  = os.getenv("LOCAL_IMAGE_NAME", "")

# ── System prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert school administrator building a conflict-free weekly class timetable.

Current state includes:
- PENDING WORK: subjects still needing scheduling per division
- FACULTY STATUS: each faculty's free slots, workload, and subjects they can teach
- RECENT ASSIGNMENTS: last few entries added

Constraints you MUST satisfy:
1. No teacher double-booking (same faculty, same slot)
2. No room double-booking (same room, same slot)
3. Faculty only teach in their available slots
4. Faculty only teach subjects they are qualified for
5. Lab subjects MUST use lab rooms; theory MUST use classrooms
6. Do not exceed faculty max workload
7. A division can only have ONE class per slot

Strategy:
- Pick the most urgent subject from PENDING WORK (most sessions remaining)
- Cross-check faculty free_slots vs division's already-used slots
- Match room type to subject (lab vs classroom)
- One action per turn

Respond with ONLY a JSON object, no explanation, no markdown:

For assigning a class:
{"action_type":"assign_class","division_id":"<id>","subject_id":"<id>","faculty_id":"<id>","room_id":"<id>","slot_id":"<id>"}

For rescheduling:
{"action_type":"reschedule_class","entry_id":"<id>","new_slot_id":"<id>","new_room_id":"<optional>","new_faculty_id":"<optional>"}

For removing:
{"action_type":"remove_assignment","entry_id":"<id>"}"""


# ── Observation → prompt text ─────────────────────────────────
def format_observation(obs) -> str:
    lines = [
        f"Step {obs.step_count} | Completion: {obs.completion_percentage:.1f}%"
        f" ({obs.assigned_sessions}/{obs.total_required_sessions} sessions)",
    ]
    if obs.pending_work:
        lines.append("PENDING WORK:")
        for pw in obs.pending_work:
            subjects = ", ".join(f"{s}x{n}" for s, n in pw.get("subjects_needed", {}).items())
            lines.append(f"  {pw['division_id']}: {subjects}")
    if obs.faculty_status:
        lines.append("FACULTY STATUS:")
        for f in obs.faculty_status:
            free = f["free_slots"]
            free_str = ", ".join(free[:10]) + (f" (+{len(free)-10} more)" if len(free) > 10 else "")
            lines.append(
                f"  {f['faculty_id']} {f['name']} [{f['current_load']}/{f['max_workload']}]"
                f" teaches={f['can_teach']} free={free_str}"
            )
    if obs.violations:
        lines.append("LAST VIOLATIONS:")
        for v in obs.violations:
            lines.append(f"  [{v.get('violation_type')}] {v.get('description','')[:100]}")
    if obs.timetable_entries:
        recent = obs.timetable_entries[-5:]
        lines.append(f"RECENT ({len(recent)} of {len(obs.timetable_entries)}):")
        for e in recent:
            lines.append(f"  [{e['entry_id']}] {e['division_id']}|{e['subject_id']}|{e['faculty_id']}|{e['room_id']}|{e['slot_id']}")
    return "\n".join(lines)


# ── LLM call ─────────────────────────────────────────────────
def get_action(llm: OpenAI, history: List[dict], obs_text: str) -> tuple[Optional[TimetableAction], str]:
    """Returns (action, raw_action_str). raw_action_str used for [STEP] logging."""
    history.append({"role": "user", "content": obs_text})
    last_error = None

    for attempt in range(1, 4):
        try:
            resp = llm.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, *history],
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content.strip()
            d = json.loads(raw)
            action = TimetableAction(**d)
            history.append({"role": "assistant", "content": raw})
            # Compact action string for [STEP] line
            atype = d.get("action_type", "unknown")
            if atype == "assign_class":
                action_str = f"assign({d.get('division_id')},{d.get('subject_id')},{d.get('slot_id')})"
            elif atype == "reschedule_class":
                action_str = f"reschedule({d.get('entry_id')},{d.get('new_slot_id')})"
            else:
                action_str = f"remove({d.get('entry_id')})"
            return action, action_str
        except Exception as e:
            last_error = str(e)
            time.sleep(1)

    return None, f"llm_error({last_error})"


# ── Main ──────────────────────────────────────────────────────
def main():
    if not API_KEY:
        # Still emit [START] and [END] so validator doesn't hang
        print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}")
        print(f"[END]   success=false steps=0 rewards=")
        sys.stderr.write("ERROR: HF_TOKEN / API_KEY / OPENAI_API_KEY not set\n")
        sys.exit(1)

    llm = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    history: List[dict] = []
    rewards: List[float] = []
    step = 0
    success = False
    last_error = None
    obs = None

    # ── [START] ───────────────────────────────────────────────
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        # Connect — supports from_docker_image if LOCAL_IMAGE_NAME is set
        if LOCAL_IMAGE:
            env_client = SchoolTimetableEnvClient.from_docker_image(LOCAL_IMAGE)
        else:
            env_client = SchoolTimetableEnvClient(base_url=ENV_URL)

        with env_client.sync() as env:
            obs = env.reset(task=TASK_NAME)
            done = False

            while not done and step < MAX_STEPS:
                obs_text = format_observation(obs)
                action, action_str = get_action(llm, history, obs_text)

                if action is None:
                    last_error = action_str
                    # Emit step with error, then break
                    print(
                        f"[STEP]  step={step+1} action=null"
                        f" reward=0.00 done=false error={last_error}",
                        flush=True,
                    )
                    break

                obs = env.step(action)
                step += 1
                reward = obs.reward
                rewards.append(reward)
                done = obs.done

                # Extract error from violations
                error_str = "null"
                if obs.violations:
                    vtype = obs.violations[0].get("violation_type", "VIOLATION")
                    error_str = vtype

                # ── [STEP] ────────────────────────────────────
                print(
                    f"[STEP]  step={step}"
                    f" action={action_str}"
                    f" reward={reward:.2f}"
                    f" done={str(done).lower()}"
                    f" error={error_str}",
                    flush=True,
                )

            success = done and obs is not None and obs.completion_percentage == 100.0

    except Exception as e:
        last_error = str(e)
        traceback.print_exc(file=sys.stderr)

    # ── [END] ─────────────────────────────────────────────────
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
