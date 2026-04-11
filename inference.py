"""
Inference Script – School Timetable Scheduling Environment
==========================================================
Runs an LLM agent (via OpenAI-compatible API) through the scheduling environment
for ALL tasks (easy, medium, hard).

MANDATORY environment variables:
    HF_TOKEN / API_KEY / OPENAI_API_KEY   (required)
    API_BASE_URL     (default: https://router.huggingface.co/v1)
    MODEL_NAME       (default: Qwen/Qwen2.5-72B-Instruct)

STDOUT FORMAT (required by OpenEnv submission validator):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations
import os
import sys
import json
import time
import traceback
from typing import Any, Dict, List, Optional

from openai import OpenAI

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import (
    SchoolTimetableEnv,
    Action,
    ActionType,
    AssignClassAction,
    RescheduleClassAction,
    RemoveAssignmentAction,
    get_task,
)


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or ""
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")   or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK    = os.getenv("BENCHMARK",    "school-timetable")
TEMPERATURE  = float(os.getenv("TEMPERATURE", "0.1"))

# All three tasks to run
ALL_TASKS = ["easy", "medium", "hard"]

SYSTEM_PROMPT = """You are an expert school administrator responsible for building a valid weekly class timetable.

You will receive the current state of the timetable (partially or fully empty) along with:
- List of divisions and their required subjects
- Faculty members with their teaching qualifications and available time slots
- Rooms and their types (classroom / lab)
- Time slots

Your goal is to assign all required classes while satisfying ALL constraints:
1. No teacher double-booking (same faculty, same slot)
2. No room double-booking (same room, same slot)
3. Faculty must only teach within their available slots
4. Faculty can only teach subjects they are qualified for
5. Lab subjects MUST use lab rooms; theory subjects MUST use classrooms
6. Do not exceed faculty max workload
7. A division can only have ONE class per slot

Strategy:
- Carefully read the "pending_work" section to know what still needs scheduling
- Check "faculty_status" to find faculty with free slots
- Check "room_status" to find available rooms
- Always verify the slot is free for BOTH the faculty AND the room AND the division
- Prioritize subjects with the most remaining sessions first

You must respond with a SINGLE JSON action in exactly this format:

For assign_class:
{
  "action_type": "assign_class",
  "assign_class": {
    "division_id": "<division_id>",
    "subject_id": "<subject_id>",
    "faculty_id": "<faculty_id>",
    "room_id": "<room_id>",
    "slot_id": "<slot_id>"
  }
}

For reschedule_class:
{
  "action_type": "reschedule_class",
  "reschedule_class": {
    "entry_id": "<entry_id>",
    "new_slot_id": "<new_slot_id>",
    "new_room_id": "<optional_room_id>",
    "new_faculty_id": "<optional_faculty_id>"
  }
}

For remove_assignment:
{
  "action_type": "remove_assignment",
  "remove_assignment": {
    "entry_id": "<entry_id>"
  }
}

Return ONLY valid JSON. No explanation. No markdown. Just the JSON object."""


# ═══════════════════════════════════════════════════════════════
# Agent
# ═══════════════════════════════════════════════════════════════

class TimetableAgent:
    def __init__(
        self,
        api_key: str,
        base_url: str = API_BASE_URL,
        model: str = MODEL_NAME,
        temperature: float = 0.1,
        max_retries: int = 3,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.conversation_history: List[Dict] = []

    def reset(self):
        """Reset conversation history for a new task."""
        self.conversation_history = []

    def get_action(self, observation_text: str) -> Optional[Dict]:
        """Send observation to LLM and parse returned action dict."""
        self.conversation_history.append({
            "role": "user",
            "content": observation_text,
        })

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        *self.conversation_history,
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                    timeout=30,
                )
                raw = response.choices[0].message.content.strip()
                action_dict = json.loads(raw)

                self.conversation_history.append({
                    "role": "assistant",
                    "content": raw,
                })

                return action_dict

            except json.JSONDecodeError:
                time.sleep(1)
            except Exception:
                time.sleep(1)

        return None

    def summarize_observation(self, obs_dict: Dict) -> str:
        """Convert full observation to a concise, prompt-friendly text."""
        hints = obs_dict.get("available_actions_hint", {})
        progress = obs_dict.get("progress", {})
        violations = obs_dict.get("recent_violations", [])

        lines = [
            f"=== TIMETABLE STATE (Step {obs_dict.get('step_count', 0)}) ===",
            f"Completion: {progress.get('completion_percentage', 0):.1f}%",
            f"Sessions assigned: {progress.get('assigned_sessions', 0)} / {progress.get('total_required_sessions', 0)}",
            "",
        ]

        # Pending work
        pending = hints.get("pending_work", [])
        if pending:
            lines.append("PENDING WORK (subjects still needing scheduling):")
            for pw in pending:
                subjects_str = ", ".join(
                    f"{subj}({sessions} sessions)"
                    for subj, sessions in pw.get("subjects_needed", {}).items()
                )
                lines.append(f"  • {pw['division_id']} ({pw['division_name']}): {subjects_str}")
            lines.append("")

        # Faculty availability
        faculty_status = hints.get("faculty_status", [])
        if faculty_status:
            lines.append("FACULTY STATUS:")
            for f in faculty_status:
                load_str = f"{f['current_load']}/{f['max_workload']}"
                free_str = ", ".join(f["free_slots"][:8])
                if len(f["free_slots"]) > 8:
                    free_str += f" (+{len(f['free_slots'])-8} more)"
                lines.append(
                    f"  • {f['faculty_id']} – {f['name']} [{load_str} sessions] "
                    f"Teaches: {f['can_teach']} | Free: {free_str}"
                )
            lines.append("")

        # Room availability
        room_status = hints.get("room_status", [])
        if room_status:
            lines.append("ROOM STATUS:")
            for r in room_status:
                free_str = ", ".join(r["free_slots"][:6])
                if len(r["free_slots"]) > 6:
                    free_str += f" (+{len(r['free_slots'])-6} more)"
                lines.append(
                    f"  • {r['room_id']} [{r['room_type']}] Free: {free_str}"
                )
            lines.append("")

        # Recent violations
        if violations:
            lines.append("LAST ACTION VIOLATIONS:")
            for v in violations:
                lines.append(f"  ⚠️  [{v.get('violation_type')}] {v.get('description')}")
            lines.append("")

        # Current entries (last 5 for context)
        entries = obs_dict.get("timetable_entries", [])
        if entries:
            lines.append(f"RECENT ASSIGNMENTS (last 5 of {len(entries)}):")
            for e in entries[-5:]:
                lines.append(
                    f"  [{e['entry_id']}] {e['division_id']} | {e['subject_id']} | "
                    f"{e['faculty_id']} | {e['room_id']} | {e['slot_id']}"
                )

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Action parsing helpers
# ═══════════════════════════════════════════════════════════════

def parse_action(d: Dict) -> Action:
    """Parse a dict from LLM into an internal Action object."""
    action_type = d.get("action_type", "")

    if action_type == "assign_class":
        payload = d.get("assign_class", {})
        return Action(
            action_type=ActionType.ASSIGN_CLASS,
            assign_class=AssignClassAction(**payload),
        )
    elif action_type == "reschedule_class":
        payload = d.get("reschedule_class", {})
        return Action(
            action_type=ActionType.RESCHEDULE_CLASS,
            reschedule_class=RescheduleClassAction(**payload),
        )
    elif action_type == "remove_assignment":
        payload = d.get("remove_assignment", {})
        return Action(
            action_type=ActionType.REMOVE_ASSIGNMENT,
            remove_assignment=RemoveAssignmentAction(**payload),
        )
    else:
        raise ValueError(f"Unknown action_type: {action_type}")


def action_to_str(d: Dict) -> str:
    """Convert action dict to a compact string for stdout."""
    atype = d.get("action_type", "unknown")
    if atype == "assign_class":
        payload = d.get("assign_class", {})
        return f"assign({payload.get('division_id','')},{payload.get('subject_id','')},{payload.get('slot_id','')})"
    elif atype == "reschedule_class":
        payload = d.get("reschedule_class", {})
        return f"reschedule({payload.get('entry_id','')},{payload.get('new_slot_id','')})"
    elif atype == "remove_assignment":
        payload = d.get("remove_assignment", {})
        return f"remove({payload.get('entry_id','')})"
    return f"unknown({atype})"


# ═══════════════════════════════════════════════════════════════
# Run a single task
# ═══════════════════════════════════════════════════════════════

def run_single_task(task_id: str, agent: TimetableAgent) -> float:
    """
    Run inference for a single task with proper [START]/[STEP]/[END] format.
    Returns the final grader score.
    """
    # [START]
    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    task_cls = get_task(task_id)
    config = task_cls.get_config()
    env = SchoolTimetableEnv(config)

    # Reset agent conversation for new task
    agent.reset()

    rewards: List[float] = []
    step = 0
    success = False
    score = 0.0

    try:
        obs = env.reset()
        done = False

        while not done and step < config.max_steps:
            obs_dict = obs.dict()
            obs_text = agent.summarize_observation(obs_dict)

            action_dict = agent.get_action(obs_text)

            if action_dict is None:
                print(
                    f"[STEP]  step={step+1} action=null"
                    f" reward=0.00 done=false error=llm_failed",
                    flush=True,
                )
                break

            try:
                action = parse_action(action_dict)
                action_str = action_to_str(action_dict)
            except Exception as e:
                error_msg = str(e)[:40].replace(" ", "_")
                print(
                    f"[STEP]  step={step+1} action=parse_error"
                    f" reward=0.00 done=false error={error_msg}",
                    flush=True,
                )
                continue

            result = env.step(action)
            step += 1
            reward = result.reward
            done = result.done
            rewards.append(reward)

            # Check violations for error field
            violations = result.info.get("violations", [])
            error_str = "null"
            if violations:
                v = violations[0]
                error_str = v.get("violation_type", "VIOLATION") if isinstance(v, dict) else str(v)

            print(
                f"[STEP]  step={step}"
                f" action={action_str}"
                f" reward={reward:.2f}"
                f" done={str(done).lower()}"
                f" error={error_str}",
                flush=True,
            )

            obs = result.observation

            if done:
                completion = obs.progress.completion_percentage
                success = completion == 100.0
                break

        # Compute grader score
        score = task_cls.grade(env.get_entries())

    except Exception:
        traceback.print_exc(file=sys.stderr)

    # [END]
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END]   success={str(success).lower()}"
        f" steps={step}"
        f" score={score:.2f}"
        f" rewards={rewards_str}",
        flush=True,
    )

    return score


# ═══════════════════════════════════════════════════════════════
# Main — Run ALL tasks
# ═══════════════════════════════════════════════════════════════

def main():
    if not API_KEY:
        print("ERROR: No API key found. Set HF_TOKEN, API_KEY, or OPENAI_API_KEY.", file=sys.stderr)
        sys.exit(1)

    agent = TimetableAgent(
        api_key=API_KEY,
        base_url=API_BASE_URL,
        model=MODEL_NAME,
        temperature=TEMPERATURE,
    )

    scores = {}
    for task_id in ALL_TASKS:
        score = run_single_task(task_id, agent)
        scores[task_id] = score

    # Summary to stderr (not parsed by validator)
    print("\n--- Summary ---", file=sys.stderr)
    for tid, sc in scores.items():
        print(f"  {tid}: {sc:.4f}", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
