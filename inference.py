"""
Inference Script – School Timetable Scheduling Environment
==========================================================
Runs an LLM agent (via OpenAI-compatible API) through the scheduling environment.

Usage:
    python inference.py --task easy --model gpt-4o
    python inference.py --task medium --debug
    python inference.py --task hard --export-csv

Environment Variables:
    OPENAI_API_KEY   (required)
    API_BASE_URL     (default: https://api.openai.com/v1)
    MODEL_NAME       (default: gpt-4o)
"""

from __future__ import annotations
import os
import sys
import json
import time
import argparse
import traceback
from typing import Any, Dict, List, Optional
from datetime import datetime

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
    export_all_faculty_timetables_csv,
    export_master_timetable_csv,
    format_timetable_text,
)


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o"

SYSTEM_PROMPT = """You are an expert school administrator responsible for building a valid weekly class timetable.

You will receive the current state of the timetable (partially or fully empty) along with:
- List of divisions and their required subjects
- Faculty members with their teaching qualifications and available time slots
- Rooms and their types (classroom / lab)
- Time slots

Your goal is to assign all required classes while satisfying ALL constraints:
1. ❌ No teacher double-booking (same faculty, same slot)
2. ❌ No room double-booking (same room, same slot)
3. ❌ Faculty must only teach within their available slots
4. ❌ Faculty can only teach subjects they are qualified for
5. ❌ Lab subjects MUST use lab rooms; theory subjects MUST use classrooms
6. ❌ Do not exceed faculty max workload
7. ❌ A division can only have ONE class per slot

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
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        debug: bool = False,
        temperature: float = 0.1,
        max_retries: int = 3,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.debug = debug
        self.temperature = temperature
        self.max_retries = max_retries
        self.conversation_history: List[Dict] = []

    def get_action(self, observation_text: str) -> Optional[Action]:
        """Send observation to LLM and parse returned action."""
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
                )
                raw = response.choices[0].message.content.strip()

                if self.debug:
                    print(f"    LLM raw response: {raw[:300]}")

                action_dict = json.loads(raw)
                action = self._parse_action(action_dict)

                self.conversation_history.append({
                    "role": "assistant",
                    "content": raw,
                })

                return action

            except json.JSONDecodeError as e:
                print(f"  ⚠️  JSON parse error (attempt {attempt}): {e}")
            except Exception as e:
                print(f"  ⚠️  LLM error (attempt {attempt}): {e}")
                if self.debug:
                    traceback.print_exc()
                time.sleep(1)

        return None

    def _parse_action(self, d: Dict) -> Action:
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
# Main Loop
# ═══════════════════════════════════════════════════════════════

def run_inference(
    task_id: str = "easy",
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    debug: bool = False,
    export_csv: bool = False,
    log_file: Optional[str] = None,
):
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    # Override from env
    base_url = os.getenv("API_BASE_URL", base_url)
    model = os.getenv("MODEL_NAME", model)

    print("=" * 65)
    print(f"  School Timetable Scheduling — OpenEnv Agent")
    print(f"  Task:  {task_id.upper()}")
    print(f"  Model: {model}")
    print(f"  Time:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # Setup
    task_cls = get_task(task_id)
    config = task_cls.get_config()
    env = SchoolTimetableEnv(config, debug=debug)
    agent = TimetableAgent(
        api_key=api_key,
        base_url=base_url,
        model=model,
        debug=debug,
    )

    log_lines: List[str] = []

    def log(msg: str):
        print(msg)
        log_lines.append(msg)

    # ── [START] ───────────────────────────────
    log("\n[START]")
    log(f"Task: {task_id} | Divisions: {len(config.divisions)} | "
        f"Faculty: {len(config.faculty)} | Max steps: {config.max_steps}")

    obs = env.reset()
    done = False
    step_rewards: List[float] = []
    invalid_count = 0
    valid_count = 0

    # ── Main Loop ─────────────────────────────
    while not done:
        obs_dict = obs.dict()
        obs_text = agent.summarize_observation(obs_dict)

        if debug:
            print("\n--- Observation ---")
            print(obs_text[:1000])
            print("------------------")

        action = agent.get_action(obs_text)

        if action is None:
            log(f"\n⚠️  Agent failed to produce a valid action at step {obs.step_count + 1}. Stopping.")
            break

        result = env.step(action)

        step_rewards.append(result.reward)
        if result.info.get("valid", True):
            valid_count += 1
        else:
            invalid_count += 1

        # ── [STEP] ────────────────────────────
        log(
            f"\n[STEP {result.observation.step_count}] "
            f"action={action.action_type} | "
            f"valid={result.info.get('valid')} | "
            f"reward={result.reward:+.3f} | "
            f"completion={result.observation.progress.completion_percentage:.1f}%"
        )

        if debug:
            rb = result.reward_breakdown
            log(
                f"  reward_breakdown: valid={rb.validity_bonus:+.2f} "
                f"efficiency={rb.efficiency_bonus:+.2f} "
                f"conflict={rb.conflict_penalty:+.2f} "
                f"redundancy={rb.redundancy_penalty:+.2f}"
            )

        if result.info.get("violations"):
            for v in result.info["violations"][:2]:
                log(f"  ⚠️  {v.get('violation_type')}: {v.get('description')[:80]}")

        obs = result.observation
        done = result.done

        if done:
            break

    # ── [END] ─────────────────────────────────
    metrics = env.get_summary_metrics()
    final_score = task_cls.grade(env.get_entries())

    log("\n[END]")
    log("=" * 65)
    log(f"  Episode complete. Reason: {obs.termination_reason}")
    log(f"  Steps taken       : {metrics['step_count']}")
    log(f"  Valid actions     : {valid_count}")
    log(f"  Invalid actions   : {invalid_count}")
    log(f"  Sessions assigned : {metrics['total_entries']}")
    log(f"  Completion        : {metrics['completion_percentage']:.1f}%")
    log(f"  Total conflicts   : {metrics['total_conflicts']}")
    log(f"  Cumul. reward     : {metrics['cumulative_reward']:+.4f}")
    log(f"  ★ FINAL SCORE     : {final_score:.4f} / 1.0000")
    log("=" * 65)

    # ── Export ────────────────────────────────
    entries = env.get_entries()

    if export_csv and entries:
        log("\n📄 Exporting timetables...")
        out_dir = f"timetables/{task_id}"
        faculty_files = export_all_faculty_timetables_csv(entries, config, out_dir)
        master_path = f"{out_dir}/master_timetable.csv"
        export_master_timetable_csv(entries, config, master_path)
        log(f"  ✓ Faculty CSVs: {out_dir}/")
        log(f"  ✓ Master CSV:   {master_path}")

        # Print a sample timetable
        if config.faculty:
            sample = config.faculty[0].name
            log(f"\n📋 Sample timetable for {sample}:")
            log(format_timetable_text(sample, entries, config))

    # ── Save log ──────────────────────────────
    if log_file:
        with open(log_file, "w") as f:
            f.write("\n".join(log_lines))
        print(f"\n📝 Log saved to {log_file}")

    return final_score


# ═══════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="School Timetable Scheduling – OpenEnv LLM Agent"
    )
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard"],
        default="easy",
        help="Task difficulty (default: easy)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LLM model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"API base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export timetable CSVs after run",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to save run log",
    )
    args = parser.parse_args()

    score = run_inference(
        task_id=args.task,
        model=args.model,
        base_url=args.base_url,
        debug=args.debug,
        export_csv=args.export_csv,
        log_file=args.log_file,
    )
    sys.exit(0 if score >= 0.8 else 1)


if __name__ == "__main__":
    main()
