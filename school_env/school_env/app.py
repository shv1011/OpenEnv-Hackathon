"""
FastAPI backend for the School Timetable Scheduling Environment UI.
Serves the static frontend and exposes REST endpoints wrapping the env.
"""

from __future__ import annotations
import os
import sys
from typing import Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import csv, io, random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import (
    SchoolTimetableEnv,
    Action, ActionType,
    AssignClassAction, RescheduleClassAction, RemoveAssignmentAction,
    get_task,
)

app = FastAPI(title="School Timetable Env")

# ── State (single session for demo) ──────────────────────────
_env: Optional[SchoolTimetableEnv] = None
_task_id: str = "easy"
_task_cls = None


def _get_env() -> SchoolTimetableEnv:
    if _env is None:
        raise HTTPException(status_code=400, detail="No active session. Call /api/reset first.")
    return _env


# ── Request models ────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: str = "easy"

class AssignRequest(BaseModel):
    division_id: str
    subject_id: str
    faculty_id: str
    room_id: str
    slot_id: str

class RescheduleRequest(BaseModel):
    entry_id: str
    new_slot_id: str
    new_room_id: Optional[str] = None
    new_faculty_id: Optional[str] = None

class RemoveRequest(BaseModel):
    entry_id: str


# ── Helpers ───────────────────────────────────────────────────

def _obs_to_dict(obs, config) -> Dict[str, Any]:
    """Serialize observation + config metadata for the frontend."""
    return {
        "step_count": obs.step_count,
        "is_terminal": obs.is_terminal,
        "termination_reason": obs.termination_reason,
        "progress": obs.progress.model_dump(),
        "resource_utilization": obs.resource_utilization.model_dump(),
        "recent_violations": [v.model_dump() for v in obs.recent_violations],
        "timetable_entries": [e.model_dump() for e in obs.timetable_entries],
        "available_actions_hint": obs.available_actions_hint,
        "config": {
            "divisions": [{"division_id": d.division_id, "name": d.name, "subjects": d.subjects} for d in config.divisions],
            "subjects": [{"subject_id": s.subject_id, "name": s.name, "requires_lab": s.requires_lab, "sessions_per_week": s.sessions_per_week} for s in config.subjects],
            "faculty": [{"faculty_id": f.faculty_id, "name": f.name, "subjects_can_teach": f.subjects_can_teach, "available_slots": f.available_slots, "max_workload": f.max_workload} for f in config.faculty],
            "rooms": [{"room_id": r.room_id, "room_type": r.room_type, "capacity": r.capacity} for r in config.rooms],
            "time_slots": [{"slot_id": s.slot_id, "day": s.day, "period": s.period} for s in config.time_slots],
            "max_steps": config.max_steps,
        },
    }


# ── API Routes ────────────────────────────────────────────────

@app.post("/api/reset")
def reset(req: ResetRequest):
    global _env, _task_id, _task_cls
    if req.task not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail="task must be easy, medium, or hard")
    _task_id = req.task
    _task_cls = get_task(req.task)
    config = _task_cls.get_config()
    _env = SchoolTimetableEnv(config)
    obs = _env.reset()
    return {"ok": True, "observation": _obs_to_dict(obs, config)}


@app.get("/api/state")
def state():
    env = _get_env()
    obs = env.state()
    return {"observation": _obs_to_dict(obs, env.config)}


@app.post("/api/assign")
def assign(req: AssignRequest):
    env = _get_env()
    action = Action(
        action_type=ActionType.ASSIGN_CLASS,
        assign_class=AssignClassAction(**req.model_dump()),
    )
    result = env.step(action)
    score = _task_cls.grade(env.get_entries()) if result.done else None
    return {
        "reward": result.reward,
        "reward_breakdown": result.reward_breakdown.model_dump(),
        "valid": result.info["valid"],
        "violations": result.info["violations"],
        "done": result.done,
        "final_score": score,
        "observation": _obs_to_dict(result.observation, env.config),
    }


@app.post("/api/reschedule")
def reschedule(req: RescheduleRequest):
    env = _get_env()
    action = Action(
        action_type=ActionType.RESCHEDULE_CLASS,
        reschedule_class=RescheduleClassAction(**req.model_dump()),
    )
    result = env.step(action)
    score = _task_cls.grade(env.get_entries()) if result.done else None
    return {
        "reward": result.reward,
        "valid": result.info["valid"],
        "violations": result.info["violations"],
        "done": result.done,
        "final_score": score,
        "observation": _obs_to_dict(result.observation, env.config),
    }


@app.post("/api/remove")
def remove(req: RemoveRequest):
    env = _get_env()
    action = Action(
        action_type=ActionType.REMOVE_ASSIGNMENT,
        remove_assignment=RemoveAssignmentAction(**req.model_dump()),
    )
    result = env.step(action)
    return {
        "reward": result.reward,
        "valid": result.info["valid"],
        "done": result.done,
        "observation": _obs_to_dict(result.observation, env.config),
    }


@app.get("/api/metrics")
def metrics():
    env = _get_env()
    m = env.get_summary_metrics()
    score = _task_cls.grade(env.get_entries())
    m["task_score"] = score
    return m


# ═══════════════════════════════════════════════════════════════
# Timetable Generator API
# ═══════════════════════════════════════════════════════════════

class SubjectInput(BaseModel):
    name: str
    lectures_per_week: int
    faculty: list[str]          # list of faculty names (one per division or shared)

class GenerateRequest(BaseModel):
    class_name: str             # e.g. "Class 10"
    num_divisions: int          # e.g. 3  → Div A, B, C
    subjects: list[SubjectInput]
    working_days: list[str]     # e.g. ["Mon","Tue","Wed","Thu","Fri"]
    periods_per_day: int        # e.g. 7
    break_after_period: int     # e.g. 4  → period 4 is break for all

# ── Generator logic ───────────────────────────────────────────

def generate_timetable(req: GenerateRequest) -> dict:
    """
    Greedy constraint-satisfying timetable generator.

    Rules:
    - Each division gets its own slot grid
    - Faculty cannot teach two divisions in the same slot
    - Break slot is the same for all divisions
    - Every subject gets exactly lectures_per_week slots per division
    - Faculty assignment: if len(faculty) == num_divisions → one per div
                          if len(faculty) == 1 → same teacher all divs (must not clash)
    """
    days = req.working_days
    ppd  = req.periods_per_day
    ndiv = req.num_divisions
    div_names = [chr(65 + i) for i in range(ndiv)]   # A, B, C ...
    brk = req.break_after_period

    # All slots in order
    all_slots = []
    for d in days:
        for p in range(1, ppd + 1):
            all_slots.append((d, p))

    # Non-break slots
    teaching_slots = [(d, p) for d, p in all_slots if p != brk]

    # Build per-division subject→faculty mapping
    # If a subject has one faculty → shared across all divs
    # If it has ndiv faculty → one per div
    subj_faculty: dict[str, list[str]] = {}   # subject → [fac_for_divA, fac_for_divB, ...]
    for s in req.subjects:
        if len(s.faculty) == 1:
            subj_faculty[s.name] = [s.faculty[0]] * ndiv
        elif len(s.faculty) >= ndiv:
            subj_faculty[s.name] = s.faculty[:ndiv]
        else:
            # cycle through provided faculty
            subj_faculty[s.name] = [s.faculty[i % len(s.faculty)] for i in range(ndiv)]

    # Timetable: div_idx → slot → subject (or "BREAK" or "FREE")
    timetable: list[dict] = [{} for _ in range(ndiv)]

    # Mark break slots
    for di in range(ndiv):
        for d in days:
            timetable[di][(d, brk)] = "BREAK"

    # Faculty occupancy: faculty_name → set of (day, period) slots used
    fac_busy: dict[str, set] = {}

    def is_slot_free(di: int, slot: tuple, faculty_name: str) -> bool:
        if slot in timetable[di]:
            return False
        busy = fac_busy.get(faculty_name, set())
        return slot not in busy

    def assign(di: int, slot: tuple, subject: str, faculty_name: str):
        timetable[di][slot] = subject
        fac_busy.setdefault(faculty_name, set()).add(slot)

    # Build assignment queue: list of (div_idx, subject_name, faculty_name, remaining_count)
    # Sort by most constrained first (fewest available faculty = shared faculty first)
    tasks = []
    for s in req.subjects:
        facs = subj_faculty[s.name]
        shared = len(set(facs)) == 1   # same teacher for all divs → most constrained
        for di in range(ndiv):
            tasks.append({
                "di": di,
                "subject": s.name,
                "faculty": facs[di],
                "remaining": s.lectures_per_week,
                "shared": shared,
            })

    # Sort: shared faculty tasks first (most constrained), then by remaining desc
    tasks.sort(key=lambda t: (not t["shared"], -t["remaining"]))

    # Greedy assignment with backtracking-lite: shuffle slots to spread across week
    shuffled_slots = list(teaching_slots)

    unresolved = []
    for task in tasks:
        di, subj, fac, needed = task["di"], task["subject"], task["faculty"], task["remaining"]
        assigned = 0
        # Try to spread: sort slots by day then period, but interleave divisions
        candidate_slots = [s for s in shuffled_slots if is_slot_free(di, s, fac)]
        # Spread across days: pick from different days first
        spread = _spread_slots(candidate_slots, needed, days)
        for slot in spread:
            if assigned >= needed:
                break
            if is_slot_free(di, slot, fac):
                assign(di, slot, subj, fac)
                assigned += 1
        if assigned < needed:
            unresolved.append({"division": div_names[di], "subject": subj, "faculty": fac,
                                "assigned": assigned, "needed": needed})

    # Build output grid
    result_grid = []   # list of {division, day, period, subject, faculty}
    for di, div in enumerate(div_names):
        for d in days:
            for p in range(1, ppd + 1):
                slot = (d, p)
                entry = timetable[di].get(slot, "FREE")
                fac_name = ""
                if entry not in ("BREAK", "FREE"):
                    fac_name = subj_faculty[entry][di]
                result_grid.append({
                    "division": div,
                    "day": d,
                    "period": p,
                    "subject": entry,
                    "faculty": fac_name,
                })

    # Faculty timetables
    all_faculty = sorted(set(f for flist in subj_faculty.values() for f in flist))
    faculty_grids = {}
    for fac in all_faculty:
        rows = []
        for d in days:
            for p in range(1, ppd + 1):
                slot = (d, p)
                divs_teaching = []
                for di, div in enumerate(div_names):
                    entry = timetable[di].get(slot, "FREE")
                    if entry not in ("BREAK", "FREE") and subj_faculty[entry][di] == fac:
                        divs_teaching.append({"division": div, "subject": entry})
                rows.append({
                    "day": d,
                    "period": p,
                    "is_break": p == brk,
                    "classes": divs_teaching,
                })
        faculty_grids[fac] = rows

    return {
        "class_name": req.class_name,
        "divisions": div_names,
        "days": days,
        "periods_per_day": ppd,
        "break_after_period": brk,
        "grid": result_grid,
        "faculty_grids": faculty_grids,
        "unresolved": unresolved,
        "subjects": [{"name": s.name, "lectures_per_week": s.lectures_per_week,
                      "faculty": subj_faculty[s.name]} for s in req.subjects],
    }


def _spread_slots(candidates: list, needed: int, days: list) -> list:
    """Pick `needed` slots spread across days as evenly as possible."""
    if len(candidates) <= needed:
        return candidates
    by_day: dict[str, list] = {d: [] for d in days}
    for slot in candidates:
        by_day[slot[0]].append(slot)
    result = []
    # Round-robin across days
    day_iters = {d: iter(by_day[d]) for d in days}
    while len(result) < needed:
        added = False
        for d in days:
            if len(result) >= needed:
                break
            try:
                result.append(next(day_iters[d]))
                added = True
            except StopIteration:
                pass
        if not added:
            break
    return result


# ── Generator endpoints ───────────────────────────────────────

_last_generated: dict = {}

@app.post("/api/generate")
def generate(req: GenerateRequest):
    global _last_generated
    if req.num_divisions < 1 or req.num_divisions > 10:
        raise HTTPException(400, "num_divisions must be 1–10")
    if not req.subjects:
        raise HTTPException(400, "At least one subject required")
    if not req.working_days:
        raise HTTPException(400, "At least one working day required")
    if req.break_after_period < 1 or req.break_after_period > req.periods_per_day:
        raise HTTPException(400, "break_after_period out of range")
    result = generate_timetable(req)
    _last_generated = result
    return result


@app.get("/api/download/faculty/{faculty_name}")
def download_faculty_csv(faculty_name: str):
    if not _last_generated:
        raise HTTPException(400, "No timetable generated yet")
    grids = _last_generated.get("faculty_grids", {})
    if faculty_name not in grids:
        raise HTTPException(404, f"Faculty '{faculty_name}' not found")
    days  = _last_generated["days"]
    ppd   = _last_generated["periods_per_day"]
    brk   = _last_generated["break_after_period"]
    rows  = grids[faculty_name]

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Day", "Period", "Subject", "Division", "Note"])
    for r in rows:
        note = "BREAK" if r["is_break"] else ""
        if r["classes"]:
            for c in r["classes"]:
                w.writerow([r["day"], r["period"], c["subject"], c["division"], note])
        else:
            w.writerow([r["day"], r["period"], "FREE" if not r["is_break"] else "BREAK", "", note])
    buf.seek(0)
    safe = faculty_name.replace(" ", "_").replace(".", "")
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{safe}_timetable.csv"'},
    )


@app.get("/api/download/all")
def download_all_csv():
    if not _last_generated:
        raise HTTPException(400, "No timetable generated yet")
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Faculty", "Day", "Period", "Subject", "Division", "Note"])
    grids = _last_generated.get("faculty_grids", {})
    brk   = _last_generated["break_after_period"]
    for fac, rows in grids.items():
        for r in rows:
            note = "BREAK" if r["is_break"] else ""
            if r["classes"]:
                for c in r["classes"]:
                    w.writerow([fac, r["day"], r["period"], c["subject"], c["division"], note])
            else:
                w.writerow([fac, r["day"], r["period"], "FREE" if not r["is_break"] else "BREAK", "", note])
        w.writerow([])
    buf.seek(0)
    cls = _last_generated.get("class_name","timetable").replace(" ","_")
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{cls}_all_faculty.csv"'},
    )


@app.get("/api/download/division/{division}")
def download_division_csv(division: str):
    if not _last_generated:
        raise HTTPException(400, "No timetable generated yet")
    grid  = _last_generated.get("grid", [])
    days  = _last_generated["days"]
    ppd   = _last_generated["periods_per_day"]
    rows  = [r for r in grid if r["division"] == division]
    if not rows:
        raise HTTPException(404, f"Division '{division}' not found")
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Day", "Period", "Subject", "Faculty"])
    for r in rows:
        w.writerow([r["day"], r["period"], r["subject"], r["faculty"]])
    buf.seek(0)
    cls = _last_generated.get("class_name","timetable").replace(" ","_")
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{cls}_Div{division}_timetable.csv"'},
    )


# ── Static frontend ───────────────────────────────────────────

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
def root():
    return FileResponse(str(static_dir / "index.html"))

@app.get("/generate")
def generator_page():
    return FileResponse(str(static_dir / "generator.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
