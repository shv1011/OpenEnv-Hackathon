"""
Timetable Export Module
=======================
Generates per-faculty and per-division timetable reports.
Supports:
    - CSV export (mandatory)
    - PDF export (optional, requires reportlab)
    - Plain-text summary
"""

from __future__ import annotations
import csv
import io
import os
from typing import List, Dict, Optional, Any
from pathlib import Path

from .models import TimetableEntry, SchoolConfig


# ─────────────────────────────────────────────
# Core Data Extraction
# ─────────────────────────────────────────────

def get_faculty_timetable(
    faculty_name: str,
    entries: List[TimetableEntry],
    config: SchoolConfig,
) -> List[Dict[str, str]]:
    """
    Returns structured timetable rows for a given faculty member.

    Parameters
    ----------
    faculty_name : str
        Display name of the faculty member.
    entries : List[TimetableEntry]
        All current timetable entries.
    config : SchoolConfig
        Environment configuration for lookups.

    Returns
    -------
    List[Dict] with keys: slot_id, day, period, division, subject, room
    """
    subjects_map = {s.subject_id: s for s in config.subjects}
    divisions_map = {d.division_id: d for d in config.divisions}
    slots_map = {s.slot_id: s for s in config.time_slots}
    rooms_map = {r.room_id: r for r in config.rooms}

    # Find faculty_id from name
    faculty = next(
        (f for f in config.faculty if f.name.lower() == faculty_name.lower()),
        None,
    )
    if faculty is None:
        raise ValueError(f"Faculty '{faculty_name}' not found in configuration.")

    rows = []
    for e in entries:
        if e.faculty_id != faculty.faculty_id:
            continue
        slot = slots_map.get(e.slot_id)
        subject = subjects_map.get(e.subject_id)
        division = divisions_map.get(e.division_id)
        room = rooms_map.get(e.room_id)

        rows.append({
            "slot_id": e.slot_id,
            "day": slot.day if slot else "?",
            "period": str(slot.period) if slot else "?",
            "division": division.name if division else e.division_id,
            "subject": subject.name if subject else e.subject_id,
            "room": e.room_id,
            "room_type": room.room_type if room else "?",
        })

    # Sort by day + period
    day_order = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4}
    rows.sort(key=lambda r: (day_order.get(r["day"], 9), int(r["period"])))
    return rows


def get_division_timetable(
    division_id: str,
    entries: List[TimetableEntry],
    config: SchoolConfig,
) -> List[Dict[str, str]]:
    """
    Returns all scheduled classes for a specific division.
    """
    subjects_map = {s.subject_id: s for s in config.subjects}
    faculty_map = {f.faculty_id: f for f in config.faculty}
    slots_map = {s.slot_id: s for s in config.time_slots}

    rows = []
    for e in entries:
        if e.division_id != division_id:
            continue
        slot = slots_map.get(e.slot_id)
        subject = subjects_map.get(e.subject_id)
        faculty = faculty_map.get(e.faculty_id)

        rows.append({
            "slot_id": e.slot_id,
            "day": slot.day if slot else "?",
            "period": str(slot.period) if slot else "?",
            "subject": subject.name if subject else e.subject_id,
            "faculty": faculty.name if faculty else e.faculty_id,
            "room": e.room_id,
        })

    day_order = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4}
    rows.sort(key=lambda r: (day_order.get(r["day"], 9), int(r["period"])))
    return rows


# ─────────────────────────────────────────────
# CSV Export
# ─────────────────────────────────────────────

def export_faculty_timetable_csv(
    faculty_name: str,
    entries: List[TimetableEntry],
    config: SchoolConfig,
    output_path: Optional[str] = None,
) -> str:
    """
    Exports a faculty's timetable as a CSV string (or to a file).

    Returns
    -------
    str : CSV content (also written to output_path if provided)
    """
    rows = get_faculty_timetable(faculty_name, entries, config)

    fieldnames = ["slot_id", "day", "period", "division", "subject", "room", "room_type"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    csv_content = buf.getvalue()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            f.write(csv_content)

    return csv_content


def export_all_faculty_timetables_csv(
    entries: List[TimetableEntry],
    config: SchoolConfig,
    output_dir: str = "timetables",
) -> Dict[str, str]:
    """
    Exports timetables for ALL faculty members to individual CSV files.

    Returns
    -------
    Dict[faculty_name → file_path]
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {}
    for faculty in config.faculty:
        safe_name = faculty.name.replace(" ", "_").replace(".", "")
        path = os.path.join(output_dir, f"{safe_name}_timetable.csv")
        export_faculty_timetable_csv(faculty.name, entries, config, path)
        results[faculty.name] = path
    return results


def export_master_timetable_csv(
    entries: List[TimetableEntry],
    config: SchoolConfig,
    output_path: str = "timetables/master_timetable.csv",
) -> str:
    """
    Exports the full timetable (all divisions, all slots) as a master CSV.
    """
    subjects_map = {s.subject_id: s for s in config.subjects}
    faculty_map = {f.faculty_id: f for f in config.faculty}
    divisions_map = {d.division_id: d for d in config.divisions}
    slots_map = {s.slot_id: s for s in config.time_slots}

    fieldnames = [
        "entry_id", "slot_id", "day", "period",
        "division", "subject", "faculty", "room",
    ]

    day_order = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4}
    sorted_entries = sorted(
        entries,
        key=lambda e: (
            day_order.get(slots_map.get(e.slot_id, type("S", (), {"day": "Zzz"})()).day, 9),
            getattr(slots_map.get(e.slot_id), "period", 99),
        ),
    )

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for e in sorted_entries:
        slot = slots_map.get(e.slot_id)
        writer.writerow({
            "entry_id": e.entry_id,
            "slot_id": e.slot_id,
            "day": slot.day if slot else "?",
            "period": slot.period if slot else "?",
            "division": divisions_map.get(e.division_id, type("D", (), {"name": e.division_id})()).name,
            "subject": subjects_map.get(e.subject_id, type("S", (), {"name": e.subject_id})()).name,
            "faculty": faculty_map.get(e.faculty_id, type("F", (), {"name": e.faculty_id})()).name,
            "room": e.room_id,
        })
    csv_content = buf.getvalue()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        f.write(csv_content)

    return csv_content


# ─────────────────────────────────────────────
# PDF Export (Optional)
# ─────────────────────────────────────────────

def export_faculty_timetable_pdf(
    faculty_name: str,
    entries: List[TimetableEntry],
    config: SchoolConfig,
    output_path: str,
) -> bool:
    """
    Exports a faculty timetable as a PDF using reportlab.
    Returns True if successful, False if reportlab not installed.
    """
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
    except ImportError:
        print("⚠️  reportlab not installed. Run: pip install reportlab")
        return False

    rows_data = get_faculty_timetable(faculty_name, entries, config)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=landscape(A4),
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )

    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(
        f"<b>Weekly Timetable – {faculty_name}</b>",
        styles["Title"],
    ))
    elements.append(Spacer(1, 0.2 * inch))

    header = ["Slot", "Day", "Period", "Division", "Subject", "Room"]
    table_data = [header] + [
        [r["slot_id"], r["day"], r["period"], r["division"], r["subject"], r["room"]]
        for r in rows_data
    ]

    if len(table_data) == 1:
        elements.append(Paragraph("No sessions scheduled.", styles["Normal"]))
    else:
        t = Table(table_data, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(t)

    doc.build(elements)
    return True


# ─────────────────────────────────────────────
# Plain Text Summary
# ─────────────────────────────────────────────

def format_timetable_text(
    faculty_name: str,
    entries: List[TimetableEntry],
    config: SchoolConfig,
) -> str:
    """Returns a formatted plain-text timetable (suitable for email body)."""
    rows = get_faculty_timetable(faculty_name, entries, config)
    if not rows:
        return f"No sessions scheduled for {faculty_name} this week."

    lines = [
        f"Weekly Timetable for {faculty_name}",
        "=" * 55,
        f"{'Slot':<8} {'Day':<8} {'Pd':>3}  {'Division':<16} {'Subject':<20} {'Room':<8}",
        "-" * 65,
    ]
    for r in rows:
        lines.append(
            f"{r['slot_id']:<8} {r['day']:<8} {r['period']:>3}  "
            f"{r['division'][:15]:<16} {r['subject'][:19]:<20} {r['room']:<8}"
        )
    lines.append("-" * 65)
    lines.append(f"Total sessions: {len(rows)}")
    return "\n".join(lines)
