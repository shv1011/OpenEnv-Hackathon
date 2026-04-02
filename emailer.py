"""
Email Module for the School Timetable Scheduling Environment.
Sends formatted timetables to faculty members via SMTP.
"""

from __future__ import annotations
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Optional, Dict
from pathlib import Path

from .models import TimetableEntry, SchoolConfig
from .export import format_timetable_text, export_faculty_timetable_csv


class TimetableMailer:
    """
    Sends timetable emails to faculty members.

    Configuration via environment variables:
        SMTP_HOST       (default: smtp.gmail.com)
        SMTP_PORT       (default: 587)
        SMTP_USER       Sender email address
        SMTP_PASSWORD   Sender email password / app password
        SCHOOL_NAME     (default: School Timetabling System)
    """

    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        school_name: Optional[str] = None,
    ):
        self.smtp_host = smtp_host or os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = smtp_user or os.getenv("SMTP_USER", "")
        self.smtp_password = smtp_password or os.getenv("SMTP_PASSWORD", "")
        self.school_name = school_name or os.getenv("SCHOOL_NAME", "School Timetabling System")

        if not self.smtp_user or not self.smtp_password:
            raise ValueError(
                "SMTP credentials not set. "
                "Provide smtp_user/smtp_password or set SMTP_USER/SMTP_PASSWORD env vars."
            )

    # ─────────────────────────────────────────
    # Public Interface
    # ─────────────────────────────────────────

    def send_faculty_email(
        self,
        faculty_name: str,
        entries: List[TimetableEntry],
        config: SchoolConfig,
        attach_csv: bool = True,
    ) -> bool:
        """
        Send timetable email to a specific faculty member.

        Parameters
        ----------
        faculty_name : str
        entries      : current timetable entries
        config       : school configuration
        attach_csv   : whether to attach CSV file

        Returns
        -------
        bool : True if sent successfully
        """
        faculty = next(
            (f for f in config.faculty if f.name.lower() == faculty_name.lower()),
            None,
        )
        if faculty is None:
            raise ValueError(f"Faculty '{faculty_name}' not found.")

        if not faculty.email:
            print(f"⚠️  No email address for {faculty_name}. Skipping.")
            return False

        subject_line = f"[{self.school_name}] Your Weekly Timetable"
        body = self._compose_body(faculty_name, entries, config)

        msg = MIMEMultipart()
        msg["From"] = self.smtp_user
        msg["To"] = faculty.email
        msg["Subject"] = subject_line
        msg.attach(MIMEText(body, "plain"))

        if attach_csv:
            csv_content = export_faculty_timetable_csv(faculty_name, entries, config)
            self._attach_csv(msg, faculty_name, csv_content)

        return self._send(faculty.email, msg)

    def send_all_faculty_emails(
        self,
        entries: List[TimetableEntry],
        config: SchoolConfig,
        attach_csv: bool = True,
    ) -> Dict[str, bool]:
        """
        Send timetable emails to ALL faculty members.

        Returns
        -------
        Dict[faculty_name → success_bool]
        """
        results = {}
        for faculty in config.faculty:
            try:
                success = self.send_faculty_email(
                    faculty.name, entries, config, attach_csv
                )
                results[faculty.name] = success
                if success:
                    print(f"  ✓ Email sent to {faculty.name} <{faculty.email}>")
            except Exception as e:
                print(f"  ✗ Failed to send to {faculty.name}: {e}")
                results[faculty.name] = False
        return results

    # ─────────────────────────────────────────
    # Email Composition
    # ─────────────────────────────────────────

    def _compose_body(
        self,
        faculty_name: str,
        entries: List[TimetableEntry],
        config: SchoolConfig,
    ) -> str:
        timetable_text = format_timetable_text(faculty_name, entries, config)

        return f"""Dear {faculty_name},

Please find your weekly class schedule below. This timetable has been
automatically generated and validated for conflicts.

{timetable_text}

If you have any concerns about your schedule, please contact the
academic office immediately.

---
This email was sent automatically by the {self.school_name}.
Please do not reply to this email.
"""

    def _attach_csv(
        self,
        msg: MIMEMultipart,
        faculty_name: str,
        csv_content: str,
    ) -> None:
        safe_name = faculty_name.replace(" ", "_").replace(".", "")
        filename = f"{safe_name}_timetable.csv"

        part = MIMEBase("application", "octet-stream")
        part.set_payload(csv_content.encode("utf-8"))
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f'attachment; filename="{filename}"',
        )
        msg.attach(part)

    def _send(self, to_email: str, msg: MIMEMultipart) -> bool:
        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.smtp_user, to_email, msg.as_string())
            return True
        except smtplib.SMTPException as e:
            print(f"  SMTP error: {e}")
            return False
        except Exception as e:
            print(f"  Unexpected error sending email: {e}")
            return False


# ─────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────

def send_faculty_email(
    faculty_name: str,
    entries: List[TimetableEntry],
    config: SchoolConfig,
    **smtp_kwargs,
) -> bool:
    """
    Standalone helper – creates a mailer and sends one email.

    Example
    -------
    send_faculty_email(
        "Dr. Sharma", entries, config,
        smtp_user="admin@school.edu",
        smtp_password="app_password",
    )
    """
    mailer = TimetableMailer(**smtp_kwargs)
    return mailer.send_faculty_email(faculty_name, entries, config)
