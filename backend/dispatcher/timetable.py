"""Timetable queries over data/courses.json."""

import json
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache

from backend.config import settings

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _tz(name):
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception:  # tzdata missing (e.g. bare Windows): SAST has no DST.
        return timezone(timedelta(hours=2), "SAST")


def _minutes(hhmm):
    hours, minutes = hhmm.split(":")
    return int(hours) * 60 + int(minutes)


class Timetable:
    def __init__(self, data):
        self.data = data
        self.is_sample = bool(data.get("is_sample", False))
        self.notice = data.get("notice", "")
        self.tz = _tz(data.get("timezone", "Africa/Johannesburg"))
        self.courses = data.get("courses", [])
        self.semesters = data.get("semesters", [])
        self.self_service_url = data.get("self_service_url", "https://self-service.wits.ac.za/")
        venues = {s["venue"] for c in self.courses for s in c.get("sessions", []) if s.get("venue")}
        self.locations = sorted(set(data.get("locations", [])) | venues)

    def now(self):
        return datetime.now(self.tz)

    def localise(self, moment):
        if moment is None:
            return self.now()
        if moment.tzinfo is None:
            return moment.replace(tzinfo=self.tz)
        return moment.astimezone(self.tz)

    def course(self, code):
        if not code:
            return None
        for course in self.courses:
            if course["code"].upper() == str(code).upper():
                return course
        return None

    def sessions_on(self, weekday):
        rows = []
        for course in self.courses:
            for session in course.get("sessions", []):
                if session["day"] == weekday:
                    rows.append((course, session))
        return sorted(rows, key=lambda row: _minutes(row[1]["start"]))

    def current_class(self, now):
        now = self.localise(now)
        if not self.in_teaching_period(now.date()):
            return None
        minute = now.hour * 60 + now.minute
        for course, session in self.sessions_on(WEEKDAYS[now.weekday()]):
            if _minutes(session["start"]) <= minute < _minutes(session["end"]):
                return course, session
        return None

    def next_class(self, now, course_code=None):
        """Next session that has not started yet, searching up to a week ahead."""
        now = self.localise(now)
        minute = now.hour * 60 + now.minute
        for offset in range(0, 8):
            day = now.date() + timedelta(days=offset)
            if not self.in_teaching_period(day):
                continue
            for course, session in self.sessions_on(WEEKDAYS[day.weekday()]):
                if course_code and course["code"] != course_code:
                    continue
                if offset == 0 and _minutes(session["start"]) <= minute:
                    continue
                return course, session, day
        return None

    def _semester_dates(self, semester):
        return date.fromisoformat(semester["start"]), date.fromisoformat(semester["end"])

    def current_semester(self, now):
        today = self.localise(now).date()
        for semester in self.semesters:
            start, end = self._semester_dates(semester)
            if start <= today <= end:
                return semester
        return None

    def upcoming_semester(self, now):
        today = self.localise(now).date()
        upcoming = [s for s in self.semesters if self._semester_dates(s)[0] > today]
        return min(upcoming, key=lambda s: s["start"]) if upcoming else None

    def in_teaching_period(self, day):
        if not self.semesters:
            return True
        for semester in self.semesters:
            start, end = self._semester_dates(semester)
            if start <= day <= end:
                return True
        return False

    def public_summary(self):
        return {
            "is_sample": self.is_sample,
            "notice": self.notice,
            "courses": [
                {
                    "code": c["code"],
                    "name": c["name"],
                    "lecturer": c.get("lecturer"),
                    "sessions": c.get("sessions", []),
                }
                for c in self.courses
            ],
            "semesters": self.semesters,
            "locations": self.locations,
        }


@lru_cache(maxsize=1)
def get_timetable():
    with open(settings.courses_file, encoding="utf-8") as handle:
        return Timetable(json.load(handle))
