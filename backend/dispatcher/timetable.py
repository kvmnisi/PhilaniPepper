"""Campus data used by the dispatcher.

Loads data/courses.json (courses with optional weekly `sessions` and dated
`exams`), data/calendar.json (academic periods), data/contacts.json and
data/locations.json. Weekly sessions only count during teaching blocks;
exams count on their own dates.
"""

import json
import re
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache

from backend.config import settings

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
# Where a day can belong to several periods (registration overlaps
# orientation), the most specific one is reported.
PERIOD_PRIORITY = {"exams": 0, "teaching": 1, "break": 2, "orientation": 3, "registration": 4}


def _tz(name):
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception:  # tzdata missing (e.g. bare Windows): SAST has no DST.
        return timezone(timedelta(hours=2), "SAST")


def minutes(hhmm):
    hours, mins = hhmm.split(":")
    return int(hours) * 60 + int(mins)


def _load(path, default=None):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        if default is None:
            raise
        return default


class Timetable:
    def __init__(self, courses, calendar=None, contacts=None, locations=None):
        self.data = courses
        self.is_sample = bool(courses.get("is_sample", False))
        self.notice = courses.get("notice", "")
        self.tz = _tz(courses.get("timezone", "Africa/Johannesburg"))
        self.courses = courses.get("courses", [])
        self.self_service_url = courses.get("self_service_url", "https://self-service.wits.ac.za/")

        calendar = calendar or {}
        self.periods = calendar.get("periods", [])
        self.calendar_events = calendar.get("events", [])

        contacts = contacts or {}
        self.contacts = contacts.get("contacts", [])
        self.default_contacts = contacts.get("default", [])
        self.school_address = contacts.get("address")

        locations = locations or {}
        self.locations = locations.get("locations", [])
        self.map_image = locations.get("map_image")
        self.east_west_note = locations.get("east_west_note", "")
        self.room_rules = [(re.compile(r["pattern"], re.IGNORECASE), r["location"]) for r in locations.get("rooms", [])]
        self.here_id = courses.get("here") or locations.get("here")

    # --- clock -------------------------------------------------------------
    def now(self):
        return datetime.now(self.tz)

    def localise(self, moment):
        if moment is None:
            return self.now()
        if moment.tzinfo is None:
            return moment.replace(tzinfo=self.tz)
        return moment.astimezone(self.tz)

    # --- lookups -------------------------------------------------------------
    def course(self, code):
        if not code:
            return None
        for course in self.courses:
            if course["code"].upper() == str(code).upper():
                return course
        return None

    def contact(self, contact_id):
        for contact in self.contacts:
            if contact["id"] == contact_id:
                return contact
        return None

    def location(self, location_id):
        for location in self.locations:
            if location["id"] == location_id:
                return location
        return None

    def room_location(self, room):
        """Location id for a room name such as MSL005, or None."""
        for pattern, location_id in self.room_rules:
            if pattern.match(room.strip()):
                return location_id
        return None

    @property
    def here(self):
        return self.location(self.here_id)

    # --- calendar ------------------------------------------------------------
    def _period_dates(self, period):
        start = date.fromisoformat(period["start"])
        end = date.fromisoformat(period["end"]) if period.get("end") else date.max
        return start, end

    def periods_on(self, day):
        hits = [p for p in self.periods if self._period_dates(p)[0] <= day <= self._period_dates(p)[1]]
        return sorted(hits, key=lambda p: PERIOD_PRIORITY.get(p["kind"], 9))

    def current_period(self, now):
        hits = self.periods_on(self.localise(now).date())
        return hits[0] if hits else None

    def next_period(self, now):
        today = self.localise(now).date()
        upcoming = [p for p in self.periods if self._period_dates(p)[0] > today]
        return min(upcoming, key=lambda p: p["start"]) if upcoming else None

    def in_teaching_period(self, day):
        if not self.periods:
            return True
        return any(p["kind"] == "teaching" for p in self.periods_on(day))

    def last_day_of_term(self):
        for event in self.calendar_events:
            if event["name"].lower() == "last day of term":
                return event["date"]
        return None

    # --- scheduled events (weekly classes + exams) ----------------------------
    def events_on(self, day):
        rows = []
        weekday = WEEKDAYS[day.weekday()]
        for course in self.courses:
            if self.in_teaching_period(day):
                for session in course.get("sessions", []):
                    if session["day"] == weekday:
                        venue = session.get("venue")
                        rows.append({
                            "kind": "class", "course": course, "date": day,
                            "start": session["start"], "end": session["end"],
                            "venues": [{"room": venue, "location": self.room_location(venue) if venue else None}] if venue else [],
                        })
            for exam in course.get("exams", []):
                if exam["date"] == day.isoformat():
                    rows.append({
                        "kind": "exam", "course": course, "date": day, "start": exam["start"], "end": exam["end"],
                        "venues": exam.get("venues", []), "exam_type": exam.get("type"), "exam_code": exam.get("code"),
                    })
        return sorted(rows, key=lambda row: (minutes(row["start"]), row["course"]["code"]))

    def current_events(self, now):
        now = self.localise(now)
        minute = now.hour * 60 + now.minute
        return [e for e in self.events_on(now.date()) if minutes(e["start"]) <= minute < minutes(e["end"])]

    def next_events(self, now, course_code=None, horizon_days=120):
        """All events sharing the earliest start time after `now`."""
        now = self.localise(now)
        minute = now.hour * 60 + now.minute
        for offset in range(0, horizon_days + 1):
            day = now.date() + timedelta(days=offset)
            rows = [e for e in self.events_on(day)
                    if (not course_code or e["course"]["code"] == course_code)
                    and (offset > 0 or minutes(e["start"]) > minute)]
            if rows:
                first = rows[0]["start"]
                return [e for e in rows if e["start"] == first]
        return []

    def public_summary(self):
        return {
            "is_sample": self.is_sample,
            "notice": self.notice,
            "courses": [
                {"code": c["code"], "name": c["name"], "lecturer": c.get("lecturer"),
                 "sessions": c.get("sessions", []), "exams": c.get("exams", [])}
                for c in self.courses
            ],
            "periods": self.periods,
            "locations": [{"id": l["id"], "name": l["name"]} for l in self.locations],
        }


@lru_cache(maxsize=1)
def get_timetable():
    return Timetable(
        _load(settings.courses_file),
        calendar=_load(settings.calendar_file, {}),
        contacts=_load(settings.contacts_file, {}),
        locations=_load(settings.locations_file, {}),
    )
