"""The shared Pepper task dispatcher.

`execute_intent(intent, slots)` maps a canonical intent (from Branch A, Branch
B, or a manual call) to a response and an action. It knows nothing about
which branch produced the intent.

Result shape:
    {
      "intent": str, "slots": dict,
      "status": "ok" | "needs_slot" | "fallback",
      "response": str,                         # what Pepper says
      "action": {"type": "say" | "show" | "open_url", ...},
      "display": {"kind": ..., ...} | None,    # what the tablet shows
    }
"""

from datetime import date, timedelta

from backend.dispatcher.timetable import WEEKDAYS, get_timetable
from backend.nlu.intents import OOS, intent_labels

FALLBACK_RESPONSE = (
    "Sorry, I can only help with the time, your classes, courses and the semester, "
    "or open Student Self-Service. Please try again."
)


def _course_label(course):
    return "%s %s" % (course["code"], course["name"])


def _session_text(session, with_day=True):
    text = "%s to %s in %s" % (session["start"], session["end"], session.get("venue", "a venue to be confirmed"))
    return ("%s %s" % (session["day"].capitalize(), text)) if with_day else text


def _join(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _course_display(course):
    return {
        "kind": "course",
        "code": course["code"],
        "name": course["name"],
        "lecturer": course.get("lecturer"),
        "sessions": course.get("sessions", []),
    }


def _say(text, display=None, status="ok"):
    return {"status": status, "response": text, "action": {"type": "say", "text": text}, "display": display}


def _show(text, display, status="ok"):
    return {"status": status, "response": text, "action": {"type": "show", "text": text}, "display": display}


def _long_date(iso):
    day = date.fromisoformat(iso)
    return "%d %s" % (day.day, day.strftime("%B %Y"))


def _resolve_weekday(weekday, now):
    if weekday in (None, "today"):
        return WEEKDAYS[now.weekday()]
    if weekday == "tomorrow":
        return WEEKDAYS[(now + timedelta(days=1)).weekday()]
    return weekday if weekday in WEEKDAYS else WEEKDAYS[now.weekday()]


def _needs_course(tt):
    text = "Which course do you mean? Please say the course name or code."
    return {
        "status": "needs_slot",
        "response": text,
        "action": {"type": "say", "text": text},
        "display": {"kind": "needs_slot", "slot": "course", "courses": [c["name"] for c in tt.courses]},
    }


def _greeting(slots, tt, now):
    return _say("Sawubona! I am the isiZulu Wits Assistant. Ask me about your classes, the time, or Self-Service.")


def _help(slots, tt, now):
    abilities = [
        "the time",
        "the current or next class",
        "who teaches a course",
        "when and where a course meets",
        "a day's timetable",
        "the list of courses",
        "the current semester",
        "opening Student Self-Service",
    ]
    return _show("You can ask me about " + _join(abilities) + ".", {"kind": "help", "items": abilities})


def _ask_time(slots, tt, now):
    clock = now.strftime("%H:%M")
    return _say("It is %s." % clock, {"kind": "time", "time": clock, "date": now.strftime("%A %d %B %Y")})


def _ask_current_class(slots, tt, now):
    found = tt.current_class(now)
    if not found:
        return _say("There is no class in progress right now.", {"kind": "message", "text": "No class in progress."})
    course, session = found
    text = "Right now it is %s in %s, until %s." % (_course_label(course), session.get("venue"), session["end"])
    display = _course_display(course)
    display["highlight"] = session
    return _show(text, display)


def _ask_next_class(slots, tt, now):
    found = tt.next_class(now, course_code=slots.get("course"))
    if not found:
        return _say("I could not find another class in the next week.", {"kind": "message", "text": "No upcoming class found."})
    course, session, day = found
    if day == now.date():
        when = "today at %s" % session["start"]
    elif day == (now + timedelta(days=1)).date():
        when = "tomorrow at %s" % session["start"]
    else:
        when = "on %s at %s" % (WEEKDAYS[day.weekday()].capitalize(), session["start"])
    text = "The next class is %s %s in %s." % (_course_label(course), when, session.get("venue"))
    display = _course_display(course)
    display["highlight"] = dict(session, date=day.isoformat())
    return _show(text, display)


def _ask_lecturer(slots, tt, now):
    course = tt.course(slots.get("course"))
    if not course:
        return _needs_course(tt)
    lecturer = course.get("lecturer") or "a lecturer who has not been confirmed yet"
    return _show("%s is taught by %s." % (_course_label(course), lecturer), _course_display(course))


def _ask_course_time(slots, tt, now):
    course = tt.course(slots.get("course"))
    if not course:
        return _needs_course(tt)
    sessions = course.get("sessions", [])
    weekday = slots.get("weekday")
    if weekday:
        day = _resolve_weekday(weekday, now)
        sessions = [s for s in sessions if s["day"] == day]
        if not sessions:
            text = "%s does not meet on %s." % (_course_label(course), day.capitalize())
            return _show(text, _course_display(course))
    if not sessions:
        return _show("%s has no scheduled sessions." % _course_label(course), _course_display(course))
    parts = ["%s %s to %s" % (s["day"].capitalize(), s["start"], s["end"]) for s in sessions]
    return _show("%s meets on %s." % (_course_label(course), _join(parts)), _course_display(course))


def _ask_course_venue(slots, tt, now):
    course = tt.course(slots.get("course"))
    if not course:
        return _needs_course(tt)
    venues = []
    for session in course.get("sessions", []):
        if session.get("venue") and session["venue"] not in venues:
            venues.append(session["venue"])
    if not venues:
        return _show("The venue for %s has not been confirmed." % _course_label(course), _course_display(course))
    if len(venues) == 1:
        text = "%s is held in %s." % (_course_label(course), venues[0])
    else:
        text = "%s is held in %s, depending on the day." % (_course_label(course), _join(venues))
    return _show(text, _course_display(course))


def _ask_day_schedule(slots, tt, now):
    day = _resolve_weekday(slots.get("weekday"), now)
    rows = tt.sessions_on(day)
    label = {"today": "today", "tomorrow": "tomorrow"}.get(slots.get("weekday"), "on " + day.capitalize())
    display = {
        "kind": "schedule",
        "weekday": day,
        "rows": [
            {"code": c["code"], "name": c["name"], "start": s["start"], "end": s["end"], "venue": s.get("venue")}
            for c, s in rows
        ],
    }
    if not rows:
        return _show("There are no classes %s." % label, display)
    parts = ["%s at %s" % (c["name"], s["start"]) for c, s in rows]
    noun = "class" if len(rows) == 1 else "classes"
    return _show("There %s %d %s %s: %s." % ("is" if len(rows) == 1 else "are", len(rows), noun, label, _join(parts)), display)


def _list_courses(slots, tt, now):
    names = [_course_label(c) for c in tt.courses]
    display = {"kind": "course_list", "courses": [{"code": c["code"], "name": c["name"]} for c in tt.courses]}
    return _show("There are %d courses: %s." % (len(names), _join(names)), display)


def _ask_current_semester(slots, tt, now):
    semester = tt.current_semester(now)
    if semester:
        text = "It is %s, which ends on %s." % (semester["name"], _long_date(semester["end"]))
        return _say(text, {"kind": "semester", "semester": semester})
    upcoming = tt.upcoming_semester(now)
    if upcoming:
        text = "We are between semesters. %s starts on %s." % (upcoming["name"], _long_date(upcoming["start"]))
        return _say(text, {"kind": "semester", "semester": None, "upcoming": upcoming})
    return _say("I do not have dates for the current semester.", {"kind": "semester", "semester": None})


def _open_self_service(slots, tt, now):
    url = tt.self_service_url
    text = "Opening Wits Student Self-Service. Please sign in on the official Wits page."
    return {
        "status": "ok",
        "response": text,
        "action": {"type": "open_url", "url": url, "text": text},
        "display": {"kind": "link", "url": url, "label": "Open Student Self-Service"},
    }


def _fallback(slots, tt, now):
    return _say(FALLBACK_RESPONSE, {"kind": "message", "text": FALLBACK_RESPONSE}, status="fallback")


HANDLERS = {
    "greeting": _greeting,
    "help": _help,
    "ask_time": _ask_time,
    "ask_current_class": _ask_current_class,
    "ask_next_class": _ask_next_class,
    "ask_lecturer": _ask_lecturer,
    "ask_course_time": _ask_course_time,
    "ask_course_venue": _ask_course_venue,
    "ask_day_schedule": _ask_day_schedule,
    "list_courses": _list_courses,
    "ask_current_semester": _ask_current_semester,
    "open_self_service": _open_self_service,
    OOS: _fallback,
}


def execute_intent(intent, slots=None, now=None, timetable=None):
    timetable = timetable or get_timetable()
    slots = dict(slots or {})
    now = timetable.localise(now)
    if intent not in intent_labels():
        intent = OOS
    result = HANDLERS.get(intent, _fallback)(slots, timetable, now)
    result.update({"intent": intent, "slots": slots, "sample_data": timetable.is_sample})
    return result
