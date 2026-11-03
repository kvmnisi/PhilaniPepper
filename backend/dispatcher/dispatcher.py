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

from backend.dispatcher.timetable import WEEKDAYS, get_timetable, minutes
from backend.nlu.intents import OOS, intent_labels

FALLBACK_RESPONSE = (
    "Sorry, I can only help with Honours courses and exams, the academic calendar, "
    "School contacts, finding places on campus, the time, or opening Student Self-Service. Please try again."
)


# --- formatting -------------------------------------------------------------

def _join(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _day_text(day, today=None):
    text = "%s %d %s" % (WEEKDAYS[day.weekday()].capitalize(), day.day, day.strftime("%B"))
    if today and day.year != today.year:
        text += " %d" % day.year
    return text


def _long_date(iso):
    day = date.fromisoformat(iso)
    return "%s %d %s %d" % (WEEKDAYS[day.weekday()].capitalize(), day.day, day.strftime("%B"), day.year)


def _course_label(course):
    return "%s %s" % (course["code"], course["name"])


def _relative_day(day, now):
    if day == now.date():
        return "today"
    if day == now.date() + timedelta(days=1):
        return "tomorrow"
    return "on " + _day_text(day, now.date())


def _venue_text(venues, tt):
    """'MSL005 and MSL006, in this building' or 'Solomon Mahlangu House, Basement 1A, on East Campus (grid D4 ...)'."""
    if not venues:
        return "a venue that has not been announced"
    groups = []
    for venue in venues:
        location_id = venue.get("location") or tt.room_location(venue.get("room") or "")
        for group in groups:
            if group[0] == location_id:
                group[1].append(venue["room"])
                break
        else:
            groups.append((location_id, [venue["room"]]))
    parts = []
    for location_id, rooms in groups:
        location = tt.location(location_id) if location_id else None
        if location_id and location_id == tt.here_id:
            parts.append("%s, in this building" % _join(rooms))
        elif location:
            text = "%s, %s" % (location["name"], _join(rooms))
            if location.get("campus") and location.get("grid"):
                text += ", on %s Campus (grid %s on the campus map)" % (location["campus"], location["grid"])
            parts.append(text)
        else:
            parts.append(_join(rooms))
    return "; ".join(parts)


def _map_for(location_ids, tt):
    """Map display data for locations outside this building, or None."""
    markers = []
    for location_id in location_ids:
        location = tt.location(location_id) if location_id else None
        if location and location.get("marker") and location_id != tt.here_id:
            if not any(m["id"] == location_id for m in markers):
                markers.append({"id": location_id, "label": location["name"],
                                "x": location["marker"]["x"], "y": location["marker"]["y"]})
    if not markers or not tt.map_image:
        return None
    here = tt.here
    return {
        "image": tt.map_image,
        "here": {"label": "You are here", "x": here["marker"]["x"], "y": here["marker"]["y"]}
        if here and here.get("marker") else None,
        "markers": markers,
    }


def _event_locations(events):
    return [v.get("location") for e in events for v in e["venues"]]


def _event_row(event, tt):
    return {
        "kind": event["kind"],
        "code": event["course"]["code"],
        "name": event["course"]["name"],
        "date": event["date"].isoformat(),
        "start": event["start"],
        "end": event["end"],
        "exam_type": event.get("exam_type"),
        "venue": _venue_text(event["venues"], tt),
    }


def _event_noun(event):
    return "exam" if event["kind"] == "exam" else "class"


def _course_display(course, tt):
    exams = course.get("exams", [])
    return {
        "kind": "course",
        "code": course["code"],
        "name": course["name"],
        "lecturer": course.get("lecturer"),
        "sessions": course.get("sessions", []),
        "exams": [
            {"code": e.get("code"), "type": e.get("type"), "date": e["date"], "day": _long_date(e["date"]),
             "start": e["start"], "end": e["end"], "venue": _venue_text(e.get("venues", []), tt)}
            for e in exams
        ],
        "map": _map_for([v.get("location") for e in exams for v in e.get("venues", [])], tt),
    }


def _contact_card(contact):
    return {key: contact.get(key) for key in ("role", "name", "email", "phone", "fax", "office") if contact.get(key)}


def _say(text, display=None, status="ok"):
    return {"status": status, "response": text, "action": {"type": "say", "text": text}, "display": display}


def _show(text, display, status="ok"):
    return {"status": status, "response": text, "action": {"type": "show", "text": text}, "display": display}


def _needs(slot, text, choices):
    return {
        "status": "needs_slot",
        "response": text,
        "action": {"type": "say", "text": text},
        "display": {"kind": "needs_slot", "slot": slot, "choices": choices},
    }


def _needs_course(tt):
    return _needs("course", "Which course do you mean? Please say the course name or code.",
                  [c["name"] for c in tt.courses])


def _resolve_day(weekday, now):
    """today / tomorrow / the next date (today included) falling on `weekday`."""
    today = now.date()
    if weekday in (None, "today"):
        return today
    if weekday == "tomorrow":
        return today + timedelta(days=1)
    if weekday in WEEKDAYS:
        return today + timedelta(days=(WEEKDAYS.index(weekday) - today.weekday()) % 7)
    return today


# --- handlers ---------------------------------------------------------------

def _greeting(slots, tt, now):
    return _say("Sawubona! I am the isiZulu Wits Assistant. Ask me about Honours exams, the calendar, "
                "School contacts or how to find a place on campus.")


def _help(slots, tt, now):
    abilities = [
        "the time",
        "the current or next exam",
        "when and where a course's exam is",
        "what is on a particular day",
        "the list of Honours courses",
        "where we are in the academic year",
        "who to contact in the School",
        "how to find a building or room",
        "opening Student Self-Service",
    ]
    return _show("You can ask me about " + _join(abilities) + ".", {"kind": "help", "items": abilities})


def _ask_time(slots, tt, now):
    clock = now.strftime("%H:%M")
    return _say("It is %s." % clock, {"kind": "time", "time": clock, "date": "%s %d" % (_day_text(now.date()), now.year)})


def _ask_current_class(slots, tt, now):
    events = tt.current_events(now)
    if not events:
        text = "No Honours exam or class is in progress right now."
        return _say(text, {"kind": "message", "text": text})
    parts = ["the %s %s %s in %s, until %s" % (e["course"]["code"], e["course"]["name"], _event_noun(e),
                                                _venue_text(e["venues"], tt), e["end"]) for e in events]
    text = "Right now: " + _join(parts) + "."
    return _show(text, {"kind": "schedule", "title": "In progress now", "rows": [_event_row(e, tt) for e in events],
                        "map": _map_for(_event_locations(events), tt)})


def _ask_next_class(slots, tt, now):
    course = tt.course(slots.get("course"))
    events = tt.next_events(now, course_code=course["code"] if course else None)
    if not events:
        text = ("There is nothing else scheduled for %s." % _course_label(course)) if course \
            else "There are no more Honours exams or classes scheduled."
        return _say(text, {"kind": "message", "text": text})
    first = events[0]
    when = "%s at %s" % (_relative_day(first["date"], now), first["start"])
    parts = ["%s, in %s" % (_course_label(e["course"]), _venue_text(e["venues"], tt)) for e in events]
    all_exams = all(e["kind"] == "exam" for e in events)
    if len(events) == 1:
        text = "The next Honours %s is %s, %s." % (_event_noun(first), parts[0], when)
    else:
        text = "The next Honours %s, %s, are %s." % ("exams" if all_exams else "sessions", when, _join(parts))
    display = {"kind": "schedule", "title": _day_text(first["date"], now.date()),
               "rows": [_event_row(e, tt) for e in events], "map": _map_for(_event_locations(events), tt)}
    return _show(text, display)


def _ask_lecturer(slots, tt, now):
    course = tt.course(slots.get("course"))
    if not course:
        return _needs_course(tt)
    if course.get("lecturer"):
        return _show("%s is taught by %s." % (_course_label(course), course["lecturer"]), _course_display(course, tt))
    coordinator = tt.contact("honours_cs")
    if coordinator:
        text = ("I don't have the lecturer for %s. For Honours course queries, contact the Honours coordinator, "
                "%s, at %s." % (_course_label(course), coordinator["name"], coordinator["email"]))
        return _show(text, {"kind": "contact", "people": [_contact_card(coordinator)]})
    return _say("I don't have the lecturer for %s." % _course_label(course), _course_display(course, tt))


def _exam_sentence(course, exam, tt, now, include_venue=False):
    day = date.fromisoformat(exam["date"])
    finished = (day, minutes(exam["end"])) <= (now.date(), now.hour * 60 + now.minute)
    kind = exam.get("type") or "exam"
    kind = kind if "exam" in kind else kind + " exam"
    text = "The %s %s %s %s from %s to %s" % (
        _course_label(course), kind, "was" if finished else "is", _relative_day(day, now), exam["start"], exam["end"])
    if include_venue:
        text += ", in " + _venue_text(exam.get("venues", []), tt)
    return text + "."


def _ask_course_time(slots, tt, now):
    course = tt.course(slots.get("course"))
    if not course:
        return _needs_course(tt)
    sentences = []
    sessions = course.get("sessions", [])
    if sessions and tt.in_teaching_period(now.date()):
        parts = ["%s %s to %s" % (s["day"].capitalize(), s["start"], s["end"]) for s in sessions]
        sentences.append("%s meets on %s." % (_course_label(course), _join(parts)))
    for exam in course.get("exams", []):
        sentences.append(_exam_sentence(course, exam, tt, now))
    if not sentences:
        sentences.append("I don't have any times for %s." % _course_label(course))
    return _show(" ".join(sentences), _course_display(course, tt))


def _ask_course_venue(slots, tt, now):
    course = tt.course(slots.get("course"))
    if not course:
        return _needs_course(tt)
    sentences = [_exam_sentence(course, exam, tt, now, include_venue=True) for exam in course.get("exams", [])]
    venues = []
    for session in course.get("sessions", []):
        if session.get("venue") and session["venue"] not in venues:
            venues.append(session["venue"])
    if venues and tt.in_teaching_period(now.date()):
        sentences.insert(0, "%s classes are held in %s." % (_course_label(course), _join(venues)))
    if not sentences:
        sentences.append("I don't have a venue for %s." % _course_label(course))
    return _show(" ".join(sentences), _course_display(course, tt))


def _ask_day_schedule(slots, tt, now):
    weekday = slots.get("weekday")
    day = _resolve_day(weekday, now)
    events = tt.events_on(day)
    if weekday in (None, "today"):
        label = "today, " + _day_text(day, now.date())
    elif weekday == "tomorrow":
        label = "tomorrow, " + _day_text(day, now.date())
    else:
        label = "on " + _day_text(day, now.date())
    display = {"kind": "schedule", "title": _day_text(day, now.date()), "rows": [_event_row(e, tt) for e in events],
               "map": _map_for(_event_locations(events), tt)}
    if not events:
        return _show("There are no Honours exams or classes %s." % label, display)
    parts = ["%s at %s in %s" % (e["course"]["name"], e["start"], _venue_text(e["venues"], tt)) for e in events]
    noun = "exam" if all(e["kind"] == "exam" for e in events) else "session"
    if len(events) == 1:
        text = "There is 1 Honours %s %s: %s." % (noun, label, parts[0])
    else:
        text = "There are %d Honours %ss %s: %s." % (len(events), noun, label, _join(parts))
    return _show(text, display)


def _list_courses(slots, tt, now):
    display = {"kind": "course_list", "courses": [{"code": c["code"], "name": c["name"]} for c in tt.courses]}
    text = "There are %d Honours Computer Science courses: %s." % (len(tt.courses), _join(c["name"] for c in tt.courses))
    return _show(text, display)


def _ask_current_semester(slots, tt, now):
    period = tt.current_period(now)
    upcoming = tt.next_period(now)
    if period:
        until = " to %s" % _long_date(period["end"]) if period.get("end") else ""
        text = "We are in the %s, from %s%s." % (period["name"], _long_date(period["start"]), until)
        last_day = tt.last_day_of_term()
        if period["kind"] == "exams" and last_day and last_day >= now.date().isoformat():
            text += " The last day of term is %s." % _long_date(last_day)
        return _say(text, {"kind": "calendar", "period": period, "next": upcoming})
    if upcoming:
        text = "The next part of the academic year is the %s, starting %s." % (upcoming["name"], _long_date(upcoming["start"]))
        return _say(text, {"kind": "calendar", "period": None, "next": upcoming})
    return _say("I don't have calendar dates for today.", {"kind": "calendar", "period": None, "next": None})


def _open_self_service(slots, tt, now):
    url = tt.self_service_url
    text = "Opening Wits Student Self-Service. Please sign in on the official Wits page."
    return {
        "status": "ok",
        "response": text,
        "action": {"type": "open_url", "url": url, "text": text},
        "display": {"kind": "link", "url": url, "label": "Open Student Self-Service"},
    }


def _ask_contact(slots, tt, now):
    contact = tt.contact(slots.get("contact")) if slots.get("contact") else None
    if contact:
        details = []
        if contact.get("email"):
            details.append("email " + contact["email"])
        if contact.get("phone"):
            details.append("phone " + contact["phone"])
        if contact.get("office"):
            details.append("office " + contact["office"])
        text = "%s: %s." % (contact["role"], contact["name"])
        if details:
            text += " " + _join(details)[0].upper() + _join(details)[1:] + "."
        return _show(text, {"kind": "contact", "people": [_contact_card(contact)]})
    people = [tt.contact(cid) for cid in tt.default_contacts if tt.contact(cid)]
    if not people:
        return _say("I don't have School contact details.")
    first = people[0]
    text = "For general questions, contact %s, %s, at %s" % (first["name"], first["role"], first["email"])
    text += (", office %s." % first["office"]) if first.get("office") else "."
    if len(people) > 1:
        text += " For Honours queries, contact %s at %s." % (people[1]["name"], people[1]["email"])
    return _show(text, {"kind": "contact", "people": [_contact_card(p) for p in people]})


def _ask_directions(slots, tt, now):
    target = slots.get("location")
    if not target:
        return _needs("location", "Where would you like to go? Please say the building or room.",
                      [l["name"] for l in tt.locations])
    room, location_id = None, target
    if not tt.location(target):
        room, location_id = target, tt.room_location(target)
    location = tt.location(location_id) if location_id else None
    if not location:
        text = "I don't know where %s is." % target
        return _say(text, {"kind": "message", "text": text})

    here = tt.here
    display = {"kind": "location", "name": room or location["name"], "building": location["name"],
               "campus": location.get("campus"), "grid": location.get("grid"), "note": location.get("note"),
               "map": None}
    if location_id == tt.here_id:
        if room:
            text = "%s is in this building, the %s." % (room, location["name"])
        else:
            grid = " It is at grid %s on the campus map." % location["grid"] if location.get("grid") else ""
            text = "You are in the %s.%s" % (location["name"], grid)
        return _show(text, display)
    if not location.get("grid"):
        return _show(("%s is not marked on the campus map. %s" % (location["name"], location.get("note") or "")).strip(),
                     display)

    display["map"] = _map_for([location_id], tt)
    name = "%s, in %s," % (room, location["name"]) if room else location["name"]
    if here and location.get("campus") == here.get("campus"):
        text = "%s is also on %s Campus, at grid %s on the campus map. You are in the %s, at grid %s." % (
            name, location["campus"], location["grid"], here["name"], here.get("grid"))
    else:
        text = "%s is on %s Campus, at grid %s on the campus map. %s" % (
            name, location["campus"], location["grid"], tt.east_west_note)
    return _show(text.strip(), display)


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
    "ask_contact": _ask_contact,
    "ask_directions": _ask_directions,
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
