"""Slot extraction by keyword matching, run after intent classification.

Both branches call the same extractor on the same isiZulu transcript (Branch A
also tries its English translation), so slot handling never differs between
branches and the A/B comparison isolates the intent decision.
"""

import re

from backend.dispatcher.timetable import get_timetable

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

# Longest forms first so that e.g. "lwesihlanu" is not shadowed by a prefix.
ZULU_WEEKDAYS = {
    "msombuluko": "monday",
    "lwesibili": "tuesday",
    "lwesithathu": "wednesday",
    "lwesine": "thursday",
    "lwesihlanu": "friday",
    "mgqibelo": "saturday",
    "ngesonto": "sunday",
}
RELATIVE_DAYS = {
    "today": "today",
    "namhlanje": "today",
    "tomorrow": "tomorrow",
    "kusasa": "tomorrow",
}

# Common isiZulu prefixes that attach to nouns ("ku-robotics", "irobotics",
# "lwakusasa", "lwangoLwesihlanu"); up to two may stack. A heuristic, not a
# morphological analyser.
_PREFIX = r"(?:(?:kwi|nge|nga|ngo|lwa|kwa|wa|ya|za|sa|ba|la|ku|ka|ye|we|se|i|e|u|o)-?){0,2}"

_COURSE_CODE = re.compile(r"(?<![a-z])([a-z]{4})\s*-?\s*(\d{4})([a-z])?(?![a-z0-9])", re.IGNORECASE)


def normalise_text(text):
    text = (text or "").lower()
    text = re.sub(r"[^\w\s\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _contains_word(text, word):
    pattern = r"(?<![a-z])" + _PREFIX + re.escape(word) + r"(?![a-z])"
    return re.search(pattern, text) is not None


def extract_course(text, timetable=None):
    timetable = timetable or get_timetable()
    norm = normalise_text(text)
    codes = {course["code"].upper(): course["code"] for course in timetable.courses}

    for match in _COURSE_CODE.finditer(norm):
        candidate = (match.group(1) + match.group(2)).upper()
        suffix = (match.group(3) or "").upper()
        if candidate + suffix in codes:
            return codes[candidate + suffix]
        if candidate in codes:
            return codes[candidate]
        for code_upper, code in codes.items():
            if code_upper.startswith(candidate):
                return code

    # Bare digits ("9001") are matched only if they identify a single course.
    for digits in re.findall(r"(?<!\d)(\d{4})(?!\d)", norm):
        hits = [code for upper, code in codes.items() if digits in upper]
        if len(hits) == 1:
            return hits[0]

    aliases = [(normalise_text(alias), course["code"])
               for course in timetable.courses for alias in [course["name"]] + course.get("aliases", [])]
    return _match_alias(norm, aliases)


def extract_weekday(text):
    norm = normalise_text(text)
    for word, value in RELATIVE_DAYS.items():
        if _contains_word(norm, word):
            return value
    for day in WEEKDAYS:
        if _contains_word(norm, day):
            return day
    for word in sorted(ZULU_WEEKDAYS, key=len, reverse=True):
        if _contains_word(norm, word):
            return ZULU_WEEKDAYS[word]
    return None


_ROOM = re.compile(r"(?<![a-z])m\s*s\s*l\s*-?\s*(\d{3})(?!\d)")


def _match_alias(norm, entries):
    """entries: [(alias, value)]. Longest alias wins."""
    for alias, value in sorted(entries, key=lambda item: -len(item[0])):
        if alias and _contains_word(norm, alias):
            return value
    return None


def extract_location(text, timetable=None):
    """A room such as "MSL005", or a location id from data/locations.json."""
    timetable = timetable or get_timetable()
    norm = normalise_text(text)
    room = _ROOM.search(norm)
    if room:
        return "MSL" + room.group(1)
    entries = [(normalise_text(alias), loc["id"])
               for loc in timetable.locations for alias in [loc["name"]] + loc.get("aliases", [])]
    return _match_alias(norm, entries)


def extract_contact(text, timetable=None):
    timetable = timetable or get_timetable()
    norm = normalise_text(text)
    entries = [(normalise_text(alias), contact["id"])
               for contact in timetable.contacts for alias in contact.get("aliases", [])]
    return _match_alias(norm, entries)


def extract_slots(*texts, timetable=None):
    """Return {course, weekday, location, contact} found in the first text that has each."""
    slots = {}
    for text in texts:
        if not text:
            continue
        if "course" not in slots:
            course = extract_course(text, timetable)
            if course:
                slots["course"] = course
        if "weekday" not in slots:
            weekday = extract_weekday(text)
            if weekday:
                slots["weekday"] = weekday
        if "location" not in slots:
            location = extract_location(text, timetable)
            if location:
                slots["location"] = location
        if "contact" not in slots:
            contact = extract_contact(text, timetable)
            if contact:
                slots["contact"] = contact
    return slots
