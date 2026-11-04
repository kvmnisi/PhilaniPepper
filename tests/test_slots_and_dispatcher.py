from datetime import datetime

import pytest

from backend.dispatcher.dispatcher import execute_intent
from backend.dispatcher.timetable import get_timetable
from backend.nlu.intents import intent_labels
from backend.nlu.slots import extract_slots

# Dates inside the November 2026 exam period (data/courses.json, data/calendar.json).
TUE_3_NOV = datetime(2026, 11, 3, 10, 0)
MON_9_NOV_EVENING = datetime(2026, 11, 9, 18, 0)
TUE_10_NOV_15H = datetime(2026, 11, 10, 15, 0)   # NLP and Secure Systems exams in progress
TUE_24_NOV = datetime(2026, 11, 24, 9, 0)        # after the last Honours exam
DEC_5 = datetime(2026, 12, 5, 9, 0)
JAN_25 = datetime(2026, 1, 25, 9, 0)


@pytest.mark.parametrize("text, expected", [
    ("ikuphi i-COMS4054", {"course": "COMS4054A"}),
    ("where is coms 4061", {"course": "COMS4061A"}),
    ("when is 4032", {"course": "COMS4032A"}),
    ("ngubani ofundisa i-computer vision", {"course": "COMS4036A"}),
    ("when is nlp on friday", {"course": "COMS4054A", "weekday": "friday"}),
    ("uhlelo lwakusasa", {"weekday": "tomorrow"}),
    ("uhlelo lwangoLwesihlanu", {"weekday": "friday"}),
    ("how do I get to Solomon Mahlangu House", {"location": "solomon_mahlangu"}),
    ("ngiya kanjani eFlower Hall", {"location": "flower_hall"}),
    ("where is msl 005", {"location": "MSL005"}),
    ("who is the head of school", {"contact": "head_of_school"}),
    ("email for the honours coordinator", {"contact": "honours_cs"}),
    ("i-email yami", {}),
])
def test_slot_extraction(text, expected):
    assert extract_slots(text) == expected


def test_aliases_do_not_match_inside_other_words():
    assert extract_slots("television and the world") == {}  # "vision" / "rl" must not fire


def test_every_canonical_intent_has_a_handler():
    slots = {"course": "COMS4054A", "location": "flower_hall", "contact": "head_of_school"}
    for label in intent_labels():
        result = execute_intent(label, slots, now=TUE_3_NOV)
        assert result["intent"] == label
        assert result["response"]
        assert result["action"]["type"] in ("say", "show", "open_url")
        assert result["sample_data"] is False


def test_unknown_intent_falls_back():
    result = execute_intent("ask_nlp_venue", {})
    assert result["intent"] == "OOS" and result["status"] == "fallback"


def test_course_intent_without_course_asks_for_it():
    result = execute_intent("ask_course_venue", {}, now=TUE_3_NOV)
    assert result["status"] == "needs_slot"
    assert "Natural Language Processing" in result["display"]["choices"]


def test_exam_time_and_venue():
    nlp = execute_intent("ask_course_venue", {"course": "COMS4054A"}, now=TUE_3_NOV)["response"]
    assert "Tuesday 10 November from 14:00 to 17:00" in nlp and "MSL005 and MSL006, in this building" in nlp
    algos = execute_intent("ask_course_venue", {"course": "COMS4032A"}, now=TUE_3_NOV)
    assert "Solomon Mahlangu House, Basement 1A, on East Campus (grid D4" in algos["response"]
    assert algos["display"]["map"]["markers"][0]["id"] == "solomon_mahlangu"


def test_past_exam_uses_past_tense():
    text = execute_intent("ask_course_time", {"course": "COMS4060A"}, now=TUE_3_NOV)["response"]
    assert "was on Monday 2 November" in text


def test_lecturer_unknown_points_to_honours_coordinator():
    result = execute_intent("ask_lecturer", {"course": "COMS4054A"}, now=TUE_3_NOV)
    assert "Dr Hairong Wang" in result["response"] and result["display"]["kind"] == "contact"


def test_current_and_next_exam():
    now = execute_intent("ask_current_class", {}, now=TUE_10_NOV_15H)
    assert [row["code"] for row in now["display"]["rows"]] == ["COMS4054A", "COMS4066A"]
    assert "No Honours exam" in execute_intent("ask_current_class", {}, now=TUE_3_NOV)["response"]
    nxt = execute_intent("ask_next_class", {}, now=TUE_3_NOV)["response"]
    assert "COMS4070A" in nxt and "Thursday 5 November at 09:00" in nxt
    both = execute_intent("ask_next_class", {}, now=MON_9_NOV_EVENING)["response"]
    assert "tomorrow at 14:00" in both and "COMS4054A" in both and "COMS4066A" in both
    assert "no more" in execute_intent("ask_next_class", {}, now=TUE_24_NOV)["response"]


def test_day_schedule_resolves_weekday_to_next_date():
    result = execute_intent("ask_day_schedule", {"weekday": "friday"}, now=TUE_3_NOV)
    assert result["display"]["title"] == "Friday 6 November"
    assert [row["code"] for row in result["display"]["rows"]] == ["COMS4050A"]
    empty = execute_intent("ask_day_schedule", {"weekday": "tuesday"}, now=TUE_3_NOV)
    assert "no Honours exams" in empty["response"]


def test_calendar_periods():
    assert "November Examination" in execute_intent("ask_current_semester", {}, now=TUE_3_NOV)["response"]
    assert "Summer Vacation" in execute_intent("ask_current_semester", {}, now=DEC_5)["response"]
    # Registration overlaps orientation in January; the more specific period wins.
    assert "Orientation" in execute_intent("ask_current_semester", {}, now=JAN_25)["response"]


def test_contacts():
    hos = execute_intent("ask_contact", {"contact": "head_of_school"})
    assert "Professor Richard Klein" in hos["response"] and "011 717 6186" in hos["response"]
    default = execute_intent("ask_contact", {})
    assert [p["name"] for p in default["display"]["people"]] == ["Ms Kgomotso Monyepote", "Dr Hairong Wang"]


def test_directions():
    east = execute_intent("ask_directions", {"location": "solomon_mahlangu"})
    assert "East Campus, at grid D4" in east["response"] and east["display"]["map"]["here"]
    west = execute_intent("ask_directions", {"location": "flower_hall"})
    assert "also on West Campus, at grid D3" in west["response"]
    room = execute_intent("ask_directions", {"location": "MSL005"})
    assert "in this building" in room["response"] and room["display"]["map"] is None
    fnb = execute_intent("ask_directions", {"location": "fnb"})
    assert "Margo Steele" in fnb["response"] and "West Campus, at grid B3" in fnb["response"]
    assert extract_slots("where is the school of accountancy") == {"location": "fnb"}
    assert execute_intent("ask_directions", {})["status"] == "needs_slot"


def test_self_service_never_handles_credentials():
    result = execute_intent("open_self_service", {})
    assert result["action"]["type"] == "open_url"
    assert result["action"]["url"] == "https://self-service.wits.ac.za/"
    assert "password" not in str(result["display"]).lower()


# --- data integrity -----------------------------------------------------------

def test_data_files_are_consistent():
    tt = get_timetable()
    assert len(tt.courses) == 11
    location_ids = {l["id"] for l in tt.locations}
    assert tt.here_id in location_ids
    for course in tt.courses:
        for exam in course["exams"]:
            assert exam["start"] < exam["end"]
            for venue in exam["venues"]:
                assert venue["location"] in location_ids, (course["code"], venue)
    assert set(tt.default_contacts) <= {c["id"] for c in tt.contacts}
    assert len({c["id"] for c in tt.contacts}) == len(tt.contacts)
    for location in tt.locations:
        if location["marker"]:
            assert 0 < location["marker"]["x"] < 1 and 0 < location["marker"]["y"] < 1
    starts = [p["start"] for p in tt.periods]
    assert starts == sorted(starts)
