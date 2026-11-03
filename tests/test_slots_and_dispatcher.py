from datetime import datetime

import pytest

from backend.dispatcher.dispatcher import execute_intent
from backend.nlu.intents import intent_labels
from backend.nlu.slots import extract_slots

MONDAY_11 = datetime(2026, 11, 2, 11, 0)   # inside Robotics (sample data)
MONDAY_17 = datetime(2026, 11, 2, 17, 0)
SATURDAY = datetime(2026, 11, 7, 9, 0)
DECEMBER = datetime(2026, 12, 10, 9, 0)   # outside both sample semesters


@pytest.mark.parametrize("text, expected", [
    ("ikuphi i-COMS9003", {"course": "COMS9003"}),
    ("where is coms 9001", {"course": "COMS9001"}),
    ("ngubani ofundisa i-machine learning", {"course": "COMS9002"}),
    ("uhlelo lwakusasa", {"weekday": "tomorrow"}),
    ("uhlelo lwangoLwesihlanu", {"weekday": "friday"}),
    ("when is nlp on friday", {"course": "COMS9004", "weekday": "friday"}),
    ("i-email yami", {}),
])
def test_slot_extraction(text, expected):
    assert extract_slots(text) == expected


def test_alias_does_not_match_inside_other_words():
    assert "course" not in extract_slots("html is fun")  # "ml" alias must not fire


def test_every_canonical_intent_has_a_handler():
    for label in intent_labels():
        result = execute_intent(label, {"course": "COMS9001"}, now=MONDAY_11)
        assert result["intent"] == label
        assert result["response"]
        assert result["action"]["type"] in ("say", "show", "open_url")


def test_unknown_intent_falls_back():
    result = execute_intent("ask_robotics_venue", {})
    assert result["intent"] == "OOS"
    assert result["status"] == "fallback"


def test_course_intent_without_course_asks_for_it():
    result = execute_intent("ask_course_venue", {}, now=MONDAY_11)
    assert result["status"] == "needs_slot"
    assert result["display"]["slot"] == "course"


def test_venue_and_lecturer():
    assert "MSL004" in execute_intent("ask_course_venue", {"course": "COMS9001"})["response"]
    assert "Prof B. Sample" in execute_intent("ask_lecturer", {"course": "COMS9002"})["response"]


def test_current_and_next_class():
    assert "COMS9001" in execute_intent("ask_current_class", {}, now=MONDAY_11)["response"]
    assert "no class" in execute_intent("ask_current_class", {}, now=MONDAY_17)["response"]
    nxt = execute_intent("ask_next_class", {}, now=MONDAY_17)
    assert "tomorrow at 08:00" in nxt["response"] and "COMS9002" in nxt["response"]
    assert "on Monday" in execute_intent("ask_next_class", {}, now=SATURDAY)["response"]


def test_no_classes_outside_semester():
    assert "no class" in execute_intent("ask_current_class", {}, now=DECEMBER)["response"]
    assert execute_intent("ask_current_semester", {}, now=DECEMBER)["display"]["semester"] is None


def test_day_schedule_relative_day():
    result = execute_intent("ask_day_schedule", {"weekday": "tomorrow"}, now=MONDAY_11)
    assert result["display"]["weekday"] == "tuesday"
    assert [row["code"] for row in result["display"]["rows"]] == ["COMS9002", "COMS9004"]


def test_self_service_never_handles_credentials():
    result = execute_intent("open_self_service", {})
    assert result["action"] == {"type": "open_url", "url": "https://self-service.wits.ac.za/", "text": result["response"]}
    assert "password" not in str(result["display"]).lower()
