"""Verb-then-object policy (backend/nlu/frame.py) applied by the dispatcher."""

import pytest

from backend.dispatcher.dispatcher import FALLBACK_RESPONSE, execute_intent
from backend.nlu.frame import analyse, decide, lexicon
from backend.nlu.slots import extract_slots


def run(text, model_intent):
    slots = extract_slots(text)
    frame = analyse(text, slots)
    return frame, execute_intent(model_intent, slots, frame=frame)


@pytest.mark.parametrize("text, verb", [
    ("ngicela ungisize", "help"), ("ungivulele", "open"), ("ngitshele", "tell"), ("ngibonise", "show"),
    ("ngiyithola kanjani", "find"), ("ngubani ofundisa", "teach"), ("ikuphi", "where"), ("sinini", "when"),
    ("ngubani", "who"), ("isikhathi sini", "what"), ("please help me", "help"), ("open it", "open"),
])
def test_verbs_are_found_inside_inflected_words(text, verb):
    assert analyse(text)["verb"] == verb


def test_isizulu_is_not_mistaken_for_the_help_verb():
    assert analyse("isizulu")["verb"] is None          # contains "siz" but is not siza


def test_known_names_are_not_unknown_subjects():
    frame = analyse("ngiya kanjani e-Solomon Mahlangu House", extract_slots("ngiya kanjani e-Solomon Mahlangu House"))
    assert frame["objects"] == ["place"] and "solomon" not in frame["unknown"]


def test_help_request_outside_scope_is_refused_with_the_verb_message():
    frame, result = run("ngi cela ungi size ngama Student fees", "help")
    assert frame["verb"] == "help" and frame["objects"] == [] and {"student", "fees"} <= set(frame["unknown"])
    assert result["response"] == "Sorry, I cannot help with that."
    assert result["policy"] == "verb_unknown_object"
    assert result["intent"] == "OOS" and result["predicted_intent"] == "help"   # both reported
    assert result["display"]["kind"] == "capabilities" and result["display"]["items"]


def test_open_request_outside_scope_is_refused_with_the_open_message():
    _, result = run("ngicela ungivulele i-library", "open_self_service")     # even if the model says open
    assert result["response"] == "Sorry, I cannot open that." and result["intent"] == "OOS"


def test_open_with_an_in_scope_object_is_carried_out():
    _, result = run("ngicela uvule i-self service", "open_self_service")
    assert result["policy"] is None and result["action"]["type"] == "open_url"


def test_object_without_a_verb_offers_every_related_capability():
    frame, result = run("blah blah blah isikhathi", "ask_time")             # model guess is not trusted
    assert frame["verb"] is None and frame["objects"] == ["time"]
    assert result["policy"] == "object_offer" and result["intent"] == "help"
    caps = lexicon()["capabilities"]
    for intent in ("ask_time", "ask_current_class", "ask_next_class", "ask_course_time", "ask_day_schedule"):
        assert caps[intent] in result["display"]["items"]
    assert "About the time" in result["response"]


def test_verb_that_cannot_apply_to_an_in_scope_object():
    _, result = run("ngicela ungivulele isikhathi", "ask_time")
    assert result["policy"] == "verb_object_mismatch"
    assert result["response"].startswith("Sorry, I cannot open that. About the time")


def test_verb_alone_offers_what_that_verb_can_do():
    _, result = run("ngicela uvule", "OOS")
    assert result["policy"] == "verb_offer" and result["display"]["items"] == ["opening Student Self-Service"]
    _, result = run("ngicela ungisize", "OOS")
    assert result["intent"] == "help" and result["display"]["kind"] == "help"


@pytest.mark.parametrize("text, model_intent", [
    ("isikhathi sini?", "ask_time"),
    ("ikuphi i-COMS4054", "ask_course_venue"),
    ("ngubani ofundisa i-computer vision", "ask_lecturer"),
    ("ngiya kanjani e-Solomon Mahlangu House", "ask_directions"),
    ("ngitshele isikhathi", "ask_time"),
])
def test_consistent_model_intent_is_carried_out(text, model_intent):
    _, result = run(text, model_intent)
    assert result["policy"] is None and result["intent"] == model_intent


def test_question_word_with_unknown_subject_is_refused():
    _, result = run("i-library ikuphi?", "ask_course_venue")
    assert result["response"] == "Sorry, I don't know where that is."


def test_no_verb_and_no_object_trusts_the_model():
    _, result = run("sawubona pepper", "greeting")
    assert result["policy"] is None and result["intent"] == "greeting"
    _, result = run("blah blah", "OOS")
    assert result["response"] == FALLBACK_RESPONSE


def test_without_a_frame_the_dispatcher_is_unchanged():
    assert decide("ask_time", None) is None
    result = execute_intent("ask_time", {})
    assert result["intent"] == result["predicted_intent"] == "ask_time" and result["policy"] is None


def test_every_verb_and_object_points_at_real_intents():
    from backend.nlu.intents import intent_labels
    labels = set(intent_labels())
    lex = lexicon()
    for entry in lex["verbs"] + lex["objects"]:
        assert set(entry["intents"]) <= labels, entry["id"]
    assert set(lex["capabilities"]) <= labels
    assert all(v["refusal"].startswith("Sorry") for v in lex["verbs"])


# Verified sentence pairs supplied by the project owner.
@pytest.mark.parametrize("text, verb", [
    ("Ngenze okuthile", "do"),               # I did something
    ("Angenzanga lutho", "do"),              # I did not do anything
    ("Ungakwenza lokho?", "do"),             # Can you do that?
    ("Angikwazi ukukwenza lokho", "do"),     # I cannot do that  (-kwazi = can, not know)
    ("Ngingakwenza lokho", "do"),            # I can do that
    ("Angiyena umfundi", None),              # I am not a student
    ("Ngingumfundi", None),                  # I am a student
    ("Anginalutho", None),                   # I don't have anything
    ("Nginezinto eziningi", None),           # I have a lot of things
])
def test_owner_supplied_pairs(text, verb):
    frame = analyse(text)
    assert frame["verb"] == verb and frame["unknown"] == []


def test_money_is_an_unknown_subject():
    assert analyse("Ingabe unayo imali?")["unknown"] == ["imali"]      # Do you have money?


def test_do_request_outside_scope_is_refused():
    _, result = run("Ungakwenza i-NSFAS yami?", "OOS")
    assert result["response"] == "Sorry, I cannot do that."


# --- question words, combined verbs and the 500 verified test sentences ---------

@pytest.mark.parametrize("text, verb", [
    ("Yiliphi ikilasi eliqhubekayo?", "which"),        # liphi = which
    ("Yisiphi isifundo esilandelayo?", "which"),       # prefixed siphi = which
    ("Sinezifundo ziphi ngoMsombuluko?", "where_which"),  # bare ziphi: where, or which
    ("ikuphi i-great hall", "where"),                   # kuphi = where
    ("Ubani umfundisi we-Computer Vision?", "who"),     # ubani was missing
    ("Lingaki ihora?", "how_many"),
    ("Ungangisiza ngani?", "what"),
    ("Unjani?", None),                                  # a greeting, not "how"
    ("Sawubona", None),                                 # not the verb -bona (see)
])
def test_question_words(text, verb):
    assert analyse(text)["verb"] == verb


def test_bare_siphi_still_allows_which_questions():
    _, result = run("Sinezifundo ziphi ngoMsombuluko?", "ask_day_schedule")
    assert result["policy"] is None and result["intent"] == "ask_day_schedule"


def test_every_question_word_counts_not_only_the_first():
    _, result = run("Sesifikile yini isikhathi sezivivinyo?", "ask_current_semester")   # fik + yini
    assert result["intent"] == "ask_current_semester"


def test_request_verbs_let_a_fitting_prediction_through():
    _, result = run("Ngicela i-imeyili kaDr Hairong Wang.", "ask_contact")
    assert result["policy"] is None and result["intent"] == "ask_contact"
    assert extract_slots("Ngicela i-imeyili kaDr Hairong Wang.")["contact"] == "honours_cs"


def test_model_out_of_scope_with_unknown_words_is_refused():
    _, result = run("Ngicela isikhathi sokuhamba kwesitimela.", "OOS")      # train departure time
    assert result["intent"] == "OOS" and result["response"].startswith("Sorry")


def test_honours_alone_is_not_a_contact():
    assert "contact" not in extract_slots("Yiziphi izifundo ze-Honours?")


def test_every_campus_map_place_is_a_location():
    from backend.dispatcher.timetable import get_timetable
    tt = get_timetable()
    assert len(tt.locations) > 100
    for text, place in [("ikuphi i-John Moffat?", "john_moffat"), ("ngiya kanjani eSturrockPark", "sturrock_park"),
                        ("ikuphi iFNB", "fnb"), ("ikuphi i-Margo Steele", "fnb"),
                        ("ikuphi i-Science Stadium Laboratories", "science_stadium_labs")]:
        assert extract_slots(text)["location"] == place
    for location in tt.locations:
        assert location["grid"][0] in "ABCDE" and location["grid"][1] in "12345"
        assert 0 < location["marker"]["x"] < 1 and 0 < location["marker"]["y"] < 1
