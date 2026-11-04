"""Verb-then-object interpretation, applied after intent classification.

Pepper first looks for the verb (what the student wants done: help, open,
show, find, contact, teach, tell; question words where/when/who/what count as
verbs), then for what it is about (time, exams,
courses, the timetable, lecturers, the calendar, Self-Service, contacts,
places, plus the course/weekday/location/contact slots). Then:

  verb + object in scope, model intent fits both  -> carry out the model's intent
  verb + an unrecognised subject                  -> the verb's refusal
                                                     ("Sorry, I cannot help with that.")
  verb + object in scope the verb cannot apply to -> refusal + what it can do about the object
  verb alone                                      -> what it can do with that verb
  no verb + object in scope                       -> everything it can do about the object
                                                     (whatever the model guessed)
  neither                                         -> trust the model (e.g. greetings)

The same analysis runs on the isiZulu transcript for both branches, so it
never favours Branch A or B. The model's prediction is still reported as is.
Vocabulary lives in data/action_lexicon.json.
"""

import json
import re
from functools import lru_cache

from backend.config import settings
from backend.dispatcher.timetable import get_timetable
from backend.nlu.slots import RELATIVE_DAYS, WEEKDAYS, ZULU_WEEKDAYS, _PREFIX, normalise_text

_ROOM = re.compile(r"(?<![a-z])m\s*s\s*l\s*-?\s*\d{3}(?!\d)")
_CODE = re.compile(r"(?<![a-z])[a-z]{4}\s*-?\s*\d{4}[a-z]?(?![a-z0-9])")
_DIGITS = re.compile(r"\d+")


@lru_cache(maxsize=1)
def lexicon():
    with open(settings.action_lexicon_file, encoding="utf-8") as handle:
        data = json.load(handle)
    for verb in data["verbs"]:
        verb["_regex"] = [re.compile(r"(?<![a-z])[a-z]{0,7}?(?:%s)(?![a-z])" % p) for p in verb.get("patterns", [])]
    data["_verbs"] = {v["id"]: v for v in data["verbs"]}
    data["_objects"] = {o["id"]: o for o in data["objects"]}
    return data


def _word_regex(word):
    return re.compile(r"(?<![a-z])" + _PREFIX + re.escape(normalise_text(word)) + r"(?![a-z])")


def _slot_words(timetable):
    words = []
    for course in timetable.courses:
        words += [course["name"]] + course.get("aliases", [])
    for location in timetable.locations:
        words += [location["name"]] + location.get("aliases", [])
    for contact in timetable.contacts:
        words += contact.get("aliases", [])
    words += WEEKDAYS + list(ZULU_WEEKDAYS) + list(RELATIVE_DAYS)
    return words


def analyse(text, slots=None, timetable=None):
    """Return {"verb", "objects", "unknown"} for a transcript."""
    lex = lexicon()
    timetable = timetable or get_timetable()
    slots = slots or {}
    norm = normalise_text(text)
    rest = " %s " % norm

    found = []
    for verb in lex["verbs"]:
        hit = False
        for regex in verb["_regex"]:
            if regex.search(norm):
                hit = True
                rest = regex.sub(" ", rest)
        for word in verb.get("words", []):
            regex = _word_regex(word)
            if regex.search(norm):
                hit = True
                rest = regex.sub(" ", rest)
        if hit:
            found.append(verb)
    verb = min(found, key=lambda v: v.get("priority", 99)) if found else None

    objects = []
    for obj in lex["objects"]:
        hit = bool(obj.get("slot") and obj["slot"] in slots)
        for word in obj.get("words", []):
            regex = _word_regex(word)
            if regex.search(norm):
                hit = True
                rest = regex.sub(" ", rest)
        if hit:
            objects.append(obj["id"])

    # Everything the assistant knows by name is not an "unknown subject".
    for word in sorted(_slot_words(timetable), key=len, reverse=True):
        if normalise_text(word):
            rest = _word_regex(word).sub(" ", rest)
    rest = _DIGITS.sub(" ", _CODE.sub(" ", _ROOM.sub(" ", rest)))
    function_words = {normalise_text(w) for w in lex.get("function_words", [])}
    unknown = [t for t in rest.replace("-", " ").split() if t not in function_words and len(t) > 1]
    return {"verb": verb["id"] if verb else None, "objects": objects, "unknown": unknown}


def _capabilities(intents):
    caps = lexicon()["capabilities"]
    return [caps[i] for i in caps if i in intents]


def decide(model_intent, frame):
    """None = carry out the model's intent; otherwise a policy dict for the dispatcher."""
    if not frame:
        return None
    lex = lexicon()
    verb = lex["_verbs"].get(frame.get("verb"))
    objects = [lex["_objects"][o] for o in frame.get("objects", []) if o in lex["_objects"]]
    object_intents = set()
    for obj in objects:
        object_intents |= set(obj["intents"])
    labels = [o["label"] for o in objects]

    if verb is None and not objects:
        return None
    if verb is None:
        # No verb or question word recognised, but the subject is in scope: offer
        # everything Pepper can do about it rather than guessing what was asked.
        return {"policy": "object_offer", "objects": labels, "capabilities": _capabilities(object_intents)}

    verb_intents = set(verb["intents"])
    if objects:
        allowed = verb_intents & object_intents
        if not allowed:
            return {"policy": "verb_object_mismatch", "verb": verb["id"], "refusal": verb["refusal"],
                    "objects": labels, "capabilities": _capabilities(object_intents)}
        if model_intent in allowed:
            return None
        return {"policy": "object_offer", "verb": verb["id"], "objects": labels, "capabilities": _capabilities(allowed)}

    if frame.get("unknown"):
        return {"policy": "verb_unknown_object", "verb": verb["id"], "refusal": verb["refusal"],
                "capabilities": _capabilities(verb_intents)}
    if model_intent in verb_intents:
        return None
    if verb["id"] == "help":
        return {"policy": "execute", "intent": "help"}
    return {"policy": "verb_offer", "verb": verb["id"], "capabilities": _capabilities(verb_intents)}
