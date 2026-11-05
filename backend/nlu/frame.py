"""Verb-then-object interpretation, applied after intent classification.

Pepper first looks for the verb (what the student wants done: open, show, find,
contact, teach, write, study, tell, help, do) and the question words (where, which,
when, who, what, how, how many), then for what the request is about (time, exams,
courses, the timetable, lecturers, the calendar, Self-Service, contacts, places,
plus the course/weekday/location/contact slots). General request verbs (ngicela,
ngifuna, ngidinga) count only when no specific verb or question word is found.
The intents of every verb and question word found are combined. Then:

  model says out of scope + unknown words          -> refusal (model and frame agree)
  verb + object in scope, model intent fits both  -> carry out the model's intent
  verb + object allow exactly one task             -> carry out that task
  verb + an unrecognised subject                  -> the verb's refusal
                                                     ("Sorry, I cannot help with that.")
  verb + object in scope the verb cannot apply to -> refusal + what it can do about the object
  verb alone                                      -> what it can do with that verb
  no verb + object in scope                       -> the model's intent if it fits the object and
                                                     every word was understood, else everything
                                                     it can do about the object
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


def _stem_regex(pattern):
    """A stem pattern with up to 10 letters of prefixes; '^pattern' must be the whole word."""
    if pattern.startswith("^"):
        return re.compile(r"(?<![a-z])(?:%s)(?![a-z])" % pattern[1:])
    return re.compile(r"(?<![a-z])[a-z]{0,10}?(?:%s)(?![a-z])" % pattern)


@lru_cache(maxsize=1)
def lexicon():
    with open(settings.action_lexicon_file, encoding="utf-8") as handle:
        data = json.load(handle)
    for entry in data["verbs"] + data["objects"]:
        entry["_regex"] = [_stem_regex(p) for p in entry.get("patterns", [])]
    data["_verbs"] = {v["id"]: v for v in data["verbs"]}
    data["_objects"] = {o["id"]: o for o in data["objects"]}
    words = sorted({normalise_text(w) for w in data.get("function_words", []) if normalise_text(w)}, key=len, reverse=True)
    # Function words may carry the usual noun prefixes (kaDr, ewits).
    data["_function_word"] = re.compile(r"^" + _PREFIX + "(?:%s)$" % "|".join(re.escape(w) for w in words))
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
    # Every specific verb or question word found counts ("Sesifikile yini ...?" has
    # both); general request verbs (ngicela, ngifuna) only when nothing else is found.
    specific = [v["id"] for v in found if v["id"] != "request"]
    verbs = specific or [v["id"] for v in found]

    objects = []
    for obj in lex["objects"]:
        slot = obj.get("slot")
        hit = bool(slot and slot in slots and slots[slot] not in obj.get("slot_ignore", []))
        for regex in obj["_regex"]:
            if regex.search(norm):
                hit = True
                rest = regex.sub(" ", rest)
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
    function_word = lex["_function_word"]
    unknown = [t for t in rest.replace("-", " ").split() if len(t) > 1 and not function_word.match(t)]
    return {"verb": verb["id"] if verb else None, "verbs": verbs, "objects": objects, "unknown": unknown}


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
    unknown = frame.get("unknown")
    if model_intent == "OOS" and unknown and (verb or objects):
        # The model says out of scope and the request contains words outside
        # Pepper's vocabulary: both agree, so refuse rather than offer.
        return {"policy": "verb_unknown_object", "verb": verb["id"] if verb else None,
                "refusal": verb["refusal"] if verb else "Sorry, I cannot help with that.",
                "capabilities": _capabilities(object_intents or set(lex["capabilities"]))}
    if verb is None:
        # No verb or question word, but the subject is in scope. If every word was
        # understood and the model's intent fits the subject, trust the model;
        # otherwise offer everything Pepper can do about the subject.
        if model_intent in object_intents and not unknown:
            return None
        return {"policy": "object_offer", "objects": labels, "capabilities": _capabilities(object_intents)}

    verb_intents = set()
    for verb_id in frame.get("verbs") or [verb["id"]]:
        verb_intents |= set(lex["_verbs"][verb_id]["intents"])
    if objects:
        allowed = verb_intents & object_intents
        if not allowed:
            return {"policy": "verb_object_mismatch", "verb": verb["id"], "refusal": verb["refusal"],
                    "objects": labels, "capabilities": _capabilities(object_intents)}
        if model_intent in allowed:
            return None
        if len(allowed) == 1 and not unknown:
            # The verb and the subject leave only one task: carry it out.
            return {"policy": "execute", "intent": next(iter(allowed))}
        return {"policy": "object_offer", "verb": verb["id"], "objects": labels, "capabilities": _capabilities(allowed)}

    if unknown:
        return {"policy": "verb_unknown_object", "verb": verb["id"], "refusal": verb["refusal"],
                "capabilities": _capabilities(verb_intents)}
    if model_intent in verb_intents:
        return None
    if verb["id"] == "help":
        return {"policy": "execute", "intent": "help"}
    return {"policy": "verb_offer", "verb": verb["id"], "capabilities": _capabilities(verb_intents)}
