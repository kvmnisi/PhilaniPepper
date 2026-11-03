"""Development stand-in classifier ("dev-rules").

Scores intents by keyword hits from data/dev_keywords.json so the pipeline,
dispatcher and UI can be tested before the real models exist. It is
deliberately simple and is not a research baseline: every result it produces
is tagged engine="dev-rules" so it can be filtered out of any evaluation.
"""

import json
from functools import lru_cache

from backend.config import settings
from backend.nlu.intents import OOS
from backend.nlu.slots import normalise_text, _contains_word

COURSE_INTENTS = {"ask_lecturer", "ask_course_time", "ask_course_venue"}

# Engines whose intent decisions come from these keyword rules (or no ASR at
# all). Their rows are excluded from reports and flagged in the UI.
STAND_IN_ENGINES = {"dev-rules", "nllb-rules", "none"}


@lru_cache(maxsize=1)
def _keywords():
    with open(settings.dev_keywords_file, encoding="utf-8") as handle:
        data = json.load(handle)
    return {"en": data["en"], "zu": data["zu"]}


def classify(text, language, slots):
    """Return (intent, scores) for `text` using the `language` keyword list."""
    norm = normalise_text(text)
    lexicon = _keywords()[language]
    scores = {}
    for intent, phrases in lexicon.items():
        hits = sum(1 for phrase in phrases if _contains_word(norm, normalise_text(phrase)))
        if hits:
            scores[intent] = float(hits)

    has_course = "course" in slots
    for intent in list(scores):
        if intent in COURSE_INTENTS:
            scores[intent] += 1.0 if has_course else -0.5
    # "What time is COMS9001?" is a course-time question, not ask_time.
    if has_course and "ask_time" in scores:
        scores["ask_course_time"] = scores.get("ask_course_time", 0.0) + scores.pop("ask_time")
    if "weekday" in slots and not has_course:
        scores["ask_day_schedule"] = scores.get("ask_day_schedule", 0.0) + 0.5

    scores = {intent: score for intent, score in scores.items() if score > 0}
    if not scores:
        return OOS, {}
    best = max(scores, key=lambda intent: scores[intent])
    return best, scores
