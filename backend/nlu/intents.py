"""Canonical intent labels, loaded from data/intents.json."""

import json
from functools import lru_cache

from backend.config import settings

OOS = "OOS"


@lru_cache(maxsize=1)
def load_intents():
    with open(settings.intents_file, encoding="utf-8") as handle:
        return json.load(handle)["intents"]


def intent_labels():
    return [item["label"] for item in load_intents()]


def intent_spec(label):
    for item in load_intents():
        if item["label"] == label:
            return item
    return None


def normalise_label(label):
    """Map a model's raw label onto a canonical one; unknown labels become OOS."""
    if label is None:
        return OOS
    cleaned = str(label).strip()
    labels = intent_labels()
    if cleaned in labels:
        return cleaned
    lowered = {item.lower(): item for item in labels}
    return lowered.get(cleaned.lower(), OOS)
