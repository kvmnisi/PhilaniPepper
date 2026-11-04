"""Branch B prompt: labelled example requests, then the new request.

    Requests that students say to Pepper, ... and the intent of each request.

    Request: ngicela usizo
    Intent: help

    ... (one example per intent, then a second round)

    Request: <isiZulu transcript>
    Intent:

The examples and each intent's label words live in data/branch_b_examples.json.
"""

import json
from functools import lru_cache

HEADER = ("Requests that students say to Pepper, a robot assistant at Wits University, "
          "and the intent of each request.\n\n")
EXAMPLE = "Request: {text}\nIntent: {word}\n\n"
QUERY = "Request: {text}\nIntent:"


def clean(text):
    return " ".join((text or "").split())


def build_prompt(text, examples=()):
    """examples: [(request text, label word)] shown before the new request."""
    shown = "".join(EXAMPLE.format(text=clean(t), word=w) for t, w in examples)
    return HEADER + shown + QUERY.format(text=clean(text))


@lru_cache(maxsize=4)
def load_examples(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def ordered_examples(data, labels):
    """[(text, label word)] in rounds: the first example of every intent, then the second, ...

    Spreading the intents evenly keeps any one label from sitting next to the new request.
    """
    words = data["label_words"]
    by_label = {label: [] for label in labels}
    for example in data["examples"]:
        if example["intent"] not in by_label:
            raise ValueError("Branch B example has an unknown intent: %r" % example["intent"])
        by_label[example["intent"]].append(example["text"])
    missing = [label for label in labels if not words.get(label)]
    if missing:
        raise ValueError("Branch B label words missing for: %s" % ", ".join(missing))
    rounds = max((len(v) for v in by_label.values()), default=0)
    ordered = []
    for index in range(rounds):
        for label in labels:
            if index < len(by_label[label]):
                ordered.append((by_label[label][index], words[label][0]))
    return ordered
