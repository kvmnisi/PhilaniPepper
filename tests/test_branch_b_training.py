import csv

import pytest

from backend.branch_b import train
from backend.branch_b.prompt import encode_example
from backend.dispatcher.timetable import get_timetable


class FakeTokenizer:
    eos_token_id = 0
    pad_token_id = None

    def __call__(self, text):
        class Out:
            pass
        out = Out()
        out.input_ids = [1] + [2 + (ord(c) % 50) for c in text]
        return out


VALUES = train.slot_values(get_timetable())


def test_templates_expand_over_every_course_name_and_alias():
    rows = train.expand_templates([{"template": "X {course} Y", "intent": "ask_course_venue"}], VALUES, max_per_template=1000)
    texts = {r["text"] for r in rows}
    assert {"X Natural Language Processing Y", "X nlp Y"} <= texts
    assert len(rows) == len(VALUES["course"])
    assert all(r["intent"] == "ask_course_venue" for r in rows)


def test_large_expansions_are_capped():
    rows = train.expand_templates([{"template": "{course} {weekday} {location}", "intent": "ask_course_time"}],
                                  VALUES, max_per_template=25)
    assert len(rows) == 25


def test_prepare_filters_unverified_dedupes_and_drops_test_overlap():
    data = [{"text": "Sawubona", "intent": "greeting", "verified": "yes"},
            {"text": "sawubona!", "intent": "greeting", "verified": "yes"},     # duplicate after normalising
            {"text": "ngisize", "intent": "help", "verified": "no"},          # not verified
            {"text": "ngisize futhi", "intent": "help"},                      # no verified column -> skipped
            {"text": "i-nlp ikuphi", "intent": "ask_course_venue", "verified": "yes"}]
    exclude = [{"reference_transcript": "I-NLP ikuphi?"}]                    # a test recording
    examples, report = train.prepare(data, [], exclude, VALUES)
    assert [e["text"] for e in examples] == ["Sawubona"]
    assert report["skipped_unverified"] == 2 and report["duplicates"] == 1 and report["dropped_test_overlap"] == 1


def test_prepare_rejects_unknown_labels():
    with pytest.raises(ValueError, match="ask_nlp_venue"):
        train.prepare([{"text": "x", "intent": "ask_nlp_venue", "verified": "yes"}], [], [], VALUES)


def test_stratified_split_keeps_rare_intents_in_training():
    examples = [{"text": "a%d" % i, "intent": "greeting"} for i in range(20)] + [{"text": "h", "intent": "help"}]
    tr, va = train.stratified_split(examples, 0.2, seed=1)
    assert len(va) == 4 and all(e["intent"] == "greeting" for e in va)
    assert any(e["intent"] == "help" for e in tr) and len(tr) + len(va) == 21


def test_loss_covers_only_the_label_tokens():
    tok = FakeTokenizer()
    feature = train.build_features(tok, [{"text": "sawubona", "intent": "greeting"}], max_length=512)[0]
    ids, prompt_len = encode_example(tok, "sawubona", "greeting")
    assert feature["input_ids"] == ids
    assert feature["labels"][:prompt_len] == [-100] * prompt_len
    assert feature["labels"][prompt_len:] == ids[prompt_len:] and ids[-1] == tok.eos_token_id


def test_dry_run_reports_without_training(tmp_path, capsys):
    data = tmp_path / "train.csv"
    with open(data, "w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(["text", "intent", "verified"])
        w.writerow(["sawubona", "greeting", "yes"])
    assert train.main(["--data", str(data), "--dry-run"]) == 0
    assert '"examples": 1' in capsys.readouterr().out
