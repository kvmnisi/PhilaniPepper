"""Branch B: few-shot prompt, label scoring, calibration, NaN handling and JSON safety."""

import json
import math

import pytest

from backend.branch_b import scoring
from backend.branch_b.branch import dtype_name, explain_failure, inkubalm_device, load_base_model
from backend.branch_b.prompt import build_prompt, load_examples, ordered_examples
from backend.config import Settings
from backend.nlu.intents import intent_labels

NAN, INF = float("nan"), float("inf")


class CharTokenizer:
    """One token per character, after a BOS token (1)."""
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text):
        class Out:
            pass
        out = Out()
        out.input_ids = [1] + [ord(c) for c in text]
        return out


def decode(ids):
    return "".join(chr(i) for i in ids[1:])


# --- examples and prompt --------------------------------------------------------

def test_examples_cover_every_intent_with_label_words():
    data = load_examples(str(Settings().branch_b_examples_file))
    labels = intent_labels()
    assert set(data["label_words"]) == set(labels)
    assert {e["intent"] for e in data["examples"]} == set(labels)          # an example for every intent
    firsts = [data["label_words"][label][0] for label in labels]
    assert len(set(firsts)) == len(firsts)                                  # label words are distinct


def test_examples_come_in_rounds_across_intents():
    labels = intent_labels()
    ordered = ordered_examples(load_examples(str(Settings().branch_b_examples_file)), labels)
    words = [w for _, w in ordered[: len(labels)]]
    assert len(set(words)) == len(labels)                                   # first round: one per intent


def test_unknown_example_intent_is_refused():
    with pytest.raises(ValueError, match="unknown intent"):
        ordered_examples({"label_words": {"help": ["help"]}, "examples": [{"text": "x", "intent": "nope"}]}, ["help"])


def test_prompt_ends_with_the_request_and_intent_cue():
    prompt = build_prompt("  ngicela   usizo ", [("sawubona", "greeting")])
    assert "Request: sawubona\nIntent: greeting\n\n" in prompt
    assert prompt.endswith("Request: ngicela usizo\nIntent:")


# --- planning -------------------------------------------------------------------

def test_single_step_plan_uses_shared_lead_and_distinct_tokens():
    tails = [[(32, 116, 105)], [(32, 104, 101)], [(32, 116, 101), (32, 99, 108)]]   # " ti", " he", " te"/" cl"
    lead, tokens = scoring.single_step_plan(tails)
    assert lead == (32,) and tokens == [116, 104, 99]                       # third label falls back to " cl"


def test_single_step_plan_gives_up_when_labels_cannot_be_told_apart():
    assert scoring.single_step_plan([[(32, 116)], [(32, 116)]]) is None


# --- scoring --------------------------------------------------------------------

LABELS = ["greeting", "help", "ask_time"]
WORDS = {"greeting": ["greeting"], "help": ["help"], "ask_time": ["time"]}


def request_of(ids):
    return decode(ids).rsplit("Request: ", 1)[1].split("\nIntent:")[0]


def make_scorer(monkeypatch, logits_for, calibrate=True):
    def fake_next(model, ids, token_ids, device, torch):
        values = logits_for(request_of(ids))
        return [values[chr(t)] for t in token_ids]
    monkeypatch.setattr(scoring, "next_token_logprobs", fake_next)
    return scoring.LabelScorer(None, CharTokenizer(), LABELS, WORDS, [("sawubona", "greeting")], "cpu", 0, None,
                               calibrate=calibrate)


def test_one_pass_scores_every_intent(monkeypatch):
    scorer = make_scorer(monkeypatch, lambda req: {"g": 0.0, "h": -1.0, "t": 2.0 if "isikhathi" in req else -3.0})
    probs = scorer.score("ngitshele isikhathi")
    assert max(probs, key=probs.get) == "ask_time" and math.isclose(sum(probs.values()), 1.0)


def test_calibration_removes_a_label_bias(monkeypatch):
    # The model always leans to "greeting"; the request itself favours help a little.
    def logits(req):
        return {"g": 3.0, "h": 1.0 if "usizo" in req else 0.0, "t": 0.0}
    assert max(make_scorer(monkeypatch, logits, calibrate=False).score("ngicela usizo").items(),
               key=lambda kv: kv[1])[0] == "greeting"
    probs = make_scorer(monkeypatch, logits).score("ngicela usizo")
    assert max(probs, key=probs.get) == "help"


def test_full_continuation_scoring_when_first_tokens_clash(monkeypatch):
    words = {"greeting": ["hello"], "help": ["hi"], "ask_time": ["hour"]}
    seen = []

    def fake_seq(model, encoded, device, pad_id, torch):
        seen.append(len(encoded))
        return [-float(len(ids)) for ids, _ in encoded]
    monkeypatch.setattr(scoring, "sequence_logprobs", fake_seq)
    monkeypatch.setattr(scoring, "single_step_plan", lambda tails: None)
    scorer = scoring.LabelScorer(None, CharTokenizer(), LABELS, words, [], "cpu", 0, None, calibrate=False)
    probs = scorer.score("x")
    assert seen == [3]                                          # one batch of three full continuations
    assert max(probs, key=probs.get) == "help" and min(probs, key=probs.get) == "greeting"   # shortest wins here


def test_long_prompts_drop_example_rounds(monkeypatch):
    monkeypatch.setattr(scoring, "next_token_logprobs", lambda m, ids, t, d, torch: [0.0] * len(t))
    examples = [("a" * 50, "greeting"), ("b" * 50, "help"), ("c" * 50, "time")] * 4
    scorer = scoring.LabelScorer(None, CharTokenizer(), LABELS, WORDS, examples, "cpu", 0, None,
                                 calibrate=False, max_tokens=400)
    base, _ = scorer._prompt("ngicela usizo")
    assert len(base) <= 400 and "a" * 50 in decode(base)


def test_stable_softmax_matches_math_and_survives_large_values():
    probs = scoring.stable_softmax([-1000.0, -1001.0, -1003.0])
    assert math.isclose(sum(probs), 1.0) and probs[0] > probs[1] > probs[2]
    assert math.isclose(probs[0] / probs[1], math.e)


@pytest.mark.parametrize("bad", [[NAN, 0.0], [-INF, 0.0], [INF, 1.0], []])
def test_stable_softmax_refuses_invalid_scores(bad):
    with pytest.raises(scoring.InvalidScores):
        scoring.stable_softmax(bad)


def test_nan_from_the_model_raises_instead_of_guessing(monkeypatch):
    scorer = make_scorer(monkeypatch, lambda req: {"g": NAN, "h": 0.0, "t": 0.0})
    with pytest.raises(scoring.InvalidScores, match="INKUBALM_DTYPE=float32"):
        scorer.score("x")


def test_padded_nan_falls_back_to_unpadded_scoring(monkeypatch):
    calls = []

    def fake(model, encoded, device, pad_id, torch):
        calls.append(len(encoded))
        return [NAN] * len(encoded) if len(encoded) > 1 else [-1.0]
    monkeypatch.setattr(scoring, "sequence_logprobs", fake)
    monkeypatch.setattr(scoring, "single_step_plan", lambda tails: None)
    scorer = scoring.LabelScorer(None, CharTokenizer(), LABELS, WORDS, [], "cpu", 0, None, calibrate=False)
    probs = scorer.score("sawubona")
    assert calls == [3, 1, 1, 1] and math.isclose(sum(probs.values()), 1.0)


# --- settings and loading -------------------------------------------------------

def test_dtype_defaults_to_float32():
    assert Settings().inkubalm_dtype == "float32"
    assert dtype_name("float32", "cuda") == "float32"
    assert dtype_name("auto", "cuda") == "bfloat16" and dtype_name("auto", "cpu") == "float32"
    with pytest.raises(ValueError):
        dtype_name("float8", "cpu")


def test_branch_b_uses_inkubalm_whenever_the_ml_libraries_are_installed(monkeypatch):
    from backend.engines import _resolve_branch_b
    settings = Settings()
    settings.branch_b_engine = "auto"
    monkeypatch.setattr("backend.engines.ml_available", lambda: True)
    assert _resolve_branch_b(settings).engine == "inkubalm"
    monkeypatch.setattr("backend.engines.ml_available", lambda: False)
    assert _resolve_branch_b(settings).engine == "dev-rules"


def test_nan_message_points_at_device_dtype_and_transformers():
    msg = explain_failure("mps", "float16")
    assert "device=mps" in msg and "INKUBALM_DEVICE=cpu" in msg and "transformers==4.44.2" in msg


class _FakeTorch:
    class cuda:
        available = False

        @classmethod
        def is_available(cls):
            return cls.available


def test_inkubalm_never_picks_apple_mps_automatically():
    _FakeTorch.cuda.available = False
    assert inkubalm_device("auto", _FakeTorch) == "cpu"
    _FakeTorch.cuda.available = True
    assert inkubalm_device("auto", _FakeTorch) == "cuda"
    assert inkubalm_device("mps", _FakeTorch) == "mps"          # only when asked for


def test_base_model_prefers_eager_attention_and_falls_back():
    seen = []

    class Auto:
        @staticmethod
        def from_pretrained(model_id, trust_remote_code, **kwargs):
            seen.append(kwargs)
            if "attn_implementation" in kwargs or "dtype" in kwargs:
                raise TypeError("unsupported")
            return "model"
    assert load_base_model(Auto, "lelapa/InkubaLM-0.4B", True, "float32") == "model"
    assert seen[0] == {"dtype": "float32", "attn_implementation": "eager"}
    assert seen[-1] == {"torch_dtype": "float32"}


# --- the browser never sees NaN -------------------------------------------------

def test_nan_never_reaches_the_browser(monkeypatch):
    from fastapi.testclient import TestClient
    import backend.main as main

    branch = main.engines.branches["B"]
    original = branch.classify

    def nan_classify(text):
        result = original(text)
        result.update(confidence=NAN, scores={"greeting": NAN})
        return result
    monkeypatch.setattr(branch, "classify", nan_classify)
    with TestClient(main.app) as client:
        streamed = client.post("/api/process", data={"text": "sawubona", "branch": "B", "stream": "true"}).text
        for line in streamed.strip().splitlines():
            json.loads(line)                                    # every line is valid JSON
        assert '"confidence": null' in streamed and "NaN" not in streamed
        plain = client.post("/api/process", data={"text": "sawubona", "branch": "B"})
        assert plain.status_code == 200 and plain.json()["confidence"] is None
        assert client.post("/api/classify/b", json={"text": "sawubona"}).json()["confidence"] is None


def test_invalid_scores_become_a_readable_error(monkeypatch):
    from fastapi.testclient import TestClient
    import backend.main as main

    def broken(text):
        raise scoring.InvalidScores("Branch B produced invalid scores (NaN or infinity).")
    monkeypatch.setattr(main.engines.branches["B"], "classify", broken)
    with TestClient(main.app) as client:
        body = client.post("/api/process", data={"text": "sawubona", "branch": "B"}).json()
    assert body["error"]["code"] == "classification_failed"
    assert "invalid scores" in body["error"]["message"]


def test_prompt_examples_never_appear_in_the_test_sentences():
    import csv
    import re

    def norm(text):
        return " ".join(re.sub(r"[^\w\s-]", " ", text.lower()).split())
    with open("data/test_sentences_500.csv", encoding="utf-8") as handle:
        test = {norm(row["text"]) for row in csv.DictReader(handle)}
    examples = load_examples(str(Settings().branch_b_examples_file))["examples"]
    assert not [e["text"] for e in examples if norm(e["text"]) in test]


# --- fine-tuned (LoRA adapter) mode --------------------------------------------

from backend.branch_b.prompt import FINETUNE_PROMPT, encode_label


def test_label_encoding_marks_where_the_label_starts():
    full, start = encode_label(CharTokenizer(), "ngicela  usizo", "help", eos_id=0)
    chars = lambda ids: "".join(chr(i) for i in ids)
    assert decode(full[:start]).endswith("### Intent:") and chars(full[start:-1]) == " help" and full[-1] == 0


def test_fine_tuned_scorer_softmaxes_full_label_scores(monkeypatch):
    seen = []

    def fake(model, encoded, device, pad_id, torch):
        seen.append(len(encoded))
        return [-1.0 if "".join(chr(i) for i in ids[start:-1]) == " help" else -5.0 for ids, start in encoded]
    monkeypatch.setattr(scoring, "sequence_logprobs", fake)
    scorer = scoring.FineTunedScorer(None, CharTokenizer(), LABELS, "cpu", 0, 0, None)
    probs = scorer.score("ngicela usizo")
    assert seen == [3] and max(probs, key=probs.get) == "help" and math.isclose(sum(probs.values()), 1.0)


def test_fine_tuned_scorer_refuses_nan(monkeypatch):
    monkeypatch.setattr(scoring, "sequence_logprobs", lambda m, e, d, p, t: [NAN] * len(e))
    with pytest.raises(scoring.InvalidScores):
        scoring.FineTunedScorer(None, CharTokenizer(), LABELS, "cpu", 0, 0, None).score("x")


def test_adapter_must_match_prompt_and_labels(tmp_path):
    from backend.branch_b.branch import BranchBUnusable, check_adapter
    labels = intent_labels()
    with pytest.raises(BranchBUnusable, match="training_info"):
        check_adapter(tmp_path, labels)
    (tmp_path / "training_info.json").write_text(json.dumps({"prompt": FINETUNE_PROMPT, "labels": labels}))
    assert check_adapter(tmp_path, labels)["labels"] == labels
    (tmp_path / "training_info.json").write_text(json.dumps({"prompt_template": "old", "labels": labels}))
    with pytest.raises(BranchBUnusable, match="older"):
        check_adapter(tmp_path, labels)


def test_adapter_setting_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("INKUBALM_ADAPTER", "none")
    assert Settings().inkubalm_adapter is None
    monkeypatch.setenv("INKUBALM_ADAPTER", "/some/adapter")
    assert Settings().inkubalm_adapter == "/some/adapter"
