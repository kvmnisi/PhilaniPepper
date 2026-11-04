"""Branch B NaN handling: scoring fallback, stable softmax, dtype, and JSON safety."""

import json
import math

import pytest

from backend.branch_b import scoring
from backend.branch_b.branch import dtype_name

NAN, INF = float("nan"), float("inf")


class FakeTokenizer:
    eos_token_id = 0

    def __call__(self, text):
        class Out:
            pass
        out = Out()
        out.input_ids = [1] + [2 + ord(c) % 50 for c in text]
        return out


def test_stable_softmax_matches_math_and_survives_large_values():
    probs = scoring.stable_softmax([-1000.0, -1001.0, -1003.0])
    assert math.isclose(sum(probs), 1.0) and probs[0] > probs[1] > probs[2]
    assert math.isclose(probs[0] / probs[1], math.e)


@pytest.mark.parametrize("bad", [[NAN, 0.0], [-INF, 0.0], [INF, 1.0], []])
def test_stable_softmax_refuses_invalid_scores(bad):
    with pytest.raises(scoring.InvalidScores):
        scoring.stable_softmax(bad)


def test_padded_nan_falls_back_to_unpadded_scoring(monkeypatch):
    calls = []

    def fake(model, encoded, device, pad_id, torch):
        calls.append(len(encoded))
        if len(encoded) > 1:                       # the padded batch is poisoned
            return [NAN] * len(encoded)
        return [-float(len(encoded[0][0]))]        # one label at a time is fine
    monkeypatch.setattr(scoring, "sequence_logprobs", fake)
    probs = scoring.score_labels(None, FakeTokenizer(), "sawubona", ["greeting", "help", "OOS"], "cpu", 0, None)
    assert calls == [3, 1, 1, 1]
    assert math.isclose(sum(probs.values()), 1.0) and all(math.isfinite(p) for p in probs.values())


def test_scores_that_stay_invalid_raise_instead_of_guessing(monkeypatch):
    monkeypatch.setattr(scoring, "sequence_logprobs", lambda m, e, d, p, t: [NAN] * len(e))
    with pytest.raises(scoring.InvalidScores, match="INKUBALM_DTYPE=float32"):
        scoring.score_labels(None, FakeTokenizer(), "x", ["greeting", "help"], "cpu", 0, None)


def test_dtype_defaults_to_float32():
    from backend.config import Settings
    assert Settings().inkubalm_dtype == "float32"
    assert dtype_name("float32", "cuda") == "float32"
    assert dtype_name("auto", "cuda") == "bfloat16" and dtype_name("auto", "cpu") == "float32"
    with pytest.raises(ValueError):
        dtype_name("float8", "cpu")


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


# --- load-time diagnosis -----------------------------------------------------

from backend.branch_b.branch import explain_failure, inkubalm_device, load_base_model


def test_damaged_adapter_weights_are_named_first():
    msg = explain_failure(["base_model.layers.0.q_proj.lora_A.weight"], False, True, "cpu", "float32")
    assert "damaged during training" in msg and "lora_A" in msg


def test_adapter_producing_nan_on_a_healthy_base():
    assert "adding the adapter produces NaN" in explain_failure([], False, True, "cuda", "float32")


def test_base_model_producing_nan_points_at_device_and_dtype():
    msg = explain_failure([], False, False, "mps", "float16")
    assert "device=mps" in msg and "INKUBALM_DEVICE=cpu" in msg


def test_healthy_model_has_no_message():
    assert explain_failure([], True, True, "cpu", "float32") is None


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
