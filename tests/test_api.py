import io
import json
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

import backend.main as main
from backend.audio import decode_audio
from backend.evaluation.logger import ResearchLog, read_rows
from backend.evaluation.report import summarise


@pytest.fixture()
def client(tmp_path, monkeypatch):
    log_path = tmp_path / "research_log.csv"
    monkeypatch.setattr(main, "LOG_PATH", log_path)
    monkeypatch.setattr(main, "research_log", ResearchLog(log_path))
    monkeypatch.setattr(main.settings, "recordings_dir", tmp_path / "recordings")
    with TestClient(main.app) as test_client:
        test_client.log_path = log_path
        yield test_client


def make_wav(seconds=0.5, rate=8000):
    samples = (np.sin(np.linspace(0, 440 * 2 * np.pi * seconds, int(rate * seconds))) * 8000).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.tobytes())
    return buffer.getvalue()


def test_health_reports_development_engines(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["engines"]["development_engines_active"] is True
    assert body["engines"]["branch_b"]["engine"] == "dev-rules"


def test_process_text_both_branches(client):
    body = client.post("/api/process", data={"text": "i-nlp ikuphi?", "branch": "both", "primary": "B"}).json()
    assert body["primary"] == "B"
    assert body["branches"]["B"]["intent"] == "ask_course_venue"
    assert body["slots"] == {"course": "COMS4054A"}
    assert body["agreement"] in (True, False)
    for key in ("asr_ms", "classification_ms", "execution_ms", "total_ms"):
        assert key in body["latency"]
    assert body["latency"]["asr_ms"] is None  # no ASR ran, so no ASR time is reported
    assert body["latency"]["total_ms"] > 0


def test_process_streams_stage_events(client):
    response = client.post("/api/process", data={"text": "what time is it", "branch": "A", "stream": "true"})
    stages = [json.loads(line)["stage"] for line in response.text.strip().splitlines()]
    assert stages == ["received", "transcribed", "classifying", "executing", "result"]


def test_audio_without_asr_model_is_a_clear_503(client):
    response = client.post("/api/process", files={"audio": ("a.wav", make_wav(), "audio/wav")}, data={"branch": "A"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "asr_unavailable"


def test_bad_audio_is_rejected(client):
    response = client.post("/api/transcribe", files={"audio": ("a.wav", b"not audio", "audio/wav")})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "bad_audio"


def test_request_needs_audio_or_text(client):
    assert client.post("/api/process", data={"branch": "A"}).status_code == 422


def test_logging_is_opt_in_and_reports_exclude_dev_rows(client):
    client.post("/api/process", data={"text": "hello", "branch": "both"})
    assert not client.log_path.exists()

    client.post("/api/process", data={
        "text": "where is the nlp exam", "branch": "both", "log": "true",
        "session_id": "S1", "trial_id": "t1", "true_intent": "ask_course_venue", "true_course": "COMS4054A",
    })
    rows = read_rows(client.log_path)
    assert [row["branch"] for row in rows] == ["A", "B"]
    assert rows[0]["correct"] == "1" and rows[0]["task_correct"] == "1"
    assert summarise(rows)["rows"] == 0                        # dev-rules excluded by default
    assert summarise(rows, include_dev=True)["paired"]["n_pairs"] == 1


def test_classify_and_execute_endpoints(client):
    a = client.post("/api/classify/a", json={"text": "how do I get to Solomon Mahlangu House"}).json()
    assert a["intent"] == "ask_directions" and a["slots"]["location"] == "solomon_mahlangu"
    executed = client.post("/api/execute", json={"intent": a["intent"], "slots": a["slots"]}).json()
    assert "East Campus" in executed["response"]


def test_courses_schedule_and_asr_eval(client):
    assert len(client.get("/api/courses").json()["courses"]) == 11
    assert client.get("/api/schedule", params={"weekday": "monday"}).json()["display"]["title"].startswith("Monday")
    assert client.get("/api/schedule", params={"weekday": "funday"}).status_code == 422
    scores = client.post("/api/evaluate/asr", json={"reference": "a b c", "hypothesis": "a b"}).json()
    assert scores["wer"]["rate"] == pytest.approx(1 / 3)


def test_wav_decoding_resamples_to_16k():
    samples, duration = decode_audio(make_wav(seconds=0.5, rate=8000))
    assert duration == pytest.approx(0.5, abs=0.01)
    assert samples.dtype == np.float32 and abs(samples.size - 8000) <= 1


def test_whisper_decoding_defaults_to_beam_search():
    from backend.asr.whisper_asr import WhisperASR
    from backend.config import Settings

    asr = WhisperASR(Settings())
    assert asr.generation_kwargs() == {"max_new_tokens": 128, "num_beams": 5}
    assert asr.decoding() == "beam5" and asr.describe()["decoding"] == "beam5"
    greedy = Settings()
    greedy.whisper_num_beams = 1
    greedy.whisper_repetition_penalty = 1.2
    assert WhisperASR(greedy).generation_kwargs() == {"max_new_tokens": 128, "num_beams": 1, "repetition_penalty": 1.2}


def test_adapter_must_match_prompt_and_labels(tmp_path):
    import json as _json

    from backend.branch_b.branch import check_adapter_compatible
    from backend.branch_b.prompt import PROMPT_TEMPLATE
    from backend.nlu.intents import intent_labels

    with pytest.raises(RuntimeError, match="training_info"):
        check_adapter_compatible(tmp_path)
    info = {"prompt_template": PROMPT_TEMPLATE, "labels": intent_labels()}
    (tmp_path / "training_info.json").write_text(_json.dumps(info))
    assert check_adapter_compatible(tmp_path)["labels"] == intent_labels()
    (tmp_path / "training_info.json").write_text(_json.dumps(dict(info, labels=intent_labels()[:-1])))
    with pytest.raises(RuntimeError, match="different intent labels"):
        check_adapter_compatible(tmp_path)


def test_policy_is_logged_separately_from_the_model_prediction(client):
    body = client.post("/api/process", data={
        "text": "ngi cela ungi size ngama Student fees", "branch": "B", "log": "true",
        "session_id": "S2", "trial_id": "t1", "true_intent": "OOS"}).json()
    assert body["response"] == "Sorry, I cannot help with that."
    assert body["executed_intent"] == "OOS" and body["policy"] == "verb_unknown_object"
    assert body["frame"]["verb"] == "help"
    row = read_rows(client.log_path)[-1]
    assert row["policy"] == "verb_unknown_object" and row["executed_intent"] == "OOS"
    assert row["task_correct"] == "1"                     # Pepper did the right thing
    assert row["correct"] == ("1" if row["predicted_intent"] == "OOS" else "0")   # model scored on its own
