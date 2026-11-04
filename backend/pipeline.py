"""End-to-end trial: audio or text -> ASR -> branch(es) -> dispatcher -> adapter.

`run_trial` is a generator of stage events so the UI can show real progress
(transcribing / classifying / executing) when the request is streamed. The
last event is either {"stage": "result", ...} or {"stage": "error", ...}.
All latencies are measured with time.perf_counter() on the server.
"""

import time
import uuid
from pathlib import Path

from backend.asr.whisper_asr import ASRUnavailable
from backend.audio import AudioDecodeError, decode_audio
from backend.dispatcher.dispatcher import execute_intent
from backend.nlu.frame import analyse
from backend.nlu.slots import extract_slots
from backend.evaluation.metrics import cer, wer


class TrialError(Exception):
    def __init__(self, code, message, stage, status=422):
        super().__init__(message)
        self.code, self.message, self.stage, self.status = code, message, stage, status


def _ms(start):
    return round((time.perf_counter() - start) * 1000.0, 1)


def _safe_name(value, fallback):
    cleaned = "".join(ch for ch in (value or "") if ch.isalnum() or ch in "-_")
    return cleaned[:64] or fallback


def new_trial_id():
    return time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]


def transcribe_audio(engines, audio):
    """Return (transcript, asr_info). Raises TrialError."""
    try:
        samples, duration = decode_audio(audio)
    except AudioDecodeError as error:
        raise TrialError("bad_audio", str(error), "transcribing")
    start = time.perf_counter()
    try:
        transcript = engines.asr.transcribe(samples)
    except ASRUnavailable as error:
        raise TrialError("asr_unavailable", str(error), "transcribing", status=503)
    engine = engines.asr.name
    if hasattr(engines.asr, "decoding"):
        engine += "-" + engines.asr.decoding()   # e.g. "whisper-beam5", logged with every trial
    return transcript, {"engine": engine, "latency_ms": _ms(start), "duration_s": round(duration, 2)}


def run_trial(engines, research_log, recordings_dir, *, audio=None, text=None, branch="A", primary=None,
              session_id=None, trial_id=None, reference_transcript=None, true_intent=None,
              true_course=None, true_weekday=None, log=False, save_audio=False):
    started = time.perf_counter()
    branches = ["A", "B"] if branch == "both" else [branch]
    primary = primary if primary in branches else branches[0]
    session_id = _safe_name(session_id, "anonymous")
    trial_id = _safe_name(trial_id, new_trial_id())
    yield {"stage": "received", "session_id": session_id, "trial_id": trial_id}

    try:
        # --- ASR ----------------------------------------------------------
        asr_info = {"engine": "text-input", "latency_ms": None, "duration_s": None}
        audio_file = None
        if audio is not None:
            yield {"stage": "transcribing"}
            if save_audio and log:
                folder = Path(recordings_dir) / session_id
                folder.mkdir(parents=True, exist_ok=True)
                audio_path = folder / (trial_id + ".wav")
                audio_path.write_bytes(audio)
                audio_file = str(audio_path)
            transcript, asr_info = transcribe_audio(engines, audio)
            source = "audio"
        else:
            transcript = (text or "").strip()
            source = "text"

        if not transcript:
            raise TrialError("empty_transcript", "No speech was recognised. Please try again.", "transcribing")
        yield {"stage": "transcribed", "transcript": transcript, "asr": asr_info}

        asr_eval = None
        if reference_transcript and source == "audio":
            asr_eval = {
                "wer": wer(reference_transcript, transcript)["rate"],
                "cer": cer(reference_transcript, transcript)["rate"],
            }

        # --- Intent classification ----------------------------------------
        yield {"stage": "classifying", "branches": branches}
        results = {}
        for name in branches:
            try:
                results[name] = engines.branches[name].classify(transcript)
            except Exception as error:
                raise TrialError("classification_failed", "Branch %s failed: %s" % (name, error), "classifying", 500)

        # --- Execution ----------------------------------------------------
        yield {"stage": "executing"}
        # Verb-then-object reading of the isiZulu transcript, identical for both branches.
        frame = analyse(transcript, extract_slots(transcript))
        execution_ms = None
        dispatched = {}
        adapter_report = None
        for name in branches:
            start = time.perf_counter()
            dispatched[name] = execute_intent(results[name]["intent"], results[name]["slots"], frame=frame)
            if name == primary:
                adapter_report = engines.adapter.deliver(dispatched[name])
                execution_ms = _ms(start)
            results[name]["dispatch"] = {
                "status": dispatched[name]["status"],
                "response": dispatched[name]["response"],
                "executed_intent": dispatched[name]["intent"],
                "policy": dispatched[name]["policy"],
            }

        total_ms = _ms(started)
        chosen, outcome = results[primary], dispatched[primary]
        latency = {
            "asr_ms": asr_info["latency_ms"],
            "translation_ms": chosen["latency"]["translation_ms"],
            "classification_ms": chosen["latency"]["classification_ms"],
            "execution_ms": execution_ms,
            "total_ms": total_ms,
        }
        agreement = (results["A"]["intent"] == results["B"]["intent"]) if len(branches) == 2 else None

        if log:
            for name in branches:
                research_log.append({
                    "session_id": session_id, "trial_id": trial_id, "source": source,
                    "audio_file": audio_file, "audio_duration_s": asr_info["duration_s"],
                    "reference_transcript": reference_transcript, "asr_transcript": transcript,
                    "asr_engine": asr_info["engine"],
                    "wer": asr_eval["wer"] if asr_eval else None,
                    "cer": asr_eval["cer"] if asr_eval else None,
                    "true_intent": true_intent, "true_course": true_course, "true_weekday": true_weekday,
                    "branch": name, "branch_engine": results[name]["engine"],
                    "translation": results[name]["translation"],
                    "predicted_intent": results[name]["intent"], "confidence": results[name]["confidence"],
                    "course_slot": results[name]["slots"].get("course"),
                    "weekday_slot": results[name]["slots"].get("weekday"),
                    "location_slot": results[name]["slots"].get("location"),
                    "executed": name == primary,
                    "executed_intent": dispatched[name]["intent"], "policy": dispatched[name]["policy"],
                    "asr_latency_ms": asr_info["latency_ms"],
                    "translation_latency_ms": results[name]["latency"]["translation_ms"],
                    "classification_latency_ms": results[name]["latency"]["classification_ms"],
                    "execution_latency_ms": execution_ms if name == primary else None,
                    "total_latency_ms": total_ms,
                    "response": dispatched[name]["response"],
                })

        yield {
            "stage": "result",
            "result": {
                "session_id": session_id,
                "trial_id": trial_id,
                "source": source,
                "transcript": transcript,
                "asr": asr_info,
                "asr_evaluation": asr_eval,
                "branch": branch,
                "primary": primary,
                "intent": chosen["intent"],
                "slots": chosen["slots"],
                "confidence": chosen["confidence"],
                "translation": chosen["translation"],
                "status": outcome["status"],
                "executed_intent": outcome["intent"],
                "policy": outcome["policy"],
                "frame": frame,
                "response": outcome["response"],
                "action": outcome["action"],
                "display": outcome["display"],
                "sample_data": outcome["sample_data"],
                "branches": results,
                "agreement": agreement,
                "latency": latency,
                "adapter": adapter_report,
                "logged": bool(log),
            },
        }
    except TrialError as error:
        yield {"stage": "error", "error": {
            "code": error.code, "message": error.message, "stage": error.stage, "status": error.status,
        }}
