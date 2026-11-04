"""FastAPI service for the isiZulu Wits Assistant.

Run from the repository root:
    uvicorn backend.main:app --reload --port 8000
"""

import inspect
import json
import math
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.config import settings
from backend.dispatcher.dispatcher import execute_intent
from backend.dispatcher.timetable import WEEKDAYS, get_timetable
from backend.engines import Engines
from backend.evaluation.logger import ResearchLog, read_rows
from backend.evaluation.metrics import cer, wer
from backend.evaluation.report import summarise
from backend.nlu.intents import load_intents
from backend.pipeline import TrialError, run_trial, transcribe_audio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

VERSION = "0.1.0"
LOG_PATH = settings.log_dir / "research_log.csv"

engines = Engines(settings)
research_log = ResearchLog(LOG_PATH)


@asynccontextmanager
async def lifespan(app):
    if settings.preload_models:
        await run_in_threadpool(engines.preload)
    log.info("Engines: %s", json.dumps(engines.describe()))
    yield


app = FastAPI(title="isiZulu Wits Assistant API", version=VERSION, lifespan=lifespan)

cors = dict(allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])
# Lets an HTTPS page (e.g. the Vercel deployment) call a backend on localhost.
if "allow_private_network" in inspect.signature(CORSMiddleware.__init__).parameters:
    cors["allow_private_network"] = True
app.add_middleware(CORSMiddleware, **cors)


# --- Request models ---------------------------------------------------------

class TextIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)


class ExecuteIn(BaseModel):
    intent: str
    slots: dict = Field(default_factory=dict)


class AsrEvalIn(BaseModel):
    reference: str
    hypothesis: str


def json_safe(value):
    """Replace NaN/Infinity (invalid in JSON) with None, recursively."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def _error_response(error):
    return JSONResponse(status_code=error.get("status", 422), content={"error": error})


def _form_bool(value):
    return str(value).lower() in ("1", "true", "yes", "on")


# --- Endpoints --------------------------------------------------------------

@app.get("/api/health")
def health():
    timetable = get_timetable()
    return {
        "status": "ok",
        "version": VERSION,
        "time": timetable.now().isoformat(timespec="seconds"),
        "engines": engines.describe(),
        "sample_timetable": timetable.is_sample,
    }


@app.get("/api/intents")
def intents():
    return {"intents": load_intents()}


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    data = await audio.read()
    if len(data) > settings.max_audio_bytes:
        raise HTTPException(413, "Audio upload is too large.")
    try:
        transcript, asr_info = await run_in_threadpool(transcribe_audio, engines, data)
    except TrialError as error:
        return _error_response({"code": error.code, "message": error.message,
                                "stage": error.stage, "status": error.status})
    return {"transcript": transcript, "asr": asr_info}


def _classify(name, body):
    start = time.perf_counter()
    result = engines.branches[name].classify(body.text)
    result["latency"]["total_ms"] = round((time.perf_counter() - start) * 1000.0, 1)
    return json_safe(result)


@app.post("/api/classify/a")
def classify_a(body: TextIn):
    return _classify("A", body)


@app.post("/api/classify/b")
def classify_b(body: TextIn):
    return _classify("B", body)


@app.post("/api/process")
async def process(
    audio: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    branch: str = Form("A"),
    primary: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    trial_id: Optional[str] = Form(None),
    reference_transcript: Optional[str] = Form(None),
    true_intent: Optional[str] = Form(None),
    true_course: Optional[str] = Form(None),
    true_weekday: Optional[str] = Form(None),
    log: str = Form("false"),
    save_audio: str = Form("false"),
    stream: str = Form("false"),
):
    branch = branch.upper() if branch.lower() != "both" else "both"
    if branch not in ("A", "B", "both"):
        raise HTTPException(422, "branch must be A, B or both")
    data = await audio.read() if audio is not None else None
    if data is not None and len(data) > settings.max_audio_bytes:
        raise HTTPException(413, "Audio upload is too large.")
    if data is None and not (text or "").strip():
        raise HTTPException(422, "Send an audio file or a text transcript.")

    def trial():
        return run_trial(
            engines, research_log, settings.recordings_dir,
            audio=data, text=text, branch=branch, primary=(primary or "").upper() or None,
            session_id=session_id, trial_id=trial_id,
            reference_transcript=(reference_transcript or "").strip() or None,
            true_intent=(true_intent or "").strip() or None,
            true_course=(true_course or "").strip() or None,
            true_weekday=(true_weekday or "").strip().lower() or None,
            log=_form_bool(log), save_audio=_form_bool(save_audio),
        )

    if _form_bool(stream):
        def ndjson():
            for event in trial():
                yield json.dumps(json_safe(event), allow_nan=False) + "\n"
        return StreamingResponse(ndjson(), media_type="application/x-ndjson")

    events = await run_in_threadpool(lambda: list(trial()))
    final = events[-1]
    if final["stage"] == "error":
        return _error_response(final["error"])
    return json_safe(final["result"])


@app.post("/api/execute")
def execute(body: ExecuteIn):
    start = time.perf_counter()
    result = execute_intent(body.intent, body.slots)
    result["adapter"] = engines.adapter.deliver(result)
    result["latency"] = {"execution_ms": round((time.perf_counter() - start) * 1000.0, 1)}
    return result


@app.get("/api/courses")
def courses():
    return get_timetable().public_summary()


@app.get("/api/schedule")
def schedule(weekday: Optional[str] = Query(None, description="monday ... sunday, today or tomorrow")):
    if weekday and weekday.lower() not in WEEKDAYS + ["today", "tomorrow"]:
        raise HTTPException(422, "weekday must be a day name, today or tomorrow")
    return execute_intent("ask_day_schedule", {"weekday": weekday.lower()} if weekday else {})


@app.post("/api/evaluate/asr")
def evaluate_asr(body: AsrEvalIn):
    return {"wer": wer(body.reference, body.hypothesis), "cer": cer(body.reference, body.hypothesis)}


@app.get("/api/research/summary")
def research_summary(include_dev: bool = False):
    return summarise(read_rows(LOG_PATH), include_dev=include_dev)


@app.get("/api/research/export")
def research_export():
    if not LOG_PATH.exists():
        raise HTTPException(404, "No research trials have been logged yet.")
    return FileResponse(LOG_PATH, media_type="text/csv", filename="research_log.csv")
