# isiZulu Wits Assistant

The tablet interface and speech-to-action pipeline for **Making Pepper Bot
Understand isiZulu**, a BSc Honours research project at the University of the
Witwatersrand. A student speaks naturally in isiZulu. Whisper transcribes it,
an intent branch (A or B) works out what the student wants, and a shared
dispatcher has Pepper perform the task.

```
Pepper App Launcher → "isiZulu Wits Assistant" → tablet page → student speaks isiZulu
  → Whisper ASR → Branch A or Branch B → canonical intent + slots → dispatcher → Pepper responds
```

## Status

The application is complete as a **pipeline and interface**. The research
models plug into it, and the ones that are not trained yet are clearly marked.
No experimental score appears anywhere in this repository.

| Stage | What | State |
| --- | --- | --- |
| 1 | Tablet frontend, with demo responses when no backend is reachable | Done |
| 2 | Python backend and `/api/health` | Done |
| 3 | Browser recording → 16 kHz WAV → upload | Done |
| 4–5 | Whisper ASR, transcript shown in the UI | Code done. Loads the DSFSI checkpoint or your fine-tuned one via `WHISPER_MODEL`. Not run here without weights. |
| 6 | Branch A: NLLB-200 translation → English classifier | Code done. Needs a trained English classifier (`BRANCH_A_CLASSIFIER`), or use the untrained `zero-shot` baseline. |
| 7 | Branch B: InkubaLM-0.4B + LoRA | Inference done (label scoring). **The LoRA fine-tune itself has not been done**, and no training script is included yet. |
| 8–9 | Shared dispatcher wired to the UI | Done |
| 10 | Choregraphe / Pepper integration | Code done (`pepper_app/`). Untested on a robot. The NAOqi version is still unconfirmed. |
| 11 | Research logging, WER/CER, macro-F1, McNemar, batch evaluation | Done |
| 12 | Vercel deployment | Configured (`frontend/vercel.json`). Not deployed from this repo. |

Until real models are configured, both branches run **`dev-rules`**: keyword
stand-ins that exist only to test the plumbing. Without the ML libraries,
Branch A's stand-in cannot translate, so isiZulu input comes out as OOS. With
them installed, Branch A defaults to **`nllb-rules`**: real translation feeding
the same English keyword stand-in. Every result, log row and
health check names the engine that produced it, and the report tool drops
`dev-rules` rows by default.

## Architecture

```mermaid
flowchart TD
    subgraph Pepper["Pepper robot (NAOqi 2.5)"]
        T["Tablet web app<br/>pepper_app/html"]
        X["Choregraphe behaviour<br/>task_executor.py"]
    end
    subgraph Laptop["Laptop / server (Python 3)"]
        API["FastAPI · backend/main.py"]
        ASR["Whisper ASR (fine-tuned isiZulu)"]
        A["Branch A<br/>NLLB zul→eng + English classifier"]
        B["Branch B<br/>InkubaLM-0.4B + LoRA"]
        S["Slot extraction<br/>course · weekday · location"]
        D["Shared dispatcher<br/>execute_intent(intent, slots)"]
        L[("Research log CSV")]
    end
    W["Browser / Vercel tablet simulator"]

    T -- "ALMemory event" --> X
    X -- "WAV over HTTP" --> API
    W -- "WAV or typed text over HTTP" --> API
    API --> ASR -- "isiZulu transcript (identical for both)" --> A & B
    A --> S
    B --> S
    S --> D
    D -- "response + action + display" --> API
    API -- "result / stage events" --> W
    API -- "result / stage events" --> X
    API -. "optional adapter" .-> X
    API --> L
```

Each branch receives the **same transcript** and returns a **canonical intent
plus slots**:

```json
{ "intent": "ask_course_venue", "slots": { "course": "COMS9001" } }
```

Slots are filled by one shared keyword extractor, so the A/B comparison
isolates the intent decision. The dispatcher never knows which branch
produced the intent. Pepper-specific execution sits behind an adapter, so the
same result drives the browser simulator, Choregraphe's virtual Pepper, and
later the physical robot.

### Intents

`greeting`, `help`, `ask_time`, `ask_current_class`, `ask_next_class`,
`ask_lecturer`, `ask_course_time`, `ask_course_venue`, `ask_day_schedule`,
`list_courses`, `ask_current_semester`, `open_self_service`, `OOS`.
Slots: `course`, `weekday` (including `today`/`tomorrow`), `location`.
Defined once in [`data/intents.json`](data/intents.json).

## Repository layout

```
frontend/            static tablet app: HTML, CSS, ES5 JavaScript, no build step
  js/ app.js (state machine) · api.js · recorder.js · pepper.js · demo.js · ui.js · config.js
  vercel.json, build-env.js
pepper_app/          Choregraphe package: manifest, behaviour, task executor, html/ (generated)
backend/
  main.py            FastAPI endpoints
  pipeline.py        one trial: ASR → branches → dispatcher → adapter → log
  asr/               Whisper engine
  branch_a/          NLLB translator + HF / zero-shot English classifier
  branch_b/          InkubaLM + LoRA label scorer, shared prompt format
  nlu/               canonical intents, slot extraction, dev-rules stand-in
  dispatcher/        execute_intent, timetable, Pepper adapter
  evaluation/        metrics, research log, report, batch evaluator
data/                intents, sample timetable, dataset template (see data/README.md)
scripts/             sync_pepper_html.py
tests/               pytest suite
```

## Run it locally

Requires Python 3.10+.

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r backend/requirements.txt
```

```bash
uvicorn backend.main:app --reload --port 8000
```

In a second terminal, serve the frontend:

```bash
python -m http.server 5173 --directory frontend
```

Open <http://localhost:5173>. The status pill reads **Assistant online**.
With no models installed, the microphone works but the backend reports that
ASR is not configured. Use research mode's **Bypass ASR** box to send typed
text through the branches. If the backend is not running, the page falls back
to a clearly labelled **demo mode** that only exercises the UI states.

**Research mode:** press and hold the crest for 1.5 seconds, or open
`?mode=research`. Students never see it.

### Turn on the models

```bash
pip install -r backend/requirements-ml.txt
```

Then configure engines with environment variables (see `.env.example`):

| Variable | Purpose |
| --- | --- |
| `ASR_ENGINE=whisper`, `WHISPER_MODEL=<hub id or path>` | Fine-tuned Whisper checkpoint. `WHISPER_LANGUAGE` is optional; the DSFSI model has no isiZulu token, so use the one your notebook detected. Repetition penalties are applied, as in the notebook. |
| `BRANCH_A_ENGINE=hf`, `BRANCH_A_CLASSIFIER=<path>` | NLLB-200 (`zul_Latn`→`eng_Latn`) plus your fine-tuned English sequence classifier. Its `id2label` must use the canonical labels; the backend refuses any others. |
| `BRANCH_A_ENGINE=nllb-rules` | The default once the ML libraries are installed and no classifier is set. Real NLLB translation, so isiZulu works end to end, but intents come from the English keyword stand-in. Excluded from reports. |
| `BRANCH_A_ENGINE=zero-shot` | NLLB-200 plus an NLI model scoring each intent's English hypothesis. A no-training baseline; report it as one. |
| `BRANCH_B_ENGINE=inkubalm`, `INKUBALM_ADAPTER=<path>` | `lelapa/InkubaLM-0.4B` with your LoRA adapter. |
| `DEVICE` | `cuda`, `mps` or `cpu` (default: auto) |
| `PEPPER_EXECUTOR_URL` | Forward executed results to the Pepper bridge |

**Branch B training format.** Inference scores every canonical label as a
continuation of the prompt in [`backend/branch_b/prompt.py`](backend/branch_b/prompt.py),
then takes a softmax over the label scores. This always yields a valid label
plus a probability. Train the LoRA with `encode_example()` from that file, with
the loss on the tokens after `prompt_len` only. Otherwise training and
inference will not match.

## Three ways the frontend runs

1. **Desktop browser** (development): as above.
2. **Vercel** (hosted research/demo UI): static files only. Whisper and
   InkubaLM stay on the Python backend, on suitable hardware.
3. **Pepper tablet**: `pepper_app/html/`, generated by
   `python scripts/sync_pepper_html.py`. This copy sets `runtime: "pepper"`
   and inlines CSS variables for the older tablet WebView. The tablet raises
   NAOqi events instead of recording. See [`pepper_app/README.md`](pepper_app/README.md).

### Deploy to Vercel

1. Import the GitHub repository in Vercel and set **Root Directory** to `frontend`.
2. Optionally set the environment variable `API_BASE_URL` to your backend's
   **HTTPS** URL. `build-env.js` writes it into `js/env.js` at build time.
3. Deploy. You can also point any deployment at a backend with
   `?api=https://…`, or with the research panel's **Backend** box (remembered
   per browser).

The hosted page is HTTPS, so browsers block calls from it to a plain-HTTP LAN
address such as `http://192.168.x.x:8000`. Calls to `http://localhost:8000`
on the same machine are allowed: the backend sends the Private Network
Access header. For another device, expose the backend over HTTPS (for
example a Cloudflare or ngrok tunnel), or run the frontend locally. The
microphone also needs HTTPS or localhost.

## API

| Method & path | Purpose |
| --- | --- |
| `GET /api/health` | Engines in use, load errors, sample-data flag |
| `POST /api/process` | **Main flow.** Multipart: `audio` (WAV preferred) or `text`, `branch` = `A`/`B`/`both`, `primary`, `session_id`, `trial_id`, optional ground truth (`reference_transcript`, `true_intent`, `true_course`, `true_weekday`), `log`, `save_audio`, `stream`. With `stream=true` it returns NDJSON stage events (`received → transcribing → transcribed → classifying → executing → result`), which the UI uses for real progress. |
| `POST /api/transcribe` | ASR only |
| `POST /api/classify/a`, `POST /api/classify/b` | `{"text": …}` → intent, slots, confidence, latency |
| `POST /api/execute` | `{"intent", "slots"}` → dispatcher result |
| `GET /api/courses`, `GET /api/schedule?weekday=` | Timetable |
| `GET /api/intents` | Canonical labels |
| `POST /api/evaluate/asr` | `{"reference", "hypothesis"}` → WER and CER with S/D/I counts |
| `GET /api/research/summary`, `GET /api/research/export` | Report JSON, raw CSV |

Example `/api/process` result (abridged):

```json
{
  "transcript": "i-robotics ikuphi?",
  "branch": "B",
  "intent": "ask_course_venue",
  "slots": { "course": "COMS9001" },
  "confidence": null,
  "response": "COMS9001 Robotics is held in MSL004.",
  "action": { "type": "show", "text": "…" },
  "display": { "kind": "course", "code": "COMS9001", "sessions": [ … ] },
  "branches": { "B": { "engine": "dev-rules", "intent": "ask_course_venue", … } },
  "latency": { "asr_ms": null, "translation_ms": null, "classification_ms": 5.1, "execution_ms": 0.1, "total_ms": 22.0 }
}
```

All latencies are measured server-side with `time.perf_counter()`. A value is
`null` when that stage did not run. The research panel also shows the
client-measured round trip.

## Research workflow

Research mode lets you choose Branch A, Branch B or **Compare both**. With
both, the branches get identical input and the panel shows predictions side
by side with agreement. You can enter optional ground truth, a reference
transcript for WER/CER, and opt in to logging and audio saving. Normal
student use logs nothing.

| Research question | Where the evidence comes from |
| --- | --- |
| RQ1: task-completion rate | `task_correct` per logged row (intent plus gold slots) |
| RQ2: constrained-domain WER | Logged audio trials with a reference transcript; `/api/evaluate/asr` |
| RQ3: Branch A macro-F1 | `report` or `batch` |
| RQ4: A vs B, paired | Exact McNemar test on trials where both branches saw the same input |
| Latency | Per-stage and total, in milliseconds |
| ASR errors downstream | Intent accuracy split by WER = 0 vs WER > 0 |

Summarise a log:

```bash
python -m backend.evaluation.report data/logs/research_log.csv
```

Run a labelled text dataset through both branches. This answers RQ3/RQ4 even
without recordings. Only `verified=yes` rows are used:

```bash
python -m backend.evaluation.batch data/intent_dataset.csv --branches both
```

The dataset follows [`data/intent_dataset_template.csv`](data/intent_dataset_template.csv).
Add `--use-audio` to transcribe `audio_path` first.

## Privacy and safety

- The assistant never asks for, receives or stores passwords, logins or
  tokens. "Open Self-Service" only links to <https://self-service.wits.ac.za/>,
  where students sign in on the official page.
- Logging and audio saving are opt-in, per trial, from research mode only.
  `data/logs/` and `data/recordings/` are git-ignored.
- No credentials or API keys are in the repository. Use `.env` (ignored)
  for local settings.

## Fonts and branding

Headings use **Grotesk Medium** (SoftMaker, `frontend/assets/fonts/Grotesk-Medium.ttf`),
which is committed for this non-commercial project. If **Europa Grotesk SH Bold** is
installed or supplied locally it takes priority; otherwise the fallback is
`"Arial Narrow", Arial, sans-serif`. See
[`frontend/assets/fonts/README.md`](frontend/assets/fonts/README.md). The Wits crest in `frontend/assets/` is used
for this University project. Confirm the use with University brand guidelines
before any public deployment.

## Tests

```bash
python -m pytest
```

The tests cover WER/CER, macro-F1, McNemar, slot extraction, every dispatcher
intent, the API (streaming, errors, opt-in logging, dev-row exclusion),
Pepper package validity, and that `pepper_app/html` is in sync with `frontend/`.

## Known limitations

- `data/courses.json` is **sample data**. Replace it with the confirmed Honours timetable.
- The isiZulu in `data/dev_keywords.json` and the slot-extraction prefixes is
  AI-drafted and **unverified**. A native speaker must check it.
- Pepper's responses are in English, through the robot's English TTS.
- The Pepper package has not been run on a robot and targets NAOqi 2.5 only.
