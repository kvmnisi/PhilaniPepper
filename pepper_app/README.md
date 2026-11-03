# Pepper application package

A Choregraphe project (NAOqi 2.5, Python 2.7) that puts the isiZulu Wits
Assistant into Pepper's App Launcher.

```
pepper_app/
├── isizulu_wits_assistant.pml   open this in Choregraphe
├── manifest.xml                 package id: isizulu-wits-assistant
├── behavior.xar                 one Python box that starts the assistant
├── icon.png                     App Launcher icon
├── html/                        tablet page, generated from ../frontend
└── scripts/
    ├── task_executor.py         records, calls the backend, speaks, updates the tablet
    └── config.json              backend URL, recording length, branch
```

> **Not yet run on hardware.** The status report notes that the lab robot's
> software version (NAOqi 2.5 with Choregraphe, or 2.9 with the Android QiSDK)
> has not been confirmed. This package targets **2.5**. On 2.9 the tablet page
> and the HTTP contract still apply, but the robot side must be rewritten as
> an Android app.

## How a turn works on the robot

```
tablet: student taps mic ──raiseEvent──▶ IsiZuluAssistant/StartListening
task_executor.py: ALAudioRecorder, front mic, 16 kHz WAV, up to record_seconds
                   (tapping again raises StopListening and ends it early)
              ──POST /api/process (stream=true)──▶ Python 3 backend on the laptop
              ◀── stage events: transcribing, classifying, executing, result
              ──raiseEvent IsiZuluAssistant/State and /Result──▶ tablet updates
              ──ALAnimatedSpeech.say(response)
```

The tablet does not record on Pepper. Its WebView has no usable microphone,
and the head microphones are the ones the research is about.

## Install on a robot

1. Regenerate the tablet copy after any frontend change:
   `python scripts/sync_pepper_html.py` (run from the repo root).
2. Edit `scripts/config.json` and set `backend_url` to the laptop's LAN
   address, e.g. `http://192.168.1.100:8000`. The robot and the laptop must be
   on the same network. Start the backend with `--host 0.0.0.0`.
3. Open `isizulu_wits_assistant.pml` in Choregraphe 2.5, connect to the robot,
   then use *Package and install* from the Robot Applications panel.
4. Disable Autonomous Life's dialogue (or the built-in English recogniser)
   while the app runs, so Pepper's own English ASR does not compete.
5. Launch **isiZulu Wits Assistant** from the App Launcher.

If Choregraphe complains about `behavior.xar` (it was written by hand, not
saved from Choregraphe), create a new project, add a single Python box, and
paste in the box script from `behavior.xar`. It only adds `scripts/` to
`sys.path` and calls `task_executor.Assistant(session, config).start()`.

## Choregraphe virtual robot (no microphone)

The virtual robot cannot hear, so the laptop's browser does the recording and
the executor runs as a bridge:

```bash
# Terminal 1, Python 2.7 with the NAOqi Python SDK on PYTHONPATH
python2 pepper_app/scripts/task_executor.py --qi-url tcp://127.0.0.1:<virtual-robot-port> --serve 8765
```

```bash
# Terminal 2, the backend forwards every executed result to the bridge
PEPPER_EXECUTOR_URL=http://127.0.0.1:8765 uvicorn backend.main:app --port 8000
```

Open the frontend in a browser. Each response is also spoken by the virtual
Pepper. The research panel's **Pepper adapter** row shows whether the result
was delivered.

## Tablet ↔ robot events (ALMemory)

| Event | Raised by | Value |
| --- | --- | --- |
| `IsiZuluAssistant/StartListening` | tablet | JSON `{branch, session_id}` |
| `IsiZuluAssistant/StopListening` | tablet | empty |
| `IsiZuluAssistant/State` | robot | `listening`, `uploading`, `transcribing`, `classifying`, `executing`, `success`, or `error:<message>` |
| `IsiZuluAssistant/Result` | robot | JSON dispatcher result, the same shape the web UI receives |
