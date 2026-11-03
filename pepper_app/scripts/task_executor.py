# -*- coding: utf-8 -*-
"""Pepper task executor for the isiZulu Wits Assistant (NAOqi 2.5, Python 2.7).

Two ways to run it:

1. Inside the Choregraphe behaviour on a physical Pepper (behavior.xar):
   the tablet raises IsiZuluAssistant/StartListening, this script records
   from the head microphones with ALAudioRecorder, POSTs the WAV to the
   Python 3 backend (/api/process), relays stage events to the tablet,
   then speaks the response.

2. As a bridge on the laptop next to Choregraphe's virtual Pepper (which has
   no microphone):
       python2 task_executor.py --qi-url tcp://127.0.0.1:<port> --serve 8765
   and start the backend with PEPPER_EXECUTOR_URL=http://127.0.0.1:8765.
   The browser simulator records on the laptop; the backend forwards each
   dispatcher result here and the virtual robot says it.

The robot only ever receives canonical dispatcher output; it never sees
which branch produced the intent. This file has not yet been run on a
physical robot: verify the NAOqi version (2.5 vs 2.9) before relying on it.
"""

from __future__ import print_function

import json
import os
import threading
import time
import uuid

try:  # Python 2.7 on the robot
    import urllib2 as urlrequest
    from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
except ImportError:  # Python 3 (tests, tooling)
    import urllib.request as urlrequest
    from http.server import BaseHTTPRequestHandler, HTTPServer

EVENTS = {
    "start": "IsiZuluAssistant/StartListening",
    "stop": "IsiZuluAssistant/StopListening",
    "state": "IsiZuluAssistant/State",
    "result": "IsiZuluAssistant/Result",
}

DEFAULT_CONFIG = {
    "backend_url": "http://192.168.1.100:8000",
    "branch": "A",
    "record_seconds": 6,
    "recordings_dir": "/home/nao/recordings/isizulu",
    # ALAudioRecorder channel mask [left, right, front, rear]; 16 kHz needs one channel.
    "microphone_channels": [0, 0, 1, 0],
    "language": "English",
    "request_timeout_s": 60,
}


def load_config(path=None):
    config = dict(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        with open(path) as handle:
            config.update(json.load(handle))
    return config


def encode_multipart(fields, file_field, filename, data):
    """Build a multipart/form-data body without third-party packages."""
    boundary = "----isizulu" + uuid.uuid4().hex
    lines = []
    for key, value in fields.items():
        if value is None:
            continue
        lines.append("--" + boundary)
        lines.append('Content-Disposition: form-data; name="%s"' % key)
        lines.append("")
        lines.append(str(value))
    head = ("\r\n".join(lines) + "\r\n").encode("utf-8") if lines else b""
    file_head = ("--%s\r\nContent-Disposition: form-data; name=\"%s\"; filename=\"%s\"\r\n"
                 "Content-Type: audio/wav\r\n\r\n" % (boundary, file_field, filename)).encode("utf-8")
    tail = ("\r\n--%s--\r\n" % boundary).encode("utf-8")
    return boundary, head + file_head + data + tail


def post_audio(config, wav_bytes, fields):
    """POST to /api/process with stream=true; yields stage events (dicts)."""
    fields = dict(fields, stream="true")
    boundary, body = encode_multipart(fields, "audio", "pepper.wav", wav_bytes)
    request = urlrequest.Request(
        config["backend_url"].rstrip("/") + "/api/process", data=body,
        headers={"Content-Type": "multipart/form-data; boundary=" + boundary},
    )
    response = urlrequest.urlopen(request, timeout=config["request_timeout_s"])
    try:
        for line in iter(response.readline, b""):
            line = line.strip()
            if line:
                yield json.loads(line.decode("utf-8"))
    finally:
        response.close()


class TaskExecutor(object):
    """Performs a dispatcher result on the robot: speech plus tablet update."""

    def __init__(self, session, config):
        self.config = config
        self.memory = session.service("ALMemory")
        try:
            self.speech = session.service("ALAnimatedSpeech")
        except Exception:
            self.speech = None
        self.tts = session.service("ALTextToSpeech")
        try:
            self.tts.setLanguage(config["language"])
        except Exception as error:
            print("Could not set TTS language:", error)

    def raise_state(self, state):
        self.memory.raiseEvent(EVENTS["state"], state)

    def raise_result(self, result):
        self.memory.raiseEvent(EVENTS["result"], json.dumps(result))

    def say(self, text):
        if not text:
            return
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        text = text.encode("utf-8") if str is bytes else text  # NAOqi 2.5 expects UTF-8 str
        if self.speech is not None:
            self.speech.say(text)
        else:
            self.tts.say(text)

    def execute(self, result):
        """Show the result on the tablet, then speak it.

        action.type: say | show | open_url. The tablet page renders `display`
        (tables, course cards, the Self-Service button); Pepper never opens a
        login page by itself and never handles credentials.
        """
        self.raise_result(result)
        self.say(result.get("response"))
        self.raise_state("success")


class Listener(object):
    """Physical robot only: records on tablet request and runs the pipeline."""

    def __init__(self, session, executor, config):
        self.session = session
        self.executor = executor
        self.config = config
        self.memory = session.service("ALMemory")
        self.recorder = session.service("ALAudioRecorder")
        self.busy = False
        self.stop_requested = threading.Event()
        self.subscribers = []

    def start(self):
        for name, handler in ((EVENTS["start"], self.on_start), (EVENTS["stop"], self.on_stop)):
            subscriber = self.memory.subscriber(name)
            subscriber.signal.connect(handler)
            self.subscribers.append(subscriber)  # keep references alive

    def stop(self):
        self.stop_requested.set()
        self.subscribers = []

    def on_start(self, value):
        if self.busy:
            return
        options = {}
        try:
            options = json.loads(value) if value else {}
        except ValueError:
            pass
        self.busy = True
        self.stop_requested.clear()
        thread = threading.Thread(target=self.run_trial, args=(options,))
        thread.daemon = True
        thread.start()

    def on_stop(self, value):
        self.stop_requested.set()

    def record(self):
        folder = self.config["recordings_dir"]
        if not os.path.isdir(folder):
            os.makedirs(folder)
        path = os.path.join(folder, "trial-%d.wav" % int(time.time() * 1000))
        try:
            self.recorder.stopMicrophonesRecording()
        except Exception:
            pass  # nothing was recording
        self.recorder.startMicrophonesRecording(path, "wav", 16000, self.config["microphone_channels"])
        self.executor.raise_state("listening")
        self.stop_requested.wait(self.config["record_seconds"])
        self.recorder.stopMicrophonesRecording()
        with open(path, "rb") as handle:
            data = handle.read()
        os.remove(path)  # audio is kept on the backend only if research logging asks for it
        return data

    def run_trial(self, options):
        try:
            audio = self.record()
            self.executor.raise_state("uploading")
            fields = {
                "branch": options.get("branch") or self.config["branch"],
                "session_id": options.get("session_id"),
            }
            for event in post_audio(self.config, audio, fields):
                stage = event.get("stage")
                if stage in ("transcribing", "classifying", "executing"):
                    self.executor.raise_state(stage)
                elif stage == "error":
                    message = event["error"]["message"]
                    self.executor.raise_result({"error": event["error"]})
                    self.executor.say("Sorry, something went wrong. Please try again.")
                    print("Pipeline error:", message)
                elif stage == "result":
                    self.executor.execute(event["result"])
        except Exception as error:
            print("Trial failed:", error)
            self.executor.raise_state("error:Could not reach the assistant service.")
            try:
                self.executor.say("Sorry, I cannot reach the assistant service.")
            except Exception:
                pass
        finally:
            self.busy = False


class Assistant(object):
    """What the Choregraphe box starts on the robot."""

    def __init__(self, session, config):
        self.executor = TaskExecutor(session, config)
        self.listener = Listener(session, self.executor, config)

    def start(self):
        self.listener.start()
        self.executor.raise_state("idle")

    def stop(self):
        self.listener.stop()


def make_bridge_handler(executor):
    class BridgeHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path.rstrip("/") != "/execute":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                result = json.loads(self.rfile.read(length).decode("utf-8"))
            except ValueError:
                self.send_error(400, "invalid JSON")
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok": true}')
            # Speak after replying so the backend is not held up by TTS.
            threading.Thread(target=executor.execute, args=(result,)).start()

        def log_message(self, fmt, *args):
            print("bridge:", fmt % args)

    return BridgeHandler


def main():
    import argparse

    import qi  # NAOqi Python SDK (pynaoqi), Python 2.7

    parser = argparse.ArgumentParser(description="isiZulu Wits Assistant task executor")
    parser.add_argument("--qi-url", default="tcp://127.0.0.1:9559", help="robot or virtual robot URL")
    parser.add_argument("--serve", type=int, default=0, help="run the HTTP bridge on this port")
    parser.add_argument("--listen", action="store_true", help="also record from the robot microphones")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"))
    args = parser.parse_args()

    app = qi.Application(["task_executor", "--qi-url=" + args.qi_url])
    app.start()
    config = load_config(args.config)
    executor = TaskExecutor(app.session, config)
    if args.listen:
        Listener(app.session, executor, config).start()
    if args.serve:
        server = HTTPServer(("127.0.0.1", args.serve), make_bridge_handler(executor))
        print("Bridge listening on http://127.0.0.1:%d/execute" % args.serve)
        server.serve_forever()
    else:
        app.run()


if __name__ == "__main__":
    main()
