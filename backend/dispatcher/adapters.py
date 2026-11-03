"""Execution adapters: where a dispatcher result goes after the web response.

The web UI always receives the result over HTTP. Optionally the backend also
forwards it to the Pepper task executor bridge (a Python 2.7 process talking
to a virtual or physical robot through NAOqi), so the same result drives the
browser simulator, Choregraphe's virtual Pepper and, later, the real robot.
"""

import json
import urllib.request


class NullAdapter:
    name = "none"

    def deliver(self, result):
        return {"adapter": self.name, "delivered": False}


class HttpPepperAdapter:
    name = "pepper-http"

    def __init__(self, url, timeout=2.0):
        self.url = url.rstrip("/") + "/execute"
        self.timeout = timeout

    def deliver(self, result):
        payload = {
            "intent": result.get("intent"),
            "slots": result.get("slots", {}),
            "response": result.get("response"),
            "action": result.get("action"),
            "display": result.get("display"),
        }
        request = urllib.request.Request(
            self.url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return {"adapter": self.name, "delivered": 200 <= response.status < 300}
        except Exception as error:  # the robot being offline must not break the UI
            return {"adapter": self.name, "delivered": False, "error": str(error)}


def build_adapter(url):
    return HttpPepperAdapter(url) if url else NullAdapter()
