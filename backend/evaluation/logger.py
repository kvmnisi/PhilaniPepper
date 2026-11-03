"""Append-only CSV research log. One row per (trial, branch).

Rows are written only when a request opts in (research mode "Log trials"),
so normal student use leaves no record. Ground-truth columns are optional.
"""

import csv
import threading
from datetime import datetime, timezone
from pathlib import Path

COLUMNS = [
    "timestamp",
    "session_id",
    "trial_id",
    "source",
    "audio_file",
    "audio_duration_s",
    "reference_transcript",
    "asr_transcript",
    "asr_engine",
    "wer",
    "cer",
    "true_intent",
    "true_course",
    "true_weekday",
    "branch",
    "branch_engine",
    "translation",
    "predicted_intent",
    "confidence",
    "course_slot",
    "weekday_slot",
    "location_slot",
    "correct",
    "task_correct",
    "executed",
    "asr_latency_ms",
    "translation_latency_ms",
    "classification_latency_ms",
    "execution_latency_ms",
    "total_latency_ms",
    "response",
]

_lock = threading.Lock()


def _blank(value):
    return "" if value is None else value


def _bool(value):
    return "" if value is None else int(bool(value))


def task_correct(row):
    """Intent right and every supplied gold slot matched; None without gold."""
    if not row.get("true_intent"):
        return None
    if row.get("predicted_intent") != row["true_intent"]:
        return False
    for gold_key, slot_key in (("true_course", "course_slot"), ("true_weekday", "weekday_slot")):
        gold = (row.get(gold_key) or "").strip().lower()
        if gold and gold != (row.get(slot_key) or "").strip().lower():
            return False
    return True


class ResearchLog:
    def __init__(self, path):
        self.path = Path(path)

    def append(self, row):
        row = dict(row)
        row.setdefault("timestamp", datetime.now(timezone.utc).isoformat(timespec="milliseconds"))
        if row.get("true_intent"):
            row["correct"] = row.get("predicted_intent") == row["true_intent"]
        else:
            row["correct"] = None
        row["task_correct"] = task_correct(row)
        for key in ("correct", "task_correct", "executed"):
            row[key] = _bool(row.get(key))
        clean = {column: _blank(row.get(column)) for column in COLUMNS}
        with _lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            new_file = not self.path.exists() or self.path.stat().st_size == 0
            with open(self.path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=COLUMNS)
                if new_file:
                    writer.writeheader()
                writer.writerow(clean)
        return clean


def read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
