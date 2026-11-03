"""Builds the ASR engine and both intent branches from settings."""

import logging

from backend.asr.whisper_asr import NoASR, WhisperASR
from backend.branch_a.branch import BranchA
from backend.branch_b.branch import BranchB
from backend.dispatcher.adapters import build_adapter
from backend.torch_utils import ml_available

log = logging.getLogger("engines")

DEV_ENGINES = {"dev-rules", "none"}


def _resolve_asr(settings):
    engine = settings.asr_engine
    if engine == "auto":
        engine = "whisper" if ml_available() else "none"
    return WhisperASR(settings) if engine == "whisper" else NoASR()


def _resolve_branch_a(settings):
    engine = settings.branch_a_engine
    if engine == "auto":
        engine = "hf" if (settings.branch_a_classifier and ml_available()) else "dev-rules"
    if engine not in ("dev-rules", "hf", "zero-shot"):
        raise ValueError("Unknown BRANCH_A_ENGINE: %s" % engine)
    return BranchA(settings, engine)


def _resolve_branch_b(settings):
    engine = settings.branch_b_engine
    if engine == "auto":
        engine = "inkubalm" if (settings.inkubalm_adapter and ml_available()) else "dev-rules"
    if engine not in ("dev-rules", "inkubalm"):
        raise ValueError("Unknown BRANCH_B_ENGINE: %s" % engine)
    return BranchB(settings, engine)


class Engines:
    def __init__(self, settings):
        self.settings = settings
        self.asr = _resolve_asr(settings)
        self.branches = {"A": _resolve_branch_a(settings), "B": _resolve_branch_b(settings)}
        self.adapter = build_adapter(settings.pepper_executor_url)
        self.load_errors = {}

    def preload(self):
        for name, engine in (("asr", self.asr), ("A", self.branches["A"]), ("B", self.branches["B"])):
            try:
                engine.load()
            except Exception as error:  # keep serving; /api/health shows the error
                log.exception("Failed to load %s", name)
                self.load_errors[name] = str(error)

    def describe(self):
        info = {
            "asr": self.asr.describe(),
            "branch_a": self.branches["A"].describe(),
            "branch_b": self.branches["B"].describe(),
            "pepper_adapter": self.adapter.name,
        }
        info["development_engines_active"] = any(
            part["engine"] in DEV_ENGINES for part in (info["asr"], info["branch_a"], info["branch_b"])
        )
        if self.load_errors:
            info["load_errors"] = self.load_errors
        return info
