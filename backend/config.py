"""Runtime configuration, read once from environment variables.

Every model is optional. With nothing configured the backend starts with
development stand-ins so the HTTP contract and the UI can be exercised; the
engine names are reported by /api/health and written into every log row.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def _env(name, default=None):
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_bool(name, default=False):
    value = _env(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name, default):
    value = _env(name)
    return float(value) if value is not None else default


def _env_int(name, default):
    value = _env(name)
    return int(value) if value is not None else default


@dataclass
class Settings:
    # --- ASR -----------------------------------------------------------
    # "auto" uses Whisper when torch + transformers are installed, else "none".
    asr_engine: str = field(default_factory=lambda: _env("ASR_ENGINE", "auto"))
    whisper_model: str = field(
        default_factory=lambda: _env("WHISPER_MODEL", "dsfsi-anv/whisper-small-anv-zulu-first-batch")
    )
    # Leave unset to use the checkpoint's own generation config. The DSFSI
    # checkpoint has no isiZulu token, so set this to whatever token your
    # fine-tuning notebook detected if decoding drifts.
    whisper_language: str = field(default_factory=lambda: _env("WHISPER_LANGUAGE"))
    whisper_max_new_tokens: int = field(default_factory=lambda: _env_int("WHISPER_MAX_NEW_TOKENS", 128))
    whisper_no_repeat_ngram: int = field(default_factory=lambda: _env_int("WHISPER_NO_REPEAT_NGRAM", 3))
    whisper_repetition_penalty: float = field(
        default_factory=lambda: _env_float("WHISPER_REPETITION_PENALTY", 1.2)
    )

    # --- Branch A: isiZulu -> English -> English intent classifier -------
    # "auto" uses hf/zero-shot when a classifier is configured, else dev-rules.
    branch_a_engine: str = field(default_factory=lambda: _env("BRANCH_A_ENGINE", "auto"))
    translator_model: str = field(
        default_factory=lambda: _env("TRANSLATOR_MODEL", "facebook/nllb-200-distilled-600M")
    )
    translator_src_lang: str = field(default_factory=lambda: _env("TRANSLATOR_SRC_LANG", "zul_Latn"))
    translator_tgt_lang: str = field(default_factory=lambda: _env("TRANSLATOR_TGT_LANG", "eng_Latn"))
    # Path or hub id of a fine-tuned English sequence classifier whose
    # id2label uses the canonical intent labels.
    branch_a_classifier: str = field(default_factory=lambda: _env("BRANCH_A_CLASSIFIER"))
    # Optional untrained baseline: an NLI model used for zero-shot labelling.
    branch_a_zero_shot_model: str = field(
        default_factory=lambda: _env("BRANCH_A_ZERO_SHOT_MODEL", "facebook/bart-large-mnli")
    )

    # --- Branch B: isiZulu -> InkubaLM-0.4B (+ LoRA) -> intent ------------
    branch_b_engine: str = field(default_factory=lambda: _env("BRANCH_B_ENGINE", "auto"))
    inkubalm_model: str = field(default_factory=lambda: _env("INKUBALM_MODEL", "lelapa/InkubaLM-0.4B"))
    inkubalm_adapter: str = field(default_factory=lambda: _env("INKUBALM_ADAPTER"))
    inkubalm_trust_remote_code: bool = field(
        default_factory=lambda: _env_bool("INKUBALM_TRUST_REMOTE_CODE", True)
    )

    # --- Shared ----------------------------------------------------------
    device: str = field(default_factory=lambda: _env("DEVICE", "auto"))
    preload_models: bool = field(default_factory=lambda: _env_bool("PRELOAD_MODELS", True))
    courses_file: Path = field(
        default_factory=lambda: Path(_env("COURSES_FILE", str(DATA_DIR / "courses.json")))
    )
    intents_file: Path = field(
        default_factory=lambda: Path(_env("INTENTS_FILE", str(DATA_DIR / "intents.json")))
    )
    dev_keywords_file: Path = field(
        default_factory=lambda: Path(_env("DEV_KEYWORDS_FILE", str(DATA_DIR / "dev_keywords.json")))
    )
    log_dir: Path = field(default_factory=lambda: Path(_env("RESEARCH_LOG_DIR", str(DATA_DIR / "logs"))))
    recordings_dir: Path = field(
        default_factory=lambda: Path(_env("RECORDINGS_DIR", str(DATA_DIR / "recordings")))
    )
    # When set, every primary dispatcher result is POSTed to the Pepper task
    # executor bridge (pepper_app/scripts/task_executor.py --serve).
    pepper_executor_url: str = field(default_factory=lambda: _env("PEPPER_EXECUTOR_URL"))
    cors_origins: list = field(
        default_factory=lambda: [o.strip() for o in _env("CORS_ORIGINS", "*").split(",") if o.strip()]
    )
    max_audio_bytes: int = field(default_factory=lambda: _env_int("MAX_AUDIO_BYTES", 10 * 1024 * 1024))


settings = Settings()
