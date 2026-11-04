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


LOCAL_WHISPER_DIR = REPO_ROOT / "WhisperModel"
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def _default_whisper_model():
    """The fine-tuned checkpoint in ./WhisperModel when its weights are present
    (they are git-ignored, so copy them in locally), else the DSFSI hub model."""
    if any((LOCAL_WHISPER_DIR / name).exists() for name in WEIGHT_FILES):
        return str(LOCAL_WHISPER_DIR)
    return "dsfsi-anv/whisper-small-anv-zulu-first-batch"


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
    whisper_model: str = field(default_factory=lambda: _env("WHISPER_MODEL") or _default_whisper_model())
    # Leave unset to use the checkpoint's own generation config. The DSFSI
    # checkpoint has no isiZulu token, so set this to whatever token your
    # fine-tuning notebook detected if decoding drifts.
    whisper_language: str = field(default_factory=lambda: _env("WHISPER_LANGUAGE"))
    whisper_max_new_tokens: int = field(default_factory=lambda: _env_int("WHISPER_MAX_NEW_TOKENS", 128))
    # Decoding. Beam size 5 matches the best reported evaluation (16.25% WER on
    # AST, checkpoint-2656). The repetition controls default to off, matching
    # the generation_config.json saved with the fine-tuned model.
    whisper_num_beams: int = field(default_factory=lambda: _env_int("WHISPER_NUM_BEAMS", 5))
    whisper_no_repeat_ngram: int = field(default_factory=lambda: _env_int("WHISPER_NO_REPEAT_NGRAM", 0))
    whisper_repetition_penalty: float = field(
        default_factory=lambda: _env_float("WHISPER_REPETITION_PENALTY", 1.0)
    )

    # --- Branch A: isiZulu -> English -> English intent classifier -------
    # "auto": hf when BRANCH_A_CLASSIFIER is set; otherwise nllb-rules when the
    # ML libraries are installed (real translation, stand-in classifier);
    # otherwise dev-rules (no translation at all).
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

    # --- Branch B: isiZulu -> InkubaLM-0.4B -> intent ---------------------
    # "auto": inkubalm when the ML libraries are installed, else dev-rules.
    branch_b_engine: str = field(default_factory=lambda: _env("BRANCH_B_ENGINE", "auto"))
    inkubalm_model: str = field(default_factory=lambda: _env("INKUBALM_MODEL", "lelapa/InkubaLM-0.4B"))
    # Labelled example requests and label words shown to InkubaLM in the prompt.
    branch_b_examples_file: Path = field(
        default_factory=lambda: Path(_env("BRANCH_B_EXAMPLES_FILE", str(DATA_DIR / "branch_b_examples.json")))
    )
    # Divide out the model's preference for some label words (content-free calibration).
    inkubalm_calibrate: bool = field(default_factory=lambda: _env_bool("INKUBALM_CALIBRATE", True))
    inkubalm_trust_remote_code: bool = field(
        default_factory=lambda: _env_bool("INKUBALM_TRUST_REMOTE_CODE", True)
    )
    # Precision for InkubaLM: float32 (default), bfloat16, float16, or auto.
    # float16 on a GPU can overflow in this model and produce NaN scores.
    inkubalm_dtype: str = field(default_factory=lambda: _env("INKUBALM_DTYPE", "float32"))
    # Device for InkubaLM: auto = cuda if available, else cpu (Apple MPS only if set explicitly).
    inkubalm_device: str = field(default_factory=lambda: _env("INKUBALM_DEVICE", "auto"))

    # --- Shared ----------------------------------------------------------
    device: str = field(default_factory=lambda: _env("DEVICE", "auto"))
    preload_models: bool = field(default_factory=lambda: _env_bool("PRELOAD_MODELS", True))
    courses_file: Path = field(
        default_factory=lambda: Path(_env("COURSES_FILE", str(DATA_DIR / "courses.json")))
    )
    calendar_file: Path = field(
        default_factory=lambda: Path(_env("CALENDAR_FILE", str(DATA_DIR / "calendar.json")))
    )
    contacts_file: Path = field(
        default_factory=lambda: Path(_env("CONTACTS_FILE", str(DATA_DIR / "contacts.json")))
    )
    locations_file: Path = field(
        default_factory=lambda: Path(_env("LOCATIONS_FILE", str(DATA_DIR / "locations.json")))
    )
    action_lexicon_file: Path = field(
        default_factory=lambda: Path(_env("ACTION_LEXICON_FILE", str(DATA_DIR / "action_lexicon.json")))
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
