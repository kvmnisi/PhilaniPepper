"""Branch B: isiZulu transcript -> InkubaLM-0.4B (+ LoRA adapter) -> intent.

Engines
  dev-rules  No model. isiZulu keyword rules. Plumbing only.
  inkubalm   lelapa/InkubaLM-0.4B with the LoRA adapter at INKUBALM_ADAPTER.
             Every canonical label is scored as a continuation of the shared
             prompt (branch_b/prompt.py); softmax over the label scores gives
             a probability, and the output is always a valid label.
"""

import json
import threading
import time
from pathlib import Path

from backend.branch_b.prompt import PROMPT_TEMPLATE
from backend.branch_b.scoring import score_labels
from backend.nlu import dev_rules
from backend.nlu.intents import intent_labels, normalise_label
from backend.nlu.slots import extract_slots


def _ms(start):
    return round((time.perf_counter() - start) * 1000.0, 1)


def check_adapter_compatible(adapter_dir):
    """Refuse an adapter trained with a different prompt or label set.

    The Colab notebook writes training_info.json next to the adapter; label
    scoring only means something if prompt and labels match this backend.
    """
    info_path = Path(adapter_dir) / "training_info.json"
    if not info_path.exists():
        raise RuntimeError("%s has no training_info.json; cannot confirm it matches this backend's prompt and labels."
                           % adapter_dir)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("prompt_template") != PROMPT_TEMPLATE:
        raise RuntimeError("Adapter was trained with a different prompt format: %r" % info.get("prompt_template"))
    if list(info.get("labels") or []) != intent_labels():
        raise RuntimeError("Adapter was trained on different intent labels: %s" % info.get("labels"))
    return info


DTYPES = ("float32", "bfloat16", "float16")


def dtype_name(setting, device):
    """Resolve INKUBALM_DTYPE. 'auto' keeps float32, except bfloat16 on GPUs that support it."""
    setting = (setting or "float32").lower()
    if setting in DTYPES:
        return setting
    if setting == "auto":
        return "bfloat16" if device == "cuda" else "float32"
    raise ValueError("INKUBALM_DTYPE must be one of %s or auto" % ", ".join(DTYPES))


def resolve_dtype(setting, device, torch):
    name = dtype_name(setting, device)
    if name == "bfloat16" and device == "cuda" and not torch.cuda.is_bf16_supported():
        name = "float32"
    return getattr(torch, name)


class BranchBUnusable(RuntimeError):
    """Branch B cannot produce valid scores; the message says why and what to do."""


def inkubalm_device(setting, torch):
    """auto -> cuda if available, else cpu. Apple MPS is used only when set explicitly,
    because InkubaLM's custom model code is not reliable on it."""
    setting = (setting or "auto").lower()
    if setting != "auto":
        return setting
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_base_model(auto_cls, model_id, trust, dtype):
    """Load InkubaLM, preferring the plain ('eager') attention implementation."""
    attempts = [dict(dtype=dtype, attn_implementation="eager"), dict(torch_dtype=dtype, attn_implementation="eager"),
                dict(dtype=dtype), dict(torch_dtype=dtype)]
    last = None
    for kwargs in attempts:
        try:
            return auto_cls.from_pretrained(model_id, trust_remote_code=trust, **kwargs)
        except (TypeError, ValueError) as error:   # argument not supported by this version / model
            last = error
    raise last


def explain_failure(bad_lora_tensors, adapted_ok, base_ok, device, dtype):
    """Turn the load-time checks into one actionable message (None = healthy)."""
    if bad_lora_tensors:
        return ("The Branch B adapter contains NaN/infinite weights (%d tensors, e.g. %s): it was damaged during "
                "training. Retrain it with the latest Colab notebook, which refuses to save a damaged adapter."
                % (len(bad_lora_tensors), bad_lora_tensors[0]))
    if adapted_ok:
        return None
    if base_ok:
        return ("The InkubaLM base model works, but adding the adapter produces NaN/infinite outputs. Retrain the "
                "adapter with the latest Colab notebook.")
    return ("InkubaLM itself produces NaN/infinite outputs on device=%s with dtype=%s. Try INKUBALM_DEVICE=cpu and "
            "INKUBALM_DTYPE=float32." % (device, dtype))


def check_model_health(model, tokenizer, device, dtype, torch, text="Sawubona"):
    """Raise BranchBUnusable with the reason if the adapted model cannot score."""
    from backend.branch_b.prompt import build_prompt

    bad = [name for name, param in model.named_parameters()
           if "lora_" in name and not bool(torch.isfinite(param).all())]

    def probe(m):
        ids = torch.tensor([tokenizer(build_prompt(text)).input_ids], device=device)
        with torch.inference_mode():
            return bool(torch.isfinite(m(input_ids=ids).logits.float()).all())

    adapted_ok = not bad and probe(model)
    base_ok = True
    if not adapted_ok and not bad:
        with model.disable_adapter():
            base_ok = probe(model)
    message = explain_failure(bad, adapted_ok, base_ok, device, dtype)
    if message:
        raise BranchBUnusable(message)


class InkubaLMClassifier:
    def __init__(self, settings):
        self.settings = settings
        self.model_id = settings.inkubalm_model
        self.adapter = settings.inkubalm_adapter
        self.loaded = False
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            if self.loaded:
                return
            if not self.adapter:
                raise RuntimeError("BRANCH_B_ENGINE=inkubalm needs INKUBALM_ADAPTER (the LoRA adapter path).")
            check_adapter_compatible(self.adapter)
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer


            trust = self.settings.inkubalm_trust_remote_code
            self.device = inkubalm_device(self.settings.inkubalm_device, torch)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=trust)
            dtype = resolve_dtype(self.settings.inkubalm_dtype, self.device, torch)
            base = load_base_model(AutoModelForCausalLM, self.model_id, trust, dtype)
            self.dtype_name = str(dtype).replace("torch.", "")
            self.model = PeftModel.from_pretrained(base, self.adapter).to(self.device).eval()
            self.pad_id = self.tokenizer.pad_token_id
            if self.pad_id is None:
                self.pad_id = self.tokenizer.eos_token_id
            check_model_health(self.model, self.tokenizer, self.device, self.dtype_name, torch)
            self._torch = torch
            self.loaded = True

    def classify(self, text):
        self.load()
        scores = score_labels(self.model, self.tokenizer, text, intent_labels(), self.device, self.pad_id, self._torch)
        best = max(scores, key=scores.get)
        return best, scores


class BranchB:
    branch = "B"

    def __init__(self, settings, engine):
        self.engine = engine
        self.classifier = InkubaLMClassifier(settings) if engine == "inkubalm" else None

    @property
    def loaded(self):
        return True if self.classifier is None else self.classifier.loaded

    def load(self):
        if self.classifier:
            self.classifier.load()

    def classify(self, transcript):
        start = time.perf_counter()
        slots = extract_slots(transcript)
        if self.classifier:
            raw, scores = self.classifier.classify(transcript)
            confidence = scores.get(raw)
        else:
            raw, scores = dev_rules.classify(transcript, "zu", slots)
            confidence = None
        return {
            "branch": self.branch,
            "engine": self.engine,
            "intent": normalise_label(raw),
            "slots": slots,
            "confidence": confidence,
            "scores": scores,
            "translation": None,
            "translation_note": None,
            "latency": {"translation_ms": None, "classification_ms": _ms(start)},
        }

    def describe(self):
        info = {"engine": self.engine, "loaded": self.loaded}
        if self.classifier:
            info.update(model=self.classifier.model_id, adapter=self.classifier.adapter,
                        device=getattr(self.classifier, "device", None),
                        dtype=getattr(self.classifier, "dtype_name", None) or self.classifier.settings.inkubalm_dtype)
        return info
