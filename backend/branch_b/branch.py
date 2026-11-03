"""Branch B: isiZulu transcript -> InkubaLM-0.4B (+ LoRA adapter) -> intent.

Engines
  dev-rules  No model. isiZulu keyword rules (unverified lexicon). Plumbing only.
  inkubalm   lelapa/InkubaLM-0.4B with the LoRA adapter at INKUBALM_ADAPTER.
             Every canonical label is scored as a continuation of the shared
             prompt (branch_b/prompt.py); softmax over the label scores gives
             a probability, and the output is always a valid label.
"""

import threading
import time

from backend.branch_b.scoring import score_labels
from backend.nlu import dev_rules
from backend.nlu.intents import intent_labels, normalise_label
from backend.nlu.slots import extract_slots


def _ms(start):
    return round((time.perf_counter() - start) * 1000.0, 1)


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
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            from backend.torch_utils import pick_device, pick_dtype

            trust = self.settings.inkubalm_trust_remote_code
            self.device = pick_device(self.settings.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=trust)
            base = AutoModelForCausalLM.from_pretrained(
                self.model_id, trust_remote_code=trust, torch_dtype=pick_dtype(self.device)
            )
            self.model = PeftModel.from_pretrained(base, self.adapter).to(self.device).eval()
            self.pad_id = self.tokenizer.pad_token_id
            if self.pad_id is None:
                self.pad_id = self.tokenizer.eos_token_id
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
            info.update(model=self.classifier.model_id, adapter=self.classifier.adapter)
        return info
