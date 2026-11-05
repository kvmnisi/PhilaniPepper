"""Branch B: isiZulu transcript -> InkubaLM-0.4B -> intent.

Engines
  dev-rules  No model. isiZulu keyword rules. Plumbing only.
  inkubalm   lelapa/InkubaLM-0.4B.
             With a LoRA adapter (./InkubaLMAdapter or INKUBALM_ADAPTER, trained
             by the Colab notebook): every canonical label is scored as the
             continuation of the fine-tune prompt (branch_b/prompt.py).
             Without one (untrained mode): few-shot label scoring with the
             labelled requests in data/branch_b_examples.json.
             Either way the output is always a valid label, with a probability.
"""

import json
from pathlib import Path

import threading
import time

from backend.branch_b.prompt import FINETUNE_PROMPT, load_examples, ordered_examples
from backend.branch_b.scoring import FineTunedScorer, InvalidScores, LabelScorer
from backend.nlu import dev_rules
from backend.nlu.intents import intent_labels, normalise_label
from backend.nlu.slots import extract_slots


def _ms(start):
    return round((time.perf_counter() - start) * 1000.0, 1)


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


def explain_failure(device, dtype):
    return ("InkubaLM produces NaN/infinite outputs on device=%s with dtype=%s. Use INKUBALM_DEVICE=cpu and "
            "INKUBALM_DTYPE=float32. If that does not help, install transformers 4.44.2 "
            "(pip install \"transformers==4.44.2\"): InkubaLM runs its own model code, written for that "
            "generation of transformers." % (device, dtype))


def check_adapter(adapter_dir, labels):
    """Refuse an adapter trained with another prompt or label set; return its training_info."""
    info_path = Path(adapter_dir) / "training_info.json"
    if not info_path.exists():
        raise BranchBUnusable("%s has no training_info.json. Use the adapter downloaded by the final cell "
                              "of the Colab notebook (InkubaLMAdapter.zip)." % adapter_dir)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("prompt") != FINETUNE_PROMPT or list(info.get("labels") or []) != list(labels):
        raise BranchBUnusable("The adapter in %s was trained with a different prompt or intent list (an older "
                              "notebook). Replace it with InkubaLMAdapter.zip from the final cell of the Colab "
                              "notebook, or set INKUBALM_ADAPTER=none for the untrained mode." % adapter_dir)
    return info


class InkubaLMClassifier:
    def __init__(self, settings):
        self.settings = settings
        self.model_id = settings.inkubalm_model
        self.adapter = settings.inkubalm_adapter
        self.adapter_info = None
        self.loaded = False
        self.examples = []
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            if self.loaded:
                return
            labels = intent_labels()
            if self.adapter:
                self.adapter_info = check_adapter(self.adapter, labels)
            else:
                data = load_examples(str(self.settings.branch_b_examples_file))
                self.examples = ordered_examples(data, labels)

            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            trust = self.settings.inkubalm_trust_remote_code
            self.device = inkubalm_device(self.settings.inkubalm_device, torch)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=trust)
            dtype = resolve_dtype(self.settings.inkubalm_dtype, self.device, torch)
            self.dtype_name = str(dtype).replace("torch.", "")
            base = load_base_model(AutoModelForCausalLM, self.model_id, trust, dtype)
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
            if self.adapter:
                try:   # newer peft checks torch.distributed.tensor, which some torch builds do not load by default
                    import torch.distributed.tensor  # noqa: F401
                except Exception:
                    pass
                from peft import PeftModel

                self.model = PeftModel.from_pretrained(base, self.adapter).to(self.device).eval()
                bad = [n for n, p in self.model.named_parameters() if "lora_" in n and not bool(torch.isfinite(p).all())]
                if bad:
                    raise BranchBUnusable("The adapter in %s contains NaN/infinite weights (%d tensors). Retrain it "
                                          "with the Colab notebook." % (self.adapter, len(bad)))
                self.scorer = FineTunedScorer(self.model, self.tokenizer, labels, self.device, pad_id,
                                              self.tokenizer.eos_token_id, torch)
            else:
                self.model = base.to(self.device).eval()
                self.scorer = LabelScorer(
                    self.model, self.tokenizer, labels, data["label_words"], self.examples, self.device, pad_id, torch,
                    calibrate=self.settings.inkubalm_calibrate,
                    max_tokens=getattr(self.model.config, "max_position_embeddings", None),
                )
            try:   # health check; also computes the calibration baseline
                self.scorer.score("Sawubona")
            except InvalidScores:
                raise BranchBUnusable(explain_failure(self.device, self.dtype_name))
            self.loaded = True

    def classify(self, text):
        self.load()
        scores = self.scorer.score(text)
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
            c = self.classifier
            if c.adapter:
                info.update(model=c.model_id, method="LoRA fine-tuned", adapter=c.adapter)
            else:
                info.update(model=c.model_id, method="few-shot label scoring (untrained)",
                            examples=len(c.examples) or None, calibrated=c.settings.inkubalm_calibrate)
            info.update(device=getattr(c, "device", None),
                        dtype=getattr(c, "dtype_name", None) or c.settings.inkubalm_dtype)
        return info
