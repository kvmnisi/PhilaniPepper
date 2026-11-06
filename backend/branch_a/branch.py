"""Branch A: isiZulu transcript -> English translation -> English intent classifier.

Engines
  dev-rules  No models. Text is passed through untranslated and classified
             with English keywords, so isiZulu input ends up as OOS.
             Plumbing tests only (type English).
  nllb-rules NLLB translation for real + the English keyword stand-in
             classifier. Lets isiZulu speech work end to end before a
             classifier is trained. The classifier is still a stand-in, so
             these results are excluded from reports.
  hf         NLLB translation + a fine-tuned English sequence classifier
             (BRANCH_A_CLASSIFIER) whose labels are the canonical intents.
  zero-shot  NLLB translation + an NLI model scoring each intent's English
             hypothesis. An untrained baseline, reported as such.

The status report recommends a dedicated translator (NLLB-200) rather than
InkubaLM for translation, so the comparison measures the route through
English rather than InkubaLM's weakness at translation.
"""

import threading
import time

from backend.nlu import dev_rules
from backend.nlu.intents import intent_labels, load_intents, normalise_label
from backend.nlu.slots import extract_slots


def _ms(start):
    return round((time.perf_counter() - start) * 1000.0, 1)


class NLLBTranslator:
    def __init__(self, settings):
        self.settings = settings
        self.model_id = settings.translator_model
        self.loaded = False
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            if self.loaded:
                return
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            from backend.torch_utils import pick_device, pick_dtype

            self.device = pick_device(self.settings.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, src_lang=self.settings.translator_src_lang)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, torch_dtype=pick_dtype(self.device))
            self.model.to(self.device).eval()
            self._torch = torch
            self.loaded = True

    def translate(self, text):
        self.load()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        target = self.tokenizer.convert_tokens_to_ids(self.settings.translator_tgt_lang)
        with self._torch.inference_mode():
            ids = self.model.generate(**inputs, forced_bos_token_id=target, max_new_tokens=64, num_beams=4)
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()


class HFIntentClassifier:
    """Fine-tuned English sequence classifier with canonical id2label."""

    def __init__(self, settings):
        self.settings = settings
        self.model_id = settings.branch_a_classifier
        self.loaded = False
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            if self.loaded:
                return
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            from backend.torch_utils import pick_device

            self.device = pick_device(self.settings.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
            self.model.to(self.device).eval()
            labels = set(self.model.config.id2label.values())
            unknown = labels - set(intent_labels())
            if unknown:
                raise ValueError("Branch A classifier has non-canonical labels: %s" % sorted(unknown))
            self._torch = torch
            self.loaded = True

    def classify(self, text):
        self.load()
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        with self._torch.inference_mode():
            probs = self._torch.softmax(self.model(**inputs).logits[0], dim=-1).tolist()
        scores = {self.model.config.id2label[i]: p for i, p in enumerate(probs)}
        best = max(scores, key=scores.get)
        return best, scores


class ZeroShotIntentClassifier:
    def __init__(self, settings):
        self.settings = settings
        self.model_id = settings.branch_a_zero_shot_model
        self.loaded = False
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            if self.loaded:
                return
            from transformers import pipeline

            from backend.torch_utils import pick_device

            device = pick_device(self.settings.device)
            self.pipe = pipeline("zero-shot-classification", model=self.model_id, device=device)
            self.hypotheses = {item["en_hypothesis"]: item["label"] for item in load_intents()}
            self.loaded = True

    def classify(self, text):
        self.load()
        out = self.pipe(text, candidate_labels=list(self.hypotheses), hypothesis_template="{}")
        scores = {self.hypotheses[h]: s for h, s in zip(out["labels"], out["scores"])}
        best = max(scores, key=scores.get)
        return best, scores


class BranchA:
    branch = "A"

    def __init__(self, settings, engine):
        self.engine = engine
        self.translator = NLLBTranslator(settings) if engine in ("nllb-rules", "hf", "zero-shot") else None
        if engine == "hf":
            self.classifier = HFIntentClassifier(settings)
        elif engine == "zero-shot":
            self.classifier = ZeroShotIntentClassifier(settings)
        else:
            self.classifier = None

    @property
    def loaded(self):
        parts = [part for part in (self.translator, self.classifier) if part is not None]
        return all(part.loaded for part in parts)

    def load(self):
        if self.translator:
            self.translator.load()
        if self.classifier:
            self.classifier.load()

    def classify(self, transcript):
        start = time.perf_counter()
        if self.translator:
            translation = self.translator.translate(transcript) if transcript.strip() else ""
            translation_note = None
        else:
            translation = transcript
            translation_note = ("No translator loaded (dev-rules): text passed through untranslated, so isiZulu "
                                "becomes OOS. Install backend/requirements-ml.txt to translate with NLLB.")
        translation_ms = _ms(start)

        start = time.perf_counter()
        slots = extract_slots(transcript, translation)
        if self.classifier:
            raw, scores = self.classifier.classify(translation)
            confidence = scores.get(raw)
        else:
            raw, scores = dev_rules.classify(translation, "en", slots)
            confidence = None  # keyword rules do not produce probabilities
        classification_ms = _ms(start)

        return {
            "branch": self.branch,
            "engine": self.engine,
            "intent": normalise_label(raw),
            "slots": slots,
            "confidence": confidence,
            "scores": scores,
            "translation": translation,
            "translation_note": translation_note,
            "latency": {"translation_ms": translation_ms, "classification_ms": classification_ms},
        }

    def describe(self):
        info = {"engine": self.engine, "loaded": self.loaded}
        if self.translator:
            info["translator"] = self.translator.model_id
        if self.classifier:
            info["classifier"] = self.classifier.model_id
        if self.engine == "nllb-rules":
            info["note"] = ("No fine-tuned English classifier found: put it in ./EnglishIntentClassifier, "
                            "or set CLASSIFIER_REPO (or BRANCH_A_CLASSIFIER) to its Hugging Face repo.")
        return info
