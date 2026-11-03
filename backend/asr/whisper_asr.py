"""Whisper ASR engine (fine-tuned isiZulu checkpoint)."""

import threading

from backend.audio import TARGET_RATE


class ASRUnavailable(RuntimeError):
    pass


class NoASR:
    """Used when no ASR model is configured. Audio requests fail clearly."""

    name = "none"
    loaded = False

    def load(self):
        pass

    def transcribe(self, samples):
        raise ASRUnavailable(
            "No ASR model is configured on the backend. Install requirements-ml.txt and set "
            "ASR_ENGINE=whisper, or use the research panel's text input to bypass ASR."
        )

    def describe(self):
        return {"engine": self.name, "loaded": False}


class WhisperASR:
    name = "whisper"

    def __init__(self, settings):
        self.settings = settings
        self.model_id = settings.whisper_model
        self.loaded = False
        self._lock = threading.Lock()
        self.device = None

    def load(self):
        with self._lock:
            if self.loaded:
                return
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            from backend.torch_utils import pick_device, pick_dtype

            self.device = pick_device(self.settings.device)
            self.dtype = pick_dtype(self.device)
            self.processor = WhisperProcessor.from_pretrained(self.model_id)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_id, torch_dtype=self.dtype)
            self.model.to(self.device).eval()
            self._torch = torch
            self.loaded = True

    def transcribe(self, samples):
        self.load()
        features = self.processor(samples, sampling_rate=TARGET_RATE, return_tensors="pt").input_features
        features = features.to(self.device, dtype=self.dtype)
        with self._torch.inference_mode():
            ids = self.model.generate(features, **self.generation_kwargs())
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def generation_kwargs(self):
        """Decoding settings passed to generate(); only non-default extras are sent."""
        s = self.settings
        kwargs = {"max_new_tokens": s.whisper_max_new_tokens, "num_beams": max(1, s.whisper_num_beams)}
        if s.whisper_no_repeat_ngram > 0:
            kwargs["no_repeat_ngram_size"] = s.whisper_no_repeat_ngram
        if s.whisper_repetition_penalty != 1.0:
            kwargs["repetition_penalty"] = s.whisper_repetition_penalty
        if s.whisper_language:
            kwargs.update(language=s.whisper_language, task="transcribe")
        return kwargs

    def decoding(self):
        beams = max(1, self.settings.whisper_num_beams)
        return "greedy" if beams == 1 else "beam%d" % beams

    def describe(self):
        return {"engine": self.name, "model": self.model_id, "loaded": self.loaded, "device": self.device,
                "decoding": self.decoding()}
