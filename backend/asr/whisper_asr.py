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
        kwargs = {
            "max_new_tokens": self.settings.whisper_max_new_tokens,
            "no_repeat_ngram_size": self.settings.whisper_no_repeat_ngram,
            "repetition_penalty": self.settings.whisper_repetition_penalty,
        }
        if self.settings.whisper_language:
            kwargs.update(language=self.settings.whisper_language, task="transcribe")
        with self._torch.inference_mode():
            ids = self.model.generate(features, **kwargs)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def describe(self):
        return {"engine": self.name, "model": self.model_id, "loaded": self.loaded, "device": self.device}
