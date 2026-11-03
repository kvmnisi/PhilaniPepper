"""Decode uploaded audio to 16 kHz mono float32 for Whisper.

The browser front end already converts recordings to 16 kHz mono WAV and
Pepper's ALAudioRecorder writes WAV, so the stdlib `wave` path covers normal
use. Other formats (webm/ogg/mp4) are converted with ffmpeg when it is on PATH.
"""

import io
import shutil
import subprocess
import wave

import numpy as np

TARGET_RATE = 16000


class AudioDecodeError(ValueError):
    pass


def _decode_wav(data):
    with wave.open(io.BytesIO(data), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if width == 1:
        samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        samples = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise AudioDecodeError("Unsupported WAV sample width: %d bytes" % width)

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate


def _decode_with_ffmpeg(data):
    if shutil.which("ffmpeg") is None:
        raise AudioDecodeError(
            "Audio is not WAV and ffmpeg is not installed. Send 16 kHz mono WAV or install ffmpeg."
        )
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-ac", "1", "-ar", str(TARGET_RATE), "-f", "s16le", "pipe:1"],
        input=data, capture_output=True, check=False,
    )
    if result.returncode != 0:
        raise AudioDecodeError("The recording could not be decoded (unsupported or damaged audio).")
    samples = np.frombuffer(result.stdout, dtype="<i2").astype(np.float32) / 32768.0
    return samples, TARGET_RATE


def resample(samples, rate, target=TARGET_RATE):
    """Linear-interpolation resample. Adequate for speech going into Whisper."""
    if rate == target or samples.size == 0:
        return samples.astype(np.float32)
    duration = samples.size / float(rate)
    target_len = int(round(duration * target))
    old_t = np.linspace(0.0, duration, num=samples.size, endpoint=False)
    new_t = np.linspace(0.0, duration, num=target_len, endpoint=False)
    return np.interp(new_t, old_t, samples).astype(np.float32)


def decode_audio(data):
    """Return (samples at 16 kHz, duration in seconds)."""
    if not data:
        raise AudioDecodeError("The audio upload was empty.")
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        try:
            samples, rate = _decode_wav(data)
        except wave.Error:
            samples, rate = _decode_with_ffmpeg(data)
    else:
        samples, rate = _decode_with_ffmpeg(data)
    samples = resample(samples, rate)
    if samples.size == 0:
        raise AudioDecodeError("The recording contained no audio samples.")
    return samples, samples.size / float(TARGET_RATE)
