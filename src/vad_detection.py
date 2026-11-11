from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from silero_vad import load_silero_vad, get_speech_timestamps
import settings

SAMPLE_RATE = settings.SR
_model = None


def load_vad():
    global _model
    if _model is None:
        _model = load_silero_vad()
    return _model


def detect_speech(audio_path, threshold: float = settings.VAD_THRESHOLD):
    """Return [{'start': sec, 'end': sec}, ...] for a 16 kHz file."""
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, got {sr} Hz")
    wav = torch.from_numpy(np.ascontiguousarray(audio))
    return get_speech_timestamps(
        wav, load_vad(), sampling_rate=SAMPLE_RATE,
        threshold=threshold, return_seconds=True
    )


if __name__ == "__main__":
    print(detect_speech(settings.CANDIDATE_TEST))
