from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


SAMPLE_RATE_HZ = 16_000
MIN_SPEECH_SECONDS = 8.0
VAD_THRESHOLD = 0.35
ENROLLMENT_CHUNK_SAMPLES = 3 * SAMPLE_RATE_HZ
ENROLLMENT_HOP_SAMPLES = ENROLLMENT_CHUNK_SAMPLES // 2
TARGET_RMS = 0.05
MAX_PEAK = 0.95


class VoiceRegistrationModels:
    def __init__(
        self,
        vad_model: Any,
        speech_timestamps: Callable[..., list[dict[str, int]]],
        audio_tensor: Callable[[np.ndarray], Any],
        encode_segment: Callable[[np.ndarray], Any],
    ) -> None:
        self._vad_model = vad_model
        self._speech_timestamps = speech_timestamps
        self._audio_tensor = audio_tensor
        self._encode_segment = encode_segment

    def enroll_array(self, waveform: np.ndarray) -> np.ndarray:
        audio = np.asarray(waveform, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("Voice enrollment requires a one-dimensional mono waveform.")
        if audio.size == 0 or not np.all(np.isfinite(audio)):
            raise ValueError("Voice enrollment waveform is empty or invalid.")

        timestamps = self._speech_timestamps(
            self._audio_tensor(audio.copy()),
            self._vad_model,
            sampling_rate=SAMPLE_RATE_HZ,
            threshold=VAD_THRESHOLD,
        )
        speech_parts = [
            audio[int(item["start"]):int(item["end"])]
            for item in timestamps
            if int(item["end"]) > int(item["start"])
        ]
        if not speech_parts:
            raise ValueError(
                "Voice enrollment contains less than 8 seconds of detected speech."
            )

        speech = np.concatenate(speech_parts).astype(np.float32, copy=False)
        if speech.size < int(MIN_SPEECH_SECONDS * SAMPLE_RATE_HZ):
            raise ValueError(
                "Voice enrollment contains less than 8 seconds of detected speech."
            )

        segments = [
            speech[start:start + ENROLLMENT_CHUNK_SAMPLES]
            for start in range(
                0,
                speech.size - ENROLLMENT_CHUNK_SAMPLES + 1,
                ENROLLMENT_HOP_SAMPLES,
            )
        ]
        if not segments:
            segments = [speech[:ENROLLMENT_CHUNK_SAMPLES]]

        embeddings = []
        for segment in segments:
            embedding = np.asarray(
                self._encode_segment(self._safe_normalize(segment)),
                dtype=np.float32,
            ).reshape(-1)
            norm = float(np.linalg.norm(embedding))
            if (
                embedding.size == 0
                or not np.all(np.isfinite(embedding))
                or not np.isfinite(norm)
                or norm == 0.0
            ):
                raise ValueError("ECAPA returned an empty or invalid embedding.")
            embeddings.append(embedding / norm)

        reference = np.mean(np.stack(embeddings), axis=0)
        norm = float(np.linalg.norm(reference))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("ECAPA returned an empty or invalid embedding.")
        return (reference / norm).astype(np.float32, copy=False)

    @staticmethod
    def _safe_normalize(waveform: np.ndarray) -> np.ndarray:
        audio = np.asarray(waveform, dtype=np.float32)
        if audio.size == 0:
            return audio
        rms = float(np.sqrt(np.mean(audio ** 2)) + 1e-9)
        scaled = audio * (TARGET_RMS / rms)
        peak = float(np.max(np.abs(scaled)))
        if peak > MAX_PEAK:
            scaled *= MAX_PEAK / peak
        return scaled.astype(np.float32, copy=False)


def load_cached_voice_registration_models(
    *,
    stage_callback: Callable[[str], None] | None = None,
) -> VoiceRegistrationModels:
    def report(stage: str) -> None:
        if stage_callback is not None:
            stage_callback(stage)

    report("AUDIO_MODEL_CACHE_CHECKING")
    from voice_vyor_Ai.ecapa_model_loader import (
        ECAPA_MODEL_FILES,
        ECAPA_MODEL_ID,
        ECAPA_MODEL_REVISION,
        load_ecapa_classifier,
    )

    hf_constants = importlib.import_module("huggingface_hub.constants")
    snapshot = (
        Path(hf_constants.HF_HUB_CACHE)
        / f"models--{ECAPA_MODEL_ID.replace('/', '--')}"
        / "snapshots"
        / ECAPA_MODEL_REVISION
    )
    missing = [name for name in ECAPA_MODEL_FILES if not (snapshot / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Cached ECAPA snapshot is incomplete; no model download was attempted. "
            f"Missing: {', '.join(missing)}"
        )

    report("AUDIO_VAD_LOADING")
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    vad_model = load_silero_vad()

    report("AUDIO_ECAPA_IMPORTING")
    from speechbrain.inference.speaker import EncoderClassifier

    report("AUDIO_ECAPA_WEIGHTS_LOADING")
    classifier = load_ecapa_classifier(
        EncoderClassifier,
        lambda **_kwargs: str(snapshot),
    )

    def encode_segment(segment: np.ndarray) -> np.ndarray:
        audio_tensor = torch.from_numpy(segment).float().unsqueeze(0)
        with torch.no_grad():
            embedding = classifier.encode_batch(audio_tensor).squeeze()
        if hasattr(embedding, "detach"):
            embedding = embedding.detach()
        if hasattr(embedding, "cpu"):
            embedding = embedding.cpu()
        if hasattr(embedding, "numpy"):
            embedding = embedding.numpy()
        return np.asarray(embedding, dtype=np.float32).reshape(-1)

    return VoiceRegistrationModels(
        vad_model=vad_model,
        speech_timestamps=get_speech_timestamps,
        audio_tensor=torch.from_numpy,
        encode_segment=encode_segment,
    )
