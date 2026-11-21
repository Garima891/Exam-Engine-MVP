from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

import numpy as np


SAMPLE_RATE_HZ = 16_000
MIN_PHONE_WINDOW_SAMPLES = SAMPLE_RATE_HZ


def _load_detector_module() -> Any:
    return importlib.import_module("voice_vyor_Ai.detector")


class AudioAdapter:
    """Analyze externally captured mono audio without owning audio hardware."""

    def __init__(
        self,
        *,
        speaker_reference_embedding: Any | None = None,
        detector_loader: Callable[[], Any] | None = None,
    ) -> None:
        self._speaker_reference = (
            None
            if speaker_reference_embedding is None
            else self._normalize_embedding(speaker_reference_embedding)
        )
        self._detector_loader = detector_loader or _load_detector_module
        self._detector: Any | None = None
        self._speech_buffer = np.empty(0, dtype=np.float32)

    def reset(self) -> None:
        """Clear speech accumulated from previous overlapping windows."""
        self._speech_buffer = np.empty(0, dtype=np.float32)

    @property
    def speaker_reference_configured(self) -> bool:
        return self._speaker_reference is not None

    def initialize(self) -> None:
        """Load the detector without opening audio hardware or processing audio."""
        self._get_detector()

    def analyze(
        self,
        audio_window: np.ndarray,
        sample_rate_hz: int = SAMPLE_RATE_HZ,
    ) -> dict[str, Any]:
        """Return independent VAD, speaker-similarity, and phone-sound results."""
        self._validate_audio(audio_window, sample_rate_hz)
        detector = self._get_detector()

        speech, self._speech_buffer, speech_duration = (
            detector.extract_evaluable_speech(
                audio_window,
                self._speech_buffer,
            )
        )
        self._speech_buffer = np.asarray(self._speech_buffer, dtype=np.float32)
        speech_duration = float(speech_duration)
        window_duration = len(audio_window) / SAMPLE_RATE_HZ
        vad_result = {
            "speech_detected": speech_duration > 0.0,
            "speech_duration_seconds": speech_duration,
            "speech_ratio": speech_duration / window_duration,
        }

        speaker_result = self._speaker_result(
            detector,
            speech,
            speech_duration,
        )
        phone_result = self._phone_result(detector, audio_window)

        return {
            "module": "audio_detection",
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "window_samples": len(audio_window),
            "vad": vad_result,
            "speaker": speaker_result,
            "phone_sound": phone_result,
        }

    def _get_detector(self) -> Any:
        if self._detector is None:
            detector = self._detector_loader()
            if detector is None:
                raise RuntimeError("Audio detector loader returned no detector.")
            self._detector = detector
        return self._detector

    def _speaker_result(
        self,
        detector: Any,
        speech: Any,
        speech_duration: float,
    ) -> dict[str, Any]:
        evaluated_speech_duration = (
            0.0 if speech is None else len(speech) / SAMPLE_RATE_HZ
        )
        result: dict[str, Any] = {
            "status": "not_evaluated",
            "reference_configured": self._speaker_reference is not None,
            "speech_duration_seconds": speech_duration,
            "evaluated_speech_duration_seconds": evaluated_speech_duration,
            "similarity": None,
            "comparison": None,
        }
        if speech_duration <= 0.0:
            result["status"] = "no_speech"
            return result
        if speech is None:
            result["status"] = "insufficient_speech"
            return result
        if self._speaker_reference is None:
            result["status"] = "reference_not_configured"
            return result

        normalized_speech = detector.safe_normalize(
            np.asarray(speech, dtype=np.float32)
        )
        embedding = self._normalize_embedding(detector.embed(normalized_speech))
        if embedding.shape != self._speaker_reference.shape:
            raise ValueError(
                "Speaker reference and extracted embedding dimensions differ."
            )

        similarity = float(np.dot(self._speaker_reference, embedding))
        low_threshold = float(detector.SIM_LOW)
        high_threshold = float(detector.SIM_HIGH)
        if similarity < low_threshold:
            comparison = "low_similarity"
        elif similarity > high_threshold:
            comparison = "high_similarity"
        else:
            comparison = "ambiguous_similarity"

        result.update(
            {
                "status": "compared",
                "similarity": similarity,
                "comparison": comparison,
                "thresholds": {
                    "low_similarity": low_threshold,
                    "high_similarity": high_threshold,
                },
            }
        )
        return result

    @staticmethod
    def _phone_result(detector: Any, audio_window: np.ndarray) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": "insufficient_audio",
            "score": None,
            "detected": None,
            "high_confidence": None,
            "threshold": float(detector.PHONE_TH),
            "high_confidence_threshold": float(detector.PHONE_HIGH_CONF),
        }
        if len(audio_window) < MIN_PHONE_WINDOW_SAMPLES:
            return result

        score = float(detector.phone_score(audio_window))
        if not np.isfinite(score):
            raise ValueError("Phone-sound detector returned a non-finite score.")

        result.update(
            {
                "status": "scored",
                "score": score,
                "detected": score >= detector.PHONE_TH,
                "high_confidence": score >= detector.PHONE_HIGH_CONF,
            }
        )
        return result

    @staticmethod
    def _normalize_embedding(embedding: Any) -> np.ndarray:
        if hasattr(embedding, "detach"):
            embedding = embedding.detach()
        if hasattr(embedding, "cpu"):
            embedding = embedding.cpu()
        if hasattr(embedding, "numpy"):
            embedding = embedding.numpy()

        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size == 0 or not np.all(np.isfinite(vector)):
            raise ValueError("Speaker embedding must be non-empty and finite.")
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError("Speaker embedding must not be a zero vector.")
        return (vector / norm).astype(np.float32, copy=False)

    @staticmethod
    def _validate_audio(audio_window: Any, sample_rate_hz: int) -> None:
        if sample_rate_hz != SAMPLE_RATE_HZ:
            raise ValueError(
                f"Expected audio sampled at {SAMPLE_RATE_HZ} Hz."
            )
        if not isinstance(audio_window, np.ndarray):
            raise ValueError("Expected audio as a NumPy float32 array.")
        if audio_window.dtype != np.float32:
            raise ValueError("Expected mono audio with dtype float32.")
        if audio_window.ndim != 1:
            raise ValueError("Expected a one-dimensional mono audio window.")
        if audio_window.size == 0:
            raise ValueError("Audio window must not be empty.")
        if not np.all(np.isfinite(audio_window)):
            raise ValueError("Audio window must contain only finite samples.")
