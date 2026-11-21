from __future__ import annotations

from collections.abc import Callable
from time import monotonic, perf_counter
from typing import Any, NoReturn

import numpy as np

from integration.config import (
    AUDIO_ANALYSIS_HOP_SAMPLES,
    AUDIO_ANALYSIS_WINDOW_SAMPLES,
    MICROPHONE_SAMPLE_RATE_HZ,
)
from integration.microphone_manager import (
    MicrophoneManager,
    MicrophoneQueueOverflowError,
)
from integration.module_adapters.audio import AudioAdapter


class AudioMonitorError(RuntimeError):
    """Base error for audio capture and window assembly failures."""


class InsufficientAudioError(TimeoutError):
    """A complete analysis window was not available before the read timed out."""


class AudioMonitor:
    """Assemble captured blocks into analysis windows and pass them to an adapter."""

    def __init__(
        self,
        microphone: MicrophoneManager,
        adapter: AudioAdapter,
        window_samples: int = AUDIO_ANALYSIS_WINDOW_SAMPLES,
        hop_samples: int | None = None,
    ) -> None:
        if (
            isinstance(window_samples, bool)
            or not isinstance(window_samples, int)
            or window_samples <= 0
        ):
            raise ValueError("Audio analysis window must contain samples.")
        if hop_samples is None:
            hop_samples = min(AUDIO_ANALYSIS_HOP_SAMPLES, window_samples)
        if (
            isinstance(hop_samples, bool)
            or not isinstance(hop_samples, int)
            or hop_samples <= 0
            or hop_samples > window_samples
        ):
            raise ValueError(
                "Audio analysis hop must be positive and no larger than the window."
            )
        if microphone.sample_rate_hz != MICROPHONE_SAMPLE_RATE_HZ:
            raise ValueError(
                f"Audio monitoring requires {MICROPHONE_SAMPLE_RATE_HZ} Hz input."
            )
        if microphone.channels != 1:
            raise ValueError("Audio monitoring requires mono microphone input.")

        self._microphone = microphone
        self._adapter = adapter
        self.window_samples = window_samples
        self.hop_samples = hop_samples
        self._opened = False
        self._pending_audio = np.empty(0, dtype=np.float32)
        self._samples_received = 0
        self._blocks_received = 0
        self._last_input_rms = 0.0
        self._next_window_start_sample = 0
        self._last_window_start_sample: int | None = None
        self._last_inference_seconds = 0.0
        self._last_window_completed_monotonic: float | None = None
        self._last_window_rms: float | None = None
        self._last_window_peak: float | None = None
        self._last_window_min: float | None = None
        self._last_window_max: float | None = None
        self._processing_callback: Callable[[], None] | None = None
        self._discontinuities = 0
        self._dropped_blocks = 0

    @property
    def pending_samples(self) -> int:
        return int(self._pending_audio.size)

    @property
    def samples_received(self) -> int:
        return self._samples_received

    @property
    def blocks_received(self) -> int:
        return self._blocks_received

    @property
    def last_input_rms(self) -> float:
        return self._last_input_rms

    @property
    def last_window_start_sample(self) -> int | None:
        return self._last_window_start_sample

    @property
    def last_inference_seconds(self) -> float:
        return self._last_inference_seconds

    @property
    def last_window_completed_monotonic(self) -> float | None:
        return self._last_window_completed_monotonic

    @property
    def last_window_rms(self) -> float | None:
        return self._last_window_rms

    @property
    def last_window_peak(self) -> float | None:
        return self._last_window_peak

    @property
    def last_window_min(self) -> float | None:
        return self._last_window_min

    @property
    def last_window_max(self) -> float | None:
        return self._last_window_max

    @property
    def discontinuities(self) -> int:
        return self._discontinuities

    @property
    def dropped_blocks(self) -> int:
        return self._dropped_blocks

    def set_processing_callback(
        self,
        callback: Callable[[], None] | None,
    ) -> None:
        self._processing_callback = callback

    def open(self) -> "AudioMonitor":
        if self._opened:
            raise RuntimeError("Audio monitor is already open.")
        self._adapter.reset()
        self._microphone.open()
        self._pending_audio = np.empty(0, dtype=np.float32)
        self._samples_received = 0
        self._blocks_received = 0
        self._last_input_rms = 0.0
        self._next_window_start_sample = 0
        self._last_window_start_sample = None
        self._last_inference_seconds = 0.0
        self._last_window_completed_monotonic = None
        self._last_window_rms = None
        self._last_window_peak = None
        self._last_window_min = None
        self._last_window_max = None
        self._discontinuities = 0
        self._dropped_blocks = 0
        self._opened = True
        return self

    def analyze_next(self, timeout: float | None = None) -> dict[str, Any]:
        if not self._opened:
            raise RuntimeError("Audio monitor is not open.")

        while self._pending_audio.size < self.window_samples:
            try:
                block = self._microphone.read(timeout=timeout)
            except MicrophoneQueueOverflowError as exc:
                self._pending_audio = np.empty(0, dtype=np.float32)
                self._adapter.reset()
                self._next_window_start_sample = self._samples_received
                self._discontinuities += 1
                self._dropped_blocks += exc.dropped_blocks
                continue
            except TimeoutError as exc:
                raise InsufficientAudioError(
                    "Timed out waiting for a complete audio analysis window "
                    f"({self._pending_audio.size}/{self.window_samples} samples)."
                ) from exc
            except Exception as exc:
                self._raise_after_cleanup(exc)

            try:
                validated_block = self._validate_block(block)
            except AudioMonitorError as exc:
                self._raise_after_cleanup(exc)
            self._blocks_received += 1
            self._samples_received += int(validated_block.size)
            float_block = validated_block.astype(np.float64, copy=False)
            self._last_input_rms = float(np.sqrt(np.mean(float_block**2)))
            self._pending_audio = np.concatenate(
                (self._pending_audio, validated_block)
            )

        audio_window = self._pending_audio[: self.window_samples].copy()
        self._pending_audio = self._pending_audio[self.hop_samples :].copy()
        self._last_window_start_sample = self._next_window_start_sample
        self._next_window_start_sample += self.hop_samples
        window64 = audio_window.astype(np.float64, copy=False)
        self._last_window_rms = float(np.sqrt(np.mean(window64**2)))
        self._last_window_peak = float(np.max(np.abs(window64)))
        self._last_window_min = float(np.min(audio_window))
        self._last_window_max = float(np.max(audio_window))

        try:
            if self._processing_callback is not None:
                self._processing_callback()
        except Exception as exc:
            self._raise_after_cleanup(exc)

        inference_started = perf_counter()
        try:
            result = self._adapter.analyze(
                audio_window,
                sample_rate_hz=MICROPHONE_SAMPLE_RATE_HZ,
            )
            self._last_window_completed_monotonic = monotonic()
            return result
        except Exception as exc:
            self._raise_after_cleanup(exc)
        finally:
            self._last_inference_seconds = perf_counter() - inference_started

    def close(self) -> None:
        try:
            self._microphone.release()
        finally:
            self._opened = False
            self._pending_audio = np.empty(0, dtype=np.float32)
            self._adapter.reset()

    def __enter__(self) -> "AudioMonitor":
        return self.open()

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> bool:
        self.close()
        return False

    def _raise_after_cleanup(self, error: Exception) -> NoReturn:
        try:
            self.close()
        except Exception as cleanup_error:
            raise AudioMonitorError(
                f"{error}; microphone cleanup failed: {cleanup_error}"
            ) from error
        raise error

    @staticmethod
    def _validate_block(block: Any) -> np.ndarray:
        if not isinstance(block, np.ndarray):
            raise AudioMonitorError("Microphone returned a non-array audio block.")
        if block.dtype != np.float32:
            raise AudioMonitorError(
                "Microphone returned audio that is not float32."
            )
        if block.ndim != 1 or block.size == 0:
            raise AudioMonitorError(
                "Microphone returned an empty or non-mono audio block."
            )
        if not np.all(np.isfinite(block)):
            raise AudioMonitorError(
                "Microphone returned audio containing non-finite samples."
            )
        return block
