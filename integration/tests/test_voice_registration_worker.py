from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
import importlib.util
import time
from unittest.mock import patch

import numpy as np

from integration.voice_registration_models import VoiceRegistrationModels
from integration.microphone_manager import (
    MicrophoneQueueOverflowError,
    MicrophoneStartupTimeoutError,
)
from integration.voice_registration_worker import (
    _resample_waveform,
    capture_voice_embedding,
)


class FakeEmbedding:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def detach(self) -> "FakeEmbedding":
        return self

    def cpu(self) -> "FakeEmbedding":
        return self

    def numpy(self) -> np.ndarray:
        return self.value


class FakeDetector:
    def __init__(self, enrollment_error: Exception | None = None) -> None:
        self.enrollment_error = enrollment_error
        self.calls: list[np.ndarray] = []
        self.microphone_released_during_processing = False
        self.microphone: FakeMicrophone | None = None

    def enroll_array(self, waveform: np.ndarray) -> FakeEmbedding:
        self.calls.append(waveform.copy())
        if self.microphone is not None:
            self.microphone_released_during_processing = self.microphone.released
        if self.enrollment_error is not None:
            raise self.enrollment_error
        return FakeEmbedding(np.array([3.0, 4.0], dtype=np.float32))


class FakeMicrophone:
    def __init__(
        self,
        stop_event: threading.Event,
        blocks: list[np.ndarray] | None = None,
        *,
        open_error: Exception | None = None,
        read_error: Exception | None = None,
        stop_after_reads: int | None = None,
    ) -> None:
        self.stop_event = stop_event
        self.blocks = list(blocks or [])
        self.open_error = open_error
        self.read_error = read_error
        self.stop_after_reads = stop_after_reads
        self.read_count = 0
        self.sample_rate_hz = 16_000
        self.input_sample_rate_hz = 16_000
        self.channels = 1
        self.selected_device_name = "Mock microphone"
        self.selected_device_index = 1
        self.selected_device_info = {
            "name": self.selected_device_name,
            "max_input_channels": 1,
            "default_samplerate": 16_000,
        }
        self.callback_count = 0
        self.open_count = 0
        self.stop_count = 0
        self.release_count = 0
        self.released = False

    def open(self) -> "FakeMicrophone":
        self.open_count += 1
        if self.open_error is not None:
            raise self.open_error
        self.callback_count = 1
        return self

    def read(self, timeout: float | None = None) -> np.ndarray:
        del timeout
        if self.read_error is not None:
            raise self.read_error
        if not self.blocks:
            self.stop_event.set()
            raise TimeoutError("No more mock audio blocks.")
        self.callback_count += 1
        self.read_count += 1
        block = self.blocks.pop(0)
        if (
            self.stop_after_reads is not None
            and self.read_count >= self.stop_after_reads
        ) or not self.blocks:
            self.stop_event.set()
        return block

    def stop_capture(self) -> None:
        self.stop_count += 1

    def release(self) -> None:
        self.release_count += 1
        self.released = True


class VoiceRegistrationWorkerTests(unittest.TestCase):
    def make_blocks(self, seconds: int) -> list[np.ndarray]:
        return [
            np.ones(16_000, dtype=np.float32) * 0.02
            for _ in range(seconds)
        ]

    def run_capture(
        self,
        microphone: FakeMicrophone,
        detector: FakeDetector | None = None,
        duration_seconds: float = 12.0,
    ) -> tuple[int, list[tuple[str, dict]]]:
        detector = detector or FakeDetector()
        output: list[tuple[str, dict]] = []
        microphone_factory = lambda **_kwargs: microphone
        detector.microphone = microphone
        result = capture_voice_embedding(
            duration_seconds,
            microphone_factory=microphone_factory,
            detector_loader=lambda: detector,
            stop_event=microphone.stop_event,
            watch_stdin=False,
            emit=lambda kind, **payload: output.append((kind, payload)),
        )
        return result, output

    def test_microphone_opens_once_and_successful_stop_processes_before_release(self) -> None:
        stop_event = threading.Event()
        microphone = FakeMicrophone(stop_event, self.make_blocks(9))
        detector = FakeDetector()

        return_code, output = self.run_capture(microphone, detector)

        self.assertEqual(return_code, 0)
        self.assertEqual(microphone.open_count, 1)
        self.assertEqual(microphone.stop_count, 1)
        self.assertEqual(microphone.release_count, 1)
        self.assertFalse(detector.microphone_released_during_processing)
        self.assertEqual(detector.calls[0].shape, (9 * 16_000,))
        self.assertEqual(output[-1][1]["status"], "VOICE CAPTURE SUCCESSFUL")
        self.assertEqual(output[-1][1]["duration_seconds"], 9.0)
        self.assertEqual(output[-1][1]["embedding_dimension"], 2)
        stages = [
            payload["stage"]
            for kind, payload in output
            if kind == "stage"
        ]
        self.assertIn("AUDIO_WORKER_STARTED", stages)
        self.assertIn("AUDIO_MODELS_LOADING", stages)
        self.assertIn("AUDIO_MODELS_READY", stages)
        self.assertIn("RECORDING_STARTED", stages)
        self.assertIn("AUDIO_WORKER_READY", stages)

    def test_default_loader_uses_registration_models_without_monitoring_detector(self) -> None:
        stop_event = threading.Event()
        microphone = FakeMicrophone(stop_event, self.make_blocks(9))
        detector = FakeDetector()
        output: list[tuple[str, dict]] = []
        microphone_factory = lambda **_kwargs: microphone
        detector.microphone = microphone

        def detector_loader() -> FakeDetector:
            self.assertFalse(
                any(
                    thread.name == "voice-registration-control"
                    for thread in threading.enumerate()
                )
            )
            return detector

        with patch(
            "integration.voice_registration_models.load_cached_voice_registration_models",
            side_effect=lambda **_kwargs: detector_loader(),
        ) as loader, patch(
            "integration.voice_registration_worker._watch_stop",
            side_effect=lambda _stop_event: None,
        ):
            return_code = capture_voice_embedding(
                12.0,
                microphone_factory=microphone_factory,
                stop_event=stop_event,
                watch_stdin=True,
                emit=lambda kind, **payload: output.append((kind, payload)),
            )

        self.assertEqual(return_code, 0)
        loader.assert_called_once()
        self.assertEqual(output[-1][1]["status"], "VOICE CAPTURE SUCCESSFUL")

    def test_registration_models_reject_short_speech_and_normalize_embedding(self) -> None:
        speech_timestamps = lambda audio, _vad, **_kwargs: [
            {"start": 0, "end": len(audio)}
        ]
        models = VoiceRegistrationModels(
            vad_model=object(),
            speech_timestamps=speech_timestamps,
            audio_tensor=lambda audio: audio,
            encode_segment=lambda _segment: np.array([3.0, 4.0], dtype=np.float32),
        )

        with self.assertRaisesRegex(ValueError, "less than 8 seconds"):
            models.enroll_array(np.ones(7 * 16_000, dtype=np.float32))

        embedding = models.enroll_array(
            np.ones(9 * 16_000, dtype=np.float32) * 0.02
        )
        np.testing.assert_allclose(embedding, [0.6, 0.8], atol=1e-6)

    def test_silence_rejection_does_not_emit_success_or_embedding(self) -> None:
        stop_event = threading.Event()
        microphone = FakeMicrophone(stop_event, self.make_blocks(9))
        detector = FakeDetector(
            ValueError(
                "Voice enrollment contains less than 8 seconds of detected speech."
            )
        )

        return_code, output = self.run_capture(microphone, detector)

        self.assertEqual(return_code, 0)
        self.assertEqual(output[-1][1]["status"], "INSUFFICIENT SPEECH")
        self.assertFalse(any(kind == "complete" for kind, _payload in output))
        self.assertEqual(microphone.release_count, 1)

    def test_short_recording_is_rejected_without_embedding_generation(self) -> None:
        stop_event = threading.Event()
        microphone = FakeMicrophone(stop_event, self.make_blocks(4))
        detector = FakeDetector()

        return_code, output = self.run_capture(microphone, detector)

        self.assertEqual(return_code, 0)
        self.assertFalse(detector.calls)
        self.assertEqual(output[-1][1]["status"], "INSUFFICIENT SPEECH")
        self.assertEqual(microphone.release_count, 1)

    def test_stop_flushes_valid_blocks_already_queued(self) -> None:
        stop_event = threading.Event()
        microphone = FakeMicrophone(
            stop_event,
            self.make_blocks(10),
            stop_after_reads=1,
        )
        detector = FakeDetector()

        return_code, output = self.run_capture(microphone, detector)

        self.assertEqual(return_code, 0)
        self.assertEqual(detector.calls[0].shape, (10 * 16_000,))
        self.assertEqual(output[-1][1]["duration_seconds"], 10.0)
        self.assertEqual(microphone.stop_count, 1)
        self.assertEqual(microphone.release_count, 1)

    def test_microphone_open_failure_reports_actual_exception_and_releases(self) -> None:
        stop_event = threading.Event()
        microphone = FakeMicrophone(
            stop_event,
            open_error=OSError("device disconnected"),
        )

        return_code, output = self.run_capture(microphone)

        self.assertEqual(return_code, 1)
        self.assertEqual(output[-1][1]["status"], "MICROPHONE ERROR")
        self.assertIn("device disconnected", output[-1][1]["error"])
        self.assertIn("OSError", output[-1][1]["diagnostic"])
        self.assertEqual(microphone.open_count, 1)
        self.assertEqual(microphone.release_count, 0)

    def test_microphone_callback_timeout_has_a_distinct_failure_status(self) -> None:
        stop_event = threading.Event()
        microphone = FakeMicrophone(
            stop_event,
            open_error=MicrophoneStartupTimeoutError(
                "No valid audio callback arrived."
            ),
        )

        return_code, output = self.run_capture(microphone)

        self.assertEqual(return_code, 1)
        self.assertEqual(
            output[-1][1]["status"],
            "MICROPHONE STARTUP TIMEOUT",
        )
        self.assertIn("No valid audio callback", output[-1][1]["error"])

    def test_queue_overflow_is_reported_as_microphone_error(self) -> None:
        stop_event = threading.Event()
        microphone = FakeMicrophone(
            stop_event,
            read_error=MicrophoneQueueOverflowError(dropped_blocks=3),
        )

        return_code, output = self.run_capture(microphone)

        self.assertEqual(return_code, 1)
        self.assertEqual(output[-1][1]["status"], "MICROPHONE ERROR")
        self.assertIn("3", output[-1][1]["error"])
        self.assertEqual(microphone.release_count, 1)

    def test_invalid_audio_block_and_unexpected_read_failure_release_stream(self) -> None:
        cases = (
            (
                [np.zeros((4, 1), dtype=np.float64)],
                None,
                "invalid mono float32",
            ),
            (
                [],
                RuntimeError("callback failed"),
                "callback failed",
            ),
        )
        for blocks, read_error, expected in cases:
            with self.subTest(expected=expected):
                stop_event = threading.Event()
                microphone = FakeMicrophone(
                    stop_event,
                    blocks,
                    read_error=read_error,
                )
                return_code, output = self.run_capture(microphone)
                self.assertEqual(return_code, 1)
                self.assertEqual(output[-1][1]["status"], "MICROPHONE ERROR")
                self.assertIn(expected, output[-1][1]["error"])
                self.assertEqual(microphone.release_count, 1)

    def test_invalid_configuration_and_detector_failure_are_explicit(self) -> None:
        stop_event = threading.Event()
        microphone = FakeMicrophone(stop_event, self.make_blocks(9))
        microphone.sample_rate_hz = 8_000
        result, output = self.run_capture(microphone)
        self.assertEqual(result, 1)
        self.assertEqual(output[-1][1]["status"], "MICROPHONE ERROR")
        self.assertIn("mono at 16000 Hz", output[-1][1]["error"])

        failed_output: list[tuple[str, dict]] = []
        failed = capture_voice_embedding(
            microphone_factory=lambda: self.fail("must not open mic"),
            detector_loader=lambda: (_ for _ in ()).throw(
                ImportError("SpeechBrain missing")
            ),
            stop_event=threading.Event(),
            watch_stdin=False,
            emit=lambda kind, **payload: failed_output.append((kind, payload)),
        )
        self.assertEqual(failed, 1)
        self.assertEqual(failed_output[-1][1]["status"], "AUDIO WORKER ERROR")
        self.assertIn("SpeechBrain missing", failed_output[-1][1]["error"])

    def test_slow_model_loading_reports_startup_timeout_before_open(self) -> None:
        output: list[tuple[str, dict]] = []
        opened = False

        def microphone_factory(**_kwargs):
            nonlocal opened
            opened = True
            raise AssertionError("microphone must not open after startup timeout")

        def slow_detector_loader():
            time.sleep(0.02)
            return FakeDetector()

        result = capture_voice_embedding(
            microphone_factory=microphone_factory,
            detector_loader=slow_detector_loader,
            stop_event=threading.Event(),
            watch_stdin=False,
            startup_timeout_seconds=0.01,
            emit=lambda kind, **payload: output.append((kind, payload)),
        )

        self.assertEqual(result, 1)
        self.assertFalse(opened)
        self.assertEqual(
            output[-1][1]["status"],
            "MICROPHONE STARTUP TIMEOUT",
        )
        self.assertIn("AUDIO_MODELS_LOADING", output[-1][1]["diagnostic"])

    @unittest.skipUnless(
        importlib.util.find_spec("scipy"),
        "SciPy is installed in the isolated audio environment, not the main app environment.",
    )
    def test_native_rate_audio_is_resampled_to_pipeline_rate(self) -> None:
        stop_event = threading.Event()
        microphone = FakeMicrophone(
            stop_event,
            [np.ones(48_000, dtype=np.float32) * 0.02 for _ in range(9)],
        )
        microphone.input_sample_rate_hz = 48_000
        microphone.selected_device_info["default_samplerate"] = 48_000
        detector = FakeDetector()

        return_code, output = self.run_capture(microphone, detector)

        self.assertEqual(return_code, 0)
        self.assertEqual(detector.calls[0].shape, (9 * 16_000,))
        self.assertEqual(output[-1][1]["duration_seconds"], 9.0)

    def test_processing_exception_does_not_report_success(self) -> None:
        stop_event = threading.Event()
        microphone = FakeMicrophone(stop_event, self.make_blocks(9))
        detector = FakeDetector(RuntimeError("ECAPA inference failed"))

        return_code, output = self.run_capture(microphone, detector)

        self.assertEqual(return_code, 0)
        self.assertEqual(output[-1][1]["status"], "VOICE CAPTURE ERROR")
        self.assertIn("ECAPA inference failed", output[-1][1]["reason"])
        self.assertFalse(any(kind == "complete" for kind, _payload in output))
        self.assertEqual(microphone.release_count, 1)

    def test_retry_uses_a_fresh_single_stream(self) -> None:
        first_event = threading.Event()
        first = FakeMicrophone(
            first_event,
            open_error=OSError("temporary disconnect"),
        )
        failed_code, failed_output = self.run_capture(first)

        second_event = threading.Event()
        second = FakeMicrophone(second_event, self.make_blocks(9))
        success_code, success_output = self.run_capture(second)

        self.assertEqual(failed_code, 1)
        self.assertEqual(failed_output[-1][1]["status"], "MICROPHONE ERROR")
        self.assertEqual(success_code, 0)
        self.assertEqual(second.open_count, 1)
        self.assertEqual(second.release_count, 1)
        self.assertEqual(success_output[-1][1]["status"], "VOICE CAPTURE SUCCESSFUL")


if __name__ == "__main__":
    unittest.main()
