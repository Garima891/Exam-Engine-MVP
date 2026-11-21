import unittest

import numpy as np

from integration.audio_monitor import (
    AudioMonitor,
    AudioMonitorError,
    InsufficientAudioError,
)
from integration.microphone_manager import (
    MicrophoneManager,
    MicrophoneQueueOverflowError,
    MicrophoneStreamError,
)


class FakeInputStream:
    def __init__(self, callbacks=(), **kwargs):
        self.kwargs = kwargs
        self.callbacks = list(callbacks)
        self.active = False
        self.stopped = False
        self.closed = False

    def start(self):
        self.active = True
        for audio, status in self.callbacks:
            self.kwargs["callback"](audio, len(audio), None, status)

    def stop(self):
        self.active = False
        self.stopped = True

    def close(self):
        self.closed = True


class FakeMicrophone:
    sample_rate_hz = 16_000
    channels = 1

    def __init__(self, reads):
        self.reads = list(reads)
        self.opened = False
        self.release_count = 0

    def open(self):
        self.opened = True

    def read(self, timeout=None):
        del timeout
        value = self.reads.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def release(self):
        self.opened = False
        self.release_count += 1


class FakeAudioAdapter:
    def __init__(self, error=None):
        self.error = error
        self.calls = []
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def analyze(self, audio_window, sample_rate_hz):
        self.calls.append((audio_window.copy(), sample_rate_hz))
        if self.error is not None:
            raise self.error
        return {"module": "audio_detection", "window_samples": len(audio_window)}


class AudioMonitorTests(unittest.TestCase):
    def test_assembles_blocks_and_passes_mono_float32_at_16khz(self):
        first = np.array([0.1, 0.2, 0.3], dtype=np.float32).reshape(-1, 1)
        second = np.array([0.4, 0.5], dtype=np.float32).reshape(-1, 1)
        stream = self._stream_for((first, None), (second, None))
        microphone = self._microphone_for(stream)
        adapter = FakeAudioAdapter()
        monitor = AudioMonitor(microphone, adapter, window_samples=5)

        with monitor:
            result = monitor.analyze_next(timeout=0.1)

        self.assertEqual(result["window_samples"], 5)
        self.assertEqual(len(adapter.calls), 1)
        received, sample_rate_hz = adapter.calls[0]
        self.assertEqual(received.dtype, np.float32)
        self.assertEqual(received.shape, (5,))
        self.assertEqual(sample_rate_hz, 16_000)
        np.testing.assert_array_equal(
            received,
            np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
        )
        self.assertTrue(stream.stopped)
        self.assertTrue(stream.closed)

    def test_carries_extra_samples_forward_to_the_next_window(self):
        microphone = FakeMicrophone(
            [
                np.arange(6, dtype=np.float32),
                np.arange(6, 10, dtype=np.float32),
            ]
        )
        adapter = FakeAudioAdapter()
        monitor = AudioMonitor(microphone, adapter, window_samples=4)

        with monitor:
            monitor.analyze_next()
            monitor.analyze_next()

        np.testing.assert_array_equal(adapter.calls[0][0], [0, 1, 2, 3])
        np.testing.assert_array_equal(adapter.calls[1][0], [4, 5, 6, 7])

    def test_uses_overlapping_windows_at_the_configured_hop(self):
        microphone = FakeMicrophone(
            [
                np.arange(6, dtype=np.float32),
                np.arange(6, 8, dtype=np.float32),
            ]
        )
        adapter = FakeAudioAdapter()
        monitor = AudioMonitor(
            microphone,
            adapter,
            window_samples=6,
            hop_samples=2,
        )

        with monitor:
            monitor.analyze_next()
            monitor.analyze_next()

        np.testing.assert_array_equal(adapter.calls[0][0], [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(adapter.calls[1][0], [2, 3, 4, 5, 6, 7])
        self.assertEqual(adapter.reset_count, 2)

    def test_exposes_capture_level_and_window_telemetry(self):
        microphone = FakeMicrophone(
            [
                np.array([0.1, 0.2], dtype=np.float32),
                np.array([0.3, 0.4], dtype=np.float32),
            ]
        )
        monitor = AudioMonitor(
            microphone,
            FakeAudioAdapter(),
            window_samples=4,
            hop_samples=2,
        )

        with monitor:
            monitor.analyze_next()
            self.assertEqual(monitor.blocks_received, 2)
            self.assertEqual(monitor.samples_received, 4)
            self.assertEqual(monitor.pending_samples, 2)
            self.assertAlmostEqual(
                monitor.last_input_rms,
                np.sqrt((0.3**2 + 0.4**2) / 2),
            )
            self.assertEqual(monitor.last_window_start_sample, 0)
            self.assertAlmostEqual(
                monitor.last_window_rms,
                np.sqrt((0.1**2 + 0.2**2 + 0.3**2 + 0.4**2) / 4),
            )
            self.assertAlmostEqual(monitor.last_window_peak, 0.4)
            self.assertAlmostEqual(monitor.last_window_min, 0.1)
            self.assertAlmostEqual(monitor.last_window_max, 0.4)
            self.assertIsNotNone(monitor.last_window_completed_monotonic)
            self.assertGreaterEqual(monitor.last_inference_seconds, 0.0)

            microphone.reads.append(
                np.array([0.5, 0.6], dtype=np.float32)
            )
            monitor.analyze_next()
            self.assertEqual(monitor.last_window_start_sample, 2)

    def test_processing_callback_runs_only_for_complete_windows(self):
        microphone = FakeMicrophone(
            [
                np.array([0.1, 0.2], dtype=np.float32),
                TimeoutError("no block yet"),
                np.array([0.3, 0.4], dtype=np.float32),
            ]
        )
        adapter = FakeAudioAdapter()
        monitor = AudioMonitor(microphone, adapter, window_samples=4)
        processing_events = []
        monitor.set_processing_callback(
            lambda: processing_events.append("processing")
        )

        with monitor:
            with self.assertRaises(InsufficientAudioError):
                monitor.analyze_next(timeout=0)
            self.assertEqual(processing_events, [])
            monitor.analyze_next(timeout=0)

        self.assertEqual(processing_events, ["processing"])

    def test_default_window_and_hop_match_standalone_detector(self):
        monitor = AudioMonitor(FakeMicrophone([]), FakeAudioAdapter())

        self.assertEqual(monitor.window_samples, 48_000)
        self.assertEqual(monitor.hop_samples, 8_000)

    def test_rejects_invalid_hop(self):
        with self.assertRaisesRegex(ValueError, "hop"):
            AudioMonitor(
                FakeMicrophone([]),
                FakeAudioAdapter(),
                window_samples=4,
                hop_samples=5,
            )

    def test_timeout_does_not_infer_partial_window_and_retains_samples(self):
        microphone = FakeMicrophone(
            [
                np.array([0.1, 0.2], dtype=np.float32),
                TimeoutError("no block yet"),
                np.array([0.3, 0.4], dtype=np.float32),
            ]
        )
        adapter = FakeAudioAdapter()
        monitor = AudioMonitor(microphone, adapter, window_samples=4)

        with monitor:
            with self.assertRaisesRegex(InsufficientAudioError, "2/4 samples"):
                monitor.analyze_next(timeout=0)
            self.assertEqual(adapter.calls, [])
            monitor.analyze_next(timeout=0)

        np.testing.assert_array_equal(
            adapter.calls[0][0],
            np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        )

    def test_invalid_blocks_release_microphone_and_do_not_reach_adapter(self):
        invalid_blocks = (
            np.array([0.1], dtype=np.float64),
            np.zeros((2, 1), dtype=np.float32),
            np.array([np.nan], dtype=np.float32),
        )

        for block in invalid_blocks:
            with self.subTest(block=block):
                microphone = FakeMicrophone([block])
                adapter = FakeAudioAdapter()
                monitor = AudioMonitor(microphone, adapter, window_samples=1)
                monitor.open()

                with self.assertRaises(AudioMonitorError):
                    monitor.analyze_next()

                self.assertEqual(microphone.release_count, 1)
                self.assertEqual(adapter.calls, [])
                self.assertFalse(monitor._opened)

    def test_stream_error_propagates_and_closes_stream(self):
        stream = self._stream_for(
            (np.ones((2, 1), dtype=np.float32), "device error")
        )
        microphone = self._microphone_for(stream)
        adapter = FakeAudioAdapter()
        monitor = AudioMonitor(microphone, adapter, window_samples=8)

        with self.assertRaises(MicrophoneStreamError):
            monitor.open()

        self.assertTrue(stream.closed)
        self.assertEqual(adapter.calls, [])

    def test_overflow_discards_partial_window_before_resuming(self):
        microphone = FakeMicrophone(
            [
                np.array([1.0, 1.0], dtype=np.float32),
                MicrophoneQueueOverflowError(dropped_blocks=3),
                np.array([3.0, 3.0], dtype=np.float32),
                np.array([4.0, 4.0], dtype=np.float32),
            ]
        )
        adapter = FakeAudioAdapter()
        monitor = AudioMonitor(microphone, adapter, window_samples=4)

        with monitor:
            monitor.analyze_next(timeout=0.1)

        np.testing.assert_array_equal(
            adapter.calls[0][0],
            np.array([3.0, 3.0, 4.0, 4.0], dtype=np.float32),
        )
        self.assertEqual(monitor.discontinuities, 1)
        self.assertEqual(monitor.dropped_blocks, 3)
        self.assertEqual(adapter.reset_count, 3)

    def test_adapter_error_releases_microphone_and_propagates(self):
        block = np.ones((4, 1), dtype=np.float32)
        stream = self._stream_for((block, None))
        microphone = self._microphone_for(stream)
        adapter = FakeAudioAdapter(error=ValueError("inference failed"))
        monitor = AudioMonitor(microphone, adapter, window_samples=4)
        monitor.open()

        with self.assertRaisesRegex(ValueError, "inference failed"):
            monitor.analyze_next()

        self.assertTrue(stream.closed)
        self.assertFalse(monitor._opened)

    def test_requires_open_monitor_and_context_cleanup_is_idempotent(self):
        stream = self._stream_for(
            (np.ones((2, 1), dtype=np.float32), None)
        )
        microphone = self._microphone_for(stream)
        monitor = AudioMonitor(microphone, FakeAudioAdapter(), window_samples=1)

        with self.assertRaisesRegex(RuntimeError, "not open"):
            monitor.analyze_next()

        with monitor:
            pass

        monitor.close()
        self.assertTrue(stream.closed)

    @staticmethod
    def _stream_for(*callbacks):
        return FakeInputStream(callbacks=callbacks)

    @staticmethod
    def _microphone_for(stream, queue_capacity=4):
        return MicrophoneManager(
            sample_rate_hz=16_000,
            block_size=0,
            queue_capacity=queue_capacity,
            input_stream_factory=lambda **kwargs: AudioMonitorTests._configure(
                stream,
                kwargs,
            ),
        )

    @staticmethod
    def _configure(stream, kwargs):
        stream.kwargs = kwargs
        return stream


if __name__ == "__main__":
    unittest.main()
