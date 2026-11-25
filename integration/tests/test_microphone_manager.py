import unittest
from types import SimpleNamespace

import numpy as np

from integration.microphone_manager import (
    MicrophoneManager,
    MicrophoneQueueOverflowError,
    MicrophoneStartupTimeoutError,
    MicrophoneStreamError,
)


class FakeInputStream:
    def __init__(self, *, callbacks=None, start_error=None, **kwargs):
        self.kwargs = kwargs
        self.callbacks = list(callbacks or [])
        self.start_error = start_error
        self.active = False
        self.stopped = False
        self.closed = False

    def start(self):
        if self.start_error is not None:
            raise self.start_error
        self.active = True
        for audio, status in self.callbacks:
            self.kwargs["callback"](audio, len(audio), None, status)

    def stop(self):
        self.active = False
        self.stopped = True

    def close(self):
        self.closed = True


def audio_block(values, dtype=np.float32):
    return np.asarray(values, dtype=dtype).reshape(-1, 1)


class MicrophoneManagerTests(unittest.TestCase):
    def test_context_opens_reads_mono_float32_and_releases_stream(self):
        source = audio_block([0.1, -0.2, 0.3])
        stream = FakeInputStream(callbacks=[(source, None)])

        with MicrophoneManager(
            sample_rate_hz=16_000,
            block_size=3,
            device_index=2,
            input_stream_factory=lambda **kwargs: self._configure(stream, kwargs),
        ) as microphone:
            block = microphone.read(timeout=0.1)
            self.assertAlmostEqual(
                microphone.last_input_rms,
                np.sqrt(np.mean(np.array([0.1, -0.2, 0.3]) ** 2)),
            )
            self.assertEqual(microphone.callback_count, 1)
            self.assertAlmostEqual(microphone.last_input_min, -0.2)
            self.assertAlmostEqual(microphone.last_input_max, 0.3)
            self.assertEqual(microphone.queue_size, 0)
            self.assertIsNotNone(microphone.seconds_since_last_callback)

        self.assertEqual(block.dtype, np.float32)
        self.assertEqual(block.shape, (3,))
        np.testing.assert_array_equal(block, source[:, 0])
        self.assertEqual(stream.kwargs["samplerate"], 16_000)
        self.assertEqual(stream.kwargs["channels"], 1)
        self.assertEqual(stream.kwargs["blocksize"], 3)
        self.assertEqual(stream.kwargs["device"], 2)
        self.assertEqual(stream.kwargs["dtype"], "float32")
        self.assertTrue(stream.stopped)
        self.assertTrue(stream.closed)
        self.assertEqual(microphone.last_input_rms, 0.0)

    def test_selected_default_device_name_is_reported(self):
        stream = FakeInputStream(callbacks=[(audio_block([0.1]), None)])
        stages = []

        class FakeSoundDevice:
            default = SimpleNamespace(device=(3, 0))

            @staticmethod
            def query_devices(device=None, kind=None):
                if device != 3 or kind != "input":
                    raise AssertionError((device, kind))
                return {"name": "Test input"}

            @staticmethod
            def InputStream(**kwargs):
                return MicrophoneManagerTests._configure(stream, kwargs)

        manager = MicrophoneManager(
            sounddevice_module=FakeSoundDevice,
            stage_callback=lambda stage, _details: stages.append(stage),
        )
        with manager:
            self.assertEqual(manager.selected_device_index, 3)
            self.assertEqual(manager.selected_device_name, "Test input")
            self.assertEqual(stream.kwargs["device"], 3)
            self.assertEqual(manager.callback_count, 1)

        self.assertIn("AUDIO_DEVICE_SELECTED", stages)
        self.assertIn("AUDIO_STREAM_OPENED", stages)
        self.assertIn("AUDIO_CALLBACK_STARTED", stages)

    def test_read_requires_open_stream_and_timeout_is_reported(self):
        manager = MicrophoneManager(input_stream_factory=lambda **_kwargs: None)
        with self.assertRaisesRegex(RuntimeError, "not open"):
            manager.read()

        stream = FakeInputStream(callbacks=[(audio_block([0.1]), None)])
        manager = MicrophoneManager(
            input_stream_factory=lambda **kwargs: self._configure(stream, kwargs)
        )
        manager.open()
        try:
            manager.read(timeout=0.1)
            with self.assertRaises(TimeoutError):
                manager.read(timeout=0)
        finally:
            manager.close()

    def test_invalid_blocks_are_reported_and_stream_is_cleaned_up(self):
        invalid_blocks = (
            (audio_block([0.1, 0.2], dtype=np.float64), "float32"),
            (np.zeros((2, 2), dtype=np.float32), "mono"),
            (audio_block([np.nan, 0.2]), "finite"),
        )

        for block, expected_error in invalid_blocks:
            with self.subTest(expected_error=expected_error):
                stream = FakeInputStream(callbacks=[(block, None)])
                manager = MicrophoneManager(
                    callback_timeout_seconds=0.02,
                    input_stream_factory=lambda **kwargs: self._configure(
                        stream,
                        kwargs,
                    )
                )
                with self.assertRaisesRegex(MicrophoneStreamError, expected_error):
                    manager.open()

                self.assertTrue(stream.closed)

    def test_callback_status_is_reported_and_stream_is_cleaned_up(self):
        stream = FakeInputStream(
            callbacks=[(audio_block([0.1]), "input overflow")]
        )
        manager = MicrophoneManager(
            callback_timeout_seconds=0.02,
            input_stream_factory=lambda **kwargs: self._configure(stream, kwargs)
        )
        with self.assertRaisesRegex(MicrophoneStreamError, "input overflow"):
            manager.open()

        self.assertTrue(stream.closed)

    def test_finished_callback_reports_stream_disconnection(self):
        stream = FakeInputStream(callbacks=[(audio_block([0.1]), None)])
        manager = MicrophoneManager(
            input_stream_factory=lambda **kwargs: self._configure(stream, kwargs)
        )
        manager.open()
        try:
            manager._finished_callback()
            with self.assertRaisesRegex(
                MicrophoneStreamError,
                "stopped unexpectedly",
            ):
                manager.read(timeout=0.1)
        finally:
            manager.close()

    def test_stream_error_clears_stale_callback_data(self):
        stream = FakeInputStream(callbacks=[(audio_block([0.1]), None)])
        manager = MicrophoneManager(
            input_stream_factory=lambda **kwargs: self._configure(stream, kwargs)
        )
        manager.open()

        manager._record_error(
            MicrophoneStreamError("microphone device disconnected")
        )

        self.assertEqual(manager.callback_count, 0)
        self.assertIsNone(manager.seconds_since_last_callback)
        self.assertEqual(manager.last_input_rms, 0.0)
        self.assertIsNone(manager.last_input_min)
        self.assertIsNone(manager.last_input_max)
        with self.assertRaisesRegex(
            MicrophoneStreamError,
            "microphone device disconnected",
        ):
            manager.read(timeout=0.1)

        self.assertTrue(stream.closed)

    def test_queue_overflow_discards_stale_audio_and_keeps_stream_open(self):
        first_block = audio_block([0.1])
        latest_block = audio_block([0.2])
        stream = FakeInputStream(
            callbacks=[(first_block, None), (latest_block, None)]
        )
        manager = MicrophoneManager(
            queue_capacity=1,
            input_stream_factory=lambda **kwargs: self._configure(stream, kwargs),
        )
        manager.open()

        with self.assertRaises(MicrophoneQueueOverflowError) as raised:
            manager.read(timeout=0.1)

        self.assertEqual(raised.exception.dropped_blocks, 1)
        self.assertEqual(manager.dropped_blocks, 1)
        self.assertTrue(stream.active)
        self.assertFalse(stream.closed)
        np.testing.assert_array_equal(manager.read(timeout=0.1), latest_block[:, 0])

        manager.close()
        self.assertTrue(stream.closed)

    def test_open_failure_reports_unavailable_device(self):
        def unavailable_factory(**_kwargs):
            raise OSError("device unavailable")

        manager = MicrophoneManager(
            device_index=7,
            input_stream_factory=unavailable_factory,
        )

        with self.assertRaisesRegex(RuntimeError, "device 7"):
            manager.open()

    def test_failed_stream_start_closes_created_stream(self):
        stream = FakeInputStream(start_error=OSError("start failed"))
        manager = MicrophoneManager(
            input_stream_factory=lambda **kwargs: self._configure(stream, kwargs)
        )

        with self.assertRaisesRegex(RuntimeError, "Could not open microphone"):
            manager.open()

        self.assertTrue(stream.closed)

    def test_stream_requires_a_valid_callback_before_open_acknowledgement(self):
        stream = FakeInputStream()
        manager = MicrophoneManager(
            callback_timeout_seconds=0.01,
            input_stream_factory=lambda **kwargs: self._configure(stream, kwargs),
        )

        with self.assertRaisesRegex(
            MicrophoneStartupTimeoutError,
            "no valid audio callback",
        ):
            manager.open()

        self.assertTrue(stream.closed)

    def test_default_device_is_resolved_to_an_explicit_index(self):
        stream = FakeInputStream(callbacks=[(audio_block([0.1]), None)])
        input_device = {
            "name": "Default input",
            "hostapi": 1,
            "max_input_channels": 1,
            "default_samplerate": 48_000,
        }

        class FakeSoundDevice:
            default = SimpleNamespace(device=(-1, 0))

            @staticmethod
            def query_devices(device=None, kind=None):
                if device is None and kind == "input":
                    return input_device
                if device is None and kind is None:
                    return [{"name": "Output", "hostapi": 0}, input_device]
                if device == 1 and kind == "input":
                    return input_device
                raise AssertionError((device, kind))

            @staticmethod
            def check_input_settings(**_kwargs):
                return None

            @staticmethod
            def InputStream(**kwargs):
                return MicrophoneManagerTests._configure(stream, kwargs)

        manager = MicrophoneManager(sounddevice_module=FakeSoundDevice)
        with manager:
            self.assertEqual(manager.selected_device_index, 1)
            self.assertEqual(manager.selected_device_name, "Default input")
            self.assertEqual(stream.kwargs["device"], 1)

    def test_unsupported_pipeline_rate_falls_back_to_native_device_rate(self):
        stream = FakeInputStream(callbacks=[(audio_block([0.1]), None)])
        device_info = {
            "name": "48 kHz input",
            "hostapi": 0,
            "max_input_channels": 2,
            "default_samplerate": 48_000,
        }

        class FakeSoundDevice:
            default = SimpleNamespace(device=(0, 0))

            @staticmethod
            def query_devices(device=None, kind=None):
                if device == 0 and kind == "input":
                    return device_info
                raise AssertionError((device, kind))

            @staticmethod
            def check_input_settings(*, samplerate, **_kwargs):
                if samplerate != 48_000:
                    raise ValueError("16 kHz unsupported")

            @staticmethod
            def InputStream(**kwargs):
                return MicrophoneManagerTests._configure(stream, kwargs)

        manager = MicrophoneManager(sounddevice_module=FakeSoundDevice)
        with manager:
            self.assertEqual(manager.sample_rate_hz, 16_000)
            self.assertEqual(manager.input_sample_rate_hz, 48_000)
            self.assertEqual(manager.input_block_size, 24_000)
            self.assertEqual(stream.kwargs["samplerate"], 48_000)
            self.assertEqual(stream.kwargs["blocksize"], 24_000)

    def test_only_mono_capture_is_supported(self):
        with self.assertRaisesRegex(ValueError, "mono"):
            MicrophoneManager(channels=2)

    @staticmethod
    def _configure(stream, kwargs):
        stream.kwargs = kwargs
        return stream


if __name__ == "__main__":
    unittest.main()
