import threading
import unittest
from unittest.mock import patch

from integration.audio_monitor import InsufficientAudioError
from integration.audio_worker import _capture_audio
from integration.microphone_manager import MicrophoneError


class AudioWorkerTests(unittest.TestCase):
    def test_microphone_open_failure_retries_without_restarting_worker(self):
        stop_event = threading.Event()

        class FakeMicrophone:
            attempts = 0

            def __init__(self, **_kwargs):
                type(self).attempts += 1
                self.attempt = type(self).attempts
                self.last_input_rms = 0.0
                self.last_input_min = None
                self.last_input_max = None
                self.callback_count = 0
                self.seconds_since_last_callback = None
                self.selected_device_index = None
                self.selected_device_name = None
                self.queue_size = 0
                self.queue_capacity = 4
                self.sample_rate_hz = 16_000
                self.block_size = 8_000

        class FakeMonitor:
            def __init__(self, _microphone, _adapter):
                self.discontinuities = 0
                self.dropped_blocks = 0
                self.blocks_received = 0
                self.samples_received = 0
                self.pending_samples = 0
                self.last_window_start_sample = None
                self.last_window_rms = None
                self.last_window_peak = None
                self.last_window_min = None
                self.last_window_max = None
                self.last_window_completed_monotonic = None

            def __enter__(self):
                if FakeMicrophone.attempts == 1:
                    raise MicrophoneError("temporary device open failure")
                return self

            def __exit__(self, *_args):
                return False

            def analyze_next(self, timeout=None):
                del timeout
                stop_event.set()
                raise InsufficientAudioError("test shutdown")

        with (
            patch("integration.audio_worker.MicrophoneManager", FakeMicrophone),
            patch("integration.audio_worker.AudioMonitor", FakeMonitor),
            patch("integration.audio_worker.RECOVERY_DELAYS_SECONDS", (0,)),
            patch("integration.audio_worker._emit"),
        ):
            window_id = _capture_audio(object(), stop_event)

        self.assertEqual(FakeMicrophone.attempts, 2)
        self.assertEqual(window_id, 0)
        self.assertTrue(stop_event.is_set())


if __name__ == "__main__":
    unittest.main()
