from __future__ import annotations

import queue
import threading
import time
import unittest
from unittest.mock import patch

from integration.exam_app import ExamEngineApp


class FakeVariable:
    def __init__(self, value: str = "") -> None:
        self.value = value

    def get(self) -> str:
        return self.value

    def set(self, value: str) -> None:
        self.value = value


class FakeButton:
    def __init__(self, state: str = "normal") -> None:
        self.state = state

    def configure(self, *, state: str) -> None:
        self.state = state


class FakeRoot:
    def after(self, _delay: int, callback) -> None:
        callback()


class QueuedRoot:
    def __init__(self) -> None:
        self.after_threads: list[int] = []

    def after(self, _delay: int, _callback) -> None:
        self.after_threads.append(threading.get_ident())


class FakeInput:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.flush_count = 0

    def write(self, value: str) -> None:
        self.lines.append(value)

    def flush(self) -> None:
        self.flush_count += 1


class FakeProcess:
    def __init__(self) -> None:
        self.stdin = FakeInput()
        self.terminated = False

    def poll(self) -> None:
        return None

    def terminate(self) -> None:
        self.terminated = True


class FakeStorage:
    def enrollment_number_available(self, _value: str) -> bool:
        return True


class ExamAppVoiceControlTests(unittest.TestCase):
    def make_app(self) -> tuple[ExamEngineApp, FakeProcess]:
        app = ExamEngineApp.__new__(ExamEngineApp)
        process = FakeProcess()
        app.root = FakeRoot()
        app._voice_process = process
        app._voice_capture_ready = False
        app._voice_stop_requested = False
        app.start_recording_button = FakeButton("disabled")
        app.stop_recording_button = FakeButton("disabled")
        app.microphone_status = FakeVariable()
        app.record_status = FakeVariable()
        app.voice_quality = FakeVariable()
        app.full_name = FakeVariable()
        app.enrollment_number = FakeVariable()
        app._captured_embeddings = []
        app._voice_embedding = None
        app._voice_duration_seconds = None
        app._voice_validation_status = "NOT VALIDATED"
        app.register_button = FakeButton("disabled")
        app.registration_status = FakeVariable()
        app.storage = FakeStorage()
        return app, process

    def test_stop_does_not_cancel_worker_while_models_are_loading(self) -> None:
        app, process = self.make_app()

        app.stop_voice_registration()

        self.assertEqual(process.stdin.lines, [])
        self.assertIn("wait until it is connected", app.record_status.get())

    def test_stop_becomes_available_after_microphone_connects(self) -> None:
        app, process = self.make_app()

        app._dispatch_voice_message(
            {"type": "status", "status": "AUDIO MODELS READY"}
        )
        self.assertEqual(app.stop_recording_button.state, "disabled")

        app._dispatch_voice_message(
            {"type": "status", "status": "MICROPHONE CONNECTED"}
        )
        self.assertEqual(app.microphone_status.get(), "MICROPHONE CONNECTED")
        self.assertEqual(app.stop_recording_button.state, "disabled")

        app._dispatch_voice_message(
            {"type": "status", "status": "AUDIO_WORKER_READY"}
        )
        self.assertEqual(app.record_status.get(), "RECORDING")
        self.assertEqual(app.stop_recording_button.state, "normal")

        app.stop_voice_registration()
        app.stop_voice_registration()

        self.assertEqual(process.stdin.lines, ["STOP\n"])
        self.assertEqual(process.stdin.flush_count, 1)
        self.assertEqual(app.stop_recording_button.state, "disabled")
        self.assertEqual(app.record_status.get(), "STOPPING RECORDING")

    def test_cancelled_worker_restores_controls_for_retry(self) -> None:
        app, process = self.make_app()
        app._captured_embeddings = [object(), object(), object()]

        app._dispatch_voice_message(
            {"type": "error", "status": "VOICE CAPTURE CANCELLED"}
        )
        app._voice_worker_finished(process)

        self.assertEqual(app.microphone_status.get(), "MICROPHONE DISCONNECTED")
        self.assertEqual(app.record_status.get(), "VOICE CAPTURE CANCELLED")
        self.assertEqual(app.start_recording_button.state, "normal")
        self.assertEqual(app.stop_recording_button.state, "disabled")
        self.assertEqual(len(app._captured_embeddings), 3)
        self.assertIsNone(app._voice_process)

    def test_stale_worker_completion_does_not_change_new_worker_controls(self) -> None:
        app, previous_process = self.make_app()
        current_process = FakeProcess()
        app._voice_process = current_process

        app._voice_worker_finished(previous_process)

        self.assertIs(app._voice_process, current_process)
        self.assertEqual(app.start_recording_button.state, "disabled")
        self.assertEqual(app.stop_recording_button.state, "disabled")

    def test_worker_messages_are_processed_from_the_main_thread_queue(self) -> None:
        app = ExamEngineApp.__new__(ExamEngineApp)
        app._voice_events = queue.Queue()
        app._closing = False
        app._voice_process = None
        app._voice_startup_started_at = None
        app._voice_capture_ready = False
        app._voice_startup_timed_out = False
        app._latest_camera_frame = None
        app._camera_frame_lock = threading.Lock()
        app._last_preview_render = time.monotonic()
        app.record_status = FakeVariable()
        app.root = QueuedRoot()

        reader = threading.Thread(
            target=lambda: app._voice_events.put(
                (
                    "voice_message",
                    {"type": "status", "status": "AUDIO MODELS READY"},
                )
            )
        )
        reader.start()
        reader.join()
        main_thread_id = threading.get_ident()

        app._poll_ui_events()

        self.assertEqual(app.record_status.get(), "CONNECTING MICROPHONE")
        self.assertEqual(app.root.after_threads, [main_thread_id])

    def test_startup_timeout_is_reported_and_worker_is_terminated(self) -> None:
        app, process = self.make_app()
        app._voice_startup_started_at = 0.0
        app._voice_startup_stage = "AUDIO_STREAM_STARTING"
        app._voice_startup_timed_out = False

        with patch("integration.exam_app.time.monotonic", return_value=61.0):
            app._check_voice_startup_timeout()

        self.assertTrue(process.terminated)
        self.assertEqual(
            app._voice_terminal_status,
            "MICROPHONE STARTUP TIMEOUT",
        )
        self.assertIn("AUDIO_STREAM_STARTING", app.record_status.get())
        self.assertEqual(app.start_recording_button.state, "disabled")

    def test_microphone_startup_error_is_visible_and_preserves_face_samples(self) -> None:
        app, _process = self.make_app()
        app._captured_embeddings = [object(), object(), object()]

        app._dispatch_voice_message(
            {
                "type": "error",
                "status": "MICROPHONE ERROR",
                "error": "PortAudioError: device busy",
                "diagnostic": "traceback details",
            }
        )

        self.assertIn("device busy", app.record_status.get())
        self.assertEqual(app.microphone_status.get(), "MICROPHONE DISCONNECTED")
        self.assertEqual(len(app._captured_embeddings), 3)


if __name__ == "__main__":
    unittest.main()
