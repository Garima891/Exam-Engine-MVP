from __future__ import annotations

import queue
import subprocess
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

from integration.exam_app import (
    ExamEngineApp,
    _resolve_audio_python,
)
from integration.config import AUDIO_INTERPRETER_PROBE_TIMEOUT_SECONDS


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
    def make_probe_process(
        self,
        *,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
        communicate=None,
    ) -> SimpleNamespace:
        process = SimpleNamespace(
            returncode=returncode,
            communicate=communicate
            or (lambda **_kwargs: (stdout, stderr)),
            killed=False,
            poll=lambda: process.returncode,
        )

        def kill():
            process.killed = True
            process.returncode = -9

        process.kill = kill
        process.wait = lambda **_kwargs: process.returncode
        return process

    def test_probe_accepts_interpreter_with_mandatory_audio_dependencies(self):
        from integration.exam_app import _probe_audio_interpreter

        communicate_timeouts = []

        def finish_probe(*, timeout):
            communicate_timeouts.append(timeout)
            return "", ""

        process = self.make_probe_process(communicate=finish_probe)
        with (
            patch("integration.exam_app.os.path.isfile", return_value=True),
            patch("integration.exam_app.os.access", return_value=True),
            patch(
                "integration.exam_app.subprocess.Popen",
                return_value=process,
            ) as popen,
        ):
            valid, error = _probe_audio_interpreter(
                r"C:\portable\audio311\python.exe"
            )

        self.assertTrue(valid)
        self.assertEqual(error, "")
        self.assertEqual(
            popen.call_args.kwargs["stdout"],
            subprocess.PIPE,
        )
        self.assertEqual(
            popen.call_args.kwargs["stderr"],
            subprocess.PIPE,
        )
        self.assertEqual(communicate_timeouts, [AUDIO_INTERPRETER_PROBE_TIMEOUT_SECONDS])
        probe_script = popen.call_args.args[0][2]
        for module_name in (
            "numpy",
            "sounddevice",
            "torch",
            "torchaudio",
            "soundfile",
            "tensorflow",
            "tensorflow_hub",
            "silero_vad",
            "speechbrain",
            "huggingface_hub",
        ):
            self.assertIn(module_name, probe_script)

    def test_probe_allows_slow_import_probe_that_exceeds_old_12_second_limit(self):
        from integration.exam_app import _probe_audio_interpreter

        calls = []
        simulated_elapsed_seconds = []

        def delayed_import_probe(*, timeout):
            calls.append(timeout)
            if timeout <= 12:
                raise subprocess.TimeoutExpired("probe", timeout)
            simulated_elapsed_seconds.append(13.0)
            return "", ""

        process = self.make_probe_process(communicate=delayed_import_probe)
        with (
            patch("integration.exam_app.os.path.isfile", return_value=True),
            patch("integration.exam_app.os.access", return_value=True),
            patch("integration.exam_app.subprocess.Popen", return_value=process),
        ):
            valid, error = _probe_audio_interpreter(
                r"C:\portable\audio311\python.exe"
            )

        self.assertTrue(valid)
        self.assertEqual(error, "")
        self.assertEqual(calls, [60.0])
        self.assertEqual(simulated_elapsed_seconds, [13.0])

    def test_probe_timeout_kills_and_reaps_process_and_reports_candidate(self):
        from integration.exam_app import _probe_audio_interpreter

        communicate_calls = []

        def timeout_then_reap(*, timeout=None):
            communicate_calls.append(timeout)
            if timeout == AUDIO_INTERPRETER_PROBE_TIMEOUT_SECONDS:
                raise subprocess.TimeoutExpired(
                    "probe",
                    timeout,
                    output="partial stdout",
                    stderr="partial stderr",
                )
            return "captured stdout", "captured stderr"

        process = self.make_probe_process(communicate=timeout_then_reap)
        process.returncode = None
        with (
            patch("integration.exam_app.os.path.isfile", return_value=True),
            patch("integration.exam_app.os.access", return_value=True),
            patch("integration.exam_app.subprocess.Popen", return_value=process),
        ):
            valid, error = _probe_audio_interpreter(
                r"C:\portable\audio311\python.exe"
            )

        self.assertFalse(valid)
        self.assertTrue(process.killed)
        self.assertEqual(
            communicate_calls,
            [AUDIO_INTERPRETER_PROBE_TIMEOUT_SECONDS, 5],
        )
        self.assertIn("timed out after 60 seconds", error)
        self.assertIn(r"C:\portable\audio311\python.exe", error)
        self.assertIn("exit code -9", error)
        self.assertIn("captured stdout", error)
        self.assertIn("captured stderr", error)

    def test_probe_reports_missing_mandatory_audio_dependency(self):
        from integration.exam_app import _probe_audio_interpreter

        process = self.make_probe_process(
            returncode=1,
            stdout=(
                "MISSING MANDATORY AUDIO DEPENDENCIES: "
                "tensorflow_hub: ModuleNotFoundError: missing"
            ),
        )
        with (
            patch("integration.exam_app.os.path.isfile", return_value=True),
            patch("integration.exam_app.os.access", return_value=True),
            patch(
                "integration.exam_app.subprocess.Popen",
                return_value=process,
            ),
        ):
            valid, error = _probe_audio_interpreter(
                r"C:\portable\audio311\python.exe"
            )

        self.assertFalse(valid)
        self.assertIn("tensorflow_hub", error)
        self.assertIn("ModuleNotFoundError", error)

    def test_invalid_configured_interpreter_and_fallbacks_report_no_candidate(self):
        with (
            patch("integration.exam_app.os.path.isfile", return_value=False),
            patch("integration.exam_app.os.access", return_value=False),
            patch("integration.exam_app._probe_audio_interpreter",
                  return_value=(False, "mandatory dependency missing")),
            patch.dict(
                "integration.exam_app.os.environ",
                {"EXAM_AUDIO_PYTHON": r"C:\invalid\python.exe"},
                clear=True,
            ),
        ):
            resolved, error = _resolve_audio_python()

        self.assertIsNone(resolved)
        self.assertIn(r"C:\invalid\python.exe", error)
        self.assertIn("mandatory dependency missing", error)

    def test_missing_environment_variable_still_uses_sys_executable_fallback(self):
        with patch.dict(
            "integration.exam_app.os.environ",
            {},
            clear=True,
        ):
            with (
                patch("integration.exam_app.os.path.isfile", return_value=True),
                patch("integration.exam_app.os.access", return_value=True),
                patch(
                    "integration.exam_app._probe_audio_interpreter",
                    return_value=(True, ""),
                ),
            ):
                resolved, error = _resolve_audio_python()

        self.assertEqual(resolved, str(__import__("os").path.abspath(
            __import__("sys").executable
        )))
        self.assertEqual(error, "")

    def test_start_exam_reports_probe_timeout_before_creating_session(self):
        app = ExamEngineApp.__new__(ExamEngineApp)
        app.student_table = SimpleNamespace(selection=lambda: ("student-1",))
        app.storage = SimpleNamespace(
            is_fully_registered=lambda _student_id: True,
            load_student=lambda _student_id: {"student_id": "student-1"},
            create_exam_session=lambda *_args: self.fail(
                "Exam session must not be created without a compatible interpreter."
            ),
        )
        with (
            patch(
                "integration.exam_app._resolve_audio_python",
                return_value=(
                    None,
                    "Interpreter probe timed out after 60 seconds.",
                ),
            ),
            patch("integration.exam_app.messagebox.showerror") as showerror,
            patch("integration.exam_app.subprocess.Popen") as popen,
        ):
            app.start_exam()

        self.assertEqual(showerror.call_args.args[0], "Start Exam")
        self.assertIn("timed out after 60 seconds", showerror.call_args.args[1])
        self.assertIn("EXAM_AUDIO_PYTHON", showerror.call_args.args[1])
        popen.assert_not_called()

    def test_portable_audio_interpreter_resolution_prefers_configured_valid_path(self):
        configured = r"C:\portable\audio311\python.exe"
        with (
            patch("integration.exam_app.os.path.isfile", return_value=True),
            patch("integration.exam_app.os.access", return_value=True),
            patch(
                "integration.exam_app._probe_audio_interpreter",
                return_value=(True, ""),
            ),
        ):
            resolved, error = _resolve_audio_python(configured)

        self.assertTrue(resolved)
        self.assertEqual(error, "")
        self.assertTrue(resolved.endswith("python.exe"))

    def test_configured_missing_interpreter_falls_back_portably(self):
        configured = r"C:\missing\python.exe"
        with (
            patch("integration.exam_app.os.path.isfile", side_effect=lambda path: str(path).endswith("python.exe") and "missing" not in str(path)),
            patch("integration.exam_app.os.access", return_value=True),
            patch(
                "integration.exam_app._probe_audio_interpreter",
                side_effect=lambda path: (
                    ("missing" not in path, "")
                    if "missing" not in path
                    else (False, "configured executable does not exist")
                ),
            ),
        ):
            resolved, error = _resolve_audio_python(configured)

        self.assertTrue(resolved)
        self.assertEqual(error, "")

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
