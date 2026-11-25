from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageTk

from integration.candidate_storage import CandidateStorage
from integration.config import (
    AUDIO_INTERPRETER_PROBE_TIMEOUT_SECONDS,
    CANDIDATE_DATABASE_PATH,
    MICROPHONE_STARTUP_TIMEOUT_SECONDS,
)
from integration.camera_manager import CameraManager
from integration.module_adapters.face_identity import FaceIdentityAdapter
from integration.student_registration import StudentRegistrationService


def _probe_audio_interpreter(python_executable: str) -> tuple[bool, str]:
    if not python_executable or not os.path.isfile(python_executable):
        return False, f"Interpreter not found: {python_executable or '(empty path)'}"
    if not os.access(python_executable, os.X_OK):
        return False, f"Interpreter is not executable: {python_executable}"
    mandatory_modules = (
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
    )
    probe_script = (
        "import importlib, sys\n"
        "missing=[]\n"
        f"for name in {mandatory_modules!r}:"
        "\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except Exception as exc:\n"
        "        missing.append(f'{name}: {type(exc).__name__}: {exc}')\n"
        "if missing:\n"
        "    print('MISSING MANDATORY AUDIO DEPENDENCIES: ' + '; '.join(missing))\n"
        "    sys.exit(1)\n"
    )
    command = [python_executable, "-c", probe_script]
    try:
        process = subprocess.Popen(
            command,
            cwd=str(CANDIDATE_DATABASE_PATH.parents[1]),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    try:
        stdout, stderr = process.communicate(
            timeout=AUDIO_INTERPRETER_PROBE_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired:
        try:
            if process.poll() is None:
                process.kill()
        except OSError as exc:
            if process.poll() is None:
                return False, (
                    "Interpreter probe timed out, but its process could not be "
                    f"terminated: {python_executable}: {type(exc).__name__}: {exc}"
                )
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired) as exc:
                return False, (
                    "Interpreter probe timed out and its process could not be "
                    f"terminated: {python_executable}: {type(exc).__name__}: {exc}"
                )
            stdout, stderr = process.communicate()
        diagnostics = _audio_probe_output(stdout, stderr)
        timeout_error = (
            "Interpreter probe timed out after "
            f"{AUDIO_INTERPRETER_PROBE_TIMEOUT_SECONDS:g} seconds: "
            f"{python_executable}. The probe process was terminated "
            f"(exit code {process.returncode})."
        )
        if diagnostics:
            timeout_error += f" Probe output: {diagnostics}"
        return False, timeout_error
    if process.returncode != 0:
        diagnostic = _audio_probe_output(stdout, stderr)
        return False, (
            f"Interpreter probe failed for {python_executable} "
            f"with exit code {process.returncode}"
            + (f": {diagnostic}" if diagnostic else "")
        )
    return True, ""


def _audio_probe_output(stdout: str | None, stderr: str | None) -> str:
    output = "\n".join(
        stream.strip()
        for stream in (stdout, stderr)
        if stream and stream.strip()
    )
    if len(output) > 2000:
        return output[:1997] + "..."
    return output


def _resolve_audio_python(configured: str | None = None) -> tuple[str | None, str]:
    candidates: list[str] = []
    configured = (
        configured
        if configured is not None
        else os.environ.get("EXAM_AUDIO_PYTHON")
    )
    if configured:
        candidates.append(os.path.expandvars(os.path.expanduser(configured.strip())))

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(
            str(
                Path(local_app_data)
                / "venvs"
                / "exam-engine-audio311"
                / "Scripts"
                / "python.exe"
            )
        )

    candidates.append(sys.executable)

    candidate_errors: list[str] = []
    last_error = "no audio Python interpreter candidate was available"
    seen_candidates: set[str] = set()
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        candidate_key = os.path.normcase(candidate)
        if candidate_key in seen_candidates:
            continue
        seen_candidates.add(candidate_key)
        valid, error = _probe_audio_interpreter(candidate)
        if valid:
            return candidate, ""
        candidate_errors.append(f"{candidate}: {error}")
    if candidate_errors:
        last_error = "No compatible audio interpreter found; attempted:\n" + "\n".join(
            candidate_errors
        )
    return None, last_error


class ExamEngineApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AI Proctored Exam Engine")
        self.root.geometry("1100x760")
        self.storage = CandidateStorage(CANDIDATE_DATABASE_PATH)

        self._camera: CameraManager | None = None
        self._camera_thread: threading.Thread | None = None
        self._camera_stop = threading.Event()
        self._camera_frame_lock = threading.Lock()
        self._latest_camera_frame: np.ndarray | None = None
        self._latest_capture_frame: np.ndarray | None = None
        self._last_preview_render = 0.0
        self._face_adapter = FaceIdentityAdapter()
        self._captured_embeddings: list[np.ndarray] = []
        self._captured_frames: list[Any] = []
        self._voice_embedding: np.ndarray | None = None
        self._voice_duration_seconds: float | None = None
        self._voice_validation_status = "NOT VALIDATED"
        self._voice_process: subprocess.Popen[str] | None = None
        self._voice_reader_thread: threading.Thread | None = None
        self._voice_capture_ready = False
        self._voice_stop_requested = False
        self._voice_events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._voice_startup_started_at: float | None = None
        self._voice_startup_stage = "NOT STARTED"
        self._voice_startup_timed_out = False
        self._voice_terminal_status: str | None = None
        self._closing = False

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True)
        self.notebook = notebook
        self.registration = ttk.Frame(notebook, padding=12)
        self.students = ttk.Frame(notebook, padding=12)
        notebook.add(self.registration, text="Student Registration")
        notebook.add(self.students, text="Registered Students")
        notebook.bind("<<NotebookTabChanged>>", self._tab_changed)

        self._build_registration()
        self._build_students()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(25, self._poll_ui_events)

    def _build_registration(self) -> None:
        left = ttk.Frame(self.registration)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(self.registration, width=330)
        right.pack(side="right", fill="y", padx=(12, 0))

        ttk.Label(left, text="Student Registration", font=("Segoe UI", 18, "bold")).pack(
            anchor="w"
        )
        details = ttk.LabelFrame(left, text="Student Details", padding=10)
        details.pack(fill="x", pady=8)
        self.full_name = tk.StringVar()
        self.enrollment_number = tk.StringVar()
        self.email = tk.StringVar()
        for label, variable in (
            ("Student Full Name *", self.full_name),
            ("Enrollment Number *", self.enrollment_number),
            ("Email (optional)", self.email),
        ):
            row = ttk.Frame(details)
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=label, width=24).pack(side="left")
            ttk.Entry(row, textvariable=variable).pack(side="left", fill="x", expand=True)

        camera_frame = ttk.LabelFrame(left, text="Face Registration", padding=8)
        camera_frame.pack(fill="both", expand=True)
        self.camera_label = ttk.Label(camera_frame)
        self.camera_label.pack(fill="both", expand=True)
        self.face_status = tk.StringVar(value="READY - press C for frame 1/3")
        ttk.Label(camera_frame, textvariable=self.face_status).pack(anchor="w")

        action = ttk.Frame(right)
        action.pack(fill="x")
        ttk.Label(action, text="Face Captures", font=("Segoe UI", 12, "bold")).pack(
            anchor="w"
        )
        self.capture_progress = tk.StringVar(value="0/3")
        ttk.Label(action, textvariable=self.capture_progress).pack(anchor="w")
        self.preview_labels: list[tk.Label] = []
        for index in range(3):
            preview = tk.Label(action, text=f"Frame {index + 1}", width=32, height=8)
            preview.pack(pady=4)
            self.preview_labels.append(preview)

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=10)
        ttk.Label(right, text="Voice Registration", font=("Segoe UI", 12, "bold")).pack(
            anchor="w"
        )
        self.microphone_status = tk.StringVar(value="MICROPHONE NOT STARTED")
        ttk.Label(right, textvariable=self.microphone_status).pack(anchor="w")
        self.record_status = tk.StringVar(value="Press A or use Start Recording")
        ttk.Label(right, textvariable=self.record_status, wraplength=300).pack(anchor="w")
        self.voice_quality = tk.StringVar(value="Voice not validated")
        ttk.Label(right, textvariable=self.voice_quality, wraplength=300).pack(
            anchor="w"
        )
        self.start_recording_button = ttk.Button(
            right,
            text="Start Recording",
            command=self.start_voice_registration,
        )
        self.start_recording_button.pack(fill="x", pady=4)
        self.stop_recording_button = ttk.Button(
            right,
            text="Stop Recording",
            command=self.stop_voice_registration,
            state="disabled",
        )
        self.stop_recording_button.pack(fill="x")
        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=10)
        self.registration_status = tk.StringVar(value="THREE FRAMES REQUIRED")
        ttk.Label(right, textvariable=self.registration_status, wraplength=300).pack(
            anchor="w"
        )
        self.register_button = ttk.Button(
            right,
            text="REGISTER STUDENT",
            command=self.register_student,
            state="disabled",
        )
        self.register_button.pack(fill="x", pady=10)

        self.root.bind("<c>", lambda _event: self.capture_face())
        self.root.bind("<C>", lambda _event: self.capture_face())
        self.root.bind("<a>", lambda _event: self._toggle_voice_registration())
        self.root.bind("<A>", lambda _event: self._toggle_voice_registration())

    def _build_students(self) -> None:
        toolbar = ttk.Frame(self.students)
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text="Registered Students", font=("Segoe UI", 18, "bold")).pack(
            side="left"
        )
        self.student_search = tk.StringVar()
        ttk.Entry(toolbar, textvariable=self.student_search, width=36).pack(
            side="right",
            padx=8,
        )
        ttk.Label(toolbar, text="Search").pack(side="right")
        self.student_search.trace_add("write", lambda *_args: self.refresh_students())

        columns = (
            "Student Name",
            "Enrollment Number",
            "Registration",
            "Face",
            "Voice",
            "Registration Date",
        )
        self.student_table = ttk.Treeview(
            self.students,
            columns=columns,
            show="headings",
            selectmode="browse",
        )
        for column in columns:
            self.student_table.heading(column, text=column)
            self.student_table.column(column, width=150, anchor="w")
        self.student_table.pack(fill="both", expand=True, pady=8)
        actions = ttk.Frame(self.students)
        actions.pack(fill="x")
        ttk.Button(actions, text="Refresh", command=self.refresh_students).pack(side="left")
        ttk.Button(actions, text="Start Exam", command=self.start_exam).pack(
            side="right"
        )

    def _tab_changed(self, _event: tk.Event) -> None:
        selected = self.notebook.index("current")
        if selected == 0:
            self._start_camera()
        else:
            self._camera_stop.set()
        if hasattr(self, "student_table"):
            self.refresh_students()

    def _close(self) -> None:
        self._closing = True
        self._camera_stop.set()
        if self._camera is not None:
            self._camera.release()
        if self._voice_process is not None and self._voice_process.poll() is None:
            try:
                self._voice_process.stdin.write("STOP\n")
                self._voice_process.stdin.flush()
            except Exception:
                pass
        self.root.destroy()

    def _start_camera(self) -> None:
        if self._camera_thread is not None and self._camera_thread.is_alive():
            return
        self._camera_stop.clear()
        self._camera_thread = threading.Thread(
            target=self._camera_loop,
            name="registration-camera",
            daemon=True,
        )
        self._camera_thread.start()

    def _camera_loop(self) -> None:
        try:
            with CameraManager() as camera:
                self._camera = camera
                while not self._camera_stop.is_set():
                    frame = camera.read()
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (640, 480))
                    with self._camera_frame_lock:
                        self._latest_camera_frame = image
                        self._latest_capture_frame = frame.copy()
        except Exception as exc:
            self._voice_events.put(
                (
                    "camera_error",
                    {
                        "error": f"{type(exc).__name__}: {exc}",
                        "diagnostic": traceback.format_exc(),
                    },
                )
            )
        finally:
            self._camera = None

    def _render_latest_camera_frame(self) -> None:
        now = time.monotonic()
        if now - self._last_preview_render < 1 / 15:
            return
        with self._camera_frame_lock:
            frame = self._latest_camera_frame
            self._latest_camera_frame = None
        if frame is None:
            return
        self._last_preview_render = now
        photo = ImageTk.PhotoImage(Image.fromarray(frame), master=self.root)
        self.camera_label.configure(image=photo)
        self._preview_photo = photo

    def capture_face(self) -> None:
        if getattr(self, "notebook", None) is not None:
            if self.notebook.index("current") != 0:
                return
        if len(self._captured_embeddings) >= 3:
            self.face_status.set("THREE FRAMES REQUIRED - complete registration")
            return
        with self._camera_frame_lock:
            frame = (
                None
                if self._latest_capture_frame is None
                else self._latest_capture_frame.copy()
            )
        if frame is None:
            self._start_camera()
            self.face_status.set(
                "CAMERA STARTING - wait for the preview before capturing"
            )
            return
        try:
            faces = self._face_adapter.extract_all_face_embeddings(frame)
        except Exception as exc:
            self.face_status.set(f"FACE ERROR: {exc}")
            return
        if not faces:
            self.face_status.set("FACE NOT DETECTED")
            return
        if len(faces) != 1:
            self.face_status.set("MULTIPLE FACES DETECTED - only one student required")
            return
        face = faces[0]
        if not face.get("landmarks_available", False):
            self.face_status.set("FACE QUALITY INVALID - landmarks unavailable")
            return
        try:
            embedding = FaceIdentityAdapter.validate_embedding(face["embedding"])
        except Exception as exc:
            self.face_status.set(f"INVALID FACE FRAME: {exc}")
            return
        if any(
            float(np.dot(embedding, prior)) >= 0.995
            for prior in self._captured_embeddings
        ):
            self.face_status.set("DUPLICATE FRAME - change position slightly")
            return

        self._captured_embeddings.append(embedding)
        self._captured_frames.append(frame.copy())
        index = len(self._captured_embeddings) - 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb).resize((250, 150))
        photo = ImageTk.PhotoImage(pil_image)
        setattr(self, f"_capture_photo_{index}", photo)
        self.preview_labels[index].configure(image=photo)
        self.capture_progress.set(f"{len(self._captured_embeddings)}/3")
        if len(self._captured_embeddings) == 3:
            self.face_status.set("FACE REGISTRATION COMPLETE")
        else:
            self.face_status.set(
                f"FRAME CAPTURED - press C for frame "
                f"{len(self._captured_embeddings) + 1}/3"
            )
        self._refresh_register_button()

    def start_voice_registration(self) -> None:
        self._log_voice_event("START_RECORDING_CLICKED")
        if getattr(self, "notebook", None) is not None:
            if self.notebook.index("current") != 0:
                return
        if self._voice_process is not None:
            if self._voice_process.poll() is None:
                self.record_status.set("Recording worker is already running.")
            else:
                self.record_status.set("Audio worker is finishing; please wait.")
            return
        audio_python, interpreter_error = _resolve_audio_python()
        if not audio_python:
            self.microphone_status.set("MICROPHONE ERROR")
            self.record_status.set(
                "AUDIO311 interpreter unavailable: "
                f"{interpreter_error}"
            )
            return
        command = [
            audio_python,
            "-u",
            "-m",
            "integration.voice_registration_worker",
            "--duration",
            "15",
            "--startup-timeout",
            str(MICROPHONE_STARTUP_TIMEOUT_SECONDS),
        ]
        self._voice_embedding = None
        self._voice_duration_seconds = None
        self._voice_validation_status = "NOT VALIDATED"
        self._voice_capture_ready = False
        self._voice_stop_requested = False
        self.voice_quality.set("Preparing audio capture...")
        self._refresh_register_button()
        try:
            self._voice_process = subprocess.Popen(
                command,
                cwd=str(CANDIDATE_DATABASE_PATH.parents[1]),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            self.microphone_status.set("MICROPHONE ERROR")
            self.record_status.set(str(exc))
            self._set_voice_controls(start_enabled=True, stop_enabled=False)
            print(
                f"VOICE REGISTRATION START ERROR | {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return
        self._set_voice_controls(start_enabled=False, stop_enabled=False)
        self.microphone_status.set("MICROPHONE STARTING")
        self.record_status.set("LOADING AUDIO MODELS")
        self._voice_startup_started_at = time.monotonic()
        self._voice_startup_stage = "AUDIO_WORKER_STARTED"
        self._voice_startup_timed_out = False
        self._voice_terminal_status = None
        self._log_voice_event("AUDIO_WORKER_STARTED", pid=self._voice_process.pid)
        self._voice_reader_thread = threading.Thread(
            target=self._voice_reader,
            name="voice-registration-reader",
            daemon=True,
        )
        self._voice_reader_thread.start()

    def stop_voice_registration(self) -> None:
        process = self._voice_process
        if process is None or process.poll() is not None:
            self.record_status.set("NO ACTIVE RECORDING")
            return
        if self._voice_stop_requested:
            self.record_status.set("STOPPING RECORDING")
            return
        if not self._voice_capture_ready:
            self.record_status.set(
                "Preparing microphone; wait until it is connected before stopping."
            )
            return
        if process.stdin is None:
            self.record_status.set("MICROPHONE ERROR: control channel unavailable")
            self._set_voice_controls(start_enabled=False, stop_enabled=False)
            print(
                "VOICE REGISTRATION STOP ERROR | worker stdin is unavailable",
                file=sys.stderr,
                flush=True,
            )
            return
        try:
            process.stdin.write("STOP\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            self.record_status.set(
                f"AUDIO WORKER ERROR: could not stop recording: {exc}"
            )
            self._set_voice_controls(start_enabled=False, stop_enabled=False)
            print(
                f"VOICE REGISTRATION STOP ERROR | {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return
        self._voice_stop_requested = True
        self._set_voice_controls(start_enabled=False, stop_enabled=False)
        self.record_status.set("STOPPING RECORDING")

    def _toggle_voice_registration(self) -> None:
        process = self._voice_process
        if process is not None and process.poll() is None:
            self.stop_voice_registration()
        else:
            self.start_voice_registration()

    def _voice_reader(self) -> None:
        process = self._voice_process
        if process is None:
            return
        try:
            if process.stdout is None:
                raise RuntimeError("Audio worker stdout pipe is unavailable.")
            for line in process.stdout:
                line = line.strip()
                prefix = "VOICE_REG_JSON:"
                if not line.startswith(prefix):
                    if line:
                        print(f"VOICE WORKER LOG | {line}", flush=True)
                    continue
                try:
                    message = json.loads(line[len(prefix):])
                except json.JSONDecodeError as exc:
                    self._voice_events.put(
                        (
                            "reader_error",
                            {
                                "error": (
                                    f"Invalid audio worker protocol JSON: "
                                    f"{type(exc).__name__}: {exc}"
                                ),
                                "diagnostic": traceback.format_exc(),
                            },
                        )
                    )
                    continue
                self._voice_events.put(("voice_message", message))
        except Exception as exc:
            self._voice_events.put(
                (
                    "reader_error",
                    {
                        "error": f"{type(exc).__name__}: {exc}",
                        "diagnostic": traceback.format_exc(),
                    },
                )
            )
        return_code = process.wait()
        self._voice_events.put(("worker_exit", (process, return_code)))

    def _poll_ui_events(self) -> None:
        if self._closing:
            return
        while True:
            try:
                event_type, payload = self._voice_events.get_nowait()
            except queue.Empty:
                break
            if event_type == "voice_message":
                self._dispatch_voice_message(payload)
            elif event_type == "worker_exit":
                process, return_code = payload
                self._handle_voice_worker_exit(process, return_code)
            elif event_type == "reader_error":
                print(
                    "VOICE WORKER READER ERROR | "
                    f"{payload['error']}\n{payload['diagnostic']}",
                    file=sys.stderr,
                    flush=True,
                )
                self._voice_terminal_status = "AUDIO WORKER ERROR"
                self.record_status.set(f"AUDIO WORKER ERROR: {payload['error']}")
                self.microphone_status.set("MICROPHONE DISCONNECTED")
            elif event_type == "camera_error":
                print(
                    "CAMERA ERROR | "
                    f"{payload['error']}\n{payload['diagnostic']}",
                    file=sys.stderr,
                    flush=True,
                )
                self.face_status.set(f"CAMERA ERROR: {payload['error']}")

        self._check_voice_startup_timeout()
        self._render_latest_camera_frame()
        if not self._closing:
            self.root.after(25, self._poll_ui_events)

    def _handle_voice_worker_exit(
        self,
        process: subprocess.Popen[str],
        return_code: int,
    ) -> None:
        if process is not self._voice_process:
            return
        if (
            return_code != 0
            and not self._voice_startup_timed_out
            and self._voice_terminal_status
            not in {
                "INSUFFICIENT SPEECH",
                "MICROPHONE ERROR",
                "MICROPHONE STARTUP TIMEOUT",
                "AUDIO WORKER ERROR",
                "VOICE CAPTURE ERROR",
            }
        ):
            self._voice_terminal_status = "AUDIO WORKER ERROR"
            self.record_status.set(
                f"AUDIO WORKER ERROR: exited with code {return_code}"
            )
            self.microphone_status.set("MICROPHONE DISCONNECTED")
        self._voice_worker_finished(process)

    def _check_voice_startup_timeout(self) -> None:
        process = self._voice_process
        started_at = self._voice_startup_started_at
        if (
            process is None
            or started_at is None
            or self._voice_capture_ready
            or self._voice_startup_timed_out
            or process.poll() is not None
        ):
            return
        elapsed = time.monotonic() - started_at
        if elapsed < MICROPHONE_STARTUP_TIMEOUT_SECONDS:
            return

        self._voice_startup_timed_out = True
        self._voice_terminal_status = "MICROPHONE STARTUP TIMEOUT"
        reason = (
            f"No valid audio callback acknowledgement after {elapsed:.1f}s; "
            f"last stage: {self._voice_startup_stage}."
        )
        self._log_voice_event(
            "MICROPHONE STARTUP TIMEOUT",
            elapsed_seconds=round(elapsed, 2),
            last_stage=self._voice_startup_stage,
        )
        self.microphone_status.set("MICROPHONE DISCONNECTED")
        self.record_status.set(f"MICROPHONE STARTUP TIMEOUT: {reason}")
        self._voice_embedding = None
        self._voice_validation_status = "NOT VALIDATED"
        self.voice_quality.set("Voice not validated")
        self._set_voice_controls(start_enabled=False, stop_enabled=False)
        self._refresh_register_button()
        try:
            process.terminate()
        except OSError as exc:
            reason = f"{reason} Could not terminate audio worker: {exc}"
            self.record_status.set(f"MICROPHONE STARTUP TIMEOUT: {reason}")
            print(
                f"VOICE WORKER TERMINATION ERROR | {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )

    def _log_voice_event(self, stage: str, **details: Any) -> None:
        timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        suffix = f" | {json.dumps(details, default=str, sort_keys=True)}" if details else ""
        print(f"VOICE REGISTRATION | {timestamp} | {stage}{suffix}", flush=True)

    def _voice_worker_finished(
        self,
        process: subprocess.Popen[str],
    ) -> None:
        if self._voice_process is not process:
            return
        self._voice_process = None
        self._voice_capture_ready = False
        self._voice_stop_requested = False
        self._voice_startup_started_at = None
        self._set_voice_controls(start_enabled=True, stop_enabled=False)
        self._refresh_register_button()

    def _set_voice_controls(
        self,
        *,
        start_enabled: bool,
        stop_enabled: bool,
    ) -> None:
        self.start_recording_button.configure(
            state="normal" if start_enabled else "disabled"
        )
        self.stop_recording_button.configure(
            state="normal" if stop_enabled else "disabled"
        )
        self._voice_capture_ready = stop_enabled

    def _schedule_ui(self, callback: Any) -> None:
        try:
            self.root.after(0, callback)
        except tk.TclError as exc:
            print(
                f"VOICE REGISTRATION UI UPDATE ERROR | {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )

    def _dispatch_voice_message(self, message: dict[str, Any]) -> None:
        message_type = message.get("type")
        status = message.get("status")
        if message_type == "stage":
            stage = str(message.get("stage", "UNKNOWN"))
            self._voice_startup_stage = stage
            details = {
                key: value
                for key, value in message.items()
                if key not in {"type", "stage", "timestamp"}
            }
            self._log_voice_event(
                stage,
                timestamp=message.get("timestamp"),
                **details,
            )
            if stage in {"AUDIO_WORKER_STARTED", "AUDIO_MODELS_LOADING"}:
                self.record_status.set("LOADING AUDIO MODELS")
            elif stage == "AUDIO_MODELS_READY":
                self.record_status.set("CONNECTING MICROPHONE")
            elif stage == "AUDIO_DEVICE_SELECTED":
                self.record_status.set(
                    f"OPENING MICROPHONE: {message.get('device_name', 'selected input')}"
                )
            elif stage in {
                "AUDIO_STREAM_OPENING",
                "AUDIO_STREAM_CREATED",
                "AUDIO_STREAM_STARTING",
            }:
                self.record_status.set("OPENING AUDIO STREAM")
            elif stage == "AUDIO_STREAM_OPENED":
                self.record_status.set("WAITING FOR AUDIO CALLBACK")
            elif stage == "AUDIO_CALLBACK_STARTED":
                self.microphone_status.set("MICROPHONE CONNECTED")
            elif stage == "RECORDING_STARTED":
                self.microphone_status.set("MICROPHONE CONNECTED")
                self.record_status.set("RECORDING")
            return

        if message_type == "complete":
            try:
                import base64

                raw = base64.b64decode(
                    message["embedding_base64"],
                    validate=True,
                )
                embedding = np.frombuffer(raw, dtype=np.float32).copy()
                norm = float(np.linalg.norm(embedding))
                duration = float(message["duration_seconds"])
                dimension = int(message["embedding_dimension"])
                if (
                    embedding.size == 0
                    or embedding.size != dimension
                    or not np.isfinite(embedding).all()
                    or not np.isfinite(norm)
                    or norm == 0.0
                    or duration < 8.0
                    or message.get("validation_status")
                    != "VALID SPEECH AND ECAPA EMBEDDING"
                ):
                    raise ValueError(
                        "Worker returned invalid voice enrollment data."
                    )
                self._voice_embedding = (embedding / norm).astype(
                    np.float32,
                    copy=False,
                )
                self._voice_duration_seconds = duration
                self._voice_validation_status = str(
                    message["validation_status"]
                )
            except (KeyError, TypeError, ValueError) as exc:
                self._voice_embedding = None
                self._voice_validation_status = "INVALID"
                reason = f"{type(exc).__name__}: {exc}"
                print(f"VOICE REGISTRATION VALIDATION ERROR | {reason}", file=sys.stderr)
                self._schedule_ui(
                    lambda value=reason: self.record_status.set(
                        f"VOICE CAPTURE ERROR: {value}"
                    )
                )
                self._schedule_ui(
                    lambda: self.microphone_status.set("MICROPHONE DISCONNECTED")
                )
                self._schedule_ui(self._refresh_register_button)
                return

            self._schedule_ui(
                lambda: self.microphone_status.set("MICROPHONE DISCONNECTED")
            )
            self._schedule_ui(
                lambda: self.record_status.set("VOICE REGISTERED SUCCESSFULLY")
            )
            self._schedule_ui(
                lambda: self.voice_quality.set(
                    f"Duration {duration:.1f}s — {self._voice_validation_status}"
                )
            )
            self._schedule_ui(self._refresh_register_button)
            return

        if message_type == "progress":
            elapsed = float(message.get("elapsed_seconds", 0.0))
            self.record_status.set(f"RECORDING - {elapsed:.1f}s")
            self.microphone_status.set("MICROPHONE CONNECTED")
            return

        if status:
            if status in {
                "MICROPHONE ERROR",
                "MICROPHONE STARTUP TIMEOUT",
                "AUDIO WORKER ERROR",
                "VOICE CAPTURE ERROR",
                "INSUFFICIENT SPEECH",
                "VOICE CAPTURE CANCELLED",
            }:
                self._voice_terminal_status = str(status)
            diagnostic = message.get("error") or message.get("reason")
            if diagnostic:
                print(
                    "VOICE REGISTRATION ERROR | "
                    f"{status} | {diagnostic}\n"
                    f"{message.get('diagnostic', '')}",
                    file=sys.stderr,
                    flush=True,
                )
            if status == "MICROPHONE CONNECTED":
                self.microphone_status.set("MICROPHONE CONNECTED")
                self.record_status.set("WAITING FOR AUDIO CALLBACK")
            elif status == "AUDIO_WORKER_READY":
                self.microphone_status.set("MICROPHONE CONNECTED")
                self.record_status.set("RECORDING")
                self._set_voice_controls(
                    start_enabled=False,
                    stop_enabled=True,
                )
            elif status == "AUDIO MODELS READY":
                self.record_status.set("CONNECTING MICROPHONE")
            elif status == "PROCESSING AUDIO":
                self.record_status.set("PROCESSING AUDIO")
                self._set_voice_controls(
                    start_enabled=False,
                    stop_enabled=False,
                )
            elif status in {
                "MICROPHONE ERROR",
                "MICROPHONE STARTUP TIMEOUT",
                "AUDIO WORKER ERROR",
                "VOICE CAPTURE ERROR",
                "INSUFFICIENT SPEECH",
                "VOICE CAPTURE CANCELLED",
            }:
                self._voice_embedding = None
                self._voice_validation_status = "NOT VALIDATED"
                detail = diagnostic or message.get("diagnostic") or ""
                self.record_status.set(
                    status if not detail else f"{status}: {detail}"
                )
                self.microphone_status.set("MICROPHONE DISCONNECTED")
                self.voice_quality.set("Voice not validated")
                self._set_voice_controls(
                    start_enabled=False,
                    stop_enabled=False,
                )
                self._refresh_register_button()

    def _refresh_register_button(self) -> None:
        valid_details = bool(
            self.full_name.get().strip()
            and self.enrollment_number.get().strip()
        )
        ready = (
            valid_details
            and len(self._captured_embeddings) == 3
            and self._voice_embedding_is_valid()
            and self.storage.enrollment_number_available(
                self.enrollment_number.get()
            )
        )
        self.register_button.configure(state="normal" if ready else "disabled")
        if ready:
            self.registration_status.set(
                "READY TO REGISTER - face and voice complete"
            )
        else:
            self.registration_status.set(
                "THREE FRAMES REQUIRED; valid voice required"
            )

    def _voice_embedding_is_valid(self) -> bool:
        if (
            self._voice_embedding is None
            or self._voice_duration_seconds is None
            or self._voice_validation_status != "VALID SPEECH AND ECAPA EMBEDDING"
        ):
            return False
        embedding = self._voice_embedding
        return bool(
            embedding.ndim == 1
            and embedding.size > 0
            and np.isfinite(embedding).all()
            and float(np.linalg.norm(embedding)) > 0.0
        )

    def register_student(self) -> None:
        if len(self._captured_embeddings) != 3 or self._voice_embedding is None:
            self.registration_status.set("THREE FRAMES AND VOICE ARE REQUIRED")
            return
        try:
            service = StudentRegistrationService(self.storage)
            result = service.register_student(
                full_name=self.full_name.get(),
                enrollment_number=self.enrollment_number.get(),
                email=self.email.get() or None,
                face_embeddings=self._captured_embeddings,
                voice_embedding=self._voice_embedding,
            )
        except Exception as exc:
            self.registration_status.set(f"REGISTRATION FAILED: {exc}")
            return
        loaded = self.storage.load_student(result["student_id"])
        if loaded is None or not self.storage.is_fully_registered(result["student_id"]):
            self.registration_status.set("REGISTRATION FAILED: reload verification failed")
            return
        self.registration_status.set("STUDENT REGISTERED SUCCESSFULLY")
        self._captured_embeddings = []
        self._captured_frames = []
        self._voice_embedding = None
        self._voice_duration_seconds = None
        self._voice_validation_status = "NOT VALIDATED"
        self.capture_progress.set("0/3")
        for preview in self.preview_labels:
            preview.configure(image="", text="Captured frame")
        self.full_name.set("")
        self.enrollment_number.set("")
        self.email.set("")
        self.refresh_students()

    def refresh_students(self) -> None:
        for item in self.student_table.get_children():
            self.student_table.delete(item)
        search = self.student_search.get().strip().casefold()
        for student in self.storage.list_students():
            searchable = " ".join(
                str(student.get(key) or "")
                for key in ("full_name", "enrollment_number", "student_id")
            ).casefold()
            if search and search not in searchable:
                continue
            self.student_table.insert(
                "",
                "end",
                iid=student["student_id"],
                values=(
                    student["full_name"],
                    student["enrollment_number"] or "",
                    student["registration_status"],
                    student["face_registration_status"],
                    student["voice_registration_status"],
                    student["created_at"],
                ),
            )

    def start_exam(self) -> None:
        selection = self.student_table.selection()
        if not selection:
            messagebox.showerror("Start Exam", "Select a registered student.")
            return
        student_id = selection[0]
        if not self.storage.is_fully_registered(student_id):
            messagebox.showerror(
                "Start Exam",
                "This student is not fully registered with three face frames "
                "and one voice embedding.",
            )
            return
        student = self.storage.load_student(student_id)
        if student is None:
            messagebox.showerror("Start Exam", "Student record could not be loaded.")
            return
        audio_python, interpreter_error = _resolve_audio_python()
        if not audio_python:
            messagebox.showerror(
                "Start Exam",
                "Audio interpreter validation failed. "
                f"{interpreter_error}\n\n"
                "Install the mandatory audio dependencies in the isolated "
                "audio environment or set EXAM_AUDIO_PYTHON to its Python "
                "executable.",
            )
            return
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        session = self.storage.create_exam_session(student_id, timestamp)
        command = [
            sys.executable,
            "-m",
            "integration.live_demo",
            "--candidate-id",
            student_id,
            "--session-id",
            session["session_id"],
        ]
        command.extend(["--audio-python", audio_python])
        try:
            subprocess.Popen(command, cwd=str(CANDIDATE_DATABASE_PATH.parents[1]))
        except Exception as exc:
            messagebox.showerror("Start Exam", f"Could not start monitor: {exc}")
            return


def main() -> int:
    root = tk.Tk()
    app = ExamEngineApp(root)
    app._start_camera()
    app.refresh_students()
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
