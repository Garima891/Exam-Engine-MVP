from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from integration.biometric_crypto import (
    ENCRYPTED_BIOMETRIC_BLOB_PREFIX,
    WindowsDpapiProtector,
    open_biometric_payload,
)
from integration.config import (
    CANDIDATE_DATABASE_PATH,
    CAMERA_INDEX,
    FACE_IDENTITY_THRESHOLD,
    REPOSITORY_ROOT,
)
from integration.candidate_storage import CandidateStorage
from integration.module_adapters.face_identity import FaceIdentityAdapter
from integration.module_adapters.phone import PhoneDetectionAdapter
from integration.proctoring_events import ProctoringEventEngine


MESSAGE_PREFIX = "PROCTOR_JSON:"
_console_lock = threading.Lock()
EVENT_HISTORY_PATH = REPOSITORY_ROOT / "integration" / "proctoring_event_history.jsonl"
AUDIO_WINDOW_STALL_TIMEOUT_SECONDS = 10.0
AUDIO_CALLBACK_STALL_TIMEOUT_SECONDS = 2.0
TEST_SCENARIOS = {
    "1": (
        "1/10 CANDIDATE FACE MATCH",
        "Show the registered face to the camera.",
    ),
    "2": (
        "2/10 DIFFERENT PERSON",
        "A consenting second person takes the camera position.",
    ),
    "3": (
        "3/10 NO FACE",
        "Move out of the camera view.",
    ),
    "4": (
        "4/10 PHONE VISIBLE",
        "Show a phone to the camera.",
    ),
    "5": (
        "5/10 REGISTERED CANDIDATE VOICE",
        "Only compare when a voice reference is explicitly linked to this candidate.",
    ),
    "6": (
        "6/10 UNKNOWN SPEAKER",
        "Only verify speaker identity when an explicitly linked voice reference is configured.",
    ),
    "7": (
        "7/10 SILENCE",
        "Remain silent; wait for several audio windows.",
    ),
    "8": (
        "8/10 PHONE RINGTONE",
        "Play a ringtone near the microphone without speaking.",
    ),
    "9": (
        "9/10 BACKGROUND NOISE",
        "Make ordinary non-speech background noise.",
    ),
    "0": (
        "10/10 INDEPENDENT EVENTS",
        "Show a phone while a different person speaks or plays a ringtone.",
    ),
}


def _print(message: str) -> None:
    with _console_lock:
        print(message, flush=True)


def _print_event(event: dict[str, Any]) -> None:
    _print("EVENT " + json.dumps(event, allow_nan=False, separators=(",", ":")))


def load_candidate_read_only(
    database_path: str | Path = CANDIDATE_DATABASE_PATH,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    """Load one registered embedding without initializing or writing the DB."""
    database_path = Path(database_path).resolve()
    if not database_path.is_file():
        raise FileNotFoundError(f"Candidate database not found: {database_path}")

    connection = sqlite3.connect(
        f"{database_path.as_uri()}?mode=ro",
        uri=True,
    )
    try:
        if candidate_id is None:
            rows = connection.execute(
                "SELECT candidate_id FROM candidates "
                "WHERE status = 'REGISTERED' ORDER BY candidate_id"
            ).fetchall()
            if len(rows) != 1:
                raise ValueError(
                    "Specify --candidate-id when the database does not contain "
                    "exactly one registered candidate."
                )
            candidate_id = rows[0][0]

        student_row = None
        if connection.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type='table' AND name='students'"
        ).fetchone():
            student_row = connection.execute(
                """
                SELECT registration_status, face_registration_status,
                       voice_registration_status
                FROM students WHERE student_id = ?
                """,
                (candidate_id,),
            ).fetchone()
        voice_row = None
        if connection.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type='table' AND name='student_voice_embeddings'"
        ).fetchone():
            voice_row = connection.execute(
                """
                SELECT embedding_blob, embedding_dimension
                FROM student_voice_embeddings WHERE student_id = ?
                """,
                (candidate_id,),
            ).fetchone()
        candidate = connection.execute(
            "SELECT candidate_id, status FROM candidates WHERE candidate_id = ?",
            (candidate_id,),
        ).fetchone()
        if candidate is None or candidate[1] != "REGISTERED":
            raise ValueError(
                f"Registered candidate was not found: {candidate_id}"
            )
        rows = connection.execute(
            """
            SELECT embedding_blob, embedding_dimension, landmarks_available
            FROM candidate_embeddings
            WHERE candidate_id = ?
            ORDER BY embedding_index
            """,
            (candidate_id,),
        ).fetchall()
    finally:
        connection.close()

    if not rows:
        raise ValueError(f"Candidate record has no embeddings: {candidate_id}")
    if student_row is not None:
        if (
            student_row[0] != "REGISTERED"
            or student_row[1] != "COMPLETE"
            or student_row[2] != "COMPLETE"
            or len(rows) != 3
            or voice_row is None
        ):
            raise ValueError(
                "Student is not fully registered; start is not allowed "
                "until three face frames and a voice embedding are enrolled."
            )
    embeddings = []
    protector = None
    for blob, dimension, landmarks_available in rows:
        if dimension != FaceIdentityAdapter.EMBEDDING_DIMENSION:
            raise ValueError("Candidate database contains an invalid embedding dimension.")
        expected_bytes = FaceIdentityAdapter.EMBEDDING_DIMENSION * np.dtype(
            np.float32
        ).itemsize
        if len(blob) != expected_bytes:
            if not blob.startswith(ENCRYPTED_BIOMETRIC_BLOB_PREFIX):
                raise ValueError("Candidate database contains a malformed embedding.")
        if not landmarks_available:
            raise ValueError("Candidate embedding has no valid landmarks.")
        if blob.startswith(ENCRYPTED_BIOMETRIC_BLOB_PREFIX):
            if protector is None:
                protector = WindowsDpapiProtector()
        plaintext = open_biometric_payload(
            blob,
            "face_embedding",
            protector,
            allow_legacy_plaintext=True,
        )
        if len(plaintext) != expected_bytes:
            raise ValueError("Candidate database contains a malformed embedding.")
        embeddings.append(
            FaceIdentityAdapter.validate_embedding(
                np.frombuffer(plaintext, dtype=np.float32).copy()
            )
        )

    voice_embedding = None
    if voice_row is not None:
        voice_values = np.frombuffer(
            open_biometric_payload(
                voice_row[0],
                "voice_embedding",
                WindowsDpapiProtector(),
            ),
            dtype=np.float32,
        ).copy()
        if (
            int(voice_row[1]) != voice_values.size
            or not np.isfinite(voice_values).all()
            or float(np.linalg.norm(voice_values)) == 0.0
        ):
            raise ValueError("Candidate database contains invalid voice embedding.")
        voice_embedding = (
            voice_values
            / float(np.linalg.norm(voice_values))
        ).astype(np.float32, copy=False)

    return {
        "candidate_id": candidate_id,
        "embedding": embeddings[0],
        "reference_embeddings": embeddings,
        "embedding_count": len(embeddings),
        "voice_embedding": voice_embedding,
    }


def _phone_output(result: Any) -> tuple[bool, bool]:
    if not isinstance(result, dict):
        return False, False
    detected = result.get("detected")
    detections = result.get("detections")
    detection_count = result.get("detection_count")
    if (
        result.get("module") != "phone_detection"
        or not isinstance(detected, bool)
        or not isinstance(detections, list)
        or isinstance(detection_count, bool)
        or not isinstance(detection_count, int)
        or detection_count != len(detections)
        or detected != bool(detections)
    ):
        return False, False
    for detection in detections:
        if (
            not isinstance(detection, dict)
            or not isinstance(detection.get("class_name"), str)
            or not _finite_number(detection.get("confidence"))
            or not isinstance(detection.get("bbox_xyxy"), list)
            or len(detection["bbox_xyxy"]) != 4
            or not all(_finite_number(value) for value in detection["bbox_xyxy"])
        ):
            return False, False
    return True, detected


def _face_output(result: Any) -> tuple[bool, bool, bool]:
    if not isinstance(result, dict):
        return False, False, False
    faces = result.get("face_results")
    total_faces = result.get("total_faces")
    identity_status = result.get("identity_status")
    if (
        not isinstance(faces, list)
        or isinstance(total_faces, bool)
        or not isinstance(total_faces, int)
        or total_faces != len(faces)
        or result.get("monitoring_status")
        not in {
            "NO_FACE_DETECTED",
            "SINGLE_FACE_DETECTED",
            "MULTIPLE_FACES_DETECTED",
        }
        or identity_status
        not in {
            "NOT_VERIFIED",
            "CANDIDATE_IDENTIFIED",
            "CANDIDATE_NOT_FOUND",
            "MULTIPLE_MATCHES_FOUND",
            "REVIEW_REQUIRED",
        }
    ):
        return False, False, False
    if total_faces == 0:
        return (
            result.get("monitoring_status") == "NO_FACE_DETECTED"
            and identity_status == "NOT_VERIFIED",
            False,
            True,
        )
    expected_monitoring_status = (
        "SINGLE_FACE_DETECTED" if total_faces == 1 else "MULTIPLE_FACES_DETECTED"
    )
    if result["monitoring_status"] != expected_monitoring_status:
        return False, False, False
    for face in faces:
        if not isinstance(face, dict):
            return False, False, False
        predicted = face.get("predicted")
        if predicted not in {"MATCH", "MISMATCH", "REVIEW_REQUIRED"}:
            return False, False, False
        if predicted in {"MATCH", "MISMATCH"} and not _finite_number(
            face.get("similarity")
        ):
            return False, False, False
        box = face.get("box")
        if box is not None and (
            not isinstance(box, list)
            or len(box) != 4
            or not all(_finite_number(value) for value in box)
        ):
            return False, False, False
    matching_count = sum(face["predicted"] == "MATCH" for face in faces)
    has_review_required = any(
        face["predicted"] == "REVIEW_REQUIRED" for face in faces
    )
    expected_identity_status = (
        "REVIEW_REQUIRED"
        if has_review_required and matching_count == 0
        else "CANDIDATE_NOT_FOUND"
        if matching_count == 0
        else "CANDIDATE_IDENTIFIED"
        if matching_count == 1
        else "MULTIPLE_MATCHES_FOUND"
    )
    if identity_status != expected_identity_status:
        return False, False, False
    mismatch = any(face["predicted"] == "MISMATCH" for face in faces)
    return True, mismatch, False


def _identity_display_status(face_result: dict[str, Any]) -> str:
    faces = face_result["face_results"]
    if not faces:
        return "NOT AVAILABLE"
    predictions = [face["predicted"] for face in faces]
    has_match = "MATCH" in predictions
    has_mismatch = "MISMATCH" in predictions
    has_review_required = "REVIEW_REQUIRED" in predictions
    if has_match and has_mismatch:
        return "REGISTERED CANDIDATE MATCHED; OTHER FACE UNMATCHED"
    if face_result["identity_status"] == "MULTIPLE_MATCHES_FOUND":
        return "MULTIPLE MATCHES FOUND"
    if has_match and has_review_required:
        return "REGISTERED CANDIDATE MATCHED; OTHER FACE REVIEW REQUIRED"
    if has_match:
        return "REGISTERED CANDIDATE MATCHED"
    if has_review_required:
        return "IDENTITY REVIEW REQUIRED"
    return "CANDIDATE NOT IDENTIFIED / UNMATCHED"


def analyze_vision_frame(
    frame: Any,
    frame_number: int,
    phone_adapter: Any,
    face_adapter: Any,
    registered_candidate: dict[str, Any],
    event_engine: ProctoringEventEngine,
    *,
    event_logger: Callable[[dict[str, Any]], None] = _print_event,
) -> dict[str, Any]:
    """Run both vision adapters on the identical frame and log valid evidence."""
    frame_id = f"frame:{frame_number}"
    try:
        phone_result = phone_adapter.detect(frame)
        phone_valid, phone_detected = _phone_output(phone_result)
        if not phone_valid:
            raise ValueError("Phone adapter returned an invalid result schema.")
        if phone_detected:
            event = event_engine.observe(
                "PHONE_DETECTED",
                frame_id,
                {
                    "detection_count": phone_result["detection_count"],
                    "detections": phone_result["detections"],
                },
            )
            if event is not None:
                event_logger(event)
        phone_status = "PHONE DETECTED" if phone_detected else "NO PHONE DETECTED"
    except Exception as exc:
        phone_result = None
        phone_status = f"ERROR: {type(exc).__name__}: {exc}"
        _print(f"PHONE MODULE ERROR | frame={frame_number} | {phone_status}")

    try:
        face_result = face_adapter.verify_registered_identity(
            registered_candidate,
            frame,
        )
        face_valid, identity_unmatched, no_face = _face_output(face_result)
        if not face_valid:
            raise ValueError("Face adapter returned an invalid result schema.")
        if no_face:
            event = event_engine.observe(
                "NO_FACE_DETECTED",
                frame_id,
                {
                    "monitoring_status": face_result["monitoring_status"],
                    "identity_status": face_result["identity_status"],
                    "total_faces": face_result["total_faces"],
                },
            )
            if event is not None:
                event_logger(event)
        elif identity_unmatched:
            event = event_engine.observe(
                "IDENTITY_UNMATCHED",
                frame_id,
                {
                    "face_results": [
                        {
                            "face_number": face.get("face_number"),
                            "predicted": face.get("predicted"),
                            "similarity": face.get("similarity"),
                            "box": face.get("box"),
                        }
                        for face in face_result["face_results"]
                        if face.get("predicted") == "MISMATCH"
                    ],
                    "identity_threshold": FACE_IDENTITY_THRESHOLD,
                },
            )
            if event is not None:
                event_logger(event)
        face_detection_status = (
            "FACE DETECTED" if face_result["total_faces"] else "NO FACE DETECTED"
        )
        identity_display_status = _identity_display_status(face_result)
        face_status = face_result["identity_status"]
    except Exception as exc:
        face_result = None
        face_status = f"ERROR: {type(exc).__name__}: {exc}"
        face_detection_status = face_status
        identity_display_status = face_status
        _print(f"FACE MODULE ERROR | frame={frame_number} | {face_status}")

    return {
        "frame_number": frame_number,
        "phone_result": phone_result,
        "phone_status": phone_status,
        "face_result": face_result,
        "face_status": face_status,
        "face_detection_status": face_detection_status,
        "candidate_identity_status": identity_display_status,
    }


def _pending_vision() -> dict[str, Any]:
    return {
        "frame_number": None,
        "phone_result": None,
        "phone_status": "WAITING FOR RESULT",
        "face_result": None,
        "face_status": "WAITING FOR RESULT",
        "face_detection_status": "WAITING FOR RESULT",
        "candidate_identity_status": "WAITING FOR RESULT",
    }


def _vision_failure(frame_number: int, error: Exception) -> dict[str, Any]:
    message = f"ERROR: {type(error).__name__}: {error}"
    _print(f"VISION WORKER ERROR | frame={frame_number} | {message}")
    return {
        "frame_number": frame_number,
        "phone_result": None,
        "phone_status": message,
        "face_result": None,
        "face_status": message,
        "face_detection_status": message,
        "candidate_identity_status": message,
    }


def _vision_result_summary(vision: dict[str, Any]) -> str:
    face_result = vision.get("face_result")
    similarities = (
        [
            f"F{item.get('face_number', '?')}={item['similarity']:.3f}"
            for item in face_result.get("face_results", [])
            if isinstance(item, dict)
            and _finite_number(item.get("similarity"))
        ]
        if isinstance(face_result, dict)
        else []
    )
    phone_result = vision.get("phone_result")
    confidences = (
        [
            f"{item['confidence']:.3f}"
            for item in phone_result.get("detections", [])
            if isinstance(item, dict)
            and _finite_number(item.get("confidence"))
        ]
        if isinstance(phone_result, dict)
        else []
    )
    return (
        f"VISION FRAME {vision.get('frame_number')} | "
        f"phone={vision.get('phone_status')} "
        f"confidence={','.join(confidences) or 'n/a'} | "
        f"face={vision.get('face_detection_status')} | "
        f"identity={vision.get('candidate_identity_status')} "
        f"similarity={','.join(similarities) or 'n/a'}"
    )


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def process_audio_result(
    window_id: Any,
    result: Any,
    event_engine: ProctoringEventEngine,
    *,
    event_logger: Callable[[dict[str, Any]], None] = _print_event,
) -> dict[str, Any]:
    """Validate one complete worker window and record supported audio events."""
    if (
        isinstance(window_id, bool)
        or not isinstance(window_id, int)
        or window_id <= 0
        or not isinstance(result, dict)
        or result.get("module") != "audio_detection"
    ):
        return {"status": "REVIEW REQUIRED", "error": "Invalid audio result envelope."}

    vad = result.get("vad")
    speaker = result.get("speaker")
    phone_sound = result.get("phone_sound")
    sound_detected = (
        None if not isinstance(phone_sound, dict) else phone_sound.get("detected")
    )
    if (
        not isinstance(vad, dict)
        or not isinstance(vad.get("speech_detected"), bool)
        or not _finite_number(vad.get("speech_duration_seconds"))
        or not isinstance(speaker, dict)
        or not isinstance(speaker.get("status"), str)
        or not isinstance(phone_sound, dict)
        or (sound_detected is not True and sound_detected is not False and sound_detected is not None)
    ):
        return {"status": "REVIEW REQUIRED", "error": "Invalid audio result schema."}

    speaker_status = speaker["status"]
    if speaker_status not in {
        "no_speech",
        "insufficient_speech",
        "reference_not_configured",
        "compared",
    }:
        return {"status": "REVIEW REQUIRED", "error": "Unknown speaker result status."}
    if speaker_status == "compared":
        if (
            speaker.get("comparison")
            not in {"low_similarity", "high_similarity", "ambiguous_similarity"}
            or not _finite_number(speaker.get("similarity"))
        ):
            return {"status": "REVIEW REQUIRED", "error": "Invalid speaker comparison."}
    elif speaker.get("similarity") is not None or speaker.get("comparison") is not None:
        return {"status": "REVIEW REQUIRED", "error": "Uncompared speaker result has a score."}

    sound_status = phone_sound.get("status")
    if sound_status == "scored":
        if sound_detected is None or not _finite_number(phone_sound.get("score")):
            return {"status": "REVIEW REQUIRED", "error": "Invalid phone-sound score."}
    elif sound_status == "insufficient_audio":
        if sound_detected is not None or phone_sound.get("score") is not None:
            return {"status": "REVIEW REQUIRED", "error": "Invalid incomplete phone-sound result."}
    else:
        return {"status": "REVIEW REQUIRED", "error": "Unknown phone-sound result status."}

    window_key = f"audio-window:{window_id}"
    comparison = speaker.get("comparison")
    similarity = speaker.get("similarity")
    if (
        vad["speech_detected"]
        and speaker["status"] == "compared"
        and comparison == "low_similarity"
        and _finite_number(similarity)
    ):
        event = event_engine.observe(
            "UNKNOWN_HUMAN_VOICE",
            window_key,
            {
                "speech_detected": True,
                "speech_duration_seconds": vad.get("speech_duration_seconds"),
                "comparison": comparison,
                "similarity": float(similarity),
                "thresholds": speaker.get("thresholds"),
            },
        )
        if event is not None:
            event_logger(event)

    score = phone_sound.get("score")
    if (
        phone_sound.get("status") == "scored"
        and phone_sound.get("detected") is True
        and _finite_number(score)
    ):
        event = event_engine.observe(
            "SUSPICIOUS_AUDIO_SOUND",
            window_key,
            {
                "detected": True,
                "score": float(score),
                "threshold": phone_sound.get("threshold"),
                "high_confidence": phone_sound.get("high_confidence"),
            },
        )
        if event is not None:
            event_logger(event)

    return {"status": "VALID", "result": result}


class AudioWorkerState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status = "AUDIO WORKER STARTING"
        self._input_rms: float | None = None
        self._input_min: float | None = None
        self._input_max: float | None = None
        self._window_rms: float | None = None
        self._window_peak: float | None = None
        self._window_min: float | None = None
        self._window_max: float | None = None
        self._callback_count = 0
        self._callback_age_seconds: float | None = None
        self._queue_size: int | None = None
        self._queue_capacity: int | None = None
        self._selected_device_index: int | None = None
        self._selected_device_name: str | None = None
        self._last_window_monotonic: float | None = None
        self._last_valid_window_monotonic: float | None = None
        self._last_window_timestamp: str | None = None
        self._last_valid_window_timestamp: str | None = None
        self._last_speech_timestamp: str | None = None
        self._connected_since_monotonic: float | None = None
        self._result: dict[str, Any] | None = None
        self._error: str | None = None
        self._warning: str | None = None
        self._discontinuities = 0

    @staticmethod
    def _audio_unavailable_status(status: str) -> bool:
        return status in {
            "MICROPHONE RECOVERING",
            "MICROPHONE ERROR",
            "AUDIO WORKER ERROR",
            "MICROPHONE OFF",
        }

    def _clear_live_audio_metrics(self) -> None:
        self._input_rms = None
        self._input_min = None
        self._input_max = None
        self._window_rms = None
        self._window_peak = None
        self._window_min = None
        self._window_max = None
        self._callback_count = 0
        self._callback_age_seconds = None
        self._queue_size = None
        self._queue_capacity = None

    def apply(self, message: dict[str, Any], event_engine: ProctoringEventEngine) -> None:
        message_type = message.get("type")
        if message_type == "audio_result":
            self._apply_audio_result(message, event_engine)
            return

        log_messages: list[str] = []
        with self._lock:
            if message_type == "status":
                previous_status = self._status
                self._status = str(message.get("status", "UNKNOWN"))
                self._apply_device_info(message)
                if self._audio_unavailable_status(self._status):
                    self._clear_live_audio_metrics()
                    self._result = None
                    self._last_window_monotonic = None
                    self._last_valid_window_monotonic = None
                    self._last_window_timestamp = None
                    self._last_valid_window_timestamp = None
                    self._connected_since_monotonic = None
                    if self._status == "MICROPHONE RECOVERING":
                        self._discontinuities = 0
                elif self._status == "MICROPHONE ON":
                    if previous_status != "MICROPHONE ON":
                        self._connected_since_monotonic = time.monotonic()
                    self._error = None
                    self._warning = None
                    self._discontinuities = 0
                    if self._selected_device_name:
                        log_messages.append(
                            "AUDIO INPUT DEVICE | "
                            f"index={self._selected_device_index} "
                            f"name={self._selected_device_name}"
                        )
                log_messages.append(f"AUDIO STATUS | {self._status}")
            elif message_type == "telemetry":
                previous_status = self._status
                self._status = str(message.get("status", "UNKNOWN"))
                self._apply_device_info(message)
                if self._status == "MICROPHONE ON":
                    if previous_status != "MICROPHONE ON":
                        self._connected_since_monotonic = time.monotonic()
                else:
                    self._connected_since_monotonic = None
                rms = message.get("input_rms")
                if self._status == "MICROPHONE ON" and _finite_number(rms):
                    self._input_rms = float(rms)
                self._input_min = (
                    float(message["input_min"])
                    if _finite_number(message.get("input_min"))
                    else None
                )
                self._input_max = (
                    float(message["input_max"])
                    if _finite_number(message.get("input_max"))
                    else None
                )
                self._window_rms = (
                    float(message["last_window_rms"])
                    if _finite_number(message.get("last_window_rms"))
                    else self._window_rms
                )
                self._window_peak = (
                    float(message["last_window_peak"])
                    if _finite_number(message.get("last_window_peak"))
                    else self._window_peak
                )
                self._window_min = (
                    float(message["last_window_min"])
                    if _finite_number(message.get("last_window_min"))
                    else self._window_min
                )
                self._window_max = (
                    float(message["last_window_max"])
                    if _finite_number(message.get("last_window_max"))
                    else self._window_max
                )
                callback_count = message.get("callback_count")
                if (
                    isinstance(callback_count, int)
                    and not isinstance(callback_count, bool)
                    and callback_count >= 0
                ):
                    self._callback_count = callback_count
                callback_age = message.get("callback_age_seconds")
                self._callback_age_seconds = (
                    float(callback_age)
                    if _finite_number(callback_age)
                    else None
                )
                queue_size = message.get("queue_size")
                queue_capacity = message.get("queue_capacity")
                if (
                    isinstance(queue_size, int)
                    and not isinstance(queue_size, bool)
                    and isinstance(queue_capacity, int)
                    and not isinstance(queue_capacity, bool)
                ):
                    self._queue_size = queue_size
                    self._queue_capacity = queue_capacity
                window_age = message.get("last_window_age_seconds")
                if _finite_number(window_age):
                    self._last_window_monotonic = (
                        time.monotonic() - float(window_age)
                    )
                discontinuities = message.get("discontinuities", 0)
                if (
                    isinstance(discontinuities, int)
                    and not isinstance(discontinuities, bool)
                    and discontinuities > self._discontinuities
                ):
                    self._result = None
                    self._last_window_monotonic = None
                    self._last_valid_window_monotonic = None
                    self._last_window_timestamp = None
                    self._last_valid_window_timestamp = None
                    self._window_rms = None
                    self._window_peak = None
                    self._window_min = None
                    self._window_max = None
                    self._connected_since_monotonic = time.monotonic()
                    self._warning = "AUDIO GAP RECOVERED"
                if isinstance(discontinuities, int) and not isinstance(
                    discontinuities, bool
                ):
                    self._discontinuities = discontinuities
            elif message_type == "warning":
                self._warning = str(
                    message.get("status", "AUDIO WARNING")
                )
                discontinuities = message.get("discontinuities")
                if (
                    isinstance(discontinuities, int)
                    and not isinstance(discontinuities, bool)
                ):
                    if discontinuities > self._discontinuities:
                        self._result = None
                        self._last_window_monotonic = None
                        self._last_valid_window_monotonic = None
                        self._last_window_timestamp = None
                        self._last_valid_window_timestamp = None
                        self._window_rms = None
                        self._window_peak = None
                        self._window_min = None
                        self._window_max = None
                        self._connected_since_monotonic = time.monotonic()
                    self._discontinuities = discontinuities
            elif message_type == "error":
                self._status = str(message.get("status", "MICROPHONE ERROR"))
                self._error = str(message.get("error", "Unknown audio worker error."))
                self._clear_live_audio_metrics()
                self._result = None
                self._last_window_monotonic = None
                self._last_valid_window_monotonic = None
                self._last_window_timestamp = None
                self._last_valid_window_timestamp = None
                self._connected_since_monotonic = None
                log_messages.append(f"AUDIO MODULE ERROR | {self._error}")
        for log_message in log_messages:
            _print(log_message)

    def _apply_audio_result(
        self,
        message: dict[str, Any],
        event_engine: ProctoringEventEngine,
    ) -> None:
        with self._lock:
            if self._status != "MICROPHONE ON":
                return
            self._apply_device_info(message)
            rms = message.get("input_rms")
            if _finite_number(rms):
                self._input_rms = float(rms)
            input_min = message.get("input_min")
            if _finite_number(input_min):
                self._input_min = float(input_min)
            input_max = message.get("input_max")
            if _finite_number(input_max):
                self._input_max = float(input_max)
            window_rms = message.get("window_rms")
            if _finite_number(window_rms):
                self._window_rms = float(window_rms)
            window_peak = message.get("window_peak")
            if _finite_number(window_peak):
                self._window_peak = float(window_peak)
            window_min = message.get("window_min")
            if _finite_number(window_min):
                self._window_min = float(window_min)
            window_max = message.get("window_max")
            if _finite_number(window_max):
                self._window_max = float(window_max)
            callback_count = message.get("callback_count")
            if (
                isinstance(callback_count, int)
                and not isinstance(callback_count, bool)
                and callback_count >= 0
            ):
                self._callback_count = callback_count
            queue_size = message.get("queue_size")
            queue_capacity = message.get("queue_capacity")
            if (
                isinstance(queue_size, int)
                and not isinstance(queue_size, bool)
                and isinstance(queue_capacity, int)
                and not isinstance(queue_capacity, bool)
            ):
                self._queue_size = queue_size
                self._queue_capacity = queue_capacity
            received_at = time.monotonic()
            self._last_window_monotonic = received_at
            timestamp = message.get("timestamp")
            self._last_window_timestamp = (
                timestamp if isinstance(timestamp, str) else None
            )

        try:
            analysis = process_audio_result(
                message.get("window_id"),
                message.get("result"),
                event_engine,
            )
        except Exception as exc:
            analysis = {
                "status": "REVIEW REQUIRED",
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }
            _print(f"AUDIO EVENT PROCESSING ERROR | {analysis['error']}")
        with self._lock:
            if self._status != "MICROPHONE ON":
                return
            if analysis["status"] == "VALID":
                self._last_valid_window_monotonic = received_at
                self._last_valid_window_timestamp = message.get("timestamp")
                vad = analysis["result"]["vad"]
                if vad["speech_detected"]:
                    self._last_speech_timestamp = message.get("timestamp")
            self._result = {
                "window_id": message.get("window_id"),
                "window_start_sample": message.get("window_start_sample"),
                "timestamp": message.get("timestamp"),
                "analysis": analysis,
            }
            log_values = {
                "input_rms": self._input_rms,
                "input_min": self._input_min,
                "input_max": self._input_max,
                "window_rms": self._window_rms,
                "window_peak": self._window_peak,
                "window_min": self._window_min,
                "window_max": self._window_max,
                "device_name": self._selected_device_name,
                "callback_count": self._callback_count,
                "queue_size": self._queue_size,
                "queue_capacity": self._queue_capacity,
            }

        if analysis["status"] == "VALID":
            result = analysis["result"]
            speaker = result["speaker"]
            sound = result["phone_sound"]
            _print(
                f"AUDIO WINDOW {message.get('window_id')} "
                f"timestamp={message.get('timestamp')} | "
                f"VAD={'SPEECH' if result['vad']['speech_detected'] else 'NO SPEECH'} | "
                f"speech_duration={result['vad']['speech_duration_seconds']:.3f}s | "
                f"speaker={speaker['status']}/{speaker.get('comparison')} | "
                f"similarity={speaker.get('similarity')} | "
                f"input_rms={log_values['input_rms']} "
                f"sample_range={log_values['input_min']}:{log_values['input_max']} | "
                f"window_rms={log_values['window_rms']} "
                f"window_peak={log_values['window_peak']} "
                f"window_range={log_values['window_min']}:"
                f"{log_values['window_max']} "
                f"device={log_values['device_name']} "
                f"callbacks={log_values['callback_count']} "
                f"queue={log_values['queue_size']}/"
                f"{log_values['queue_capacity']} | "
                f"phone_sound={sound['status']}/"
                f"{sound.get('detected')} score={sound.get('score')}"
            )
        else:
            _print(
                "AUDIO ANALYSIS REVIEW REQUIRED | "
                + analysis.get("error", "Invalid result.")
            )

    def mark_worker_exit(self, return_code: int | None) -> None:
        with self._lock:
            if self._status not in {
                "MICROPHONE OFF",
                "MICROPHONE ERROR",
                "AUDIO WORKER ERROR",
            }:
                self._status = "AUDIO WORKER ERROR"
                self._error = f"Audio worker exited unexpectedly ({return_code})."
                self._clear_live_audio_metrics()
                self._result = None
                self._connected_since_monotonic = None

    def mark_startup_timeout(self, timeout: float) -> None:
        with self._lock:
            self._status = "AUDIO WORKER ERROR"
            self._error = (
                f"Audio worker did not become ready within {timeout:.0f} seconds."
            )
            self._clear_live_audio_metrics()
            self._result = None
            self._connected_since_monotonic = None

    def mark_worker_start_error(self, error: Exception) -> None:
        with self._lock:
            self._status = "AUDIO WORKER ERROR"
            self._error = f"{type(error).__name__}: {error}"
            self._clear_live_audio_metrics()
            self._result = None
            self._connected_since_monotonic = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            result = None if self._result is None else dict(self._result)
            now = time.monotonic()
            return {
                "status": self._status,
                "input_rms": self._input_rms,
                "input_min": self._input_min,
                "input_max": self._input_max,
                "window_rms": self._window_rms,
                "window_peak": self._window_peak,
                "window_min": self._window_min,
                "window_max": self._window_max,
                "callback_count": self._callback_count,
                "callback_age_seconds": self._callback_age_seconds,
                "queue_size": self._queue_size,
                "queue_capacity": self._queue_capacity,
                "selected_device_index": self._selected_device_index,
                "selected_device_name": self._selected_device_name,
                "last_window_age_seconds": (
                    None
                    if self._last_window_monotonic is None
                    else max(0.0, now - self._last_window_monotonic)
                ),
                "last_window_timestamp": self._last_window_timestamp,
                "last_valid_window_age_seconds": (
                    None
                    if self._last_valid_window_monotonic is None
                    else max(0.0, now - self._last_valid_window_monotonic)
                ),
                "last_valid_window_timestamp": self._last_valid_window_timestamp,
                "last_speech_timestamp": self._last_speech_timestamp,
                "connected_age_seconds": (
                    None
                    if self._connected_since_monotonic is None
                    else max(0.0, now - self._connected_since_monotonic)
                ),
                "result": result,
                "error": self._error,
                "warning": self._warning,
            }

    def _apply_device_info(self, message: dict[str, Any]) -> None:
        selected_index = message.get("selected_device_index")
        if isinstance(selected_index, int) and not isinstance(
            selected_index,
            bool,
        ):
            self._selected_device_index = selected_index
        selected_name = message.get("selected_device_name")
        if isinstance(selected_name, str) and selected_name:
            self._selected_device_name = selected_name


class AudioWorkerClient:
    def __init__(
        self,
        python_executable: str | Path,
        audio_reference: str | Path | None,
        event_engine: ProctoringEventEngine,
        state: AudioWorkerState,
        speaker_reference_base64: str | None = None,
    ) -> None:
        self.python_executable = Path(python_executable).resolve()
        self.audio_reference = (
            None if audio_reference is None else Path(audio_reference).resolve()
        )
        self.speaker_reference_base64 = speaker_reference_base64
        self.event_engine = event_engine
        self.state = state
        self.process: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None

    def start(self) -> None:
        if not self.python_executable.is_file():
            raise FileNotFoundError(
                f"Audio Python interpreter not found: {self.python_executable}"
            )
        if self.audio_reference is not None and not self.audio_reference.is_file():
            raise FileNotFoundError(
                f"Audio reference WAV not found: {self.audio_reference}"
            )
        command = [
            str(self.python_executable),
            "-u",
            "-m",
            "integration.audio_worker",
        ]
        if self.audio_reference is not None:
            command.extend(["--audio-reference", str(self.audio_reference)])
        elif self.speaker_reference_base64 is not None:
            command.extend([
                "--speaker-reference-base64",
                self.speaker_reference_base64,
            ])
        self.process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._reader = threading.Thread(
            target=self._read_output,
            name="audio-worker-output",
            daemon=True,
        )
        self._reader.start()

    def _read_output(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        for line in self.process.stdout:
            line = line.rstrip("\r\n")
            if line.startswith(MESSAGE_PREFIX):
                try:
                    message = json.loads(line[len(MESSAGE_PREFIX) :])
                except json.JSONDecodeError as exc:
                    _print(f"AUDIO WORKER PROTOCOL ERROR | {exc}")
                    continue
                if not isinstance(message, dict):
                    _print("AUDIO WORKER PROTOCOL ERROR | message is not an object")
                    continue
                self.state.apply(message, self.event_engine)
            elif line:
                with _console_lock:
                    print(f"[audio worker] {line}", file=sys.stderr, flush=True)
        self.state.mark_worker_exit(self.process.poll())

    def wait_until_ready(self, timeout: float = 600.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snapshot = self.state.snapshot()
            if snapshot["status"] == "MICROPHONE ON":
                return True
            if snapshot["status"] in {
                "MICROPHONE ERROR",
                "AUDIO WORKER ERROR",
            }:
                return False
            if self.process is not None and self.process.poll() is not None:
                self.state.mark_worker_exit(self.process.returncode)
                return False
            time.sleep(0.1)
        self.state.mark_startup_timeout(timeout)
        return False

    def stop(self) -> None:
        if self.process is None:
            return
        process = self.process
        if process.poll() is None and process.stdin is not None:
            try:
                process.stdin.write("STOP\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if self._reader is not None:
            self._reader.join(timeout=2)


_DASHBOARD_WIDTH = 1280
_DASHBOARD_HEIGHT = 760
_COLORS = {
    "background": (27, 38, 55),
    "header": (20, 32, 49),
    "card": (250, 251, 253),
    "text": (28, 43, 61),
    "muted": (107, 121, 139),
    "border": (216, 223, 231),
    "green": (51, 151, 83),
    "red": (44, 67, 207),
    "orange": (0, 145, 239),
    "grey": (116, 128, 145),
    "white": (255, 255, 255),
}


def _audio_dashboard_state(audio: dict[str, Any]) -> dict[str, Any]:
    worker_status = str(audio.get("status", "AUDIO WORKER STARTING"))
    diagnostics = {
        "device": audio.get("selected_device_name"),
        "device_index": audio.get("selected_device_index"),
        "input_min": audio.get("input_min"),
        "input_max": audio.get("input_max"),
        "window_rms": audio.get("window_rms"),
        "window_peak": audio.get("window_peak"),
        "window_min": audio.get("window_min"),
        "window_max": audio.get("window_max"),
        "callback_count": audio.get("callback_count", 0),
        "callback_age_seconds": audio.get("callback_age_seconds"),
        "queue_size": audio.get("queue_size"),
        "queue_capacity": audio.get("queue_capacity"),
        "last_speech_timestamp": audio.get("last_speech_timestamp"),
        "last_window_timestamp": audio.get("last_window_timestamp"),
        "last_window_age_seconds": audio.get("last_window_age_seconds"),
        "last_valid_window_timestamp": audio.get("last_valid_window_timestamp"),
        "last_valid_window_age_seconds": audio.get(
            "last_valid_window_age_seconds"
        ),
    }
    if worker_status == "MICROPHONE ON":
        microphone_status, microphone_color = "CONNECTED", _COLORS["green"]
    elif worker_status == "MICROPHONE RECOVERING":
        microphone_status, microphone_color = "RECONNECTING", _COLORS["orange"]
    elif worker_status in {"MICROPHONE ERROR", "AUDIO WORKER ERROR"}:
        microphone_status, microphone_color = "ERROR", _COLORS["red"]
    elif worker_status == "MICROPHONE OFF":
        microphone_status, microphone_color = "DISCONNECTED", _COLORS["red"]
    else:
        microphone_status, microphone_color = "STARTING", _COLORS["grey"]

    active = worker_status == "MICROPHONE ON"
    callback_age = audio.get("callback_age_seconds")
    if (
        active
        and _finite_number(callback_age)
        and callback_age > AUDIO_CALLBACK_STALL_TIMEOUT_SECONDS
    ):
        return {
            "microphone": "INPUT STALLED",
            "microphone_color": _COLORS["red"],
            "voice": "AUDIO UNAVAILABLE",
            "voice_color": _COLORS["red"],
            "speaker": "AUDIO UNAVAILABLE",
            "speaker_color": _COLORS["grey"],
            "sound": "AUDIO UNAVAILABLE",
            "sound_color": _COLORS["grey"],
            "rms": None,
            "voice_detail": (
                f"No microphone callback for {callback_age:.1f}s"
            ),
            "speaker_detail": "",
            "sound_detail": "",
            "warning": audio.get("warning"),
            **diagnostics,
        }
    last_result = audio.get("result") if active else None
    analysis = (
        last_result.get("analysis")
        if isinstance(last_result, dict)
        else None
    )
    raw_result = (
        analysis.get("result")
        if isinstance(analysis, dict) and analysis.get("status") == "VALID"
        else None
    )
    if not isinstance(raw_result, dict):
        if (
            active
            and isinstance(analysis, dict)
            and analysis.get("status") != "VALID"
        ):
            return {
                "microphone": microphone_status,
                "microphone_color": microphone_color,
                "voice": "AUDIO REVIEW REQUIRED",
                "voice_color": _COLORS["orange"],
                "speaker": "AUDIO REVIEW REQUIRED",
                "speaker_color": _COLORS["orange"],
                "sound": "AUDIO REVIEW REQUIRED",
                "sound_color": _COLORS["orange"],
                "rms": audio.get("input_rms"),
                "voice_detail": "",
                "speaker_detail": analysis.get("error", ""),
                "sound_detail": analysis.get("error", ""),
                "warning": audio.get("warning"),
                **diagnostics,
            }
        age = audio.get("last_valid_window_age_seconds")
        if not _finite_number(age):
            age = audio.get("connected_age_seconds")
        if (
            active
            and _finite_number(age)
            and age > AUDIO_WINDOW_STALL_TIMEOUT_SECONDS
        ):
            stalled_detail = (
                f"No complete audio window for {age:.1f}s"
            )
            return {
                "microphone": microphone_status,
                "microphone_color": microphone_color,
                "voice": "AUDIO PROCESSING STALLED",
                "voice_color": _COLORS["orange"],
                "speaker": "AUDIO PROCESSING STALLED",
                "speaker_color": _COLORS["orange"],
                "sound": "AUDIO PROCESSING STALLED",
                "sound_color": _COLORS["orange"],
                "rms": audio.get("input_rms"),
                "voice_detail": stalled_detail,
                "speaker_detail": stalled_detail,
                "sound_detail": stalled_detail,
                "warning": audio.get("warning"),
                **diagnostics,
            }
        unavailable = worker_status in {
            "MICROPHONE RECOVERING",
            "MICROPHONE ERROR",
            "AUDIO WORKER ERROR",
            "MICROPHONE OFF",
        }
        waiting = active or not unavailable
        state = "WAITING FOR AUDIO" if waiting else "AUDIO UNAVAILABLE"
        state_color = (
            _COLORS["orange"]
            if worker_status == "MICROPHONE RECOVERING"
            else _COLORS["red"]
            if unavailable
            else _COLORS["grey"]
        )
        return {
            "microphone": microphone_status,
            "microphone_color": microphone_color,
            "voice": state if active or unavailable else "AUDIO UNAVAILABLE",
            "voice_color": state_color,
            "speaker": "AUDIO UNAVAILABLE" if unavailable else "NOT AVAILABLE",
            "speaker_color": state_color if unavailable else _COLORS["grey"],
            "sound": state if active or unavailable else "AUDIO UNAVAILABLE",
            "sound_color": state_color,
            "rms": audio.get("input_rms") if active else None,
            "voice_detail": "Waiting for a complete analysis window" if active else "",
            "speaker_detail": "",
            "sound_detail": "",
            "warning": audio.get("warning"),
            **diagnostics,
        }

    age = audio.get("last_valid_window_age_seconds")
    if not _finite_number(age):
        age = audio.get("connected_age_seconds")
    if (
        active
        and _finite_number(age)
        and age > AUDIO_WINDOW_STALL_TIMEOUT_SECONDS
    ):
        stalled_detail = f"No complete audio window for {age:.1f}s"
        return {
            "microphone": microphone_status,
            "microphone_color": microphone_color,
            "voice": "AUDIO PROCESSING STALLED",
            "voice_color": _COLORS["orange"],
            "speaker": "AUDIO PROCESSING STALLED",
            "speaker_color": _COLORS["orange"],
            "sound": "AUDIO PROCESSING STALLED",
            "sound_color": _COLORS["orange"],
            "rms": audio.get("input_rms"),
            "voice_detail": stalled_detail,
            "speaker_detail": stalled_detail,
            "sound_detail": stalled_detail,
            "warning": audio.get("warning"),
            **diagnostics,
        }

    vad = raw_result["vad"]
    speaker = raw_result["speaker"]
    sound = raw_result["phone_sound"]
    voice = "SPEECH DETECTED" if vad["speech_detected"] else "NO SPEECH"
    voice_color = _COLORS["green"]
    voice_detail = (
        f"VAD speech {vad['speech_duration_seconds']:.3f}s"
        + (
            f" | window RMS {audio['window_rms']:.4f}"
            if _finite_number(audio.get("window_rms"))
            else ""
        )
    )
    if _finite_number(audio.get("window_min")) and _finite_number(
        audio.get("window_max")
    ):
        voice_detail += (
            f" | window range {audio['window_min']:.4f}:"
            f"{audio['window_max']:.4f}"
        )
    last_speech_timestamp = audio.get("last_speech_timestamp")
    if isinstance(last_speech_timestamp, str):
        voice_detail += f" | last speech {last_speech_timestamp[:19]}"
    last_window_timestamp = (
        audio.get("last_window_timestamp")
        or audio.get("last_valid_window_timestamp")
    )
    if isinstance(last_window_timestamp, str):
        voice_detail += f" | window {last_window_timestamp[11:19]}"
    last_window_age = audio.get("last_window_age_seconds")
    if _finite_number(last_window_age):
        voice_detail += f" | age {last_window_age:.1f}s"

    comparison = speaker.get("comparison")
    similarity = speaker.get("similarity")
    if speaker.get("status") == "compared":
        speaker_status, speaker_color = {
            "low_similarity": ("UNKNOWN SPEAKER", _COLORS["red"]),
            "high_similarity": ("REGISTERED VOICE MATCH", _COLORS["green"]),
            "ambiguous_similarity": ("VOICE UNCERTAIN", _COLORS["orange"]),
        }.get(comparison, ("REVIEW REQUIRED", _COLORS["orange"]))
        speaker_detail = (
            f"Similarity {similarity:.3f}"
            if _finite_number(similarity)
            else ""
        )
    elif speaker.get("status") == "reference_not_configured":
        speaker_status = "REFERENCE NOT CONFIGURED"
        speaker_color = _COLORS["grey"]
        speaker_detail = ""
    elif speaker.get("status") == "insufficient_speech":
        speaker_status = "INSUFFICIENT SPEECH"
        speaker_color = _COLORS["grey"]
        speaker_detail = ""
    else:
        speaker_status = "NOT APPLICABLE" if not vad["speech_detected"] else "REVIEW REQUIRED"
        speaker_color = _COLORS["grey"] if not vad["speech_detected"] else _COLORS["orange"]
        speaker_detail = ""

    score = sound.get("score")
    if sound.get("status") == "scored":
        sound_status = (
            "SUSPICIOUS SOUND DETECTED"
            if sound.get("detected") is True
            else "NO SUSPICIOUS SOUND"
        )
        sound_color = (
            _COLORS["orange"]
            if sound.get("detected") is True
            else _COLORS["green"]
        )
        sound_detail = (
            f"Score {score:.3f} / threshold {sound.get('threshold')}"
            if _finite_number(score)
            else ""
        )
    else:
        sound_status = "AUDIO UNAVAILABLE"
        sound_color = _COLORS["grey"]
        sound_detail = ""

    rms = audio.get("input_rms")
    return {
        "microphone": microphone_status,
        "microphone_color": microphone_color,
        "voice": voice,
        "voice_color": voice_color,
        "voice_detail": voice_detail,
        "speaker": speaker_status,
        "speaker_color": speaker_color,
        "sound": sound_status,
        "sound_color": sound_color,
        "rms": rms if _finite_number(rms) else None,
        "speaker_detail": speaker_detail,
        "sound_detail": sound_detail,
        "warning": audio.get("warning"),
        **diagnostics,
    }


def _counter_text(event_engine: ProctoringEventEngine, event_type: str) -> str:
    limit = event_engine.confirmation_observations
    count = min(event_engine.counters[event_type], limit)
    if event_type in event_engine.confirmed_events:
        return f"CONFIRMED  {count}/{limit}"
    return f"OBSERVATIONS  {count}/{limit}"


def _live_clock_text(now: datetime | None = None) -> str:
    current_time = datetime.now().astimezone() if now is None else now.astimezone()
    return current_time.strftime("%Y-%m-%d %H:%M:%S")


def _put_fitted_text(
    cv2_module: Any,
    image: Any,
    text: str,
    origin: tuple[int, int],
    max_width: int,
    color: tuple[int, int, int],
    scale: float,
    thickness: int = 1,
) -> None:
    font = cv2_module.FONT_HERSHEY_SIMPLEX
    display_text = str(text)
    fitted_scale = scale
    while display_text:
        width = cv2_module.getTextSize(
            display_text, font, fitted_scale, thickness
        )[0][0]
        if width <= max_width:
            break
        if fitted_scale > 0.30:
            fitted_scale = max(0.30, fitted_scale - 0.02)
            continue
        display_text = display_text[:-4].rstrip() + "..."
    cv2_module.putText(
        image,
        display_text,
        origin,
        font,
        fitted_scale,
        color,
        thickness,
        cv2_module.LINE_AA,
    )


def _draw_status_card(
    cv2_module: Any,
    canvas: Any,
    origin: tuple[int, int],
    size: tuple[int, int],
    title: str,
    status: str,
    status_color: tuple[int, int, int],
    detail: str = "",
    counter: str = "",
) -> None:
    x, y = origin
    width, height = size
    cv2_module.rectangle(
        canvas, (x, y), (x + width, y + height), _COLORS["card"], -1
    )
    cv2_module.rectangle(
        canvas,
        (x, y),
        (x + width, y + height),
        _COLORS["border"],
        1,
    )
    cv2_module.rectangle(canvas, (x, y), (x + 5, y + height), status_color, -1)
    _put_fitted_text(
        cv2_module,
        canvas,
        title.upper(),
        (x + 13, y + 20),
        width - 25,
        _COLORS["muted"],
        0.39,
        1,
    )
    _put_fitted_text(
        cv2_module,
        canvas,
        status,
        (x + 13, y + 43),
        width - 25,
        status_color,
        0.43,
        1,
    )
    if detail:
        _put_fitted_text(
            cv2_module,
            canvas,
            detail,
            (x + 13, y + 63),
            width - 25,
            _COLORS["text"],
            0.34,
            1,
        )
    if counter:
        _put_fitted_text(
            cv2_module,
            canvas,
            counter,
            (x + 13, y + height - 7),
            width - 25,
            _COLORS["red"]
            if counter.startswith("CONFIRMED")
            else _COLORS["muted"],
            0.32,
            1,
        )


def _draw_frame_overlays(cv2_module: Any, frame: Any, vision: dict[str, Any]) -> Any:
    preview = frame.copy()
    phone_result = vision.get("phone_result")
    if isinstance(phone_result, dict):
        for detection in phone_result.get("detections", []):
            box = detection.get("bbox_xyxy")
            if not isinstance(box, list) or len(box) != 4:
                continue
            x1, y1, x2, y2 = (int(value) for value in box)
            color = _COLORS["orange"]
            cv2_module.rectangle(preview, (x1, y1), (x2, y2), color, 2)
            cv2_module.putText(
                preview,
                f"PHONE {float(detection['confidence']):.2f}",
                (x1, max(16, y1 - 6)),
                cv2_module.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2_module.LINE_AA,
            )

    face_result = vision.get("face_result")
    if isinstance(face_result, dict):
        for face in face_result.get("face_results", []):
            box = face.get("box")
            if not isinstance(box, list) or len(box) != 4:
                continue
            predicted = face.get("predicted")
            color = {
                "MATCH": _COLORS["green"],
                "MISMATCH": _COLORS["red"],
                "REVIEW_REQUIRED": _COLORS["orange"],
            }.get(predicted, _COLORS["grey"])
            label = {
                "MATCH": "REGISTERED CANDIDATE MATCHED",
                "MISMATCH": "CANDIDATE UNMATCHED",
                "REVIEW_REQUIRED": "IDENTITY REVIEW REQUIRED",
            }.get(predicted, "IDENTITY UNAVAILABLE")
            similarity = face.get("similarity")
            if _finite_number(similarity):
                label += f" {similarity:.3f}"
            x1, y1, x2, y2 = (int(value) for value in box)
            cv2_module.rectangle(preview, (x1, y1), (x2, y2), color, 2)
            cv2_module.putText(
                preview,
                label,
                (x1, max(16, y1 - 6)),
                cv2_module.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                2,
                cv2_module.LINE_AA,
            )
    return preview


def _event_model_summary(event: dict[str, Any]) -> str:
    output = event.get("model_output", {})
    event_type = event.get("event_type")
    if event_type == "PHONE_DETECTED":
        detections = output.get("detections", [])
        confidences = [
            item.get("confidence")
            for item in detections
            if isinstance(item, dict) and _finite_number(item.get("confidence"))
        ]
        return (
            f"max confidence {max(confidences):.3f}"
            if confidences
            else "phone detected"
        )
    if event_type == "IDENTITY_UNMATCHED":
        faces = output.get("face_results", [])
        similarities = [
            item.get("similarity")
            for item in faces
            if isinstance(item, dict) and _finite_number(item.get("similarity"))
        ]
        return (
            "similarity " + ", ".join(f"{value:.3f}" for value in similarities)
            if similarities
            else "identity unmatched"
        )
    if event_type == "UNKNOWN_HUMAN_VOICE":
        similarity = output.get("similarity")
        return (
            f"similarity {similarity:.3f}"
            if _finite_number(similarity)
            else "speaker mismatch"
        )
    if event_type == "SUSPICIOUS_AUDIO_SOUND":
        score = output.get("score")
        return f"sound score {score:.3f}" if _finite_number(score) else "sound detected"
    return "no face detected"


def render_frame(
    cv2_module: Any,
    frame: Any,
    vision: dict[str, Any],
    audio: dict[str, Any],
    event_engine: ProctoringEventEngine,
    camera_status: str,
    scenario: tuple[str, str] | None = None,
    captured_frame_number: int | None = None,
) -> Any:
    canvas = np.full(
        (_DASHBOARD_HEIGHT, _DASHBOARD_WIDTH, 3),
        _COLORS["background"],
        dtype=np.uint8,
    )
    cv2_module.rectangle(
        canvas, (0, 0), (_DASHBOARD_WIDTH, 56), _COLORS["header"], -1
    )
    cv2_module.putText(
        canvas,
        "AI PROCTORING MONITOR",
        (18, 37),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.65,
        _COLORS["white"],
        2,
        cv2_module.LINE_AA,
    )
    cv2_module.circle(canvas, (244, 29), 5, _COLORS["green"], -1)
    cv2_module.putText(
        canvas,
        "LIVE",
        (256, 34),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.39,
        _COLORS["green"],
        1,
        cv2_module.LINE_AA,
    )
    _put_fitted_text(
        cv2_module,
        canvas,
        f"SESSION {event_engine.session_id}",
        (390, 35),
        430,
        (216, 226, 238),
        0.39,
    )
    _put_fitted_text(
        cv2_module,
        canvas,
        _live_clock_text(),
        (730, 34),
        145,
        (216, 226, 238),
        0.35,
    )
    cv2_module.circle(
        canvas,
        (887, 29),
        6,
        _COLORS["green"] if camera_status == "OPEN" else _COLORS["grey"],
        -1,
    )
    cv2_module.putText(
        canvas,
        "CAMERA CONNECTED" if camera_status == "OPEN" else "CAMERA UNAVAILABLE",
        (900, 34),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.36,
        _COLORS["white"],
        1,
        cv2_module.LINE_AA,
    )
    audio_state = _audio_dashboard_state(audio)
    cv2_module.circle(
        canvas,
        (1085, 29),
        6,
        audio_state["microphone_color"],
        -1,
    )
    _put_fitted_text(
        cv2_module,
        canvas,
        f"MIC {audio_state['microphone']}",
        (1098, 34),
        164,
        _COLORS["white"],
        0.36,
    )

    camera_box = (16, 68, 740, 400)
    cv2_module.rectangle(
        canvas,
        (camera_box[0], camera_box[1]),
        (camera_box[0] + camera_box[2], camera_box[1] + camera_box[3]),
        _COLORS["card"],
        -1,
    )
    result_frame_number = vision.get("frame_number")
    camera_label = (
        f"LIVE CAMERA | CAPTURED {captured_frame_number} | RESULT "
        f"{result_frame_number if result_frame_number is not None else 'WAITING'}"
        if captured_frame_number is not None
        else f"LIVE CAMERA | FRAME {result_frame_number or 0}"
    )
    cv2_module.putText(
        canvas,
        camera_label,
        (28, 92),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.42,
        _COLORS["text"],
        1,
        cv2_module.LINE_AA,
    )
    overlays_match_frame = (
        captured_frame_number is None
        or captured_frame_number == result_frame_number
    )
    annotated = _draw_frame_overlays(
        cv2_module,
        frame,
        vision if overlays_match_frame else {},
    )
    frame_height, frame_width = annotated.shape[:2]
    target_width, target_height = 716, 340
    scale = min(target_width / frame_width, target_height / frame_height)
    resized_width = max(1, int(frame_width * scale))
    resized_height = max(1, int(frame_height * scale))
    resized = cv2_module.resize(
        annotated,
        (resized_width, resized_height),
        interpolation=cv2_module.INTER_AREA,
    )
    image_x = 28 + (target_width - resized_width) // 2
    image_y = 101 + (target_height - resized_height) // 2
    canvas[
        image_y : image_y + resized_height,
        image_x : image_x + resized_width,
    ] = resized
    cv2_module.rectangle(
        canvas,
        (image_x - 1, image_y - 1),
        (image_x + resized_width, image_y + resized_height),
        _COLORS["border"],
        1,
    )
    scenario_text = (
        f"SCENARIO: {scenario[0]} - {scenario[1]}"
        if scenario is not None
        else "Press 1-9 or 0 to label a manual scenario; counters remain cumulative."
    )
    _put_fitted_text(
        cv2_module,
        canvas,
        scenario_text,
        (28, 458),
        716,
        _COLORS["text"],
        0.34,
    )

    phone_result = vision.get("phone_result")
    if vision.get("phone_status") == "PHONE DETECTED":
        phone_confidences = [
            item.get("confidence")
            for item in phone_result.get("detections", [])
            if isinstance(item, dict) and _finite_number(item.get("confidence"))
        ]
        phone_detail = (
            f"Confidence {max(phone_confidences):.3f}"
            if phone_confidences
            else "Detection output received"
        )
        phone_color = (
            _COLORS["red"]
            if "PHONE_DETECTED" in event_engine.confirmed_events
            else _COLORS["orange"]
        )
    elif vision.get("phone_status") == "NO PHONE DETECTED":
        phone_detail = "No phone in current frame"
        phone_color = _COLORS["green"]
    else:
        phone_detail = str(vision.get("phone_status", "WAITING"))
        phone_color = _COLORS["red"]

    face_result = vision.get("face_result")
    face_count = (
        face_result.get("total_faces", 0)
        if isinstance(face_result, dict)
        else 0
    )
    face_detection = vision.get("face_detection_status", "WAITING FOR RESULT")
    if face_detection == "FACE DETECTED":
        face_detail = f"{face_count} face(s) detected"
        face_color = _COLORS["green"]
    elif face_detection == "NO FACE DETECTED":
        face_detail = "No face available for identity comparison"
        face_color = _COLORS["orange"]
    else:
        face_detail = str(face_detection)
        face_color = (
            _COLORS["red"]
            if str(face_detection).startswith("ERROR")
            else _COLORS["grey"]
        )

    identity_status = vision.get(
        "candidate_identity_status",
        "NOT AVAILABLE" if face_count == 0 else "WAITING FOR RESULT",
    )
    if identity_status == "REGISTERED CANDIDATE MATCHED":
        identity_color = _COLORS["green"]
    elif "UNMATCHED" in identity_status:
        identity_color = _COLORS["red"]
    elif "REVIEW" in identity_status or "MULTIPLE" in identity_status:
        identity_color = _COLORS["orange"]
    else:
        identity_color = _COLORS["grey"]
    identity_detail = (
        f"Unmatched {_counter_text(event_engine, 'IDENTITY_UNMATCHED').split()[-1]}  |  "
        f"No face {_counter_text(event_engine, 'NO_FACE_DETECTED').split()[-1]}"
    )
    if isinstance(face_result, dict):
        face_scores = [
            f"F{item.get('face_number', '?')}:{item['similarity']:.3f}"
            for item in face_result.get("face_results", [])
            if isinstance(item, dict) and _finite_number(item.get("similarity"))
        ]
        if face_scores:
            identity_detail = (
                f"{', '.join(face_scores[:2])}  |  "
                f"{identity_detail}"
            )

    mic_level = (
        f"RMS {audio_state['rms']:.4f}"
        if _finite_number(audio_state["rms"])
        else "RMS unavailable"
    )
    sample_min = audio_state.get("input_min")
    sample_max = audio_state.get("input_max")
    sample_range = (
        f" | range {sample_min:.4f}:{sample_max:.4f}"
        if _finite_number(sample_min) and _finite_number(sample_max)
        else ""
    )
    callbacks = audio_state.get("callback_count", 0)
    queue_size = audio_state.get("queue_size")
    queue_capacity = audio_state.get("queue_capacity")
    queue_text = (
        f" | Q {queue_size}/{queue_capacity}"
        if isinstance(queue_size, int) and isinstance(queue_capacity, int)
        else ""
    )
    mic_detail = (
        f"{mic_level}{sample_range} | callbacks {callbacks}{queue_text}"
    )
    device_name = audio_state.get("device") or "unresolved"
    device_index = audio_state.get("device_index")
    mic_device_detail = f"Device {device_name} | index {device_index}"
    if audio_state.get("warning"):
        mic_detail += "  |  GAP RECOVERED"

    speaker_counter = _counter_text(event_engine, "UNKNOWN_HUMAN_VOICE")
    speaker_detail = (
        f"{audio_state['speaker_detail']}  |  {speaker_counter}".strip(" |")
    )
    sound_counter = _counter_text(event_engine, "SUSPICIOUS_AUDIO_SOUND")
    sound_detail = f"{audio_state['sound_detail']}  |  {sound_counter}".strip(" |")
    if _finite_number(audio_state.get("window_peak")):
        sound_detail += f"  |  peak {audio_state['window_peak']:.4f}"
    sound_color = (
        _COLORS["red"]
        if "SUSPICIOUS_AUDIO_SOUND" in event_engine.confirmed_events
        else audio_state["sound_color"]
    )

    _draw_status_card(
        cv2_module, canvas, (772, 68), (241, 92),
        "Face Detection", face_detection, face_color, face_detail,
    )
    _draw_status_card(
        cv2_module, canvas, (1023, 68), (241, 92),
        "Candidate Identity", identity_status, identity_color, identity_detail,
    )
    _draw_status_card(
        cv2_module, canvas, (772, 168), (241, 92),
        "Phone Detection", vision.get("phone_status", "WAITING"),
        phone_color, phone_detail,
        _counter_text(event_engine, "PHONE_DETECTED"),
    )
    _draw_status_card(
        cv2_module, canvas, (1023, 168), (241, 92),
        "Microphone Status", audio_state["microphone"],
        audio_state["microphone_color"], mic_detail, mic_device_detail,
    )
    _draw_status_card(
        cv2_module, canvas, (772, 268), (241, 92),
        "Voice Detection", audio_state["voice"],
        audio_state["voice_color"], audio_state.get("voice_detail", ""),
    )
    _draw_status_card(
        cv2_module, canvas, (1023, 268), (241, 92),
        "Speaker Verification", audio_state["speaker"],
        audio_state["speaker_color"], speaker_detail,
    )
    _draw_status_card(
        cv2_module, canvas, (772, 368), (492, 100),
        "Suspicious Audio", audio_state["sound"],
        sound_color, sound_detail,
    )

    cv2_module.rectangle(canvas, (16, 484), (356, 744), _COLORS["card"], -1)
    cv2_module.rectangle(canvas, (16, 484), (356, 744), _COLORS["border"], 1)
    cv2_module.putText(
        canvas, "CURRENT SESSION COUNTERS / ALERTS", (28, 509),
        cv2_module.FONT_HERSHEY_SIMPLEX, 0.46, _COLORS["text"], 1,
        cv2_module.LINE_AA,
    )
    for index, event_type in enumerate(event_engine.counters):
        count = min(
            event_engine.counters[event_type],
            event_engine.confirmation_observations,
        )
        confirmed = event_type in event_engine.confirmed_events
        counter_status = "CONFIRMED" if confirmed else "OBSERVATIONS"
        _put_fitted_text(
            cv2_module,
            canvas,
            f"{event_type}  {count}/{event_engine.confirmation_observations}  "
            f"{counter_status}",
            (28, 542 + index * 36),
            316,
            _COLORS["red"] if confirmed else _COLORS["text"],
            0.34,
        )

    cv2_module.rectangle(canvas, (368, 484), (810, 744), _COLORS["card"], -1)
    cv2_module.rectangle(canvas, (368, 484), (810, 744), _COLORS["border"], 1)
    cv2_module.putText(
        canvas, "CURRENT SESSION EVENT HISTORY", (382, 509),
        cv2_module.FONT_HERSHEY_SIMPLEX, 0.46, _COLORS["text"], 1,
        cv2_module.LINE_AA,
    )
    for index, event in enumerate(event_engine.event_log[-5:]):
        timestamp = str(event.get("timestamp", ""))
        summary = (
            f"{timestamp[:19]}  {event['event_type']}  "
            f"{event['observation_count']}/{event_engine.confirmation_observations} "
            f"{event['status']}"
        )
        _put_fitted_text(
            cv2_module,
            canvas,
            summary,
            (382, 540 + index * 39),
            414,
            _COLORS["red"] if event["status"] == "CONFIRMED" else _COLORS["text"],
            0.30,
        )
        _put_fitted_text(
            cv2_module,
            canvas,
            f"{event['module']} | {_event_model_summary(event)}",
            (382, 555 + index * 39),
            414,
            _COLORS["muted"],
            0.29,
        )
    if event_engine.history_error:
        _put_fitted_text(
            cv2_module,
            canvas,
            "HISTORY WRITE ERROR: " + event_engine.history_error,
            (382, 729),
            414,
            _COLORS["red"],
            0.29,
        )

    cv2_module.rectangle(canvas, (822, 484), (1264, 744), _COLORS["card"], -1)
    cv2_module.rectangle(canvas, (822, 484), (1264, 744), _COLORS["border"], 1)
    cv2_module.putText(
        canvas, "HISTORICAL CONFIRMED EVENTS", (836, 509),
        cv2_module.FONT_HERSHEY_SIMPLEX, 0.46, _COLORS["text"], 1,
        cv2_module.LINE_AA,
    )
    historical = event_engine.historical_confirmed_events[-5:]
    if not historical:
        cv2_module.putText(
            canvas, "No previous confirmed events", (836, 542),
            cv2_module.FONT_HERSHEY_SIMPLEX, 0.4, _COLORS["muted"], 1,
            cv2_module.LINE_AA,
        )
    for index, event in enumerate(historical):
        y = 540 + index * 39
        _put_fitted_text(
            cv2_module,
            canvas,
            f"{event['event_type']} | {event['module']} | "
            f"{event['observation_count']}/"
            f"{event_engine.confirmation_observations} CONFIRMED",
            (836, y),
            414,
            _COLORS["red"],
            0.30,
        )
        _put_fitted_text(
            cv2_module,
            canvas,
            f"{str(event['timestamp'])[:19]} | SESSION {event['session_id']}",
            (836, y + 15),
            414,
            _COLORS["muted"],
            0.29,
        )
    return canvas


class LiveProctoringDemo:
    def __init__(
        self,
        registered_candidate: dict[str, Any],
        audio_python: str | Path,
        audio_reference: str | Path | None = None,
        *,
        session_id: str | None = None,
        session_storage: CandidateStorage | None = None,
        camera_factory: Callable[[], Any] | None = None,
        phone_adapter: Any | None = None,
        face_adapter: Any | None = None,
        audio_worker_factory: Callable[..., Any] = AudioWorkerClient,
        cv2_module: Any | None = None,
        event_engine: ProctoringEventEngine | None = None,
    ) -> None:
        self.registered_candidate = registered_candidate
        self.audio_python = audio_python
        self.audio_reference = audio_reference
        voice_embedding = registered_candidate.get("voice_embedding")
        self.speaker_reference_base64 = (
            None
            if voice_embedding is None
            else base64.b64encode(
                np.asarray(voice_embedding, dtype=np.float32).tobytes()
            ).decode("ascii")
        )
        self.camera_factory = camera_factory
        self.phone_adapter = phone_adapter
        self.face_adapter = face_adapter
        self.audio_worker_factory = audio_worker_factory
        self.cv2 = cv2_module
        self.event_engine = event_engine or ProctoringEventEngine()
        self.session_id = session_id
        self.session_storage = session_storage
        self.audio_state = AudioWorkerState()

    def run(self) -> int:
        if self.cv2 is None:
            import cv2

            self.cv2 = cv2
        if self.phone_adapter is None:
            self.phone_adapter = PhoneDetectionAdapter()
        if self.face_adapter is None:
            self.face_adapter = FaceIdentityAdapter()
        if self.camera_factory is None:
            from integration.camera_manager import CameraManager

            self.camera_factory = lambda: CameraManager(camera_index=CAMERA_INDEX)

        worker = None
        frame_number = 0
        camera_status = "CLOSED"
        vision_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="live-vision",
        )
        vision_future: Future[dict[str, Any]] | None = None
        latest_vision = _pending_vision()
        latest_reported_history_error: str | None = None
        try:
            _print(f"DEMO SESSION {self.event_engine.session_id}")
            _print(
                "REGISTERED FACE RECORD | "
                f"candidate_id={self.registered_candidate.get('candidate_id')} "
                f"embedding_count={self.registered_candidate.get('embedding_count', 1)} "
                f"embedding_dimension={FaceIdentityAdapter.EMBEDDING_DIMENSION} "
                f"similarity_threshold={FACE_IDENTITY_THRESHOLD}"
            )
            _print("Starting the isolated audio worker and live camera.")
            try:
                worker = self.audio_worker_factory(
                    self.audio_python,
                    self.audio_reference,
                    self.event_engine,
                    self.audio_state,
                    self.speaker_reference_base64,
                )
                worker.start()
            except Exception as exc:
                self.audio_state.mark_worker_start_error(exc)
                _print(
                    "AUDIO WORKER ERROR | "
                    f"{type(exc).__name__}: {exc}; continuing with vision."
                )
            _print(
                "Camera and vision are running independently of audio. "
                "Press 1-9 or 0 to label a test scenario; Q stops the demo."
            )
            with self.camera_factory() as camera:
                camera_status = "OPEN"
                scenario = None
                last_vision_report = 0.0
                while True:
                    frame = camera.read()
                    frame_number += 1
                    if vision_future is not None and vision_future.done():
                        try:
                            latest_vision = vision_future.result()
                        except Exception as exc:
                            latest_vision = _vision_failure(
                                frame_number,
                                exc,
                            )
                        vision_future = None
                        if time.monotonic() - last_vision_report >= 1.0:
                            _print(_vision_result_summary(latest_vision))
                            last_vision_report = time.monotonic()
                    if vision_future is None:
                        vision_future = vision_executor.submit(
                            analyze_vision_frame,
                            frame.copy(),
                            frame_number,
                            self.phone_adapter,
                            self.face_adapter,
                            self.registered_candidate,
                            self.event_engine,
                        )
                    audio = self.audio_state.snapshot()
                    preview = render_frame(
                        self.cv2,
                        frame,
                        latest_vision,
                        audio,
                        self.event_engine,
                        camera_status,
                        scenario,
                        captured_frame_number=frame_number,
                    )
                    self.cv2.imshow("AI Proctoring Monitor - Q stops", preview)
                    key = self.cv2.waitKey(1) & 0xFF
                    if (
                        self.event_engine.history_error
                        and self.event_engine.history_error
                        != latest_reported_history_error
                    ):
                        latest_reported_history_error = (
                            self.event_engine.history_error
                        )
                        _print(
                            "EVENT HISTORY ERROR | "
                            f"{latest_reported_history_error}"
                        )
                    if key in {ord("q"), ord("Q"), 27}:
                        break
                    key_text = chr(key) if 0 <= key < 128 else ""
                    if key_text in TEST_SCENARIOS:
                        scenario = TEST_SCENARIOS[key_text]
                        _print(f"SCENARIO {scenario[0]} | {scenario[1]}")
                        _print("Event counters are cumulative; they are not reset.")
        finally:
            camera_status = "CLOSED"
            try:
                if vision_future is not None:
                    vision_future.cancel()
                vision_executor.shutdown(wait=True)
                if worker is not None:
                    worker.stop()
            finally:
                self.cv2.destroyAllWindows()
                if self.face_adapter is not None:
                    close = getattr(
                        getattr(self.face_adapter, "landmarker", None),
                        "close",
                        None,
                    )
                    if callable(close):
                        close()
        if self.session_storage is not None and self.session_id is not None:
            try:
                self.session_storage.end_exam_session(
                    self.session_id,
                    datetime.now().astimezone().isoformat(timespec="milliseconds"),
                )
            except Exception as exc:
                _print(f"SESSION END ERROR | {type(exc).__name__}: {exc}")
        _print(f"DEMO SESSION STOPPED {self.event_engine.session_id}")
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the unified phone, face identity, and audio demo."
    )
    parser.add_argument(
        "--audio-python",
        default=os.environ.get("EXAM_AUDIO_PYTHON"),
        help="Python executable in the isolated audio environment.",
    )
    parser.add_argument("--candidate-id")
    parser.add_argument("--session-id")
    parser.add_argument(
        "--audio-reference",
        type=Path,
        help="Optional WAV explicitly linked to the registered candidate below.",
    )
    parser.add_argument(
        "--audio-reference-candidate-id",
        help="Registered candidate ID that the optional voice WAV belongs to.",
    )
    args = parser.parse_args(argv)
    if not args.audio_python:
        parser.error(
            "Provide --audio-python or set EXAM_AUDIO_PYTHON to the audio "
            "environment's Python executable."
        )
    candidate = load_candidate_read_only(
        CANDIDATE_DATABASE_PATH,
        args.candidate_id,
    )
    if (args.audio_reference is None) != (
        args.audio_reference_candidate_id is None
    ):
        parser.error(
            "--audio-reference and --audio-reference-candidate-id must be "
            "provided together."
        )
    if args.audio_reference is not None:
        if not args.audio_reference.is_file():
            parser.error(f"Audio reference does not exist: {args.audio_reference}")
        if args.audio_reference_candidate_id != candidate["candidate_id"]:
            parser.error(
                "Audio reference candidate ID must match the selected "
                f"registered candidate ({candidate['candidate_id']})."
            )
    storage = CandidateStorage(CANDIDATE_DATABASE_PATH)
    session_id = args.session_id
    if session_id is not None:
        session = storage.load_exam_session(session_id)
        if session is None:
            parser.error(f"Exam session not found: {session_id}")
        if session["student_id"] != candidate["candidate_id"]:
            parser.error(
                "Exam session does not belong to the selected student."
            )

    def persist_event(event: dict[str, Any]) -> None:
        if session_id is None:
            return
        model_output = event["model_output"]
        model_score = model_output.get("confidence")
        if model_score is None:
            model_score = model_output.get("score")
        if model_score is None:
            model_score = model_output.get("similarity")
        if not isinstance(model_score, (int, float)):
            model_score = None
        storage.record_proctoring_event(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            student_id=candidate["candidate_id"],
            event_type=event["event_type"],
            module=event["module"],
            observation_id=event["observation_id"],
            observation_count=event["observation_count"],
            timestamp=event["timestamp"],
            model_score=float(model_score) if model_score is not None else None,
            status=event["status"],
            reason=model_output.get("reason"),
        )

    event_engine = ProctoringEventEngine(
        session_id=session_id,
        history_path=EVENT_HISTORY_PATH,
        event_persistence_callback=persist_event if session_id else None,
    )
    demo = LiveProctoringDemo(
        candidate,
        args.audio_python,
        args.audio_reference,
        session_id=session_id,
        session_storage=storage,
        event_engine=event_engine,
    )
    return demo.run()


if __name__ == "__main__":
    raise SystemExit(main())
