import os
import threading
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from integration.live_demo import (
    AudioWorkerState,
    LiveProctoringDemo,
    TEST_SCENARIOS,
    _audio_dashboard_state,
    _live_clock_text,
    _pending_vision,
    analyze_vision_frame,
    render_frame,
    load_candidate_read_only,
    process_audio_result,
)
from integration.candidate_storage import CandidateStorage
from integration.proctoring_events import ProctoringEventEngine


class FakePhoneAdapter:
    def __init__(self, result):
        self.result = result
        self.frames = []

    def detect(self, frame):
        self.frames.append(frame)
        return self.result


class FakeFaceAdapter:
    def __init__(self, result):
        self.result = result
        self.frames = []

    def verify_registered_identity(self, candidate, frame):
        self.asserted_candidate = candidate
        self.frames.append(frame)
        return self.result


def phone_result(detected=False):
    detections = (
        [{"class_name": "phone", "confidence": 0.82, "bbox_xyxy": [1, 2, 3, 4]}]
        if detected
        else []
    )
    return {
        "module": "phone_detection",
        "detected": detected,
        "detection_count": len(detections),
        "detections": detections,
    }


def face_result(predicted="MATCH"):
    return {
        "monitoring_status": "SINGLE_FACE_DETECTED",
        "identity_status": (
            "CANDIDATE_IDENTIFIED"
            if predicted == "MATCH"
            else "CANDIDATE_NOT_FOUND"
        ),
        "total_faces": 1,
        "face_results": [
            {
                "face_number": 1,
                "predicted": predicted,
                "similarity": 0.8 if predicted == "MATCH" else 0.1,
                "box": [1, 2, 3, 4],
            }
        ],
    }


def audio_result(
    *,
    speech=True,
    speaker_status="compared",
    comparison="low_similarity",
    similarity=0.1,
    phone_detected=True,
    phone_score=0.4,
):
    return {
        "module": "audio_detection",
        "vad": {
            "speech_detected": speech,
            "speech_duration_seconds": 0.9 if speech else 0.0,
        },
        "speaker": {
            "status": speaker_status,
            "comparison": comparison,
            "similarity": similarity,
            "thresholds": {"low_similarity": 0.22, "high_similarity": 0.38},
        },
        "phone_sound": {
            "status": "scored",
            "detected": phone_detected,
            "score": phone_score,
            "threshold": 0.2,
            "high_confidence": False,
        },
    }


def _audio_state_for_test(result, *, input_rms):
    return _audio_dashboard_state(
        {
            "status": "MICROPHONE ON",
            "input_rms": input_rms,
            "result": {
                "analysis": {"status": "VALID", "result": result},
            },
        }
    )


class LiveDemoTests(unittest.TestCase):
    def test_test_scenarios_have_visible_number_keys(self):
        self.assertEqual(set(TEST_SCENARIOS), set("1234567890"))
        self.assertTrue(TEST_SCENARIOS["5"][0].startswith("5/10"))
        self.assertTrue(TEST_SCENARIOS["0"][0].startswith("10/10"))

    def test_phone_and_identity_adapters_receive_the_same_frame(self):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        candidate = {"candidate_id": "candidate-1", "embedding": np.ones(128)}
        phone = FakePhoneAdapter(phone_result(detected=True))
        face = FakeFaceAdapter(face_result("MISMATCH"))
        engine = ProctoringEventEngine()
        logged = []

        result = analyze_vision_frame(
            frame,
            7,
            phone,
            face,
            candidate,
            engine,
            event_logger=logged.append,
        )

        self.assertIs(phone.frames[0], frame)
        self.assertIs(face.frames[0], frame)
        self.assertIs(face.asserted_candidate, candidate)
        self.assertEqual(result["phone_status"], "PHONE DETECTED")
        self.assertEqual(result["face_status"], "CANDIDATE_NOT_FOUND")
        self.assertEqual(result["face_detection_status"], "FACE DETECTED")
        self.assertEqual(
            result["candidate_identity_status"],
            "CANDIDATE NOT IDENTIFIED / UNMATCHED",
        )
        self.assertEqual(engine.counters["PHONE_DETECTED"], 1)
        self.assertEqual(engine.counters["IDENTITY_UNMATCHED"], 1)
        self.assertEqual(len(logged), 2)

    def test_face_absence_and_identity_mismatch_are_separate_events(self):
        engine = ProctoringEventEngine()
        candidate = {"candidate_id": "candidate-1", "embedding": np.ones(128)}
        phone = FakePhoneAdapter(phone_result())

        class FaceSequence:
            def __init__(self):
                self.results = [
                    {
                        "monitoring_status": "NO_FACE_DETECTED",
                        "identity_status": "NOT_VERIFIED",
                        "total_faces": 0,
                        "face_results": [],
                    },
                    face_result("MISMATCH"),
                ]

            def verify_registered_identity(self, _candidate, _frame):
                return self.results.pop(0)

        face = FaceSequence()
        for frame_number in (1, 2):
            analyze_vision_frame(
                np.zeros((8, 8, 3), dtype=np.uint8),
                frame_number,
                phone,
                face,
                candidate,
                engine,
                event_logger=lambda _event: None,
            )

        self.assertEqual(engine.counters["NO_FACE_DETECTED"], 1)
        self.assertEqual(engine.counters["IDENTITY_UNMATCHED"], 1)

    def test_no_face_has_unavailable_identity_not_unmatched(self):
        vision = analyze_vision_frame(
            np.zeros((8, 8, 3), dtype=np.uint8),
            1,
            FakePhoneAdapter(phone_result()),
            FakeFaceAdapter(
                {
                    "monitoring_status": "NO_FACE_DETECTED",
                    "identity_status": "NOT_VERIFIED",
                    "total_faces": 0,
                    "face_results": [],
                }
            ),
            {"candidate_id": "candidate-1"},
            ProctoringEventEngine(),
            event_logger=lambda _event: None,
        )

        self.assertEqual(vision["face_detection_status"], "NO FACE DETECTED")
        self.assertEqual(vision["candidate_identity_status"], "NOT AVAILABLE")
        self.assertEqual(vision["phone_status"], "NO PHONE DETECTED")

    def test_candidate_match_is_valid_and_does_not_count_as_unmatched(self):
        engine = ProctoringEventEngine()
        result = analyze_vision_frame(
            np.zeros((8, 8, 3), dtype=np.uint8),
            1,
            FakePhoneAdapter(phone_result()),
            FakeFaceAdapter(face_result("MATCH")),
            {"candidate_id": "candidate-1"},
            engine,
            event_logger=lambda _event: None,
        )

        self.assertEqual(result["face_status"], "CANDIDATE_IDENTIFIED")
        self.assertEqual(result["candidate_identity_status"], "REGISTERED CANDIDATE MATCHED")
        self.assertEqual(engine.counters["IDENTITY_UNMATCHED"], 0)
        self.assertEqual(engine.counters["NO_FACE_DETECTED"], 0)

    def test_contradictory_face_aggregate_is_rejected(self):
        result = face_result("MATCH")
        result["identity_status"] = "CANDIDATE_NOT_FOUND"
        engine = ProctoringEventEngine()
        vision = analyze_vision_frame(
            np.zeros((8, 8, 3), dtype=np.uint8),
            1,
            FakePhoneAdapter(phone_result()),
            FakeFaceAdapter(result),
            {"candidate_id": "candidate-1"},
            engine,
            event_logger=lambda _event: None,
        )

        self.assertIsNone(vision["face_result"])
        self.assertIn("ERROR:", vision["candidate_identity_status"])
        self.assertEqual(engine.counters["IDENTITY_UNMATCHED"], 0)

    def test_invalid_vision_outputs_are_visible_but_do_not_count(self):
        engine = ProctoringEventEngine()
        result = analyze_vision_frame(
            np.zeros((8, 8, 3), dtype=np.uint8),
            1,
            FakePhoneAdapter({"detected": True}),
            FakeFaceAdapter({"identity_status": "CANDIDATE_NOT_FOUND"}),
            {"candidate_id": "candidate-1"},
            engine,
            event_logger=lambda _event: None,
        )

        self.assertTrue(result["phone_status"].startswith("ERROR:"))
        self.assertTrue(result["face_status"].startswith("ERROR:"))
        self.assertEqual(sum(engine.counters.values()), 0)

    def test_audio_events_are_independent_deduplicated_and_confirm_at_five(self):
        engine = ProctoringEventEngine()
        logged = []
        for window_id in range(1, 6):
            result = process_audio_result(
                window_id,
                audio_result(),
                engine,
                event_logger=logged.append,
            )
            self.assertEqual(result["status"], "VALID")
        process_audio_result(
            5,
            audio_result(),
            engine,
            event_logger=logged.append,
        )

        self.assertEqual(engine.counters["UNKNOWN_HUMAN_VOICE"], 5)
        self.assertEqual(engine.counters["SUSPICIOUS_AUDIO_SOUND"], 5)
        self.assertIn("UNKNOWN_HUMAN_VOICE", engine.confirmed_events)
        self.assertIn("SUSPICIOUS_AUDIO_SOUND", engine.confirmed_events)
        self.assertEqual(len(logged), 10)
        self.assertEqual(
            [event["status"] for event in logged[-2:]],
            ["CONFIRMED", "CONFIRMED"],
        )

    def test_incomplete_speaker_reference_and_inert_scores_do_not_count(self):
        engine = ProctoringEventEngine()
        unresolved = audio_result(
            speaker_status="reference_not_configured",
            comparison=None,
            similarity=None,
            phone_detected=False,
            phone_score=0.99,
        )
        result = process_audio_result(
            1,
            unresolved,
            engine,
            event_logger=lambda _event: None,
        )
        self.assertEqual(result["status"], "VALID")
        self.assertEqual(engine.counters["UNKNOWN_HUMAN_VOICE"], 0)
        self.assertEqual(engine.counters["SUSPICIOUS_AUDIO_SOUND"], 0)

    def test_high_similarity_speech_is_not_counted_as_unknown_voice(self):
        engine = ProctoringEventEngine()
        result = process_audio_result(
            1,
            audio_result(
                comparison="high_similarity",
                similarity=0.8,
                phone_detected=False,
                phone_score=0.0,
            ),
            engine,
            event_logger=lambda _event: None,
        )

        self.assertEqual(result["status"], "VALID")
        self.assertEqual(engine.counters["UNKNOWN_HUMAN_VOICE"], 0)

    def test_audio_worker_error_clears_previous_result_and_blocks_later_results(self):
        state = AudioWorkerState()
        state.apply(
            {"type": "status", "status": "MICROPHONE ON"},
            ProctoringEventEngine(),
        )
        state.apply(
            {
                "type": "audio_result",
                "window_id": 1,
                "input_rms": 0.1,
                "result": audio_result(
                    speaker_status="no_speech",
                    comparison=None,
                    similarity=None,
                    speech=False,
                    phone_detected=False,
                    phone_score=0.0,
                ),
            },
            ProctoringEventEngine(),
        )
        engine = ProctoringEventEngine()
        state.apply(
            {
                "type": "error",
                "status": "MICROPHONE ERROR",
                "error": "device unavailable",
            },
            engine,
        )
        state.apply(
            {
                "type": "audio_result",
                "window_id": 2,
                "input_rms": 0.1,
                "result": audio_result(),
            },
            engine,
        )

        snapshot = state.snapshot()
        self.assertEqual(snapshot["status"], "MICROPHONE ERROR")
        self.assertEqual(snapshot["error"], "device unavailable")
        self.assertIsNone(snapshot["result"])
        self.assertIsNone(snapshot["input_rms"])
        self.assertEqual(engine.counters["UNKNOWN_HUMAN_VOICE"], 0)

    def test_unavailable_audio_is_not_reported_as_silence(self):
        state = AudioWorkerState()
        state.apply(
            {
                "type": "error",
                "status": "AUDIO WORKER ERROR",
                "error": "worker failed",
            },
            ProctoringEventEngine(),
        )

        display = _audio_dashboard_state(state.snapshot())
        self.assertEqual(display["microphone"], "ERROR")
        self.assertEqual(display["voice"], "AUDIO UNAVAILABLE")
        self.assertEqual(display["sound"], "AUDIO UNAVAILABLE")
        self.assertIsNone(display["rms"])

    def test_connected_silence_and_speech_have_distinct_voice_states(self):
        silence = audio_result(
            speech=False,
            speaker_status="no_speech",
            comparison=None,
            similarity=None,
            phone_detected=False,
            phone_score=0.0,
        )
        silence_display = _audio_state_for_test(silence, input_rms=0.0)
        self.assertEqual(silence_display["microphone"], "CONNECTED")
        self.assertEqual(silence_display["voice"], "NO SPEECH")
        self.assertEqual(silence_display["rms"], 0.0)

        speech = audio_result(
            speech=True,
            speaker_status="reference_not_configured",
            comparison=None,
            similarity=None,
            phone_detected=False,
            phone_score=0.0,
        )
        speech_display = _audio_state_for_test(speech, input_rms=0.1)
        self.assertEqual(speech_display["voice"], "SPEECH DETECTED")
        self.assertEqual(speech_display["speaker"], "REFERENCE NOT CONFIGURED")

    def test_audio_gap_clears_stale_result_until_next_complete_window(self):
        state = AudioWorkerState()
        engine = ProctoringEventEngine()
        silence = audio_result(
            speech=False,
            speaker_status="no_speech",
            comparison=None,
            similarity=None,
            phone_detected=False,
            phone_score=0.0,
        )
        state.apply({"type": "status", "status": "MICROPHONE ON"}, engine)
        state.apply(
            {
                "type": "audio_result",
                "window_id": 1,
                "input_rms": 0.0,
                "result": silence,
            },
            engine,
        )
        state.apply(
            {
                "type": "telemetry",
                "status": "MICROPHONE ON",
                "input_rms": 0.02,
                "discontinuities": 1,
            },
            engine,
        )

        self.assertIsNone(state.snapshot()["result"])
        self.assertEqual(
            _audio_dashboard_state(state.snapshot())["voice"],
            "WAITING FOR AUDIO",
        )

        state.apply(
            {
                "type": "audio_result",
                "window_id": 2,
                "input_rms": 0.02,
                "result": silence,
            },
            engine,
        )
        state.apply(
            {
                "type": "telemetry",
                "status": "MICROPHONE ON",
                "input_rms": 0.02,
                "discontinuities": 1,
            },
            engine,
        )

        self.assertIsNotNone(state.snapshot()["result"])
        self.assertEqual(
            _audio_dashboard_state(state.snapshot())["voice"],
            "NO SPEECH",
        )

    def test_audio_result_staleness_is_not_displayed_as_no_speech(self):
        silence = audio_result(
            speech=False,
            speaker_status="no_speech",
            comparison=None,
            similarity=None,
            phone_detected=False,
            phone_score=0.0,
        )
        display = _audio_dashboard_state(
            {
                "status": "MICROPHONE ON",
                "input_rms": 0.0,
                "connected_age_seconds": 25.0,
                "last_valid_window_age_seconds": 25.0,
                "result": {
                    "analysis": {"status": "VALID", "result": silence},
                },
            }
        )

        self.assertEqual(display["microphone"], "CONNECTED")
        self.assertEqual(display["voice"], "AUDIO PROCESSING STALLED")
        self.assertEqual(display["speaker"], "AUDIO PROCESSING STALLED")
        self.assertEqual(display["sound"], "AUDIO PROCESSING STALLED")

    def test_missing_microphone_callbacks_are_not_displayed_as_silence(self):
        display = _audio_dashboard_state(
            {
                "status": "MICROPHONE ON",
                "callback_age_seconds": 3.0,
                "last_valid_window_age_seconds": 0.1,
                "result": {
                    "analysis": {
                        "status": "VALID",
                        "result": audio_result(
                            speech=False,
                            speaker_status="no_speech",
                            comparison=None,
                            similarity=None,
                            phone_detected=False,
                            phone_score=0.0,
                        ),
                    },
                },
            }
        )

        self.assertEqual(display["microphone"], "INPUT STALLED")
        self.assertEqual(display["voice"], "AUDIO UNAVAILABLE")

    def test_audio_state_exposes_device_callback_and_vad_diagnostics(self):
        state = AudioWorkerState()
        engine = ProctoringEventEngine()
        state.apply(
            {
                "type": "status",
                "status": "MICROPHONE ON",
                "selected_device_index": 3,
                "selected_device_name": "Test input",
            },
            engine,
        )
        state.apply(
            {
                "type": "telemetry",
                "status": "MICROPHONE ON",
                "input_rms": 0.0004,
                "input_min": -0.001,
                "input_max": 0.001,
                "callback_count": 24,
                "callback_age_seconds": 0.01,
                "queue_size": 1,
                "queue_capacity": 4,
            },
            engine,
        )
        speech = audio_result(
            speech=True,
            speaker_status="reference_not_configured",
            comparison=None,
            similarity=None,
            phone_detected=False,
            phone_score=0.0,
        )
        state.apply(
            {
                "type": "audio_result",
                "window_id": 1,
                "timestamp": "2026-01-01T00:00:00+00:00",
                "input_rms": 0.0004,
                "window_rms": 0.0003,
                "window_peak": 0.001,
                "window_min": -0.0008,
                "window_max": 0.0009,
                "input_min": -0.001,
                "input_max": 0.001,
                "callback_count": 24,
                "selected_device_index": 3,
                "selected_device_name": "Test input",
                "result": speech,
            },
            engine,
        )

        snapshot = state.snapshot()
        display = _audio_dashboard_state(snapshot)
        self.assertEqual(snapshot["selected_device_name"], "Test input")
        self.assertEqual(snapshot["callback_count"], 24)
        self.assertEqual(snapshot["queue_size"], 1)
        self.assertEqual(display["voice"], "SPEECH DETECTED")
        self.assertIn("VAD speech 0.900s", display["voice_detail"])
        self.assertAlmostEqual(display["window_rms"], 0.0003)
        self.assertAlmostEqual(display["window_min"], -0.0008)
        self.assertEqual(display["last_speech_timestamp"], "2026-01-01T00:00:00+00:00")

    def test_audio_event_processing_does_not_hold_dashboard_state_lock(self):
        state = AudioWorkerState()
        engine = ProctoringEventEngine()
        state.apply({"type": "status", "status": "MICROPHONE ON"}, engine)
        processing_started = threading.Event()
        release_processing = threading.Event()

        def blocked_processing(*_args, **_kwargs):
            processing_started.set()
            release_processing.wait(timeout=2)
            return {"status": "VALID", "result": audio_result()}

        with patch(
            "integration.live_demo.process_audio_result",
            side_effect=blocked_processing,
        ):
            worker = threading.Thread(
                target=state.apply,
                args=(
                    {
                        "type": "audio_result",
                        "window_id": 1,
                        "result": audio_result(),
                    },
                    engine,
                ),
            )
            worker.start()
            self.assertTrue(processing_started.wait(timeout=1))
            self.assertEqual(state.snapshot()["status"], "MICROPHONE ON")
            release_processing.set()
            worker.join(timeout=1)

        self.assertFalse(worker.is_alive())

    def test_audio_event_processing_failure_is_visible_not_silently_dropped(self):
        state = AudioWorkerState()
        state.apply(
            {"type": "status", "status": "MICROPHONE ON"},
            ProctoringEventEngine(),
        )
        with patch(
            "integration.live_demo.process_audio_result",
            side_effect=RuntimeError("event persistence failed"),
        ):
            state.apply(
                {
                    "type": "audio_result",
                    "window_id": 1,
                    "result": audio_result(),
                },
                ProctoringEventEngine(),
            )

        snapshot = state.snapshot()
        self.assertEqual(snapshot["status"], "MICROPHONE ON")
        self.assertEqual(
            snapshot["result"]["analysis"]["status"],
            "REVIEW REQUIRED",
        )
        self.assertIn(
            "event persistence failed",
            snapshot["result"]["analysis"]["error"],
        )

    def test_live_clock_renders_new_time_without_waiting_for_new_event(self):
        class RecordingCV2:
            FONT_HERSHEY_SIMPLEX = cv2.FONT_HERSHEY_SIMPLEX
            LINE_AA = cv2.LINE_AA
            INTER_AREA = cv2.INTER_AREA

            def __init__(self):
                self.texts = []

            def __getattr__(self, name):
                return getattr(cv2, name)

            def putText(self, image, text, *args, **kwargs):
                self.texts.append(str(text))
                return cv2.putText(image, text, *args, **kwargs)

        cv2_module = RecordingCV2()
        clock_values = [
            datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
        ]
        with patch("integration.live_demo.datetime") as mocked_datetime:
            mocked_datetime.now.side_effect = clock_values
            for _ in clock_values:
                render_frame(
                    cv2_module,
                    np.zeros((80, 120, 3), dtype=np.uint8),
                    _pending_vision(),
                    AudioWorkerState().snapshot(),
                    ProctoringEventEngine(session_id="clock-test"),
                    "OPEN",
                )

        first_clock = clock_values[0].astimezone().strftime("%Y-%m-%d %H:%M:%S")
        second_clock = clock_values[1].astimezone().strftime("%Y-%m-%d %H:%M:%S")
        self.assertIn(first_clock, cv2_module.texts)
        self.assertIn(second_clock, cv2_module.texts)
        self.assertEqual(
            _live_clock_text(clock_values[0]),
            first_clock,
        )

    def test_dashboard_separates_current_counters_from_prior_confirmations(self):
        with tempfile.TemporaryDirectory() as directory:
            history_path = Path(directory) / "events.jsonl"
            previous = ProctoringEventEngine(
                session_id="previous-session-id",
                history_path=history_path,
            )
            for index in range(5):
                previous.observe(
                    "PHONE_DETECTED",
                    f"previous-frame:{index}",
                    {"confidence": 0.8},
                )
            current = ProctoringEventEngine(
                session_id="current-session-id",
                history_path=history_path,
            )

            labels = []

            class RecordingCV2:
                FONT_HERSHEY_SIMPLEX = cv2.FONT_HERSHEY_SIMPLEX
                LINE_AA = cv2.LINE_AA
                INTER_AREA = cv2.INTER_AREA

                def __getattr__(self, name):
                    return getattr(cv2, name)

                def putText(self, image, text, *args, **kwargs):
                    labels.append(str(text))
                    return cv2.putText(image, text, *args, **kwargs)

            render_frame(
                RecordingCV2(),
                np.zeros((80, 120, 3), dtype=np.uint8),
                _pending_vision(),
                AudioWorkerState().snapshot(),
                current,
                "OPEN",
            )

        self.assertIn("PHONE_DETECTED  0/5  OBSERVATIONS", labels)
        self.assertIn("PHONE_DETECTED | phone_detection | 5/5 CONFIRMED", labels)
        self.assertTrue(
            any("previous-session-id" in label for label in labels)
        )

    def test_vision_inference_does_not_block_dashboard_refresh(self):
        inference_started = threading.Event()
        release_inference = threading.Event()
        frame = np.zeros((80, 120, 3), dtype=np.uint8)

        class SlowPhoneAdapter(FakePhoneAdapter):
            def detect(self, input_frame):
                inference_started.set()
                release_inference.wait(timeout=2)
                return super().detect(input_frame)

        class FakeCamera:
            def __init__(self):
                self.read_count = 0

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                self.read_count += 1
                return frame

        class ResponsiveCV2:
            FONT_HERSHEY_SIMPLEX = cv2.FONT_HERSHEY_SIMPLEX
            LINE_AA = cv2.LINE_AA
            INTER_AREA = cv2.INTER_AREA

            def __init__(self):
                self.rendered = 0

            def __getattr__(self, name):
                return getattr(cv2, name)

            def imshow(self, *_args):
                self.rendered += 1

            def waitKey(self, _delay):
                if self.rendered >= 2:
                    release_inference.set()
                    return ord("q")
                return -1

            @staticmethod
            def destroyAllWindows():
                return None

        camera = FakeCamera()
        face = FakeFaceAdapter(face_result("MATCH"))
        phone = SlowPhoneAdapter(phone_result())
        demo = LiveProctoringDemo(
            {"candidate_id": "candidate-1", "embedding": np.ones(128)},
            "unused-audio-python",
            camera_factory=lambda: camera,
            phone_adapter=phone,
            face_adapter=face,
            audio_worker_factory=lambda *_args: SimpleNamespace(
                start=lambda: None,
                stop=lambda: None,
            ),
            cv2_module=ResponsiveCV2(),
        )

        self.assertEqual(demo.run(), 0)
        self.assertTrue(inference_started.is_set())
        self.assertGreaterEqual(camera.read_count, 2)
        self.assertEqual(len(phone.frames), 1)
        self.assertEqual(len(face.frames), 1)

    def test_dashboard_draws_unmatched_face_without_match_label(self):
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        engine = ProctoringEventEngine()
        vision = analyze_vision_frame(
            frame,
            1,
            FakePhoneAdapter(phone_result()),
            FakeFaceAdapter(face_result("MISMATCH")),
            {"candidate_id": "candidate-1"},
            engine,
            event_logger=lambda _event: None,
        )
        labels = []

        class RecordingCV2:
            FONT_HERSHEY_SIMPLEX = cv2.FONT_HERSHEY_SIMPLEX
            LINE_AA = cv2.LINE_AA
            INTER_AREA = cv2.INTER_AREA

            def __getattr__(self, name):
                return getattr(cv2, name)

            def putText(self, image, text, *args, **kwargs):
                labels.append(str(text))
                return cv2.putText(image, text, *args, **kwargs)

        dashboard = render_frame(
            RecordingCV2(),
            frame,
            vision,
            AudioWorkerState().snapshot(),
            engine,
            "OPEN",
        )

        self.assertEqual(dashboard.shape, (760, 1280, 3))
        self.assertIn("CANDIDATE UNMATCHED 0.100", labels)
        self.assertIn("CANDIDATE NOT IDENTIFIED / UNMATCHED", labels)
        self.assertNotIn("REGISTERED CANDIDATE MATCHED", labels)

    def test_audio_worker_command_omits_unlinked_voice_reference(self):
        from integration.live_demo import AudioWorkerClient

        with tempfile.TemporaryDirectory() as directory:
            interpreter = Path(directory) / "python.exe"
            interpreter.touch()
            process = SimpleNamespace(
                stdout=iter(()),
                stdin=None,
                poll=lambda: 0,
            )
            with patch(
                "integration.live_demo.subprocess.Popen",
                return_value=process,
            ) as popen:
                client = AudioWorkerClient(
                    interpreter,
                    None,
                    ProctoringEventEngine(),
                    AudioWorkerState(),
                )
                client.start()
                client._reader.join(timeout=1)

            command = popen.call_args.args[0]
            self.assertNotIn("--audio-reference", command)

    def test_audio_worker_start_error_does_not_stop_vision(self):
        frame = np.zeros((80, 120, 3), dtype=np.uint8)

        class FakeCamera:
            def __init__(self):
                self.read_count = 0

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                self.read_count += 1
                return frame

        class FailedWorker:
            def __init__(self, *_args):
                pass

            def start(self):
                raise OSError("audio interpreter unavailable")

            def stop(self):
                pass

        class HeadlessCV2:
            FONT_HERSHEY_SIMPLEX = cv2.FONT_HERSHEY_SIMPLEX
            LINE_AA = cv2.LINE_AA
            INTER_AREA = cv2.INTER_AREA

            def __getattr__(self, name):
                return getattr(cv2, name)

            @staticmethod
            def imshow(*_args):
                return None

            @staticmethod
            def waitKey(_delay):
                return ord("q")

            @staticmethod
            def destroyAllWindows():
                return None

        camera = FakeCamera()
        phone = FakePhoneAdapter(phone_result())
        face = FakeFaceAdapter(
            {
                "monitoring_status": "NO_FACE_DETECTED",
                "identity_status": "NOT_VERIFIED",
                "total_faces": 0,
                "face_results": [],
            }
        )
        demo = LiveProctoringDemo(
            {"candidate_id": "candidate-1"},
            "unused-audio-python",
            camera_factory=lambda: camera,
            phone_adapter=phone,
            face_adapter=face,
            audio_worker_factory=FailedWorker,
            cv2_module=HeadlessCV2(),
        )

        self.assertEqual(demo.run(), 0)
        self.assertEqual(camera.read_count, 1)
        self.assertEqual(len(phone.frames), 1)
        self.assertEqual(len(face.frames), 1)
        self.assertEqual(demo.audio_state.snapshot()["status"], "AUDIO WORKER ERROR")

    def test_candidate_embedding_load_uses_existing_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "candidate.sqlite3"
            embedding = np.zeros(128, dtype=np.float32)
            embedding[0] = 1.0
            connection = sqlite3.connect(database)
            try:
                connection.execute(
                    "CREATE TABLE candidates "
                    "(candidate_id TEXT PRIMARY KEY, status TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE candidate_embeddings "
                    "(candidate_id TEXT, embedding_index INTEGER, "
                    "embedding_blob BLOB, embedding_dimension INTEGER, "
                    "landmarks_available INTEGER)"
                )
                connection.execute(
                    "INSERT INTO candidates VALUES (?, ?)",
                    ("candidate-1", "REGISTERED"),
                )
                connection.execute(
                    "INSERT INTO candidate_embeddings VALUES (?, ?, ?, ?, ?)",
                    ("candidate-1", 0, embedding.tobytes(), 128, 1),
                )
                connection.commit()
            finally:
                connection.close()

            candidate = load_candidate_read_only(database)
            self.assertEqual(candidate["candidate_id"], "candidate-1")
            np.testing.assert_array_equal(candidate["embedding"], embedding)

            connection = sqlite3.connect(
                f"{database.resolve().as_uri()}?mode=ro",
                uri=True,
            )
            try:
                with self.assertRaises(sqlite3.OperationalError):
                    connection.execute(
                        "UPDATE candidates SET status='CHANGED'"
                    )
            finally:
                connection.close()

    @unittest.skipUnless(os.name == "nt", "Windows DPAPI is required.")
    def test_candidate_embedding_loader_reads_dpapi_encrypted_data(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "candidate.sqlite3"
            storage = CandidateStorage(database)
            embeddings = []
            for index in (3, 4, 5):
                embedding = np.zeros(128, dtype=np.float32)
                embedding[index] = 1.0
                embeddings.append(embedding)
            voice_embedding = np.array([1.0, 2.0], dtype=np.float32)
            storage.register_student(
                student_id="candidate-dpapi",
                full_name="DPAPI Candidate",
                enrollment_number="ENR-DPAPI",
                email=None,
                face_embeddings=embeddings,
                voice_embedding=voice_embedding,
                created_at="2026-01-01T00:00:00+00:00",
            )

            candidate = load_candidate_read_only(
                database,
                candidate_id="candidate-dpapi",
            )

            self.assertEqual(candidate["candidate_id"], "candidate-dpapi")
            np.testing.assert_array_equal(candidate["embedding"], embeddings[0])
            self.assertEqual(candidate["embedding_count"], 3)
            self.assertEqual(len(candidate["reference_embeddings"]), 3)
            self.assertIsNotNone(candidate["voice_embedding"])


if __name__ == "__main__":
    unittest.main()
