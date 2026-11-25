import threading
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from integration.proctoring_events import ProctoringEventEngine


EVENT_TYPES = (
    "PHONE_DETECTED",
    "IDENTITY_UNMATCHED",
    "NO_FACE_DETECTED",
    "UNKNOWN_HUMAN_VOICE",
    "SUSPICIOUS_AUDIO_SOUND",
)


class ProctoringEventEngineTests(unittest.TestCase):
    def test_five_observations_within_window_confirm_event(self):
        engine = ProctoringEventEngine(
            session_id="window-demo",
            confirmation_window_seconds=30.0,
        )
        clock = [0.0]
        with patch(
            "integration.proctoring_events.time.monotonic",
            side_effect=lambda: clock[0],
        ):
            records = []
            for index in range(1, 6):
                records.append(
                    engine.observe(
                        "PHONE_DETECTED",
                        f"frame:{index}",
                        {"confidence": 0.8},
                    )
                )
                clock[0] += 1.0

        self.assertEqual(records[4]["status"], "CONFIRMED")
        self.assertEqual(records[4]["observation_count"], 5)
        self.assertIn("PHONE_DETECTED", engine.confirmed_events)

    def test_all_event_types_throttle_rapid_observations(self):
        for event_type in EVENT_TYPES:
            with self.subTest(event_type=event_type):
                engine = ProctoringEventEngine(session_id=f"rapid-{event_type}")
                clock = [0.0]
                with patch(
                    "integration.proctoring_events.time.monotonic",
                    side_effect=lambda: clock[0],
                ):
                    for index in range(5):
                        engine.observe(event_type, f"observation:{index}", {})
                        clock[0] += 0.1

                self.assertEqual(engine.counters[event_type], 1)
                self.assertNotIn(event_type, engine.confirmed_events)

    def test_all_event_types_count_once_per_eligible_second(self):
        for event_type in EVENT_TYPES:
            with self.subTest(event_type=event_type):
                engine = ProctoringEventEngine(session_id=f"interval-{event_type}")
                clock = [0.0]
                with patch(
                    "integration.proctoring_events.time.monotonic",
                    side_effect=lambda: clock[0],
                ):
                    for index in range(11):
                        clock[0] = index / 10
                        engine.observe(event_type, f"observation:{index}", {})

                self.assertEqual(engine.counters[event_type], 2)

    def test_all_event_types_confirm_on_fifth_eligible_observation(self):
        for event_type in EVENT_TYPES:
            with self.subTest(event_type=event_type):
                engine = ProctoringEventEngine(session_id=f"confirm-{event_type}")
                clock = [0.0]
                with patch(
                    "integration.proctoring_events.time.monotonic",
                    side_effect=lambda: clock[0],
                ):
                    records = []
                    for index in range(5):
                        records.append(
                            engine.observe(
                                event_type,
                                f"observation:{index}",
                                {},
                            )
                        )
                        clock[0] = float(index + 1)

                self.assertEqual(records[-1]["status"], "CONFIRMED")
                self.assertEqual(records[-1]["observation_count"], 5)
                self.assertIn(event_type, engine.confirmed_events)

    def test_all_event_types_expire_observations_outside_rolling_window(self):
        for event_type in EVENT_TYPES:
            with self.subTest(event_type=event_type):
                engine = ProctoringEventEngine(session_id=f"expiry-{event_type}")
                clock = [0.0]
                with patch(
                    "integration.proctoring_events.time.monotonic",
                    side_effect=lambda: clock[0],
                ):
                    for index in range(4):
                        engine.observe(
                            event_type,
                            f"stale:{index}",
                            {},
                        )
                        clock[0] = float(index + 1)

                    clock[0] = 34.0
                    engine.observe(event_type, "fresh:0", {})

                self.assertEqual(engine.counters[event_type], 1)
                self.assertNotIn(event_type, engine.confirmed_events)

    def test_event_type_observation_queues_are_independent(self):
        engine = ProctoringEventEngine(session_id="independent-types")
        clock = [0.0]
        with patch(
            "integration.proctoring_events.time.monotonic",
            side_effect=lambda: clock[0],
        ):
            for index, event_type in enumerate(EVENT_TYPES):
                engine.observe(event_type, f"first:{index}", {})

        self.assertEqual(
            engine.counters,
            {event_type: 1 for event_type in EVENT_TYPES},
        )
        self.assertEqual(engine.confirmed_events, ())

    def test_stale_observations_expire_and_next_observations_start_fresh(self):
        engine = ProctoringEventEngine(
            session_id="expiry-demo",
        )
        clock = [0.0]
        with patch(
            "integration.proctoring_events.time.monotonic",
            side_effect=lambda: clock[0],
        ):
            for index in range(1, 5):
                engine.observe(
                    "NO_FACE_DETECTED",
                    f"old-frame:{index}",
                    {"total_faces": 0},
                )
                clock[0] += 1.0

            clock[0] += 31.0
            engine.observe(
                "NO_FACE_DETECTED",
                "fresh-frame:1",
                {"total_faces": 0},
            )

            self.assertEqual(engine.counters["NO_FACE_DETECTED"], 1)
            self.assertNotIn("NO_FACE_DETECTED", engine.confirmed_events)

            for index in range(2, 6):
                clock[0] += 1.0
                engine.observe(
                    "NO_FACE_DETECTED",
                    f"fresh-frame:{index}",
                    {"total_faces": 0},
                )

        self.assertEqual(engine.counters["NO_FACE_DETECTED"], 5)
        self.assertIn("NO_FACE_DETECTED", engine.confirmed_events)

    def test_expired_deduplication_slot_allows_a_fresh_observation_id(self):
        engine = ProctoringEventEngine(
            session_id="dedup-demo",
            confirmation_window_seconds=5.0,
        )
        stale = time.monotonic() - 6.0
        engine._seen["IDENTITY_UNMATCHED"].add("face:1")
        engine._seen_times["IDENTITY_UNMATCHED"]["face:1"] = stale

        fresh = engine.observe(
            "IDENTITY_UNMATCHED",
            "face:1",
            {"similarity": 0.1},
        )

        self.assertIsNotNone(fresh)
        self.assertEqual(fresh["observation_count"], 1)
        self.assertIsNone(
            engine.observe(
                "IDENTITY_UNMATCHED",
                "face:1",
                {"similarity": 0.2},
            )
        )

    def test_counts_are_independent_and_duplicate_observations_are_ignored(self):
        engine = ProctoringEventEngine(session_id="demo-1")

        first = engine.observe(
            "PHONE_DETECTED",
            "frame:1",
            {"confidence": 0.8},
        )
        duplicate = engine.observe(
            "PHONE_DETECTED",
            "frame:1",
            {"confidence": 0.9},
        )
        face_event = engine.observe(
            "IDENTITY_UNMATCHED",
            "frame:1",
            {"similarity": 0.1},
        )

        self.assertEqual(first["observation_count"], 1)
        self.assertIsNone(duplicate)
        self.assertEqual(face_event["observation_count"], 1)
        self.assertEqual(engine.counters["PHONE_DETECTED"], 1)
        self.assertEqual(engine.counters["IDENTITY_UNMATCHED"], 1)
        self.assertEqual(engine.counters["NO_FACE_DETECTED"], 0)
        self.assertEqual(first["session_id"], "demo-1")
        self.assertEqual(first["module"], "phone_detection")
        self.assertEqual(first["status"], "OBSERVED")

    def test_only_the_fifth_distinct_observation_confirms_its_event(self):
        engine = ProctoringEventEngine(session_id="demo-2")
        clock = [0.0]
        with patch(
            "integration.proctoring_events.time.monotonic",
            side_effect=lambda: clock[0],
        ):
            records = []
            for index in range(1, 6):
                records.append(
                    engine.observe(
                        "NO_FACE_DETECTED",
                        f"frame:{index}",
                        {"total_faces": 0},
                    )
                )
                clock[0] += 1.0

        self.assertEqual([record["status"] for record in records[:4]], ["OBSERVED"] * 4)
        self.assertEqual(records[4]["status"], "CONFIRMED")
        self.assertEqual(records[4]["observation_count"], 5)
        self.assertIn("NO_FACE_DETECTED", engine.confirmed_events)
        self.assertNotIn("IDENTITY_UNMATCHED", engine.confirmed_events)
        self.assertEqual(len(engine.event_log), 5)
        self.assertTrue(records[4]["timestamp"])
        self.assertEqual(records[4]["model_output"], {"total_faces": 0})

    def test_no_face_calls_within_one_second_do_not_confirm(self):
        engine = ProctoringEventEngine(session_id="no-face-fast")
        clock = [0.0]
        with patch(
            "integration.proctoring_events.time.monotonic",
            side_effect=lambda: clock[0],
        ):
            records = []
            for index in range(5):
                records.append(
                    engine.observe(
                        "NO_FACE_DETECTED",
                        f"frame:{index}",
                        {"total_faces": 0},
                    )
                )
                clock[0] += 0.2

        self.assertEqual(sum(event is not None for event in records), 1)
        self.assertEqual(engine.counters["NO_FACE_DETECTED"], 1)
        self.assertNotIn("NO_FACE_DETECTED", engine.confirmed_events)

    def test_no_face_calls_every_100ms_are_throttled(self):
        engine = ProctoringEventEngine(session_id="no-face-100ms")
        clock = [0.0]
        with patch(
            "integration.proctoring_events.time.monotonic",
            side_effect=lambda: clock[0],
        ):
            for index in range(20):
                engine.observe(
                    "NO_FACE_DETECTED",
                    f"frame:{index}",
                    {"total_faces": 0},
                )
                clock[0] += 0.1

        self.assertEqual(engine.counters["NO_FACE_DETECTED"], 2)
        self.assertNotIn("NO_FACE_DETECTED", engine.confirmed_events)

    def test_no_face_five_eligible_observations_confirm_after_four_seconds(self):
        engine = ProctoringEventEngine(session_id="no-face-eligible")
        clock = [0.0]
        with patch(
            "integration.proctoring_events.time.monotonic",
            side_effect=lambda: clock[0],
        ):
            records = []
            for index in range(5):
                records.append(
                    engine.observe(
                        "NO_FACE_DETECTED",
                        f"frame:{index}",
                        {"total_faces": 0},
                    )
                )
                clock[0] += 1.0

        self.assertEqual(records[-1]["status"], "CONFIRMED")
        self.assertEqual(records[-1]["observation_count"], 5)
        self.assertIn("NO_FACE_DETECTED", engine.confirmed_events)

    def test_confirmed_event_logs_once_and_counter_stays_capped(self):
        engine = ProctoringEventEngine()
        clock = [0.0]
        with patch(
            "integration.proctoring_events.time.monotonic",
            side_effect=lambda: clock[0],
        ):
            for index in range(1, 6):
                engine.observe("PHONE_DETECTED", f"frame:{index}", {})
                clock[0] += 1.0

            self.assertIsNone(
                engine.observe("PHONE_DETECTED", "frame:6", {"confidence": 0.9})
            )
        self.assertEqual(engine.counters["PHONE_DETECTED"], 5)
        self.assertEqual(len(engine.event_log), 5)
        self.assertEqual(engine.event_log[-1]["status"], "CONFIRMED")

    def test_invalid_event_and_non_json_model_output_are_rejected(self):
        engine = ProctoringEventEngine()
        with self.assertRaisesRegex(ValueError, "Unsupported event"):
            engine.observe("UNKNOWN", "frame:1", {})
        with self.assertRaises(ValueError):
            engine.observe("PHONE_DETECTED", "frame:1", {"score": float("nan")})
        self.assertEqual(engine.counters["PHONE_DETECTED"], 0)

    def test_new_session_starts_fresh_and_keeps_prior_confirmed_history(self):
        with tempfile.TemporaryDirectory() as directory:
            history_path = Path(directory) / "events.jsonl"
            previous = ProctoringEventEngine(
                session_id="previous-session",
                history_path=history_path,
            )
            clock = [0.0]
            with patch(
                "integration.proctoring_events.time.monotonic",
                side_effect=lambda: clock[0],
            ):
                for index in range(5):
                    previous.observe(
                        "PHONE_DETECTED",
                        f"previous-frame:{index}",
                        {"confidence": 0.8},
                    )
                    clock[0] += 1.0

                current = ProctoringEventEngine(
                    session_id="current-session",
                    history_path=history_path,
                )

                self.assertTrue(all(count == 0 for count in current.counters.values()))
                self.assertEqual(current.event_log, [])
                self.assertEqual(len(current.historical_events), 5)
                self.assertEqual(
                    len(current.historical_confirmed_events),
                    1,
                )
                self.assertEqual(
                    current.historical_confirmed_events[0]["session_id"],
                    "previous-session",
                )

                first = current.observe(
                    "PHONE_DETECTED",
                    "current-frame:1",
                    {"confidence": 0.9},
                )
                self.assertEqual(first["observation_count"], 1)
                self.assertEqual(current.counters["PHONE_DETECTED"], 1)
                self.assertEqual(len(current.historical_events), 5)

    def test_history_write_failure_is_exposed_without_losing_session_event(self):
        with tempfile.TemporaryDirectory() as directory:
            history_path = Path(directory) / "events.jsonl"
            history_path.mkdir()
            engine = ProctoringEventEngine(history_path=history_path)

            event = engine.observe(
                "NO_FACE_DETECTED",
                "frame:1",
                {"total_faces": 0},
            )

            self.assertEqual(event["status"], "OBSERVED")
            self.assertEqual(engine.counters["NO_FACE_DETECTED"], 1)
            self.assertIn("Could not persist event history", engine.history_error)
            self.assertEqual(engine.persistence_status, "FAILED")

    def test_database_persistence_failure_is_failed_not_persisted(self):
        def failing_callback(_event):
            raise RuntimeError("database unavailable")

        engine = ProctoringEventEngine(
            session_id="db-failure",
            event_persistence_callback=failing_callback,
        )
        event = engine.observe(
            "PHONE_DETECTED",
            "frame:db",
            {"confidence": 0.8},
        )

        self.assertIsNotNone(event)
        self.assertEqual(engine.persistence_status, "FAILED")
        self.assertIn("database unavailable", engine.history_error)
        self.assertIn("RuntimeError", engine.history_error)

    def test_slow_history_write_does_not_block_current_session_snapshot(self):
        started = threading.Event()
        release = threading.Event()
        engine = ProctoringEventEngine()

        def slow_append(_event):
            started.set()
            release.wait(timeout=2)

        with patch.object(engine, "_append_history", side_effect=slow_append):
            engine.history_path = Path("unused.jsonl")
            worker = threading.Thread(
                target=engine.observe,
                args=("PHONE_DETECTED", "frame:1", {"confidence": 0.8}),
            )
            worker.start()
            self.assertTrue(started.wait(timeout=1))
            self.assertEqual(engine.counters["PHONE_DETECTED"], 1)
            self.assertEqual(len(engine.event_log), 1)
            release.set()
            worker.join(timeout=1)

        self.assertFalse(worker.is_alive())


if __name__ == "__main__":
    unittest.main()
