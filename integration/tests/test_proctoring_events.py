import threading
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from integration.proctoring_events import ProctoringEventEngine


class ProctoringEventEngineTests(unittest.TestCase):
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
        records = [
            engine.observe(
                "NO_FACE_DETECTED",
                f"frame:{index}",
                {"total_faces": 0},
            )
            for index in range(1, 6)
        ]

        self.assertEqual([record["status"] for record in records[:4]], ["OBSERVED"] * 4)
        self.assertEqual(records[4]["status"], "CONFIRMED")
        self.assertEqual(records[4]["observation_count"], 5)
        self.assertIn("NO_FACE_DETECTED", engine.confirmed_events)
        self.assertNotIn("IDENTITY_UNMATCHED", engine.confirmed_events)
        self.assertEqual(len(engine.event_log), 5)
        self.assertTrue(records[4]["timestamp"])
        self.assertEqual(records[4]["model_output"], {"total_faces": 0})

    def test_confirmed_event_logs_once_and_counter_stays_capped(self):
        engine = ProctoringEventEngine()
        for index in range(1, 6):
            engine.observe("PHONE_DETECTED", f"frame:{index}", {})

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
            for index in range(5):
                previous.observe(
                    "PHONE_DETECTED",
                    f"previous-frame:{index}",
                    {"confidence": 0.8},
                )

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
