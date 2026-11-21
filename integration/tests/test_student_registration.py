from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from integration.biometric_crypto import WindowsDpapiProtector
from integration.candidate_storage import CandidateStorage
from integration.module_adapters.face_identity import FaceIdentityAdapter
from integration.student_registration import StudentRegistrationService
from integration.tests.biometric_fakes import FakeBiometricProtector


def face_embedding(index: int) -> np.ndarray:
    vector = np.zeros(128, dtype=np.float32)
    vector[index] = 1.0
    return vector


class StudentRegistrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.tempdir.name) / "students.sqlite3"
        self.protector = FakeBiometricProtector()
        self.storage = CandidateStorage(
            self.database_path,
            biometric_protector=self.protector,
        )
        self.service = StudentRegistrationService(self.storage)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def register(self, **overrides) -> dict:
        arguments = {
            "full_name": "Test Student",
            "enrollment_number": "ENR-001",
            "email": "student@example.test",
            "face_embeddings": [
                face_embedding(0),
                face_embedding(1),
                face_embedding(2),
            ],
            "voice_embedding": np.array([3.0, 4.0, 0.0], dtype=np.float32),
        }
        arguments.update(overrides)
        return self.service.register_student(**arguments)

    def test_registration_persists_three_separate_faces_and_one_voice(self) -> None:
        result = self.register()

        self.assertEqual(result["registration_status"], "REGISTERED")
        self.assertEqual(result["face_registration_status"], "COMPLETE")
        self.assertEqual(result["voice_registration_status"], "COMPLETE")
        self.assertNotIn("face_embeddings", result)
        self.assertNotIn("voice_embedding", result)

        loaded = self.storage.load_student(result["student_id"])
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(len(loaded["face_embeddings"]), 3)
        self.assertEqual(loaded["voice_embedding"].shape, (3,))
        self.assertTrue(
            np.allclose(
                loaded["face_embeddings"],
                [
                    face_embedding(0),
                    face_embedding(1),
                    face_embedding(2),
                ],
            )
        )
        self.assertTrue(
            np.allclose(
                loaded["voice_embedding"],
                np.array([0.6, 0.8, 0.0], dtype=np.float32),
            )
        )
        listed = self.storage.list_students()
        self.assertEqual(len(listed), 1)
        self.assertNotIn("face_embeddings", listed[0])
        self.assertNotIn("voice_embedding", listed[0])

    def test_enrollment_number_uniqueness_ignores_case_and_outer_whitespace(self) -> None:
        self.register()

        with self.assertRaisesRegex(ValueError, "Enrollment number already exists"):
            self.register(enrollment_number="  enr-001 ")

        self.assertEqual(len(self.storage.list_students()), 1)

    def test_selected_student_loads_only_its_own_biometrics(self) -> None:
        first = self.register()
        second = self.register(
            full_name="Second Student",
            enrollment_number="ENR-002",
            face_embeddings=[
                face_embedding(3),
                face_embedding(4),
                face_embedding(5),
            ],
            voice_embedding=np.array([0.0, 0.0, 7.0], dtype=np.float32),
        )

        loaded = self.storage.load_student(first["student_id"])

        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded["student_id"], first["student_id"])
        self.assertNotEqual(loaded["student_id"], second["student_id"])
        self.assertTrue(
            np.allclose(
                loaded["face_embeddings"][0],
                face_embedding(0),
            )
        )
        self.assertTrue(
            np.allclose(
                loaded["voice_embedding"],
                np.array([0.6, 0.8, 0.0], dtype=np.float32),
            )
        )

    def test_registration_requires_details_three_distinct_faces_and_voice(self) -> None:
        invalid_cases = (
            ({"full_name": "  "}, "full name"),
            ({"enrollment_number": " "}, "Enrollment number"),
            (
                {"face_embeddings": [face_embedding(0), face_embedding(1)]},
                "Exactly three",
            ),
            (
                {
                    "face_embeddings": [
                        face_embedding(0),
                        face_embedding(0) * 2,
                        face_embedding(2),
                    ]
                },
                "distinct captures",
            ),
            (
                {"voice_embedding": np.zeros(3, dtype=np.float32)},
                "zero vector",
            ),
            (
                {"voice_embedding": np.array([1.0, np.nan], dtype=np.float32)},
                "finite",
            ),
        )
        for overrides, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.register(**overrides)

        self.assertEqual(self.storage.list_students(), [])

    def test_legacy_candidate_is_migrated_without_losing_its_embedding(self) -> None:
        legacy_path = Path(self.tempdir.name) / "legacy.sqlite3"
        legacy_embedding = face_embedding(7)
        connection = sqlite3.connect(str(legacy_path))
        try:
            connection.executescript(
                """
                CREATE TABLE candidates (
                    candidate_id TEXT PRIMARY KEY,
                    candidate_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE candidate_embeddings (
                    candidate_id TEXT NOT NULL,
                    embedding_index INTEGER NOT NULL,
                    embedding_blob BLOB NOT NULL,
                    embedding_dimension INTEGER NOT NULL,
                    landmarks_available INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (candidate_id, embedding_index),
                    FOREIGN KEY (candidate_id)
                        REFERENCES candidates(candidate_id)
                        ON DELETE CASCADE
                );
                """
            )
            connection.execute(
                """
                INSERT INTO candidates VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "legacy-student",
                    "Legacy Student",
                    "REGISTERED",
                    "2026-01-01T00:00:00+00:00",
                    "2026-01-01T00:00:00+00:00",
                ),
            )
            connection.execute(
                """
                INSERT INTO candidate_embeddings VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "legacy-student",
                    0,
                    legacy_embedding.tobytes(),
                    128,
                    1,
                    "2026-01-01T00:00:00+00:00",
                ),
            )
            connection.commit()
        finally:
            connection.close()

        migrated = CandidateStorage(
            legacy_path,
            biometric_protector=self.protector,
        )
        student = migrated.load_student("legacy-student")
        candidate = migrated.load_candidate("legacy-student")
        self.assertIsNotNone(student)
        self.assertEqual(student["full_name"], "Legacy Student")
        self.assertEqual(student["registration_status"], "LEGACY_INCOMPLETE")
        self.assertEqual(student["face_registration_status"], "LEGACY_INCOMPLETE")
        self.assertEqual(student["voice_registration_status"], "NOT_REGISTERED")
        self.assertIsNone(student["enrollment_number"])
        self.assertEqual(len(student["face_embeddings"]), 1)
        self.assertIsNone(student["voice_embedding"])
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertTrue(np.allclose(candidate["embedding"], legacy_embedding))
        self.assertEqual(migrated.list_candidates()[0]["status"], "REGISTERED")

        connection = sqlite3.connect(str(legacy_path))
        try:
            stored_blob = connection.execute(
                """
                SELECT embedding_blob FROM candidate_embeddings
                WHERE candidate_id = ?
                """,
                ("legacy-student",),
            ).fetchone()[0]
            schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
        finally:
            connection.close()
        self.assertTrue(stored_blob.startswith(CandidateStorage._ENCRYPTED_BLOB_PREFIX))
        self.assertNotEqual(stored_blob, legacy_embedding.tobytes())
        self.assertEqual(schema_version, CandidateStorage._SCHEMA_VERSION)

    def test_legacy_migration_failure_rolls_back_schema_and_plaintext(self) -> None:
        legacy_path = Path(self.tempdir.name) / "failed-legacy.sqlite3"
        legacy_embedding = face_embedding(9)
        connection = sqlite3.connect(str(legacy_path))
        try:
            connection.executescript(
                """
                CREATE TABLE candidates (
                    candidate_id TEXT PRIMARY KEY,
                    candidate_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE candidate_embeddings (
                    candidate_id TEXT NOT NULL,
                    embedding_index INTEGER NOT NULL,
                    embedding_blob BLOB NOT NULL,
                    embedding_dimension INTEGER NOT NULL,
                    landmarks_available INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (candidate_id, embedding_index)
                );
                """
            )
            connection.execute(
                "INSERT INTO candidates VALUES (?, ?, ?, ?, ?)",
                (
                    "legacy-student",
                    "Legacy Student",
                    "REGISTERED",
                    "2026-01-01T00:00:00+00:00",
                    "2026-01-01T00:00:00+00:00",
                ),
            )
            connection.execute(
                "INSERT INTO candidate_embeddings VALUES (?, ?, ?, ?, ?, ?)",
                (
                    "legacy-student",
                    0,
                    legacy_embedding.tobytes(),
                    128,
                    1,
                    "2026-01-01T00:00:00+00:00",
                ),
            )
            connection.commit()
        finally:
            connection.close()

        with self.assertRaisesRegex(OSError, "test encryption failure"):
            CandidateStorage(
                legacy_path,
                biometric_protector=FakeBiometricProtector(
                    failing_purpose="face_embedding",
                ),
            )

        connection = sqlite3.connect(str(legacy_path))
        try:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            stored_blob = connection.execute(
                "SELECT embedding_blob FROM candidate_embeddings"
            ).fetchone()[0]
            schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
        finally:
            connection.close()
        self.assertNotIn("students", tables)
        self.assertEqual(stored_blob, legacy_embedding.tobytes())
        self.assertEqual(schema_version, 0)

    def test_encryption_failure_rolls_back_every_registration_record(self) -> None:
        failing_path = Path(self.tempdir.name) / "failing.sqlite3"
        failing_storage = CandidateStorage(
            failing_path,
            biometric_protector=FakeBiometricProtector(
                failing_purpose="voice_embedding",
            ),
        )
        failing_service = StudentRegistrationService(failing_storage)

        with self.assertRaisesRegex(OSError, "test encryption failure"):
            failing_service.register_student(
                "Rollback Student",
                "ENR-ROLLBACK",
                [face_embedding(0), face_embedding(1), face_embedding(2)],
                np.array([1.0, 0.0], dtype=np.float32),
            )

        self.assertEqual(failing_storage.list_students(), [])
        self.assertEqual(failing_storage.list_candidates(), [])

    def test_windows_dpapi_round_trip(self) -> None:
        if os.name != "nt":
            self.skipTest("Windows DPAPI is available only on Windows.")

        protector = WindowsDpapiProtector()
        plaintext = np.array([0.25, 0.75], dtype=np.float32).tobytes()
        ciphertext = protector.protect(plaintext, "test_embedding")

        self.assertNotEqual(ciphertext, plaintext)
        self.assertEqual(
            protector.unprotect(ciphertext, "test_embedding"),
            plaintext,
        )
        with self.assertRaises(OSError):
            protector.unprotect(ciphertext, "different_purpose")

    @unittest.skipUnless(os.name == "nt", "Windows DPAPI is required.")
    def test_new_student_face_and_voice_blobs_are_dpapi_protected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "dpapi-student.sqlite3"
            storage = CandidateStorage(database)
            service = StudentRegistrationService(storage)
            student = service.register_student(
                "DPAPI Student",
                "ENR-DPAPI",
                [
                    face_embedding(0),
                    face_embedding(1),
                    face_embedding(2),
                ],
                np.array([3.0, 4.0, 0.0], dtype=np.float32),
            )

            connection = sqlite3.connect(str(database))
            try:
                face_blobs = connection.execute(
                    """
                    SELECT embedding_blob FROM candidate_embeddings
                    WHERE candidate_id = ? ORDER BY embedding_index
                    """,
                    (student["student_id"],),
                ).fetchall()
                voice_blob = connection.execute(
                    """
                    SELECT embedding_blob FROM student_voice_embeddings
                    WHERE student_id = ?
                    """,
                    (student["student_id"],),
                ).fetchone()[0]
            finally:
                connection.close()

            self.assertEqual(len(face_blobs), 3)
            self.assertTrue(
                all(
                    blob[0].startswith(CandidateStorage._ENCRYPTED_BLOB_PREFIX)
                    for blob in face_blobs
                )
            )
            self.assertTrue(
                voice_blob.startswith(CandidateStorage._ENCRYPTED_BLOB_PREFIX)
            )
            loaded = storage.load_student(student["student_id"])
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(len(loaded["face_embeddings"]), 3)
            self.assertIsNotNone(loaded["voice_embedding"])

    def test_exam_session_and_events_are_student_scoped(self) -> None:
        student = self.register()
        timestamp = "2026-11-21T12:00:00+05:30"
        session = self.storage.create_exam_session(
            student["student_id"],
            timestamp,
        )

        event = self.storage.record_proctoring_event(
            event_id="event-1",
            session_id=session["session_id"],
            student_id=student["student_id"],
            event_type="PHONE_DETECTED",
            module="phone_detection",
            observation_id="frame-123",
            observation_count=1,
            timestamp=timestamp,
            model_score=0.82,
            status="OBSERVED",
            reason="Phone confidence exceeded threshold.",
        )

        self.assertEqual(event["student_id"], student["student_id"])
        self.assertEqual(
            self.storage.list_session_events(session["session_id"])[0]["observation_id"],
            "frame-123",
        )
        self.storage.end_exam_session(session["session_id"], timestamp)
        ended = self.storage.load_exam_session(session["session_id"])
        self.assertIsNotNone(ended)
        assert ended is not None
        self.assertEqual(ended["status"], "ENDED")

    def test_legacy_student_cannot_create_exam_session(self) -> None:
        legacy_path = Path(self.tempdir.name) / "legacy-session.sqlite3"
        connection = sqlite3.connect(str(legacy_path))
        try:
            connection.executescript(
                """
                CREATE TABLE candidates (
                    candidate_id TEXT PRIMARY KEY,
                    candidate_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE candidate_embeddings (
                    candidate_id TEXT NOT NULL,
                    embedding_index INTEGER NOT NULL,
                    embedding_blob BLOB NOT NULL,
                    embedding_dimension INTEGER NOT NULL,
                    landmarks_available INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (candidate_id, embedding_index)
                );
                """
            )
            connection.execute(
                "INSERT INTO candidates VALUES (?, ?, ?, ?, ?)",
                (
                    "legacy-student",
                    "Legacy Student",
                    "REGISTERED",
                    "2026-01-01T00:00:00+00:00",
                    "2026-01-01T00:00:00+00:00",
                ),
            )
            connection.execute(
                "INSERT INTO candidate_embeddings VALUES (?, ?, ?, ?, ?, ?)",
                (
                    "legacy-student",
                    0,
                    face_embedding(0).tobytes(),
                    128,
                    1,
                    "2026-01-01T00:00:00+00:00",
                ),
            )
            connection.commit()
        finally:
            connection.close()

        storage = CandidateStorage(
            legacy_path,
            biometric_protector=self.protector,
        )
        with self.assertRaisesRegex(ValueError, "not fully registered"):
            storage.create_exam_session(
                "legacy-student",
                "2026-11-21T12:00:00+05:30",
            )


if __name__ == "__main__":
    unittest.main()
