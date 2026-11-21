import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from integration.candidate_identity import CandidateIdentityService
from integration.candidate_storage import CandidateStorage
from integration.module_adapters.face_identity import FaceIdentityAdapter
from integration.tests.biometric_fakes import FakeBiometricProtector


class FakeFaceAdapter:
    def __init__(self, face_sets):
        self.face_sets = list(face_sets)
        self.calls = 0

    def extract_all_face_embeddings(self, _frame):
        index = min(self.calls, len(self.face_sets) - 1)
        self.calls += 1
        return self.face_sets[index]


def embedding(value=1.0, offset=0.0):
    vector = np.zeros(128, dtype=np.float32)
    vector[0] = value
    if offset:
        vector[1] = offset
    return FaceIdentityAdapter.validate_embedding(vector)


class CandidateIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.tempdir.name) / "candidates.sqlite3"
        self.storage = CandidateStorage(
            self.database_path,
            biometric_protector=FakeBiometricProtector(),
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_registration_aggregates_normalizes_and_reloads(self):
        adapter = FakeFaceAdapter([
            [{
                "face_number": 1,
                "box": [1, 2, 3, 4],
                "embedding": np.array([1.0] + [0.0] * 127, dtype=np.float32),
                "landmarks_available": True,
            }],
        ])
        service = CandidateIdentityService(adapter, self.storage)
        record = service.register_candidate(
            "candidate-1",
            "Test Candidate",
            [object(), object()],
        )

        self.assertEqual(record["registration_status"], "REGISTERED")
        self.assertEqual(record["valid_sample_count"], 2)
        candidate = self.storage.load_candidate("candidate-1")
        self.assertEqual(candidate["embedding"].shape, (128,))
        self.assertAlmostEqual(
            float(np.linalg.norm(candidate["embedding"])),
            1.0,
            places=5,
        )

    def test_exact_sample_count_is_required_before_persistence(self):
        adapter = FakeFaceAdapter([[
            {
                "face_number": 1,
                "box": [1, 2, 3, 4],
                "embedding": embedding(),
                "landmarks_available": True,
            },
        ]])
        service = CandidateIdentityService(adapter, self.storage)

        with self.assertRaisesRegex(ValueError, "Expected 3 valid samples"):
            service.register_candidate(
                "candidate-incomplete",
                "Test Candidate",
                [object(), object()],
                expected_sample_count=3,
            )

        self.assertIsNone(self.storage.load_candidate("candidate-incomplete"))

    def test_expected_sample_count_accepts_exactly_three_samples(self):
        adapter = FakeFaceAdapter([[
            {
                "face_number": 1,
                "box": [1, 2, 3, 4],
                "embedding": embedding(),
                "landmarks_available": True,
            },
        ]])
        service = CandidateIdentityService(adapter, self.storage)

        record = service.register_candidate(
            "candidate-three-samples",
            "Test Candidate",
            [object(), object(), object()],
            expected_sample_count=3,
        )

        self.assertEqual(record["valid_sample_count"], 3)
        self.assertEqual(
            self.storage.load_candidate("candidate-three-samples")[
                "embedding"
            ].shape,
            (128,),
        )

    def test_registration_rejects_no_face_and_multiple_faces(self):
        adapter = FakeFaceAdapter([
            [],
            [
                {"face_number": 1, "embedding": embedding(), "landmarks_available": True},
                {"face_number": 2, "embedding": embedding(), "landmarks_available": True},
            ],
        ])
        service = CandidateIdentityService(adapter, self.storage)
        with self.assertRaisesRegex(ValueError, "no valid single-face"):
            service.register_candidate("candidate-1", "Test Candidate", [1, 2])

    def test_duplicate_candidate_requires_explicit_overwrite(self):
        self.storage.register_candidate(
            "candidate-1",
            "Test Candidate",
            [embedding()],
            "2026-01-01T00:00:00+00:00",
        )
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.storage.register_candidate(
                "candidate-1",
                "Other Candidate",
                [embedding()],
                "2026-01-02T00:00:00+00:00",
            )
        overwritten = self.storage.register_candidate(
            "candidate-1",
            "Replacement Candidate",
            [embedding()],
            "2026-01-02T00:00:00+00:00",
            overwrite=True,
        )
        self.assertEqual(overwritten["candidate_name"], "Replacement Candidate")

    def test_missing_candidate_and_no_face_statuses(self):
        adapter = FakeFaceAdapter([[]])
        service = CandidateIdentityService(adapter, self.storage)
        missing = service.verify_frame("missing", object())
        self.assertEqual(missing["identity_status"], "CANDIDATE_NOT_FOUND")

        self.storage.register_candidate(
            "candidate-1",
            "Test Candidate",
            [embedding()],
            "2026-01-01T00:00:00+00:00",
        )
        no_face = service.verify_frame("candidate-1", object())
        self.assertEqual(no_face["identity_status"], "CANDIDATE_NOT_FOUND")
        self.assertEqual(no_face["total_faces"], 0)

    def test_identified_candidate_and_unmatched_other_face(self):
        adapter = FakeFaceAdapter([[
            {
                "face_number": 1,
                "box": [0, 0, 10, 10],
                "embedding": embedding(),
                "landmarks_available": True,
            },
            {
                "face_number": 2,
                "box": [20, 20, 30, 30],
                "embedding": embedding(0.0, 1.0),
                "landmarks_available": True,
            },
        ]])
        service = CandidateIdentityService(adapter, self.storage)
        self.storage.register_candidate(
            "candidate-1",
            "Test Candidate",
            [embedding()],
            "2026-01-01T00:00:00+00:00",
        )
        result = service.verify_frame("candidate-1", object())

        self.assertEqual(result["identity_status"], "CANDIDATE_IDENTIFIED")
        self.assertEqual(len(result["matched_faces"]), 1)
        self.assertEqual(len(result["unmatched_faces"]), 1)
        self.assertEqual(result["unexpected_faces"][0]["face_number"], 2)
        self.assertNotEqual(
            result["identity_status"],
            "MULTIPLE_MATCHES_FOUND",
        )

    def test_unmatched_and_review_statuses(self):
        adapter = FakeFaceAdapter([[
            {
                "face_number": 1,
                "box": [0, 0, 10, 10],
                "embedding": np.zeros(128, dtype=np.float32),
                "landmarks_available": True,
            },
        ]])
        service = CandidateIdentityService(adapter, self.storage)
        self.storage.register_candidate(
            "candidate-1",
            "Test Candidate",
            [embedding()],
            "2026-01-01T00:00:00+00:00",
        )
        result = service.verify_frame("candidate-1", object())
        self.assertEqual(result["identity_status"], "REVIEW_REQUIRED")

        self.storage.register_candidate(
            "candidate-2",
            "Another Candidate",
            [embedding(0.0, 1.0)],
            "2026-01-01T00:00:00+00:00",
        )
        adapter.face_sets = [[{
            "face_number": 1,
            "box": [0, 0, 10, 10],
            "embedding": np.array([0.0] + [1.0] + [0.0] * 126, dtype=np.float32),
            "landmarks_available": True,
        }]]
        adapter.calls = 0
        unmatched = service.verify_frame("candidate-1", object())
        self.assertEqual(unmatched["identity_status"], "CANDIDATE_UNMATCHED")

    def test_multiple_candidate_matches_only_when_genuine(self):
        adapter = FakeFaceAdapter([[
            {
                "face_number": 1,
                "box": [0, 0, 10, 10],
                "embedding": embedding(),
                "landmarks_available": True,
            },
        ]])
        service = CandidateIdentityService(adapter, self.storage)
        for candidate_id in ("candidate-1", "candidate-2"):
            self.storage.register_candidate(
                candidate_id,
                candidate_id,
                [embedding()],
                "2026-01-01T00:00:00+00:00",
            )
        result = service.verify_frame("candidate-1", object())
        self.assertEqual(result["identity_status"], "MULTIPLE_MATCHES_FOUND")
        self.assertEqual(
            set(result["multiple_matches"][0]["candidate_ids"]),
            {"candidate-1", "candidate-2"},
        )

    def test_corrupted_storage_record_is_safe(self):
        self.storage.register_candidate(
            "candidate-1",
            "Test Candidate",
            [embedding()],
            "2026-01-01T00:00:00+00:00",
        )
        connection = sqlite3.connect(str(self.database_path))
        try:
            connection.execute(
                """
                UPDATE candidate_embeddings
                SET embedding_blob = ?
                WHERE candidate_id = ?
                """,
                (np.zeros(127, dtype=np.float32).tobytes(), "candidate-1"),
            )
            connection.commit()
        finally:
            connection.close()
        with self.assertRaisesRegex(ValueError, "Corrupted embedding"):
            self.storage.load_candidate("candidate-1")

    def test_invalid_dimensions_and_candidate_id(self):
        with self.assertRaisesRegex(ValueError, "dimension"):
            FaceIdentityAdapter.validate_embedding(np.ones(127))
        with self.assertRaisesRegex(ValueError, "Candidate ID"):
            self.storage.register_candidate(
                "",
                "Test Candidate",
                [embedding()],
                "2026-01-01T00:00:00+00:00",
            )


if __name__ == "__main__":
    unittest.main()
