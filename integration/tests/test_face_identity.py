import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from integration.config import (
    FACE_IDENTITY_THRESHOLD,
    FACE_LANDMARKER_MODEL_PATH,
    FACE_RECOGNITION_MODEL_PATH,
)
from integration.module_adapters.face_identity import FaceIdentityAdapter


class FakeCV2:
    COLOR_BGR2RGB = 1

    def __init__(self, image=None):
        self.reference_image = image
        self.input_frames = []

    def cvtColor(self, frame, _conversion):
        self.input_frames.append(frame)
        return frame

    def imread(self, _path):
        return self.reference_image


class FakeMediaPipe:
    ImageFormat = SimpleNamespace(SRGB=1)

    @staticmethod
    def Image(**kwargs):
        return SimpleNamespace(**kwargs)


class FakeLandmarker:
    def __init__(self, face_count=1):
        self.face_count = face_count

    def detect(self, _image):
        points = [
            SimpleNamespace(x=0.5, y=0.5)
            for _ in range(478)
        ]
        points[0] = SimpleNamespace(x=0.2, y=0.2)
        points[1] = SimpleNamespace(x=0.8, y=0.8)
        return SimpleNamespace(
            face_landmarks=[points for _ in range(self.face_count)]
        )


class FakeRecognizer:
    def __init__(self, embeddings=None):
        padded = []
        for embedding in embeddings or [np.array([1.0, 0.0])]:
            vector = np.zeros(128, dtype=np.float32)
            vector[:len(embedding)] = np.asarray(embedding, dtype=np.float32)
            padded.append(vector)
        self.embeddings = padded
        self.aligned_count = 0

    def alignCrop(self, _image, _face_data):
        self.aligned_count += 1
        return self.aligned_count

    def feature(self, aligned_face):
        index = (aligned_face - 1) % len(self.embeddings)
        return np.asarray(self.embeddings[index], dtype=np.float32)


def make_adapter(face_count=1, embeddings=None, reference_image=None):
    cv2_module = FakeCV2(reference_image)
    adapter = FaceIdentityAdapter(
        landmarker=FakeLandmarker(face_count),
        recognizer=FakeRecognizer(embeddings),
        cv2_module=cv2_module,
        mediapipe_module=FakeMediaPipe,
    )
    return adapter, cv2_module


class FaceIdentityAdapterTests(unittest.TestCase):
    def test_embedding_validation_rejects_bad_vectors(self):
        valid = np.ones(128, dtype=np.float32)
        normalized = FaceIdentityAdapter.validate_embedding(valid)
        self.assertEqual(normalized.shape, (128,))
        self.assertAlmostEqual(float(np.linalg.norm(normalized)), 1.0, places=5)

        with self.assertRaisesRegex(ValueError, "dimension"):
            FaceIdentityAdapter.validate_embedding(np.ones(127, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "zero vector"):
            FaceIdentityAdapter.validate_embedding(np.zeros(128, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "finite"):
            FaceIdentityAdapter.validate_embedding(
                np.full(128, np.nan, dtype=np.float32)
            )

    def test_uses_existing_model_paths_and_threshold(self):
        self.assertTrue(FACE_LANDMARKER_MODEL_PATH.is_file())
        self.assertTrue(FACE_RECOGNITION_MODEL_PATH.is_file())
        self.assertEqual(FACE_IDENTITY_THRESHOLD, 0.3822)
        adapter, _ = make_adapter()
        self.assertEqual(adapter.identity_threshold, 0.3822)

    def test_initializes_models_from_existing_paths(self):
        calls = {}

        class FakeBaseOptions:
            def __init__(self, model_asset_path):
                calls["landmarker_path"] = model_asset_path

        class FakeFaceLandmarkerOptions:
            def __init__(self, **kwargs):
                calls["landmarker_options"] = kwargs

        class FakeFaceLandmarkerFactory:
            @staticmethod
            def create_from_options(options):
                calls["landmarker_instance_options"] = options
                return object()

        def create_recognizer(model_path, config_path):
            calls["recognizer_path"] = model_path
            calls["recognizer_config"] = config_path
            return object()

        cv2_module = FakeCV2()
        cv2_module.FaceRecognizerSF = SimpleNamespace(
            create=create_recognizer
        )
        mediapipe_module = SimpleNamespace(
            tasks=SimpleNamespace(
                BaseOptions=FakeBaseOptions,
                vision=SimpleNamespace(
                    FaceLandmarkerOptions=FakeFaceLandmarkerOptions,
                    RunningMode=SimpleNamespace(IMAGE="image"),
                    FaceLandmarker=FakeFaceLandmarkerFactory,
                ),
            )
        )

        adapter = FaceIdentityAdapter(
            cv2_module=cv2_module,
            mediapipe_module=mediapipe_module,
        )

        self.assertEqual(
            calls["landmarker_path"],
            str(FACE_LANDMARKER_MODEL_PATH),
        )
        self.assertEqual(
            calls["recognizer_path"],
            str(FACE_RECOGNITION_MODEL_PATH),
        )
        self.assertEqual(calls["recognizer_config"], "")
        self.assertEqual(
            calls["landmarker_options"]["running_mode"],
            "image",
        )
        self.assertEqual(calls["landmarker_options"]["num_faces"], 5)
        self.assertEqual(adapter.identity_threshold, 0.3822)

    def test_registers_one_face_and_identifies_candidate(self):
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        adapter, cv2_module = make_adapter(reference_image=frame)
        candidate = adapter.register_candidate(
            "candidate-reference.jpg",
            "candidate-1",
        )

        result = adapter.verify_registered_identity(candidate, frame)

        self.assertEqual(candidate["candidate_id"], "candidate-1")
        self.assertEqual(
            result["monitoring_status"],
            "SINGLE_FACE_DETECTED",
        )
        self.assertEqual(result["identity_status"], "CANDIDATE_IDENTIFIED")
        self.assertEqual(result["matching_faces_count"], 1)
        self.assertEqual(result["best_match"]["similarity"], 1.0)
        self.assertIs(result["image"], frame)
        self.assertIs(cv2_module.input_frames[-1], frame)

    def test_identity_threshold_separates_match_from_mismatch(self):
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        candidate = {
            "candidate_id": "candidate-1",
            "embedding": np.array([1.0] + [0.0] * 127, dtype=np.float32),
        }

        matching, _ = make_adapter(
            embeddings=[np.array([0.40, 0.9165151])],
        )
        mismatching, _ = make_adapter(
            embeddings=[np.array([0.38, 0.9252027])],
        )

        self.assertEqual(
            matching.verify_registered_identity(candidate, frame)[
                "identity_status"
            ],
            "CANDIDATE_IDENTIFIED",
        )
        self.assertEqual(
            mismatching.verify_registered_identity(candidate, frame)[
                "identity_status"
            ],
            "CANDIDATE_NOT_FOUND",
        )

    def test_existing_threshold_still_accepts_known_impostor_score(self):
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        candidate = {
            "candidate_id": "candidate-1",
            "embedding": np.array([1.0] + [0.0] * 127, dtype=np.float32),
        }
        impostor_embedding = np.array(
            [0.408207, np.sqrt(1.0 - 0.408207**2)],
            dtype=np.float32,
        )
        adapter, _ = make_adapter(embeddings=[impostor_embedding])

        result = adapter.verify_registered_identity(candidate, frame)

        self.assertAlmostEqual(
            result["face_results"][0]["similarity"],
            0.408207,
            places=5,
        )
        self.assertEqual(result["face_results"][0]["predicted"], "MATCH")
        self.assertEqual(result["identity_status"], "CANDIDATE_IDENTIFIED")

    def test_threshold_boundary_is_inclusive_and_timings_are_reported(self):
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        candidate = {
            "candidate_id": "candidate-1",
            "embedding": np.array([1.0] + [0.0] * 127, dtype=np.float32),
        }
        threshold = FACE_IDENTITY_THRESHOLD
        matching, _ = make_adapter(
            embeddings=[
                np.array(
                    [threshold, np.sqrt(1.0 - threshold**2)],
                    dtype=np.float32,
                )
            ],
        )

        result = matching.verify_registered_identity(candidate, frame)

        score = result["face_results"][0]["similarity"]
        self.assertGreaterEqual(score, threshold)
        self.assertEqual(result["face_results"][0]["predicted"], "MATCH")
        self.assertIn("face_detection_ms", result["timings_ms"])
        self.assertIn("embedding_extraction_ms", result["timings_ms"])
        self.assertIn("similarity_comparison_ms", result["timings_ms"])

    def test_invalid_reference_and_bad_frame_are_rejected_or_reviewed(self):
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        adapter, _ = make_adapter()
        with self.assertRaisesRegex(ValueError, "Invalid registered candidate"):
            adapter.verify_registered_identity(
                {"candidate_id": "candidate-1", "embedding": np.zeros(128)},
                frame,
            )
        with self.assertRaisesRegex(ValueError, "non-empty BGR"):
            adapter.verify_registered_identity(
                {
                    "candidate_id": "candidate-1",
                    "embedding": np.array([1.0] + [0.0] * 127),
                },
                np.zeros((0, 30, 3), dtype=np.uint8),
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            adapter.verify_registered_identity(
                {
                    "candidate_id": "candidate-1",
                    "embedding": np.array([1.0] + [0.0] * 127),
                },
                np.full((20, 30, 3), np.nan, dtype=np.float32),
            )

        invalid_faces, _ = make_adapter(
            embeddings=[np.zeros(128, dtype=np.float32)],
        )
        result = invalid_faces.verify_registered_identity(
            {
                "candidate_id": "candidate-1",
                "embedding": np.array([1.0] + [0.0] * 127),
            },
            frame,
        )
        self.assertEqual(result["face_results"][0]["predicted"], "REVIEW_REQUIRED")
        self.assertIsNotNone(result["face_results"][0]["box"])

    def test_reports_absent_and_multiple_faces(self):
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        candidate = {
            "candidate_id": "candidate-1",
            "embedding": np.array([1.0] + [0.0] * 127, dtype=np.float32),
        }

        absent, _ = make_adapter(face_count=0)
        absent_result = absent.verify_registered_identity(candidate, frame)
        self.assertEqual(absent_result["monitoring_status"], "NO_FACE_DETECTED")
        self.assertEqual(absent_result["identity_status"], "NOT_VERIFIED")

        multiple, _ = make_adapter(
            face_count=2,
            embeddings=[
                np.array([1.0] + [0.0] * 127),
                np.array([1.0] + [0.0] * 127),
            ],
        )
        multiple_result = multiple.verify_registered_identity(candidate, frame)
        self.assertEqual(
            multiple_result["monitoring_status"],
            "MULTIPLE_FACES_DETECTED",
        )
        self.assertEqual(
            multiple_result["identity_status"],
            "MULTIPLE_MATCHES_FOUND",
        )

    def test_registration_requires_exactly_one_face(self):
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        absent, _ = make_adapter(face_count=0, reference_image=frame)
        with self.assertRaisesRegex(ValueError, "No face detected"):
            absent.register_candidate("reference.jpg", "candidate-1")

        multiple, _ = make_adapter(face_count=2, reference_image=frame)
        with self.assertRaisesRegex(ValueError, "Multiple faces detected"):
            multiple.register_candidate("reference.jpg", "candidate-1")

    def test_rejects_non_color_frame(self):
        adapter, _ = make_adapter()
        with self.assertRaisesRegex(ValueError, "three channels"):
            adapter.verify_registered_identity(
                {
                    "candidate_id": "candidate-1",
                    "embedding": np.array([1.0] + [0.0] * 127),
                },
                np.zeros((20, 30), dtype=np.uint8),
            )


if __name__ == "__main__":
    unittest.main()
