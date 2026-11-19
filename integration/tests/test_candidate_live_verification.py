import unittest

import numpy as np

from integration.candidate_live_verification import (
    GREEN,
    RED,
    YELLOW,
    LiveMetrics,
    adapter_result_to_candidate_result,
    annotate_frame,
    capture_enrollment_frames,
    face_annotations,
    run_live_session,
)


class FakeCV2:
    FONT_HERSHEY_SIMPLEX = 0

    def __init__(self, keys=None):
        self.keys = list(keys or [])
        self.rectangles = []
        self.texts = []
        self.images = []

    def putText(self, image, text, *_args):
        self.texts.append(text)
        return image

    def imshow(self, window, image):
        self.images.append((window, image.copy()))
        return None

    def waitKey(self, _delay):
        return self.keys.pop(0) if self.keys else -1

    def rectangle(self, image, first, second, color, _thickness):
        self.rectangles.append((first, second, color))
        return image


class FakeCamera:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.read_count = 0
        self.released = False

    def __enter__(self):
        return self

    def read(self):
        self.read_count += 1
        return next(self.frames)

    def release(self):
        self.released = True

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.release()
        return False


class FakeFaceAdapter:
    def __init__(self, results):
        self.results = iter(results)

    def extract_all_face_embeddings(self, _frame):
        return next(self.results)


class FakeVerifier:
    def __init__(self, result):
        self.result = result

    def verify_registered_identity(self, _candidate, _frame):
        return self.result


def face(face_number, box, status, similarity):
    record = {
        "face_number": face_number,
        "box": box,
        "similarity": similarity,
    }
    if status == "CANDIDATE_IDENTIFIED":
        return record
    return {**record, "status": status}


class CandidateLiveVerificationTests(unittest.TestCase):
    def test_face_results_remain_bound_to_their_own_boxes(self):
        result = {
            "matched_faces": [
                face(1, [65, 10, 95, 40], "CANDIDATE_IDENTIFIED", 0.82),
            ],
            "unmatched_faces": [
                face(2, [5, 10, 35, 40], "CANDIDATE_UNMATCHED", 0.33),
            ],
        }

        annotations = face_annotations(result, frame_width=100)

        self.assertEqual(
            [(item["face_number"], item["box"], item["color"]) for item in annotations],
            [
                (1, [65, 10, 95, 40], GREEN),
                (2, [5, 10, 35, 40], RED),
            ],
        )
        self.assertEqual(annotations[0]["ground_truth_role"], "student")
        self.assertEqual(annotations[1]["ground_truth_role"], "volunteer")
        self.assertIsNone(annotations[0]["test_outcome"])
        self.assertIsNone(annotations[1]["test_outcome"])

    def test_actual_match_display_is_not_changed_by_test_position(self):
        result = {
            "matched_faces": [
                face(1, [5, 10, 35, 40], "CANDIDATE_IDENTIFIED", 0.82),
                face(2, [65, 10, 95, 40], "CANDIDATE_IDENTIFIED", 0.408207),
            ],
            "unmatched_faces": [],
        }

        left_candidate_ground_truth = face_annotations(
            result,
            frame_width=100,
            candidate_side="left",
        )
        right_candidate_ground_truth = face_annotations(
            result,
            frame_width=100,
            candidate_side="right",
        )

        self.assertEqual(left_candidate_ground_truth[0]["color"], GREEN)
        self.assertEqual(left_candidate_ground_truth[1]["color"], GREEN)
        self.assertEqual(
            left_candidate_ground_truth[1]["label"],
            "FACE MATCHED / IDENTIFIED",
        )
        self.assertEqual(
            left_candidate_ground_truth[1]["system_status"],
            "CANDIDATE_IDENTIFIED",
        )
        self.assertEqual(
            left_candidate_ground_truth[1]["test_outcome"],
            "FALSE ACCEPTANCE",
        )
        self.assertEqual(
            [item["label"] for item in right_candidate_ground_truth],
            [item["label"] for item in left_candidate_ground_truth],
        )
        self.assertEqual(
            [item["color"] for item in right_candidate_ground_truth],
            [item["color"] for item in left_candidate_ground_truth],
        )
        self.assertEqual(
            [item["ground_truth_role"] for item in left_candidate_ground_truth],
            ["student", "volunteer"],
        )
        self.assertEqual(
            [item["ground_truth_role"] for item in right_candidate_ground_truth],
            ["volunteer", "student"],
        )

    def test_false_acceptance_shows_measured_match_and_test_failure(self):
        cv2 = FakeCV2()
        frame = np.zeros((60, 100, 3), dtype=np.uint8)
        result = {
            "matched_faces": [
                face(1, [5, 10, 35, 40], "CANDIDATE_IDENTIFIED", 0.408207),
            ],
            "unmatched_faces": [],
        }

        _, annotations = annotate_frame(cv2, frame, result)

        self.assertEqual(annotations[0]["label"], "FACE MATCHED / IDENTIFIED")
        self.assertEqual(annotations[0]["color"], GREEN)
        self.assertEqual(annotations[0]["test_outcome"], "FALSE ACCEPTANCE")
        self.assertTrue(any("0.408" in text for text in cv2.texts))
        self.assertIn("FALSE ACCEPTANCE", cv2.texts)

    def test_failed_live_frame_stays_visible_without_more_camera_reads(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        camera = FakeCamera([frame])
        cv2 = FakeCV2([ord("q")])
        verifier = FakeVerifier({
            "candidate_id": "candidate-d2",
            "identity_status": "CANDIDATE_NOT_FOUND",
            "total_faces": 2,
            "face_results": [
                {
                    "face_number": 1,
                    "box": [5, 20, 30, 55],
                    "similarity": 0.714,
                    "predicted": "MATCH",
                },
                {
                    "face_number": 2,
                    "box": [65, 20, 90, 55],
                    "similarity": 0.344,
                    "predicted": "MISMATCH",
                },
            ],
        })

        metrics = run_live_session(
            camera,
            verifier,
            {"candidate_id": "candidate-d2"},
            cv2,
        )

        self.assertEqual(camera.read_count, 1)
        self.assertTrue(camera.released)
        self.assertEqual(len(cv2.images), 1)
        self.assertIn("stopped result", cv2.images[0][0])
        self.assertTrue(
            any("FACE MATCHED / IDENTIFIED" in text for text in cv2.texts)
        )
        self.assertIn("FALSE ACCEPTANCE", cv2.texts)
        self.assertTrue(any("FACE UNMATCHED" in text for text in cv2.texts))
        self.assertIn("FALSE REJECTION", cv2.texts)
        self.assertEqual(metrics.false_acceptance_count, 1)
        self.assertEqual(metrics.false_rejection_count, 1)

    def test_adapter_predictions_keep_candidate_id_scores_and_boxes_per_face(self):
        adapter_result = {
            "candidate_id": "candidate-d2",
            "identity_status": "MULTIPLE_MATCHES_FOUND",
            "total_faces": 2,
            "face_results": [
                {
                    "face_number": 1,
                    "box": [5, 10, 35, 40],
                    "similarity": 0.82,
                    "predicted": "MATCH",
                },
                {
                    "face_number": 2,
                    "box": [65, 10, 95, 40],
                    "similarity": 0.33,
                    "predicted": "MISMATCH",
                },
            ],
        }

        result = adapter_result_to_candidate_result(adapter_result)
        annotations = face_annotations(result, frame_width=100)

        self.assertEqual(result["candidate_id"], "candidate-d2")
        self.assertEqual(
            [(item["face_number"], item["box"], item["system_status"]) for item in annotations],
            [
                (1, [5, 10, 35, 40], "CANDIDATE_IDENTIFIED"),
                (2, [65, 10, 95, 40], "CANDIDATE_UNMATCHED"),
            ],
        )
        self.assertEqual([item["similarity"] for item in annotations], [0.82, 0.33])

    def test_adapter_review_prediction_preserves_face_box_and_timings(self):
        adapter_result = {
            "candidate_id": "candidate-d2",
            "identity_status": "REVIEW_REQUIRED",
            "total_faces": 1,
            "timings_ms": {
                "face_detection_ms": 4.0,
                "embedding_extraction_ms": 6.0,
                "similarity_comparison_ms": 0.1,
                "verification_total_ms": 10.1,
            },
            "face_results": [
                {
                    "face_number": 1,
                    "box": [5, 10, 35, 40],
                    "similarity": None,
                    "predicted": "REVIEW_REQUIRED",
                },
            ],
        }

        result = adapter_result_to_candidate_result(adapter_result)
        annotations = face_annotations(result, frame_width=100)

        self.assertEqual(
            (annotations[0]["box"], annotations[0]["color"], annotations[0]["label"]),
            ([5, 10, 35, 40], YELLOW, "REVIEW REQUIRED"),
        )
        self.assertEqual(result["timings_ms"], adapter_result["timings_ms"])

    def test_review_status_is_yellow_and_missing_faces_leave_no_stale_boxes(self):
        cv2 = FakeCV2()
        frame = np.zeros((60, 100, 3), dtype=np.uint8)
        review_result = {
            "matched_faces": [],
            "unmatched_faces": [
                face(1, [10, 10, 30, 30], "REVIEW_REQUIRED", None),
            ],
        }
        first_image, first_annotations = annotate_frame(
            cv2,
            frame,
            review_result,
        )
        second_image, second_annotations = annotate_frame(
            cv2,
            frame,
            {"identity_status": "CANDIDATE_NOT_FOUND", "total_faces": 0},
        )

        self.assertEqual(first_annotations[0]["color"], YELLOW)
        self.assertEqual(len(cv2.rectangles), 1)
        self.assertEqual(len(second_annotations), 0)
        self.assertIsNot(first_image, frame)
        self.assertIsNot(second_image, frame)
        self.assertTrue(np.all(frame == 0))

    def test_exactly_three_enrollment_attempts_are_kept_in_memory(self):
        frames = [
            np.full((2, 2, 3), value, dtype=np.uint8)
            for value in (1, 2, 3)
        ]
        embedding = np.array([1.0] + [0.0] * 127, dtype=np.float32)
        adapter = FakeFaceAdapter([
            [{
                "embedding": embedding,
                "landmarks_available": True,
            }]
            for _ in frames
        ])
        cv2 = FakeCV2([ord("c"), ord("c"), ord("c")])

        accepted, rejected = capture_enrollment_frames(
            FakeCamera(frames),
            adapter,
            cv2,
        )

        self.assertEqual(len(accepted), 3)
        self.assertEqual(rejected, [])
        self.assertTrue(
            all(np.array_equal(sample, original) for sample, original in zip(accepted, frames))
        )

    def test_invalid_enrollment_frame_is_rejected(self):
        frames = [
            np.full((2, 2, 3), value, dtype=np.uint8)
            for value in (1, 2, 3)
        ]
        adapter = FakeFaceAdapter([[]])
        cv2 = FakeCV2([ord("c"), ord("c"), ord("c")])
        camera = FakeCamera(frames)
        valid_embedding = np.array([1.0] + [0.0] * 127, dtype=np.float32)
        adapter.results = iter([
            [],
            [{
                "embedding": valid_embedding,
                "landmarks_available": True,
            }],
            [{
                "embedding": valid_embedding,
                "landmarks_available": True,
            }],
        ])

        accepted, rejected = capture_enrollment_frames(camera, adapter, cv2)

        self.assertEqual(len(accepted), 2)
        self.assertEqual(rejected, ["No face detected."])

    def test_duplicate_enrollment_frame_is_rejected(self):
        frame = np.ones((2, 2, 3), dtype=np.uint8)
        third_frame = np.full((2, 2, 3), 2, dtype=np.uint8)
        embedding = np.array([1.0] + [0.0] * 127, dtype=np.float32)
        face_result = {
            "embedding": embedding,
            "landmarks_available": True,
        }
        adapter = FakeFaceAdapter([
            [face_result],
            [face_result],
        ])

        accepted, rejected = capture_enrollment_frames(
            FakeCamera([frame, frame.copy(), third_frame]),
            adapter,
            FakeCV2([ord("c"), ord("c"), ord("c")]),
        )

        self.assertEqual(len(accepted), 2)
        self.assertEqual(rejected, ["Duplicate frame."])

    def test_metrics_count_false_acceptance_and_correct_frames(self):
        metrics = LiveMetrics()
        correct = {
            "total_faces": 2,
            "matched_faces": [
                face(2, [65, 10, 95, 40], "CANDIDATE_IDENTIFIED", 0.82),
            ],
            "unmatched_faces": [
                face(1, [5, 10, 35, 40], "CANDIDATE_UNMATCHED", 0.33),
            ],
        }
        false_acceptance = {
            "total_faces": 2,
            "matched_faces": [
                face(2, [65, 10, 95, 40], "CANDIDATE_IDENTIFIED", 0.82),
                face(1, [5, 10, 35, 40], "CANDIDATE_IDENTIFIED", 0.408207),
            ],
            "unmatched_faces": [],
        }
        correct["timings_ms"] = {
            "face_detection_ms": 3.0,
            "embedding_extraction_ms": 5.0,
            "similarity_comparison_ms": 0.1,
            "verification_total_ms": 8.5,
        }
        false_acceptance["timings_ms"] = correct["timings_ms"]

        metrics.record(correct, 20.0, 100)
        metrics.record(false_acceptance, 25.0, 100)

        self.assertEqual(metrics.frame_count, 2)
        self.assertEqual(metrics.detected_face_count, 4)
        self.assertEqual(metrics.student_match_count, 2)
        self.assertEqual(metrics.volunteer_unmatched_count, 1)
        self.assertEqual(metrics.false_acceptance_count, 1)
        self.assertEqual(metrics.both_labels_correct_count, 1)
        self.assertAlmostEqual(metrics.average_latency_ms, 22.5)
        self.assertAlmostEqual(metrics.processing_fps, 1000.0 / 22.5)
        self.assertEqual(len(metrics.face_scores), 4)
        self.assertAlmostEqual(metrics.average_face_detection_ms, 3.0)
        self.assertAlmostEqual(metrics.average_embedding_extraction_ms, 5.0)
        self.assertAlmostEqual(metrics.average_similarity_ms, 0.1)
        self.assertAlmostEqual(metrics.average_verification_ms, 8.5)

    def test_right_registered_participant_mapping_counts_screenshot_correctly(self):
        metrics = LiveMetrics()
        result = {
            "total_faces": 2,
            "matched_faces": [
                face(1, [400, 100, 550, 300], "CANDIDATE_IDENTIFIED", 0.469),
            ],
            "unmatched_faces": [
                face(2, [20, 100, 170, 300], "CANDIDATE_UNMATCHED", 0.176),
            ],
        }

        annotations = metrics.record(result, 250.0, frame_width=640)

        self.assertEqual(
            [(item["ground_truth_role"], item["system_status"]) for item in annotations],
            [
                ("student", "CANDIDATE_IDENTIFIED"),
                ("volunteer", "CANDIDATE_UNMATCHED"),
            ],
        )
        self.assertEqual(metrics.student_match_count, 1)
        self.assertEqual(metrics.volunteer_unmatched_count, 1)
        self.assertEqual(metrics.false_acceptance_count, 0)
        self.assertEqual(metrics.false_rejection_count, 0)
        self.assertEqual(metrics.both_labels_correct_count, 1)


if __name__ == "__main__":
    unittest.main()
