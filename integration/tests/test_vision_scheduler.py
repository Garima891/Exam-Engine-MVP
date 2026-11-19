import threading
import unittest
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from integration.vision_benchmark import (
    MeasuredOutput,
    _box_iou,
    _compare_landmarks,
    _match_boxes,
    _run_paired_mode,
)
from integration.vision_scheduler import (
    CapturedFrame,
    LatestFrameMailbox,
    VisionFrameProcessor,
)


class RecordingAdapter:
    def __init__(self):
        self.frames = []

    def detect(self, frame):
        self.frames.append(frame)
        return "phone"

    def extract_all_face_embeddings(self, frame):
        self.frames.append(frame)
        return "face"


class VisionSchedulerTests(unittest.TestCase):
    def test_mailbox_keeps_only_the_latest_pending_frame(self):
        mailbox = LatestFrameMailbox()
        first = CapturedFrame(1, 1.0, bytearray(b"1"))
        latest = CapturedFrame(2, 2.0, bytearray(b"2"))

        self.assertFalse(mailbox.publish(first))
        self.assertTrue(mailbox.publish(latest))
        self.assertEqual(mailbox.replaced_count, 1)
        self.assertIs(mailbox.take(timeout=0), latest)
        mailbox.close()
        self.assertIsNone(mailbox.take(timeout=0))

    def test_sequential_adapters_receive_the_same_unmodified_frame(self):
        frame_data = bytearray(b"original")
        phone = RecordingAdapter()
        face = RecordingAdapter()
        processor = VisionFrameProcessor(phone, face, mode="sequential")
        try:
            processed = processor.process(CapturedFrame(7, 0.0, frame_data))
        finally:
            processor.close()

        self.assertIs(phone.frames[0], frame_data)
        self.assertIs(face.frames[0], frame_data)
        self.assertEqual(frame_data, bytearray(b"original"))
        self.assertEqual(processed.phone.value, "phone")
        self.assertEqual(processed.face.value, "face")

    def test_concurrent_adapters_receive_the_same_frame_and_overlap(self):
        frame_data = bytearray(b"original")
        barrier = threading.Barrier(2)
        received = []

        class ConcurrentAdapter:
            def detect(self, frame):
                received.append(frame)
                barrier.wait(timeout=2)
                return "phone"

            def extract_all_face_embeddings(self, frame):
                received.append(frame)
                barrier.wait(timeout=2)
                return "face"

        processor = VisionFrameProcessor(
            ConcurrentAdapter(),
            ConcurrentAdapter(),
            mode="concurrent",
        )
        try:
            processed = processor.process(CapturedFrame(8, 0.0, frame_data))
        finally:
            processor.close()

        self.assertEqual(len(received), 2)
        self.assertTrue(all(frame is frame_data for frame in received))
        self.assertEqual(frame_data, bytearray(b"original"))
        self.assertEqual(processed.phone.value, "phone")
        self.assertEqual(processed.face.value, "face")

    def test_paired_modes_receive_the_same_captured_frame_object(self):
        frame_data = bytearray(b"paired-frame")
        captured_frame = CapturedFrame(9, 0.0, frame_data)
        phone = RecordingAdapter()
        face = RecordingAdapter()
        sequential = VisionFrameProcessor(phone, face, mode="sequential")
        concurrent = VisionFrameProcessor(phone, face, mode="concurrent")
        try:
            sequential_result = sequential.process(captured_frame)
            concurrent_result = concurrent.process(captured_frame)
        finally:
            sequential.close()
            concurrent.close()

        self.assertIs(sequential_result.frame, captured_frame)
        self.assertIs(concurrent_result.frame, captured_frame)
        self.assertEqual(len(phone.frames), 2)
        self.assertEqual(len(face.frames), 2)
        self.assertTrue(all(frame is frame_data for frame in phone.frames + face.frames))
        self.assertEqual(frame_data, bytearray(b"paired-frame"))

    def test_box_overlap_comparison_reports_iou_and_unmatched_boxes(self):
        self.assertEqual(_box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        comparison = _match_boxes(
            [
                {"class_id": 0, "bbox": [0, 0, 10, 10], "confidence": 0.8},
                {"class_id": 0, "bbox": [20, 20, 30, 30], "confidence": 0.7},
            ],
            [{"class_id": 0, "bbox": [0, 0, 10, 10], "confidence": 0.75}],
        )

        self.assertEqual(comparison["pairs"][0]["iou"], 1.0)
        self.assertAlmostEqual(comparison["pairs"][0]["confidence_delta"], 0.05)
        self.assertEqual(comparison["unmatched_sequential"], [1])
        self.assertEqual(comparison["unmatched_concurrent"], [])

    def test_landmark_comparison_reports_exactness_and_coordinate_delta(self):
        sequential = [[(0.1, 0.2, 0.0), (0.3, 0.4, 0.1)]]
        identical = [[(0.1, 0.2, 0.0), (0.3, 0.4, 0.1)]]
        changed = [[(0.1, 0.2, 0.0), (0.3, 0.42, 0.1)]]

        self.assertTrue(_compare_landmarks(sequential, identical)["exact_match"])
        changed_comparison = _compare_landmarks(sequential, changed)
        self.assertFalse(changed_comparison["exact_match"])
        self.assertAlmostEqual(changed_comparison["max_abs_delta"], 0.02)
        self.assertIsNone(_compare_landmarks([], [])["exact_match"])

    def test_paired_benchmark_compares_both_modes_on_one_retained_frame(self):
        source_frame = np.zeros((2, 2, 3), dtype=np.uint8)
        packet = CapturedFrame(10, perf_counter() - 0.01, source_frame, 2.0)
        received_phone_frames = []
        received_face_frames = []

        class FakePhone:
            def detect(self, frame):
                received_phone_frames.append(frame)
                return MeasuredOutput(
                    {
                        "detections": [
                            {
                                "class_id": 0,
                                "class_name": "phone",
                                "confidence": 0.8,
                                "bbox_xyxy": [0.0, 0.0, 1.0, 1.0],
                            }
                        ]
                    },
                    {},
                )

        class FakeFace:
            def __init__(self, detected=True):
                self.detected = detected

            def extract_all_face_embeddings(self, frame):
                received_face_frames.append(frame)
                if not self.detected:
                    return MeasuredOutput([], {}, [])
                return MeasuredOutput(
                    [{"embedding": [1.0], "box": [0, 0, 1, 1]}],
                    {},
                    [[(0.1, 0.2, 0.0)]],
                )

        class FakeWorker:
            def __init__(self):
                self.mailbox = LatestFrameMailbox()
                self.mailbox.publish(packet)
                self.camera_opened = True
                self.failed_reads = 0
                self.read_count = 1

            def raise_if_failed(self):
                return None

            def stop(self):
                self.mailbox.close()

        class FakeProcess:
            def __init__(self):
                self.cpu = 0.0

            def cpu_times(self):
                self.cpu += 0.01
                return SimpleNamespace(user=self.cpu, system=0.0)

        summary = _run_paired_mode(
            FakeWorker(),
            FakePhone(),
            FakeFace(),
            cv2=None,
            process=FakeProcess(),
            scenario="phone_near_face",
            frames=1,
            warmup=0,
            preview=False,
        )

        self.assertEqual(summary["measured_frames"], 1)
        self.assertEqual(summary["frame_integrity_failures"], 0)
        self.assertTrue(summary["frames"][0]["same_frame_object"])
        self.assertTrue(
            all(
                frame is received_phone_frames[0]
                for frame in received_phone_frames + received_face_frames
            )
        )
        self.assertEqual(
            summary["paired_comparison"]["phone_box_iou"]["mean"],
            1.0,
        )
        self.assertEqual(
            summary["paired_comparison"]["exact_landmark_match_frames"],
            1,
        )

        no_face_summary = _run_paired_mode(
            FakeWorker(),
            FakePhone(),
            FakeFace(detected=False),
            cv2=None,
            process=FakeProcess(),
            scenario="no_face",
            frames=1,
            warmup=0,
            preview=False,
        )
        self.assertEqual(
            no_face_summary["paired_comparison"]["exact_landmark_match_frames"],
            0,
        )
        self.assertEqual(
            no_face_summary["paired_comparison"][
                "landmark_availability_agreement_frames"
            ],
            1,
        )

    def test_paired_benchmark_preserves_two_independent_faces(self):
        source_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        packet = CapturedFrame(11, perf_counter() - 0.01, source_frame, 2.0)

        class FakePhone:
            def detect(self, frame):
                return MeasuredOutput(
                    {"detections": []},
                    {},
                )

        class FakeFace:
            def extract_all_face_embeddings(self, frame):
                return MeasuredOutput(
                    [
                        {"embedding": [1.0, 0.0], "box": [0, 0, 4, 4]},
                        {"embedding": [0.0, 1.0], "box": [5, 5, 9, 9]},
                    ],
                    {},
                    [
                        [(0.1, 0.1, 0.0), (0.2, 0.2, 0.0)],
                        [(0.7, 0.7, 0.0), (0.8, 0.8, 0.0)],
                    ],
                )

        class FakeWorker:
            def __init__(self):
                self.mailbox = LatestFrameMailbox()
                self.mailbox.publish(packet)
                self.camera_opened = True
                self.failed_reads = 0
                self.read_count = 1
                self.mailbox_replaced_count = 0

            def raise_if_failed(self):
                return None

            def stop(self):
                self.mailbox.close()

        class FakeProcess:
            def cpu_times(self):
                return SimpleNamespace(user=0.01, system=0.0)

        summary = _run_paired_mode(
            FakeWorker(),
            FakePhone(),
            FakeFace(),
            cv2=None,
            process=FakeProcess(),
            scenario="two_faces_no_phone",
            frames=1,
            warmup=0,
            preview=False,
        )
        row = summary["frames"][0]

        self.assertEqual(row["sequential"]["face_count"], 2)
        self.assertEqual(row["concurrent"]["face_count"], 2)
        self.assertTrue(row["phone_count_agrees"])
        self.assertTrue(row["face_count_agrees"])
        self.assertTrue(row["sequential"]["landmarks_available"])
        self.assertTrue(row["sequential"]["embedding_available"])
        self.assertTrue(row["concurrent"]["embedding_available"])
        self.assertEqual(
            summary["paired_comparison"]["face_count_agreement_frames"],
            1,
        )
        self.assertEqual(
            summary["mode_results"]["sequential"]["face_extra_face_frames"],
            0,
        )

    def test_paired_benchmark_reports_partial_two_face_misses(self):
        source_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        packet = CapturedFrame(12, perf_counter() - 0.01, source_frame, 2.0)

        class FakePhone:
            def detect(self, frame):
                return MeasuredOutput({"detections": []}, {})

        class FakeOneFace:
            def extract_all_face_embeddings(self, frame):
                return MeasuredOutput(
                    [{"embedding": [1.0, 0.0], "box": [0, 0, 4, 4]}],
                    {},
                    [[(0.1, 0.1, 0.0), (0.2, 0.2, 0.0)]],
                )

        class FakeWorker:
            def __init__(self):
                self.mailbox = LatestFrameMailbox()
                self.mailbox.publish(packet)
                self.camera_opened = True
                self.failed_reads = 0
                self.read_count = 1

            def raise_if_failed(self):
                return None

            def stop(self):
                self.mailbox.close()

        class FakeProcess:
            def cpu_times(self):
                return SimpleNamespace(user=0.01, system=0.0)

        summary = _run_paired_mode(
            FakeWorker(),
            FakePhone(),
            FakeOneFace(),
            cv2=None,
            process=FakeProcess(),
            scenario="two_faces_no_phone",
            frames=1,
            warmup=0,
            preview=False,
        )

        self.assertEqual(
            summary["mode_results"]["sequential"]["face_missed_frames"],
            1,
        )
        self.assertEqual(
            summary["mode_results"]["concurrent"]["face_missed_frames"],
            1,
        )

    def test_processor_rejects_a_second_outstanding_frame(self):
        entered = threading.Event()
        release = threading.Event()

        class BlockingAdapter:
            def detect(self, frame):
                entered.set()
                if not release.wait(timeout=2):
                    raise TimeoutError("test release was not signalled")
                return "phone"

            def extract_all_face_embeddings(self, frame):
                return "face"

        processor = VisionFrameProcessor(
            BlockingAdapter(),
            BlockingAdapter(),
            mode="sequential",
        )
        outcomes = []

        def process_first():
            try:
                outcomes.append(
                    processor.process(CapturedFrame(1, 0.0, bytearray(b"a")))
                )
            except Exception as exc:
                outcomes.append(exc)

        thread = threading.Thread(target=process_first)
        thread.start()
        try:
            self.assertTrue(entered.wait(timeout=2))
            with self.assertRaisesRegex(RuntimeError, "already being processed"):
                processor.process(CapturedFrame(2, 0.0, bytearray(b"b")))
        finally:
            release.set()
            thread.join(timeout=2)
            processor.close()

        self.assertFalse(thread.is_alive())
        self.assertEqual(len(outcomes), 1)
        self.assertFalse(isinstance(outcomes[0], Exception))


if __name__ == "__main__":
    unittest.main()
