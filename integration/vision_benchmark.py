"""Opt-in live benchmark for the shared phone and face vision adapters."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread, local
from typing import Any, Callable

from integration.camera_manager import CameraManager
from integration.vision_scheduler import (
    CapturedFrame,
    LatestFrameMailbox,
    VisionFrameProcessor,
)


ROOT = Path(__file__).resolve().parents[1]
THREAD_CONFIGS = ("1/1", "2/1", "4/1")
SCENARIOS = {
    "phone_full": "Hold one phone fully visible and steady; keep your face out of view.",
    "phone_far": "Hold one phone far from the camera, fully visible; keep your face out of view.",
    "phone_partial": "Show only part of one phone at the edge of the frame; keep your face out of view.",
    "phone_angle": "Hold one phone fully visible at an oblique angle; keep your face out of view.",
    "phone_near_face": "Hold one phone close beside your face, keeping both visible.",
    "no_face": "Show no face and no phone in the camera view.",
    "one_face": "Show exactly one face; do not hold a phone.",
    "face_partial": "Show one face partly outside the frame; do not hold a phone.",
    "two_faces_no_phone": "Show exactly two people, with two faces visible and no phone.",
    "two_faces_with_phone": "Show exactly two people with two faces visible and one phone visible.",
}
PHONE_PRESENT_SCENARIOS = {
    "phone_full",
    "phone_far",
    "phone_partial",
    "phone_angle",
    "phone_near_face",
    "two_faces_with_phone",
}
FACE_EXPECTED_COUNTS = {
    "phone_near_face": 1,
    "one_face": 1,
    "face_partial": 1,
    "two_faces_no_phone": 2,
    "two_faces_with_phone": 2,
}


@dataclass
class MeasuredOutput:
    value: Any
    stages_ms: dict[str, float]
    landmarks: list[list[tuple[float, float, float]]] | None = None


class StageRecorder:
    def __init__(self) -> None:
        self._local = local()

    def begin(self) -> None:
        self._local.values = {}

    def add(self, name: str, duration_ms: float) -> None:
        values = getattr(self._local, "values", None)
        if values is not None:
            values[name] = values.get(name, 0.0) + duration_ms

    def finish(self) -> dict[str, float]:
        values = getattr(self._local, "values", {})
        self._local.values = None
        return dict(values)


class TimedYoloModel:
    def __init__(self, model: Any, recorder: StageRecorder) -> None:
        self._model = model
        self._recorder = recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        predictions = self._model.predict(*args, **kwargs)
        self._recorder.add("predict_wall_ms", (time.perf_counter() - started) * 1000)
        if predictions:
            speed = getattr(predictions[0], "speed", None)
            if isinstance(speed, dict):
                for stage in ("preprocess", "inference", "postprocess"):
                    value = speed.get(stage)
                    if isinstance(value, (int, float)):
                        self._recorder.add(f"predict_{stage}_ms", float(value))
        return predictions


class TimedPhoneAdapter:
    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter
        self._recorder = StageRecorder()
        adapter.model = TimedYoloModel(adapter.model, self._recorder)

    def detect(self, frame: Any) -> MeasuredOutput:
        self._recorder.begin()
        started = time.perf_counter()
        output = self._adapter.detect(frame)
        adapter_ms = (time.perf_counter() - started) * 1000
        stages = self._recorder.finish()
        stages["adapter_ms"] = adapter_ms
        stages["adapter_other_ms"] = max(
            0.0, adapter_ms - stages.get("predict_wall_ms", 0.0)
        )
        return MeasuredOutput(output, stages)


class TimedCV2:
    def __init__(self, cv2_module: Any, recorder: StageRecorder) -> None:
        self._cv2 = cv2_module
        self._recorder = recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cv2, name)

    def cvtColor(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        result = self._cv2.cvtColor(*args, **kwargs)
        self._recorder.add("bgr_to_rgb_ms", (time.perf_counter() - started) * 1000)
        return result


class TimedMediaPipe:
    def __init__(self, module: Any, recorder: StageRecorder) -> None:
        self._module = module
        self._recorder = recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)

    def Image(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        image = self._module.Image(*args, **kwargs)
        self._recorder.add("mediapipe_image_ms", (time.perf_counter() - started) * 1000)
        return image


class TimedLandmarker:
    def __init__(
        self,
        landmarker: Any,
        recorder: StageRecorder,
        landmark_recorder: "LandmarkRecorder",
    ) -> None:
        self._landmarker = landmarker
        self._recorder = recorder
        self._landmark_recorder = landmark_recorder

    def detect(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        result = self._landmarker.detect(*args, **kwargs)
        self._recorder.add("mediapipe_detect_ms", (time.perf_counter() - started) * 1000)
        self._landmark_recorder.record(result)
        return result

    def close(self) -> Any:
        return self._landmarker.close()


class LandmarkRecorder:
    def __init__(self) -> None:
        self._local = local()

    def begin(self) -> None:
        self._local.faces = None

    def record(self, result: Any) -> None:
        landmarks = getattr(result, "face_landmarks", ())
        self._local.faces = [
            [
                (
                    float(point.x),
                    float(point.y),
                    float(getattr(point, "z", 0.0)),
                )
                for point in face
            ]
            for face in landmarks
        ]

    def finish(self) -> list[list[tuple[float, float, float]]] | None:
        faces = getattr(self._local, "faces", None)
        self._local.faces = None
        return faces


class TimedRecognizer:
    def __init__(self, recognizer: Any, recorder: StageRecorder) -> None:
        self._recognizer = recognizer
        self._recorder = recorder

    def alignCrop(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        result = self._recognizer.alignCrop(*args, **kwargs)
        self._recorder.add("sface_align_ms", (time.perf_counter() - started) * 1000)
        return result

    def feature(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        result = self._recognizer.feature(*args, **kwargs)
        self._recorder.add("sface_feature_ms", (time.perf_counter() - started) * 1000)
        return result


class TimedFaceAdapter:
    def __init__(self, adapter: Any, cv2_module: Any, mediapipe_module: Any) -> None:
        self._adapter = adapter
        self._recorder = StageRecorder()
        self._landmark_recorder = LandmarkRecorder()
        adapter._cv2 = TimedCV2(cv2_module, self._recorder)
        adapter._mp = TimedMediaPipe(mediapipe_module, self._recorder)
        adapter.landmarker = TimedLandmarker(
            adapter.landmarker,
            self._recorder,
            self._landmark_recorder,
        )
        adapter.recognizer = TimedRecognizer(adapter.recognizer, self._recorder)

    def extract_all_face_embeddings(self, frame: Any) -> MeasuredOutput:
        self._recorder.begin()
        self._landmark_recorder.begin()
        started = time.perf_counter()
        output = self._adapter.extract_all_face_embeddings(frame)
        adapter_ms = (time.perf_counter() - started) * 1000
        stages = self._recorder.finish()
        landmarks = self._landmark_recorder.finish()
        stages["adapter_ms"] = adapter_ms
        stages["adapter_other_ms"] = max(
            0.0,
            adapter_ms
            - sum(
                duration
                for name, duration in stages.items()
                if name.endswith("_ms")
            ),
        )
        return MeasuredOutput(output, stages, landmarks)


class CaptureWorker:
    """Owns the sole CameraManager and publishes only the newest available frame."""

    def __init__(self, camera_factory: Callable[[], Any]) -> None:
        self.mailbox = LatestFrameMailbox()
        self._camera_factory = camera_factory
        self._stop = Event()
        self._ready = Event()
        self._thread = Thread(target=self._capture, name="vision-camera-capture", daemon=True)
        self.error: BaseException | None = None
        self.camera_opened = False
        self.failed_reads = 0
        self.read_count = 0
        self._sequence = 0

    def start(self) -> None:
        self._thread.start()
        if not self._ready.wait(timeout=15):
            raise TimeoutError("CameraManager did not finish opening the camera within 15 seconds.")
        self.raise_if_failed()
        if not self.camera_opened:
            raise RuntimeError("CameraManager did not open the webcam.")

    def _capture(self) -> None:
        try:
            with self._camera_factory() as camera:
                self.camera_opened = True
                self._ready.set()
                while not self._stop.is_set():
                    started = time.perf_counter()
                    try:
                        image = camera.read()
                    except Exception:
                        self.failed_reads += 1
                        raise
                    captured_at = time.perf_counter()
                    capture_ms = (captured_at - started) * 1000
                    self.read_count += 1
                    if image is None or getattr(image, "size", 0) == 0:
                        self.failed_reads += 1
                        raise RuntimeError("CameraManager returned an empty frame.")
                    if self._stop.is_set():
                        break
                    self.mailbox.publish(
                        CapturedFrame(self._sequence, captured_at, image, capture_ms)
                    )
                    self._sequence += 1
        except Exception as exc:
            self.error = exc
        finally:
            self._ready.set()
            self.mailbox.close()

    def raise_if_failed(self) -> None:
        if self.error is not None:
            raise RuntimeError("Webcam capture failed.") from self.error

    def stop(self) -> None:
        self._stop.set()
        self.mailbox.close()
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise TimeoutError("Camera capture thread did not stop within 10 seconds.")
        self.raise_if_failed()


class CpuSampler:
    def __init__(self, process: Any) -> None:
        self._process = process
        self._stop = Event()
        self._samples: list[float] = []
        self._thread: Thread | None = None

    def start(self) -> None:
        self._process.cpu_percent(interval=None)

        def sample() -> None:
            while not self._stop.wait(0.1):
                self._samples.append(float(self._process.cpu_percent(interval=None)))

        self._thread = Thread(target=sample, name="vision-cpu-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> list[float]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        return self._samples


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "max": None}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _box_iou(left: list[float], right: list[float]) -> float:
    if len(left) != 4 or len(right) != 4:
        raise ValueError("Expected XYXY boxes with four coordinates.")
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _match_boxes(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates = []
    for left_index, left_detection in enumerate(left):
        left_box = left_detection.get("bbox")
        if left_box is None:
            continue
        for right_index, right_detection in enumerate(right):
            right_box = right_detection.get("bbox")
            if right_box is None:
                continue
            if (
                "class_id" in left_detection
                and "class_id" in right_detection
                and left_detection["class_id"] != right_detection["class_id"]
            ):
                continue
            candidates.append((_box_iou(left_box, right_box), left_index, right_index))

    candidates.sort(reverse=True)
    used_left: set[int] = set()
    used_right: set[int] = set()
    pairs = []
    for overlap, left_index, right_index in candidates:
        if overlap <= 0 or left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        pair = {
            "sequential_index": left_index,
            "concurrent_index": right_index,
            "iou": overlap,
        }
        left_confidence = left[left_index].get("confidence")
        right_confidence = right[right_index].get("confidence")
        if isinstance(left_confidence, (int, float)) and isinstance(
            right_confidence, (int, float)
        ):
            pair["confidence_delta"] = abs(left_confidence - right_confidence)
        pairs.append(pair)

    return {
        "pairs": pairs,
        "unmatched_sequential": [
            index for index in range(len(left)) if index not in used_left
        ],
        "unmatched_concurrent": [
            index for index in range(len(right)) if index not in used_right
        ],
    }


def _compare_landmarks(
    sequential: list[list[tuple[float, float, float]]] | None,
    concurrent: list[list[tuple[float, float, float]]] | None,
) -> dict[str, Any]:
    if sequential is None or concurrent is None:
        return {
            "sequential_available": sequential is not None,
            "concurrent_available": concurrent is not None,
            "same_shape": False,
            "exact_match": False,
            "max_abs_delta": None,
        }
    if len(sequential) != len(concurrent):
        return {
            "sequential_available": bool(sequential),
            "concurrent_available": bool(concurrent),
            "same_shape": False,
            "exact_match": False,
            "max_abs_delta": None,
        }
    if not sequential:
        return {
            "sequential_available": False,
            "concurrent_available": False,
            "same_shape": True,
            "exact_match": None,
            "max_abs_delta": None,
        }

    deltas = []
    for left_face, right_face in zip(sequential, concurrent):
        if len(left_face) != len(right_face):
            return {
                "sequential_available": bool(sequential),
                "concurrent_available": bool(concurrent),
                "same_shape": False,
                "exact_match": False,
                "max_abs_delta": None,
            }
        for left_point, right_point in zip(left_face, right_face):
            deltas.extend(abs(left - right) for left, right in zip(left_point, right_point))

    max_delta = max(deltas, default=0.0)
    return {
        "sequential_available": bool(sequential),
        "concurrent_available": bool(concurrent),
        "same_shape": True,
        "exact_match": max_delta == 0.0,
        "max_abs_delta": max_delta,
    }


def _serialize_phone(measured: MeasuredOutput) -> list[dict[str, Any]]:
    output = measured.value
    detections = output.get("detections", []) if isinstance(output, dict) else []
    serialized = []
    for detection in detections:
        box = detection.get("bbox", detection.get("box"))
        if box is None:
            box = detection.get("bbox_xyxy")
        serialized.append(
            {
                "class_id": detection.get("class_id"),
                "class_name": detection.get("class_name"),
                "confidence": detection.get("confidence"),
                "bbox": box,
            }
        )
    return serialized


def _serialize_faces(measured: MeasuredOutput) -> list[dict[str, Any]]:
    faces = measured.value if isinstance(measured.value, list) else []
    return [
        {
            "bbox": face.get("bbox", face.get("box")),
            "landmarks_available": True,
            "embedding_available": face.get("embedding") is not None,
        }
        for face in faces
    ]


def _annotate(cv2: Any, frame: Any, row: dict[str, Any], mode: str) -> None:
    image = frame.copy()
    height, width = image.shape[:2]
    for detection in row["phones"]:
        box = detection.get("bbox")
        if box is None or len(box) != 4:
            continue
        x1, y1, x2, y2 = (int(value) for value in box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            image,
            f"phone {float(detection['confidence'] or 0):.2f}",
            (max(0, x1), max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )
    for face in row["faces"]:
        box = face.get("bbox")
        if box is None or len(box) != 4:
            continue
        x1, y1, x2, y2 = (int(value) for value in box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)
    cv2.putText(
        image,
        f"{mode}: phone={len(row['phones'])}, face={len(row['faces'])}",
        (10, max(22, height - 14)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )
    cv2.imshow("Vision benchmark - press q to stop", image)
    cv2.waitKey(1)


def _run_mode(
    worker: CaptureWorker,
    phone: TimedPhoneAdapter,
    face: TimedFaceAdapter,
    cv2: Any,
    process: Any,
    scenario: str,
    mode: str,
    frames: int,
    warmup: int,
    preview: bool,
) -> dict[str, Any]:
    processor = VisionFrameProcessor(phone, face, mode=mode)
    cpu_sampler = CpuSampler(process)
    mailbox_replaced_start = worker.mailbox.replaced_count
    cpu_sampler.start()
    rows: list[dict[str, Any]] = []
    captured_frames = 0
    first_dispatch: float | None = None
    last_completion: float | None = None
    integrity_checks = 0
    integrity_failures = 0
    started = time.perf_counter()
    try:
        while captured_frames < warmup + frames:
            worker.raise_if_failed()
            packet = worker.mailbox.take(timeout=1.0)
            if packet is None:
                worker.raise_if_failed()
                raise RuntimeError("Frame mailbox closed before the requested frame count was reached.")
            before_crc = zlib.crc32(memoryview(packet.image))
            result = processor.process(packet)
            after_crc = zlib.crc32(memoryview(packet.image))
            integrity_checks += 1
            if before_crc != after_crc:
                integrity_failures += 1
                raise RuntimeError(
                    f"Adapter modified the shared frame buffer at sequence {packet.sequence}."
                )
            captured_frames += 1
            row = {
                "sequence": packet.sequence,
                "width": int(packet.image.shape[1]),
                "height": int(packet.image.shape[0]),
                "capture_ms": packet.capture_ms,
                "queue_delay_ms": result.queue_delay_ms,
                "frame_age_ms": result.frame_age_ms,
                "phone_ms": result.phone.duration_ms,
                "face_ms": result.face.duration_ms,
                "combined_processing_ms": result.processing_ms,
                "phone_stages_ms": result.phone.value.stages_ms,
                "face_stages_ms": result.face.value.stages_ms,
                "phones": _serialize_phone(result.phone.value),
                "faces": _serialize_faces(result.face.value),
            }
            if captured_frames > warmup:
                rows.append(row)
                if first_dispatch is None:
                    first_dispatch = result.dispatched_at
                last_completion = result.completed_at
            if preview:
                _annotate(cv2, packet.image, row, mode)
    finally:
        processing_wall_ms = (time.perf_counter() - started) * 1000
        cpu_samples = cpu_sampler.stop()
        processor.close()

    worker.raise_if_failed()
    phone_times = [row["phone_ms"] for row in rows]
    face_times = [row["face_ms"] for row in rows]
    combined_times = [row["combined_processing_ms"] for row in rows]
    capture_times = [row["capture_ms"] for row in rows]
    queue_delays = [row["queue_delay_ms"] for row in rows]
    frame_ages = [row["frame_age_ms"] for row in rows]
    measured_wall = (
        max(0.0, last_completion - first_dispatch)
        if first_dispatch is not None and last_completion is not None
        else 0.0
    )
    phone_counts = [len(row["phones"]) for row in rows]
    face_counts = [len(row["faces"]) for row in rows]
    confidence_scores = [
        detection["confidence"]
        for row in rows
        for detection in row["phones"]
        if isinstance(detection.get("confidence"), (int, float))
    ]
    face_stages: dict[str, list[float]] = {}
    phone_stages: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row["face_stages_ms"].items():
            face_stages.setdefault(key, []).append(value)
        for key, value in row["phone_stages_ms"].items():
            phone_stages.setdefault(key, []).append(value)

    summary = {
        "scenario": scenario,
        "mode": mode,
        "valid_measured_frames": len(rows),
        "warmup_frames": warmup,
        "phone_counts_per_frame": phone_counts,
        "phone_detected_frames": sum(count > 0 for count in phone_counts),
        "phone_detection_rate": (
            sum(count > 0 for count in phone_counts) / len(rows) if rows else None
        ),
        "phone_confidence_scores": confidence_scores,
        "face_counts_per_frame": face_counts,
        "face_landmarks_available_per_frame": [bool(row["faces"]) for row in rows],
        "phone_inference_ms": _summary(phone_times),
        "face_adapter_ms": _summary(face_times),
        "combined_processing_ms": _summary(combined_times),
        "camera_capture_ms_on_processed_frames": _summary(capture_times),
        "queue_delay_ms": _summary(queue_delays),
        "frame_age_ms": _summary(frame_ages),
        "delayed_processed_reads_over_100ms": sum(value > 100 for value in capture_times),
        "mailbox_replacements_during_mode": (
            worker.mailbox.replaced_count - mailbox_replaced_start
        ),
        "processing_fps_by_busy_time": (
            len(rows) / (sum(combined_times) / 1000) if sum(combined_times) else None
        ),
        "consumer_throughput_fps": (
            len(rows) / measured_wall if measured_wall else None
        ),
        "cpu_percent": _summary(cpu_samples),
        "phone_stage_ms": {name: _summary(values) for name, values in phone_stages.items()},
        "face_stage_ms": {name: _summary(values) for name, values in face_stages.items()},
        "processing_wall_ms_including_warmup": processing_wall_ms,
        "frame_integrity_checks": integrity_checks,
        "frame_integrity_failures": integrity_failures,
        "camera_opened": worker.camera_opened,
        "capture_read_count_so_far": worker.read_count,
        "failed_reads_so_far": worker.failed_reads,
        "frames": rows,
    }
    return summary


def _run_paired_mode(
    worker: CaptureWorker,
    phone: TimedPhoneAdapter,
    face: TimedFaceAdapter,
    cv2: Any,
    process: Any,
    scenario: str,
    frames: int,
    warmup: int,
    preview: bool,
) -> dict[str, Any]:
    processors = {
        mode: VisionFrameProcessor(phone, face, mode=mode)
        for mode in ("sequential", "concurrent")
    }
    measured_rows: list[dict[str, Any]] = []
    checksum_checks = 0
    checksum_failures = 0
    capture_replacements_start = worker.mailbox.replaced_count
    captured_count = 0
    expected_phone = scenario in PHONE_PRESENT_SCENARIOS
    expected_faces = FACE_EXPECTED_COUNTS.get(scenario, 0)

    def run_one_mode(mode: str, frame: CapturedFrame) -> tuple[Any, float | None]:
        cpu_before = process.cpu_times()
        result = processors[mode].process(frame)
        cpu_after = process.cpu_times()
        cpu_seconds = (
            (cpu_after.user - cpu_before.user)
            + (cpu_after.system - cpu_before.system)
        )
        wall_seconds = result.processing_ms / 1000
        cpu_percent = 100 * cpu_seconds / wall_seconds if wall_seconds > 0 else None
        if result.frame is not frame:
            raise RuntimeError(f"{mode} processor returned a different CapturedFrame.")
        return result, cpu_percent

    try:
        while captured_count < warmup + frames:
            worker.raise_if_failed()
            camera_frame = worker.mailbox.take(timeout=1.0)
            if camera_frame is None:
                worker.raise_if_failed()
                raise RuntimeError(
                    "Frame mailbox closed before the paired frame count was reached."
                )

            source_crc = zlib.crc32(memoryview(camera_frame.image))
            retained_image = camera_frame.image.copy()
            retained_crc = zlib.crc32(memoryview(retained_image))
            checksum_checks += 1
            if retained_crc != source_crc:
                checksum_failures += 1
                raise RuntimeError(
                    f"Captured frame {camera_frame.sequence} changed while being retained."
                )

            paired_frame = CapturedFrame(
                camera_frame.sequence,
                camera_frame.captured_at,
                retained_image,
                camera_frame.capture_ms,
            )
            frame_crc = retained_crc
            order = (
                ("sequential", "concurrent")
                if captured_count % 2 == 0
                else ("concurrent", "sequential")
            )
            outputs: dict[str, tuple[Any, float | None]] = {}
            for mode in order:
                result, cpu_percent = run_one_mode(mode, paired_frame)
                outputs[mode] = (result, cpu_percent)
                checksum_checks += 1
                if zlib.crc32(memoryview(paired_frame.image)) != frame_crc:
                    checksum_failures += 1
                    raise RuntimeError(
                        f"{mode} adapter modified paired frame "
                        f"{paired_frame.sequence}."
                    )

            captured_count += 1
            if captured_count <= warmup:
                continue

            sequential_result, sequential_cpu = outputs["sequential"]
            concurrent_result, concurrent_cpu = outputs["concurrent"]
            sequential_phones = _serialize_phone(sequential_result.phone.value)
            concurrent_phones = _serialize_phone(concurrent_result.phone.value)
            sequential_faces = _serialize_faces(sequential_result.face.value)
            concurrent_faces = _serialize_faces(concurrent_result.face.value)
            phone_boxes = _match_boxes(sequential_phones, concurrent_phones)
            face_boxes = _match_boxes(sequential_faces, concurrent_faces)
            landmark_comparison = _compare_landmarks(
                sequential_result.face.value.landmarks,
                concurrent_result.face.value.landmarks,
            )

            seq_phone_count = len(sequential_phones)
            con_phone_count = len(concurrent_phones)
            seq_face_count = len(sequential_faces)
            con_face_count = len(concurrent_faces)
            row = {
                "sequence": paired_frame.sequence,
                "width": int(paired_frame.image.shape[1]),
                "height": int(paired_frame.image.shape[0]),
                "capture_ms": paired_frame.capture_ms,
                "frame_integrity_verified": True,
                "same_frame_object": (
                    sequential_result.frame is concurrent_result.frame
                    and sequential_result.frame.image is concurrent_result.frame.image
                ),
                "sequential": {
                    "phones": sequential_phones,
                    "faces": sequential_faces,
                    "phone_count": seq_phone_count,
                    "face_count": seq_face_count,
                    "landmarks_available": bool(
                        sequential_result.face.value.landmarks
                    ),
                    "embedding_available": any(
                        item["embedding_available"] for item in sequential_faces
                    ),
                    "phone_ms": sequential_result.phone.duration_ms,
                    "face_ms": sequential_result.face.duration_ms,
                    "processing_ms": sequential_result.processing_ms,
                    "processing_fps": (
                        1000 / sequential_result.processing_ms
                        if sequential_result.processing_ms > 0
                        else None
                    ),
                    "cpu_percent": sequential_cpu,
                    "queue_delay_ms": sequential_result.queue_delay_ms,
                    "frame_age_ms": sequential_result.frame_age_ms,
                },
                "concurrent": {
                    "phones": concurrent_phones,
                    "faces": concurrent_faces,
                    "phone_count": con_phone_count,
                    "face_count": con_face_count,
                    "landmarks_available": bool(
                        concurrent_result.face.value.landmarks
                    ),
                    "embedding_available": any(
                        item["embedding_available"] for item in concurrent_faces
                    ),
                    "phone_ms": concurrent_result.phone.duration_ms,
                    "face_ms": concurrent_result.face.duration_ms,
                    "processing_ms": concurrent_result.processing_ms,
                    "processing_fps": (
                        1000 / concurrent_result.processing_ms
                        if concurrent_result.processing_ms > 0
                        else None
                    ),
                    "cpu_percent": concurrent_cpu,
                    "queue_delay_ms": concurrent_result.queue_delay_ms,
                    "frame_age_ms": concurrent_result.frame_age_ms,
                },
                "phone_count_agrees": seq_phone_count == con_phone_count,
                "phone_box_comparison": phone_boxes,
                "face_count_agrees": seq_face_count == con_face_count,
                "face_box_comparison": face_boxes,
                "landmark_comparison": landmark_comparison,
                "expected_phone_present": expected_phone,
                "expected_face_count": expected_faces,
            }
            if not row["same_frame_object"]:
                raise RuntimeError(
                    f"Paired modes did not receive the same image object "
                    f"for frame {paired_frame.sequence}."
                )
            measured_rows.append(row)
            if preview:
                _annotate(cv2, paired_frame.image, {
                    "phones": sequential_phones,
                    "faces": sequential_faces,
                }, "sequential")
                _annotate(cv2, paired_frame.image, {
                    "phones": concurrent_phones,
                    "faces": concurrent_faces,
                }, "concurrent")
    finally:
        for processor in processors.values():
            processor.close()

    worker.stop()
    worker.raise_if_failed()
    mode_metrics: dict[str, Any] = {}
    for mode in ("sequential", "concurrent"):
        outputs = [row[mode] for row in measured_rows]
        phone_counts = [item["phone_count"] for item in outputs]
        face_counts = [item["face_count"] for item in outputs]
        latencies = [item["processing_ms"] for item in outputs]
        queue_delays = [item["queue_delay_ms"] for item in outputs]
        frame_ages = [item["frame_age_ms"] for item in outputs]
        cpu_values = [
            item["cpu_percent"] for item in outputs if item["cpu_percent"] is not None
        ]
        confidences = [
            detection["confidence"]
            for item in outputs
            for detection in item["phones"]
            if isinstance(detection.get("confidence"), (int, float))
        ]
        phone_detected_frames = sum(count > 0 for count in phone_counts)
        face_detected_frames = sum(count > 0 for count in face_counts)
        mode_metrics[mode] = {
            "phone_counts_per_frame": phone_counts,
            "phone_detected_frames": phone_detected_frames,
            "phone_missed_frames": (
                sum(count == 0 for count in phone_counts) if expected_phone else 0
            ),
            "phone_false_positive_frames": (
                phone_detected_frames if not expected_phone else 0
            ),
            "phone_extra_boxes": (
                sum(max(0, count - 1) for count in phone_counts)
                if expected_phone
                else 0
            ),
            "phone_confidence_scores": confidences,
            "face_counts_per_frame": face_counts,
            "face_detected_frames": face_detected_frames,
            "face_missed_frames": (
                sum(count < expected_faces for count in face_counts)
                if expected_faces > 0
                else 0
            ),
            "face_false_positive_frames": (
                sum(count > expected_faces for count in face_counts)
                if expected_faces > 0
                else face_detected_frames
            ),
            "face_extra_faces": sum(
                max(0, count - expected_faces) for count in face_counts
            ),
            "face_extra_face_frames": sum(
                count > expected_faces for count in face_counts
            ),
            "face_embedding_available_frames": sum(
                item["embedding_available"] for item in outputs
            ),
            "latency_ms": _summary(latencies),
            "phone_ms": _summary([item["phone_ms"] for item in outputs]),
            "face_ms": _summary([item["face_ms"] for item in outputs]),
            "processing_fps": (
                len(outputs) / (sum(latencies) / 1000) if sum(latencies) else None
            ),
            "cpu_percent": _summary(cpu_values),
            "queue_delay_ms": _summary(queue_delays),
            "frame_age_ms": _summary(frame_ages),
        }

    phone_iou = [
        pair["iou"]
        for row in measured_rows
        for pair in row["phone_box_comparison"]["pairs"]
    ]
    face_iou = [
        pair["iou"]
        for row in measured_rows
        for pair in row["face_box_comparison"]["pairs"]
    ]
    landmark_deltas = [
        row["landmark_comparison"]["max_abs_delta"]
        for row in measured_rows
        if row["landmark_comparison"]["max_abs_delta"] is not None
    ]
    confidence_deltas = [
        pair["confidence_delta"]
        for row in measured_rows
        for pair in row["phone_box_comparison"]["pairs"]
        if "confidence_delta" in pair
    ]
    summary = {
        "scenario": scenario,
        "thread_config": None,
        "warmup_frames": warmup,
        "measured_frames": len(measured_rows),
        "expected_phone_present": expected_phone,
        "expected_face_count": expected_faces,
        "same_frame_capture_count": captured_count,
        "frame_integrity_checks": checksum_checks,
        "frame_integrity_failures": checksum_failures,
        "latest_mailbox_replacements_during_capture": (
            worker.mailbox.replaced_count - capture_replacements_start
        ),
        "camera_opened": worker.camera_opened,
        "capture_read_count": worker.read_count,
        "failed_reads": worker.failed_reads,
        "mode_results": mode_metrics,
        "paired_comparison": {
            "phone_count_agreement_frames": sum(
                row["phone_count_agrees"] for row in measured_rows
            ),
            "phone_box_iou_pairs": len(phone_iou),
            "phone_box_iou": _summary(phone_iou),
            "phone_unmatched_sequential_boxes": sum(
                len(row["phone_box_comparison"]["unmatched_sequential"])
                for row in measured_rows
            ),
            "phone_unmatched_concurrent_boxes": sum(
                len(row["phone_box_comparison"]["unmatched_concurrent"])
                for row in measured_rows
            ),
            "phone_confidence_delta": _summary(confidence_deltas),
            "face_count_agreement_frames": sum(
                row["face_count_agrees"] for row in measured_rows
            ),
            "face_box_iou_pairs": len(face_iou),
            "face_box_iou": _summary(face_iou),
            "landmark_shape_agreement_frames": sum(
                row["landmark_comparison"]["same_shape"] for row in measured_rows
            ),
            "landmark_availability_agreement_frames": sum(
                row["landmark_comparison"]["sequential_available"]
                == row["landmark_comparison"]["concurrent_available"]
                for row in measured_rows
            ),
            "exact_landmark_match_frames": sum(
                row["landmark_comparison"]["exact_match"] is True
                for row in measured_rows
            ),
            "embedding_availability_agreement_frames": sum(
                row["sequential"]["embedding_available"]
                == row["concurrent"]["embedding_available"]
                for row in measured_rows
            ),
            "landmark_max_abs_delta": _summary(landmark_deltas),
        },
        "frame_age_note": (
            "Paired modes run consecutively on the same captured image. The second "
            "mode's frame age includes the first mode's processing time; mode order "
            "alternates to balance this effect."
        ),
        "frames": measured_rows,
    }
    return summary


def _parse_threads(value: str) -> tuple[int, int]:
    try:
        intra, inter = (int(part) for part in value.split("/", maxsplit=1))
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("Thread settings must use INTRA/INTER format.") from exc
    if intra < 1 or inter < 1:
        raise argparse.ArgumentTypeError("Thread counts must be positive integers.")
    return intra, inter


def _launch_sweep(args: argparse.Namespace) -> None:
    print(f"Scenario: {args.scenario}: {SCENARIOS[args.scenario]}")
    print("Launching one isolated Python process per thread configuration.")
    for configuration in THREAD_CONFIGS:
        command = [
            sys.executable,
            "-B",
            "-m",
            "integration.vision_benchmark",
            "--scenario",
            args.scenario,
            "--threads",
            configuration,
            "--mode",
            args.mode,
            "--frames",
            str(args.frames),
            "--warmup",
            str(args.warmup),
        ]
        if args.no_preview:
            command.append("--no-preview")
        if args.paired:
            command.append("--paired")
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            raise SystemExit(
                f"Benchmark failed for thread configuration {configuration} "
                f"(exit code {completed.returncode}); stopping the sweep."
            )


def _run_benchmark(args: argparse.Namespace) -> None:
    intra_threads, inter_threads = _parse_threads(args.threads)
    os.environ["ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS"] = "1"
    os.environ["YOLO_AUTOINSTALL"] = "false"

    import torch

    torch.set_num_threads(intra_threads)
    torch.set_num_interop_threads(inter_threads)

    import cv2
    import psutil

    from integration.module_adapters.face_identity import FaceIdentityAdapter
    from integration.module_adapters.phone import PhoneDetectionAdapter

    phone_adapter = PhoneDetectionAdapter()
    face_adapter = FaceIdentityAdapter()
    phone = TimedPhoneAdapter(phone_adapter)
    face = TimedFaceAdapter(face_adapter, cv2, face_adapter._mp)
    process = psutil.Process()
    worker = CaptureWorker(CameraManager)
    summaries = []
    try:
        worker.start()
        if args.paired:
            summary = _run_paired_mode(
                worker,
                phone,
                face,
                cv2,
                process,
                args.scenario,
                args.frames,
                args.warmup,
                not args.no_preview,
            )
            summary["thread_config"] = args.threads
            summaries.append(summary)
            print("PAIRED_SUMMARY " + json.dumps(summary, separators=(",", ":")))
        else:
            modes = ("sequential", "concurrent") if args.mode == "compare" else (args.mode,)
            for mode in modes:
                summary = _run_mode(
                    worker,
                    phone,
                    face,
                    cv2,
                    process,
                    args.scenario,
                    mode,
                    args.frames,
                    args.warmup,
                    not args.no_preview,
                )
                summary["thread_config"] = args.threads
                summaries.append(summary)
                print("PASS_SUMMARY " + json.dumps(summary, separators=(",", ":")))
    finally:
        try:
            worker.stop()
        finally:
            face_adapter.landmarker.close()
            cv2.destroyAllWindows()

    if len(summaries) == 2:
        sequential, concurrent = summaries
        comparison = {
            "scenario": args.scenario,
            "thread_config": args.threads,
            "same_live_scene_required": True,
            "sequential_fps": sequential["consumer_throughput_fps"],
            "concurrent_fps": concurrent["consumer_throughput_fps"],
            "sequential_mean_latency_ms": sequential["combined_processing_ms"]["mean"],
            "concurrent_mean_latency_ms": concurrent["combined_processing_ms"]["mean"],
            "sequential_phone_detection_rate": sequential["phone_detection_rate"],
            "concurrent_phone_detection_rate": concurrent["phone_detection_rate"],
            "sequential_phone_confidence_scores": sequential["phone_confidence_scores"],
            "concurrent_phone_confidence_scores": concurrent["phone_confidence_scores"],
            "sequential_face_counts": sequential["face_counts_per_frame"],
            "concurrent_face_counts": concurrent["face_counts_per_frame"],
            "note": (
                "Sequential and concurrent passes consume different live frames; "
                "compare condition-level results, not pixel-identical predictions."
            ),
        }
        print("MODE_COMPARISON " + json.dumps(comparison, separators=(",", ":")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), required=True)
    parser.add_argument("--mode", choices=("compare", "sequential", "concurrent"), default="compare")
    parser.add_argument("--threads", choices=THREAD_CONFIGS, default="2/1")
    parser.add_argument("--sweep-threads", action="store_true")
    parser.add_argument(
        "--paired",
        action="store_true",
        help="Run sequential and concurrent modes on the same captured frame objects.",
    )
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 0:
        parser.error("--frames must be positive and --warmup cannot be negative.")
    if args.paired and args.mode != "compare":
        parser.error("--paired requires --mode compare.")
    if args.paired and args.frames < 20:
        parser.error("--paired requires at least 20 measured frames per scenario.")
    if args.sweep_threads:
        _launch_sweep(args)
    else:
        print(f"Scenario: {args.scenario}: {SCENARIOS[args.scenario]}")
        _run_benchmark(args)


if __name__ == "__main__":
    main()
