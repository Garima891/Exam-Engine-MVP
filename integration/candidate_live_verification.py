"""Diagnostic-only candidate enrollment and live identity verification runner."""

from __future__ import annotations

import argparse
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from integration.camera_manager import CameraManager
from integration.candidate_identity import CandidateIdentityService
from integration.candidate_storage import CandidateStorage
from integration.config import (
    CANDIDATE_DATABASE_PATH,
    FACE_IDENTITY_THRESHOLD,
)
from integration.module_adapters.face_identity import FaceIdentityAdapter


ENROLLMENT_FRAME_COUNT = 3
LIVE_FRAME_COUNT = 60
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 191, 255)
LIVE_VERIFICATION_WINDOW = "D.2 live identity verification - Q stops"
FAILED_VERIFICATION_WINDOW = "D.2 stopped result - Q or ESC closes"


def validate_enrollment_frame(
    face_adapter: FaceIdentityAdapter,
    frame: Any,
) -> str | None:
    """Return a rejection reason, or None when one sample is usable."""
    try:
        faces = face_adapter.extract_all_face_embeddings(frame)
    except (TypeError, ValueError) as exc:
        return str(exc)
    if len(faces) != 1:
        return "No face detected." if not faces else "Multiple faces detected."

    face = faces[0]
    if not face.get("landmarks_available", False):
        return "Landmarks unavailable."
    try:
        embedding = FaceIdentityAdapter.validate_embedding(face["embedding"])
    except (KeyError, TypeError, ValueError) as exc:
        return str(exc)
    if not np.isclose(float(np.linalg.norm(embedding)), 1.0, atol=1e-5):
        return "Embedding is not L2-normalized."
    return None


def capture_enrollment_frames(
    camera: Any,
    face_adapter: FaceIdentityAdapter,
    cv2_module: Any,
) -> tuple[list[Any], list[str]]:
    """Capture exactly three attempts; invalid samples are rejected, not replaced."""
    accepted: list[Any] = []
    rejected: list[str] = []
    seen_frames: set[bytes] = set()
    attempts = 0
    deadline = time.monotonic() + 300

    while attempts < ENROLLMENT_FRAME_COUNT:
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for three enrollment captures.")
        frame = camera.read()
        preview = frame.copy()
        cv2_module.putText(
            preview,
            f"Enrollment: press C for sample {attempts + 1}/3; Q cancels",
            (12, 28),
            cv2_module.FONT_HERSHEY_SIMPLEX,
            0.6,
            GREEN,
            2,
        )
        cv2_module.imshow("D.2 candidate enrollment", preview)
        key = cv2_module.waitKey(1) & 0xFF
        if key == ord("q"):
            raise RuntimeError("Enrollment cancelled by user.")
        if key != ord("c"):
            continue

        attempts += 1
        sample = frame.copy()
        frame_digest = hashlib.sha256(sample.tobytes()).digest()
        if frame_digest in seen_frames:
            rejected.append("Duplicate frame.")
            print(
                f"ENROLLMENT_SAMPLE {attempts}/3 REJECTED: Duplicate frame.",
                flush=True,
            )
            continue
        seen_frames.add(frame_digest)
        rejection_reason = validate_enrollment_frame(face_adapter, sample)
        if rejection_reason is not None:
            rejected.append(rejection_reason)
            print(
                f"ENROLLMENT_SAMPLE {attempts}/3 REJECTED: {rejection_reason}",
                flush=True,
            )
            continue

        accepted.append(sample)
        print(
            f"ENROLLMENT_SAMPLE {attempts}/3 ACCEPTED: "
            "one face, landmarks available, valid normalized 128-D embedding.",
            flush=True,
        )

    return accepted, rejected


def _per_face_statuses(
    verification_result: dict[str, Any],
) -> dict[int, tuple[str, float | None]]:
    statuses: dict[int, tuple[str, float | None]] = {}
    for face in verification_result.get("matched_faces", []):
        face_number = face.get("face_number")
        if face_number is not None:
            statuses[int(face_number)] = (
                "CANDIDATE_IDENTIFIED",
                face.get("similarity"),
            )
    for face in verification_result.get("unmatched_faces", []):
        face_number = face.get("face_number")
        if face_number is not None:
            statuses[int(face_number)] = (
                face.get("status", "REVIEW_REQUIRED"),
                face.get("similarity"),
            )
    return statuses


def face_annotations(
    verification_result: dict[str, Any],
    frame_width: int,
    candidate_side: str = "right",
) -> list[dict[str, Any]]:
    """Build per-box display records; role-by-side is test ground truth only."""
    if candidate_side not in {"left", "right"}:
        raise ValueError("candidate_side must be 'left' or 'right'.")

    statuses = _per_face_statuses(verification_result)
    annotations = []
    for face_number, (status, similarity) in statuses.items():
        face_record = next(
            (
                face
                for face in (
                    verification_result.get("matched_faces", [])
                    + verification_result.get("unmatched_faces", [])
                )
                if face.get("face_number") == face_number
            ),
            None,
        )
        if face_record is None:
            continue
        box = face_record.get("box")
        if box is None or len(box) != 4:
            continue

        x1, _, x2, _ = box
        face_side = "left" if (float(x1) + float(x2)) / 2 < frame_width / 2 else "right"
        ground_truth_role = (
            "student" if face_side == candidate_side else "volunteer"
        )
        if status == "CANDIDATE_IDENTIFIED":
            label, color = "FACE MATCHED / IDENTIFIED", GREEN
        elif status == "CANDIDATE_UNMATCHED":
            label, color = "FACE UNMATCHED", RED
        else:
            label, color = "REVIEW REQUIRED", YELLOW

        if (
            ground_truth_role == "volunteer"
            and status == "CANDIDATE_IDENTIFIED"
        ):
            test_outcome = "FALSE ACCEPTANCE"
        elif (
            ground_truth_role == "student"
            and status == "CANDIDATE_UNMATCHED"
        ):
            test_outcome = "FALSE REJECTION"
        else:
            test_outcome = None

        annotations.append(
            {
                "face_number": face_number,
                "box": box,
                "ground_truth_role": ground_truth_role,
                "system_status": status,
                "similarity": similarity,
                "label": label,
                "color": color,
                "test_outcome": test_outcome,
            }
        )
    return annotations


def annotate_frame(
    cv2_module: Any,
    frame: Any,
    verification_result: dict[str, Any],
    candidate_side: str = "right",
) -> tuple[Any, list[dict[str, Any]]]:
    """Render only current-frame boxes so missing faces cannot leave stale labels."""
    image = frame.copy()
    height, width = image.shape[:2]
    annotations = face_annotations(
        verification_result,
        frame_width=width,
        candidate_side=candidate_side,
    )
    for annotation in annotations:
        x1, y1, x2, y2 = (int(value) for value in annotation["box"])
        cv2_module.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            annotation["color"],
            2,
        )
        similarity = annotation["similarity"]
        score_text = "" if similarity is None else f" {similarity:.3f}"
        cv2_module.putText(
            image,
            f"{annotation['label']}{score_text}",
            (max(0, x1), max(18, y1 - 5)),
            cv2_module.FONT_HERSHEY_SIMPLEX,
            0.5,
            annotation["color"],
            2,
        )
        if annotation["test_outcome"] is not None:
            cv2_module.putText(
                image,
                annotation["test_outcome"],
                (max(0, x1), min(height - 4, max(18, y1 + 18))),
                cv2_module.FONT_HERSHEY_SIMPLEX,
                0.5,
                RED,
                2,
            )
    if not annotations:
        message = verification_result.get("reason", "No valid face detected.")
        if verification_result.get("identity_status") == "REVIEW_REQUIRED":
            message = "REVIEW REQUIRED"
        cv2_module.putText(
            image,
            str(message),
            (10, max(22, height - 14)),
            cv2_module.FONT_HERSHEY_SIMPLEX,
            0.6,
            YELLOW,
            2,
        )
    return image, annotations


def adapter_result_to_candidate_result(
    adapter_result: dict[str, Any],
) -> dict[str, Any]:
    """Keep each adapter prediction, score, and box attached to its face."""
    matched_faces = []
    unmatched_faces = []
    for face in adapter_result.get("face_results", []):
        observation = {
            "face_number": face.get("face_number"),
            "box": face.get("box"),
            "similarity": face.get("similarity"),
        }
        if face.get("predicted") == "MATCH":
            matched_faces.append(observation)
        elif face.get("predicted") == "MISMATCH":
            unmatched_faces.append({
                **observation,
                "status": "CANDIDATE_UNMATCHED",
            })
        else:
            unmatched_faces.append({
                **observation,
                "status": "REVIEW_REQUIRED",
            })
    return {
        "candidate_id": adapter_result.get("candidate_id"),
        "identity_status": adapter_result.get("identity_status"),
        "total_faces": adapter_result.get("total_faces", 0),
        "matched_faces": matched_faces,
        "unmatched_faces": unmatched_faces,
        "timings_ms": adapter_result.get("timings_ms", {}),
    }


@dataclass
class LiveMetrics:
    frame_count: int = 0
    detected_face_count: int = 0
    student_match_count: int = 0
    volunteer_unmatched_count: int = 0
    false_acceptance_count: int = 0
    false_rejection_count: int = 0
    both_labels_correct_count: int = 0
    processing_latencies_ms: list[float] = field(default_factory=list)
    face_detection_latencies_ms: list[float] = field(default_factory=list)
    embedding_extraction_latencies_ms: list[float] = field(default_factory=list)
    similarity_latencies_ms: list[float] = field(default_factory=list)
    verification_latencies_ms: list[float] = field(default_factory=list)
    face_scores: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        verification_result: dict[str, Any],
        latency_ms: float,
        frame_width: int,
        candidate_side: str = "right",
    ) -> list[dict[str, Any]]:
        annotations = face_annotations(
            verification_result,
            frame_width,
            candidate_side,
        )
        self.frame_count += 1
        self.detected_face_count += int(verification_result.get("total_faces", 0))
        self.processing_latencies_ms.append(float(latency_ms))
        timings = verification_result.get("timings_ms", {})
        for key, target in (
            ("face_detection_ms", self.face_detection_latencies_ms),
            ("embedding_extraction_ms", self.embedding_extraction_latencies_ms),
            ("similarity_comparison_ms", self.similarity_latencies_ms),
            ("verification_total_ms", self.verification_latencies_ms),
        ):
            if key in timings:
                target.append(float(timings[key]))

        student = [
            face for face in annotations
            if face["ground_truth_role"] == "student"
        ]
        volunteer = [
            face for face in annotations
            if face["ground_truth_role"] == "volunteer"
        ]
        self.student_match_count += sum(
            face["system_status"] == "CANDIDATE_IDENTIFIED"
            for face in student
        )
        self.volunteer_unmatched_count += sum(
            face["system_status"] == "CANDIDATE_UNMATCHED"
            for face in volunteer
        )
        self.false_acceptance_count += sum(
            face["system_status"] == "CANDIDATE_IDENTIFIED"
            for face in volunteer
        )
        self.false_rejection_count += sum(
            face["system_status"] == "CANDIDATE_UNMATCHED"
            for face in student
        )
        if (
            len(annotations) == 2
            and len(student) == 1
            and len(volunteer) == 1
            and student[0]["system_status"] == "CANDIDATE_IDENTIFIED"
            and volunteer[0]["system_status"] == "CANDIDATE_UNMATCHED"
        ):
            self.both_labels_correct_count += 1

        self.face_scores.extend(
            {
                "frame": self.frame_count,
                "face_number": face["face_number"],
                "ground_truth_role": face["ground_truth_role"],
                "system_status": face["system_status"],
                "similarity": face["similarity"],
            }
            for face in annotations
        )
        return annotations

    @property
    def average_latency_ms(self) -> float:
        if not self.processing_latencies_ms:
            return 0.0
        return sum(self.processing_latencies_ms) / len(self.processing_latencies_ms)

    @property
    def processing_fps(self) -> float:
        if self.average_latency_ms == 0.0:
            return 0.0
        return 1000.0 / self.average_latency_ms

    @staticmethod
    def _average(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @property
    def average_face_detection_ms(self) -> float:
        return self._average(self.face_detection_latencies_ms)

    @property
    def average_embedding_extraction_ms(self) -> float:
        return self._average(self.embedding_extraction_latencies_ms)

    @property
    def average_similarity_ms(self) -> float:
        return self._average(self.similarity_latencies_ms)

    @property
    def average_verification_ms(self) -> float:
        return self._average(self.verification_latencies_ms)


def show_registration_preview() -> None:
    import cv2

    with CameraManager() as camera:
        print("Preview only; no frames will be enrolled. Press Q to close.", flush=True)
        while True:
            frame = camera.read()
            cv2.imshow("D.2 registration preview - Q closes", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()


def show_verification_preview(
    candidate_id: str,
    database_path: str | Path = CANDIDATE_DATABASE_PATH,
) -> None:
    import cv2

    storage = CandidateStorage(database_path)
    candidate = storage.load_candidate(candidate_id)
    if candidate is None or candidate["candidate_id"] != candidate_id:
        raise KeyError(f"Registered candidate not found: {candidate_id}")
    with CameraManager() as camera:
        print(
            "Session 2 preview only; no face processing or samples. "
            "Press Q to close.",
            flush=True,
        )
        while True:
            frame = camera.read()
            cv2.imshow("D.2 verification preview - Q closes", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()


def hold_failed_verification_frame(
    cv2_module: Any,
    camera: CameraManager,
) -> None:
    """Release the camera but keep the failed result visible until dismissed."""
    camera.release()
    print(
        "Verification stopped. Inspect the final classified frame; "
        "press Q or ESC to close it.",
        flush=True,
    )
    while True:
        key = cv2_module.waitKey(50) & 0xFF
        if key in {ord("q"), 27}:
            return


def run_live_session(
    camera: CameraManager,
    face_adapter: FaceIdentityAdapter,
    registered_candidate: dict[str, Any],
    cv2_module: Any,
    *,
    candidate_side: str = "right",
    max_frames: int = LIVE_FRAME_COUNT,
) -> LiveMetrics:
    metrics = LiveMetrics()
    with camera as opened_camera:
        print(
            "Live test: keep the enrolled student on the configured side and "
            "the consenting volunteer on the other; press Q to stop.",
            flush=True,
        )
        while metrics.frame_count < max_frames:
            frame_started = time.perf_counter()
            frame = opened_camera.read()
            try:
                adapter_result = face_adapter.verify_registered_identity(
                    registered_candidate,
                    frame,
                )
                result = adapter_result_to_candidate_result(adapter_result)
            except (TypeError, ValueError) as exc:
                result = {
                    "candidate_id": registered_candidate["candidate_id"],
                    "identity_status": "REVIEW_REQUIRED",
                    "reason": str(exc),
                    "total_faces": 0,
                    "matched_faces": [],
                    "unmatched_faces": [],
                }
            shown, annotations = annotate_frame(
                cv2_module,
                frame,
                result,
                candidate_side,
            )
            total_frame_ms = (time.perf_counter() - frame_started) * 1000.0
            metrics.record(
                result,
                total_frame_ms,
                frame.shape[1],
                candidate_side,
            )
            for face in annotations:
                score = face["similarity"]
                print(
                    f"FRAME={metrics.frame_count} face={face['face_number']} "
                    f"ground_truth_role={face['ground_truth_role']} "
                    f"system={face['system_status']} "
                    f"label={face['label']} "
                    f"test_outcome={face['test_outcome'] or 'NONE'} "
                    f"similarity={score if score is not None else 'n/a'} "
                    f"box={face['box']}",
                    flush=True,
                )
            if metrics.false_acceptance_count or metrics.false_rejection_count:
                cv2_module.imshow(FAILED_VERIFICATION_WINDOW, shown)
                print(
                    "STOPPING: the measured frame contains a false acceptance "
                    "or false rejection.",
                    flush=True,
                )
                hold_failed_verification_frame(cv2_module, opened_camera)
                break
            cv2_module.imshow(LIVE_VERIFICATION_WINDOW, shown)
            if cv2_module.waitKey(1) & 0xFF == ord("q"):
                break

    return metrics


def run_enrollment(
    candidate_id: str,
    database_path: str | Path = CANDIDATE_DATABASE_PATH,
) -> bool:
    import cv2

    storage = CandidateStorage(database_path)
    if storage.load_candidate(candidate_id) is not None:
        raise ValueError(
            f"Candidate ID already exists; refusing to overwrite: {candidate_id}"
        )

    face_adapter = FaceIdentityAdapter()
    service = CandidateIdentityService(
        face_adapter,
        storage,
        threshold=FACE_IDENTITY_THRESHOLD,
    )
    try:
        with CameraManager() as camera:
            accepted, rejected = capture_enrollment_frames(
                camera,
                face_adapter,
                cv2,
            )
        print(
            f"ENROLLMENT_TOTAL accepted={len(accepted)} rejected={len(rejected)} "
            f"attempted={ENROLLMENT_FRAME_COUNT}",
            flush=True,
        )
        if len(accepted) != ENROLLMENT_FRAME_COUNT:
            print(
                "ENROLLMENT_FAILED: all three frames must be accepted; "
                "live verification will not start.",
                flush=True,
            )
            return False

        registration = service.register_candidate(
            candidate_id,
            "D2 authorized test student",
            accepted,
            overwrite=False,
            expected_sample_count=ENROLLMENT_FRAME_COUNT,
        )
        if registration["valid_sample_count"] != ENROLLMENT_FRAME_COUNT:
            print(
                "ENROLLMENT_FAILED: registration service did not accept all "
                "three samples; live verification will not start.",
                flush=True,
            )
            return False
        stored = storage.load_candidate(candidate_id)
        if (
            stored is None
            or stored["candidate_id"] != candidate_id
            or stored["embedding"].shape != (128,)
        ):
            raise RuntimeError("Registered embedding could not be validated.")
        if not np.isclose(
            float(np.linalg.norm(stored["embedding"])),
            1.0,
            atol=1e-5,
        ):
            raise RuntimeError("Stored reference embedding is not normalized.")
        print(
            f"REGISTRATION_SUCCEEDED candidate_id={candidate_id} "
            f"accepted=3 rejected={len(rejected)} "
            "stored_embedding_valid=true raw_images_saved=false",
            flush=True,
        )
        return True
    finally:
        face_adapter.landmarker.close()
        cv2.destroyAllWindows()


def run_verification(
    candidate_id: str,
    candidate_side: str,
    max_frames: int = LIVE_FRAME_COUNT,
    database_path: str | Path = CANDIDATE_DATABASE_PATH,
) -> LiveMetrics:
    import cv2

    storage = CandidateStorage(database_path)
    registered_candidate = storage.load_candidate(candidate_id)
    if (
        registered_candidate is None
        or registered_candidate["candidate_id"] != candidate_id
    ):
        raise KeyError(f"Registered candidate not found: {candidate_id}")

    face_adapter = FaceIdentityAdapter()
    try:
        return run_live_session(
            CameraManager(),
            face_adapter,
            registered_candidate,
            cv2,
            candidate_side=candidate_side,
            max_frames=max_frames,
        )
    finally:
        face_adapter.landmarker.close()
        cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--enrollment-preview",
        action="store_true",
        help="show the Session 1 camera preview only",
    )
    mode.add_argument(
        "--enroll",
        action="store_true",
        help="capture three enrollment attempts after the Session 1 preview approval",
    )
    mode.add_argument(
        "--verification-preview",
        action="store_true",
        help="show the Session 2 camera preview only for a registered candidate",
    )
    mode.add_argument(
        "--verify",
        action="store_true",
        help="run measured live verification after Session 2 preview approval",
    )
    parser.add_argument("--candidate-id")
    parser.add_argument("--max-frames", type=int, default=LIVE_FRAME_COUNT)
    parser.add_argument(
        "--candidate-side",
        choices=("left", "right"),
        default="right",
    )
    args = parser.parse_args()

    if args.enrollment_preview:
        show_registration_preview()
        return 0
    if args.enroll:
        if not args.candidate_id:
            parser.error("--candidate-id is required with --enroll")
        return 0 if run_enrollment(args.candidate_id) else 1
    if not args.candidate_id:
        parser.error("--candidate-id is required with verification modes")
    if args.verification_preview:
        show_verification_preview(args.candidate_id)
        return 0
    if args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    metrics = run_verification(
        args.candidate_id,
        args.candidate_side,
        args.max_frames,
    )
    failed = (
        metrics.frame_count == 0
        or metrics.false_acceptance_count > 0
        or metrics.false_rejection_count > 0
        or metrics.both_labels_correct_count == 0
    )
    print(
        f"LIVE_RESULT={'FAIL' if failed else 'PASS'} "
        "LIVE_SUMMARY "
        f"candidate_id={args.candidate_id} "
        f"threshold={FACE_IDENTITY_THRESHOLD:.4f} "
        f"frames={metrics.frame_count} "
        f"faces={metrics.detected_face_count} "
        f"student_matches={metrics.student_match_count} "
        f"volunteer_unmatched={metrics.volunteer_unmatched_count} "
        f"false_acceptances={metrics.false_acceptance_count} "
        f"false_rejections={metrics.false_rejection_count} "
        f"both_correct_frames={metrics.both_labels_correct_count} "
        f"face_detection_ms={metrics.average_face_detection_ms:.3f} "
        f"embedding_extraction_ms={metrics.average_embedding_extraction_ms:.3f} "
        f"similarity_ms={metrics.average_similarity_ms:.3f} "
        f"verification_ms={metrics.average_verification_ms:.3f} "
        f"average_latency_ms={metrics.average_latency_ms:.3f} "
        f"processing_fps={metrics.processing_fps:.3f}",
        flush=True,
    )
    for score in metrics.face_scores:
        print(f"FACE_SCORE {score}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
